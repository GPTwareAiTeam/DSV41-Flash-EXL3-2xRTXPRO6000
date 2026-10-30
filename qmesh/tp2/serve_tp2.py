"""OpenAI-compatible HTTP front end for the two-rank TP2 generator.

Launch one process per rank with the same distributed environment as run_tp2.
Only rank zero listens for HTTP; a single scheduler drives both ranks in lockstep.
"""
from __future__ import annotations

import copy
import hmac
import json
import os
import queue
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("DSPARK", "1")
os.environ.setdefault("NUM_DRAFT", "5")
os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/.cache/torch_extensions")

import torch
import torch.distributed as dist
from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import ComboSampler
from qmesh.tp2 import run_tp2 as tp
import content_seed
from encoding import encode_messages, eos_token, parse_message_from_completion_text

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-v4.1-flash")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
API_KEY = os.environ.get("API_KEY", "")


class AgreedSampler(ComboSampler):
    def forward(self, *args, **kwargs):
        token = super().forward(*args, **kwargs)
        dist.broadcast(token, src=0)
        return token


def setup():
    if tp.DSPARK_POLICY not in ("always", "auto"):
        raise ValueError("DSPARK_POLICY must be always or auto")
    if tp.DSPARK_POLICY == "auto" and (not tp.DSPARK or tp.BATCH != 1 or tp.DSPARK_DYNAMIC):
        raise ValueError("DSPARK_POLICY=auto requires DSPARK=1, BATCH=1 and DSPARK_DYNAMIC=0")
    torch.cuda.set_device(tp.RANK)
    dist.init_process_group(backend="nccl", rank=tp.RANK, world_size=tp.WORLD,
                            init_method=os.environ["INIT_METHOD"])
    device = torch.device("cuda", tp.RANK)
    config = Config.from_directory(os.path.join(tp.OVERLAY, f"rank{tp.RANK}"))
    if config.tp_size != tp.WORLD or config.tp_rank != tp.RANK:
        raise ValueError("checkpoint TP layout does not match launch")
    model = Model.from_config(config)
    max_chunk, _ = tp.resolve_max_chunk(device)
    agreed = torch.tensor(max_chunk, dtype=torch.int32, device=device)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    max_chunk = int(agreed.item())
    cache = Cache(model, max_num_tokens=tp.CACHE_TOKENS, max_history=tp.MAX_HISTORY,
                  max_batch_size=tp.BATCH, **tp.KV_CACHE_KWARGS)
    model.load(device=device, max_chunk_size=max_chunk, max_output_size=tp.MAX_NEW,
               max_batch_size=tp.BATCH, progressbar=False)
    if tp.COMM in ("p2p", "hybrid"):
        from exllamav3.tp2.p2p import P2PComm
        from qmesh.tp2.prefill_ar import wrap_prefill_ar
        p2p = tp.TPBackendP2P(P2PComm(tp.RANK, world=tp.WORLD,
                                      rendezvous_path=os.environ["P2P_RDV"], device=tp.RANK,
                                      stats=tp.P2P_STATS_OUT is not None))
        vocab_backend = p2p if tp.COMM == "p2p" else tp.TPBackendHybrid(tp.TPBackendNCCL(), p2p)
        backend = wrap_prefill_ar(vocab_backend, p2p)
    elif tp.COMM == "nccl":
        if os.environ.get("PREFILL_AR", "0") != "0":
            raise ValueError("PREFILL_AR requires COMM=p2p or hybrid")
        backend = vocab_backend = tp.TPBackendNCCL()
    else:
        raise ValueError("COMM must be p2p, hybrid or nccl")
    if os.environ.get("TP_COLLECTIVE_TRACE") == "1":
        backend = tp.TPBackendTrace(backend)
    tp.set_tp_reduce(model, backend)
    tp.wrap_vocab_split(model, tp.RANK, tp.WORLD, vocab_backend)
    tokenizer = Tokenizer.from_config(config)
    draft_model = draft_cache = None
    if tp.DSPARK:
        draft_model = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=tp.CACHE_TOKENS,
                            max_history=tp.MAX_HISTORY, max_batch_size=tp.BATCH, **tp.KV_CACHE_KWARGS)
        draft_model.load(device=device, progressbar=False)
        tp.set_tp_reduce(draft_model, backend)
        original = draft_model.sample_from_state
        def sample(state, params):
            ids = original(state, params)
            dist.broadcast(ids, src=0)
            if "draft_confidence_len" in params:
                length = torch.tensor(params["draft_confidence_len"], dtype=torch.int32, device=device)
                dist.broadcast(length, src=0)
                params["draft_confidence_len"] = int(length.item())
            if "draft_conf" in params:
                dist.broadcast(params["draft_conf"], src=0)
            return ids
        draft_model.sample_from_state = sample
    generator = Generator(model, cache, tokenizer, max_batch_size=tp.BATCH, max_chunk_size=max_chunk,
                          draft_model=draft_model, draft_cache=draft_cache,
                          recurrent_cache_size=tp.PREFIX_CACHE_CAPACITY if tp.PREFIX_CACHE else 0,
                          recurrent_checkpoint_interval=tp.PREFIX_CACHE_INTERVAL if tp.PREFIX_CACHE else None,
                          recurrent_checkpoint_interval_pp=tp.PREFIX_CACHE_INTERVAL if tp.PREFIX_CACHE else 32768,
                          num_draft_tokens=tp.NUM_DRAFT if tp.DSPARK else None,
                          dynamic_draft_tokens=tp.DSPARK and tp.DSPARK_DYNAMIC,
                          draft_confidence=tp.DSPARK_TARGET_ACC,
                          record_draft_stats=tp.DSPARK and (tp.DSPARK_DYNAMIC or tp.DSPARK_POLICY == "auto"))
    generator.prefix_cache_min_match = tp.PREFIX_CACHE_MIN_MATCH if tp.PREFIX_CACHE else None
    generator.prefill_chunk_limit = tp.PrefillChunkLimiter(device) if tp.PREFILL_ADAPTIVE else None
    generator.observe_draft_confidence = tp.DSPARK_POLICY == "auto"
    generator.count_draft_forwards = tp.DSPARK_COUNT_FORWARDS
    generator.draft_forward_count = 0
    if isinstance(backend, tp.TPBackendTrace):
        original_trace_iterate = generator.iterate
        def trace_iterate(*args, **kwargs):
            backend.step += 1
            tp.log(f"COLLECTIVE_STEP rank={tp.RANK} step={backend.step} sequence={backend.sequence}")
            return original_trace_iterate(*args, **kwargs)
        generator.iterate = trace_iterate
    dist.barrier()
    if tp.WARM_TOKENS:
        warm_text = encode_messages([{"role": "user", "content": "Hello"}], thinking_mode="chat")
        warm_ids = tokenizer.encode(warm_text, add_bos=False, encode_special_tokens=True)
        for _ in range(tp.BATCH):
            generator.enqueue(Job(input_ids=warm_ids, max_new_tokens=tp.WARM_TOKENS,
                                  min_new_tokens=tp.WARM_TOKENS, sampler=AgreedSampler(temperature=0),
                                  stop_conditions=[]))
        while generator.num_remaining_jobs():
            for _ in generator.iterate():
                pass
    tp.log("SERVER_READY")
    return generator, tokenizer


def prepare(data, chat):
    if not isinstance(data, dict):
        raise ValueError("request must be an object")
    if data.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError("unknown model")
    if data.get("n", 1) != 1:
        raise ValueError("only n=1 is supported")
    max_tokens = data.get("max_completion_tokens", data.get("max_tokens", tp.MAX_NEW))
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    temperature = float(data.get("temperature", 1.0))
    top_p = float(data.get("top_p", 1.0))
    presence = float(data.get("presence_penalty", 0.0))
    frequency = float(data.get("frequency_penalty", 0.0))
    if not 0 <= temperature <= 2 or not 0 < top_p <= 1 or not -2 <= presence <= 2 or not -2 <= frequency <= 2:
        raise ValueError("invalid sampling parameters")
    stop = data.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or len(stop) > 4 or any(not isinstance(s, str) or not s for s in stop):
        raise ValueError("stop must be a string or up to four nonempty strings")
    tools = data.get("tools") or []
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    if chat:
        messages = copy.deepcopy(data.get("messages"))
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty array")
        if tools:
            definitions = tools
            if messages[0].get("role") == "system":
                messages[0]["tools"] = definitions
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": definitions})
        prompt, media = encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)
        if media["images"]:
            raise ValueError("image input is not supported")
    else:
        if tools:
            raise ValueError("tools are only supported for chat completions")
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
    seed = data.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("seed must be an integer")
    return {"prompt": prompt, "chat": chat, "tools": bool(tools), "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "presence": presence, "frequency": frequency,
            "stop": stop, "seed": seed, "stream": bool(data.get("stream", False)),
            "include_usage": bool((data.get("stream_options") or {}).get("include_usage", False))}


class Engine:
    def __init__(self, generator, tokenizer):
        self.generator = generator
        self.tokenizer = tokenizer
        self.pending = queue.Queue()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.failure = None
        self.on_failure = None

    def submit(self, request):
        events = queue.Queue()
        with self.lock:
            if self.failure is not None or self.stopping.is_set():
                return None, self.failure or "Server is shutting down"
            self.pending.put((request, events))
        return events, None

    def stop(self):
        with self.lock:
            self.stopping.set()

    def validate_request(self, request):
        ids = self.tokenizer.encode(request["prompt"], add_bos=False, encode_special_tokens=True)
        if ids.numel() == 0:
            raise ValueError("prompt must contain at least one token")
        if ids.shape[-1] + request["max_tokens"] + 1 + self.generator.max_draft_tokens > self.generator.max_total_tokens:
            raise ValueError("prompt and requested completion exceed cache capacity")
        request["input_ids"] = ids[0].tolist()

    def run(self):
        active = {}
        try:
            while True:
                entries = []
                command = None
                if tp.RANK == 0:
                    if self.stopping.is_set() and not active and self.pending.empty():
                        command = {"shutdown": True}
                    else:
                        if not active:
                            try:
                                entries.append(self.pending.get(timeout=0.1))
                            except queue.Empty:
                                continue
                        while len(entries) + len(active) < tp.BATCH:
                            try:
                                entries.append(self.pending.get_nowait())
                            except queue.Empty:
                                break
                        command = {"requests": [e[0] for e in entries]}
                payload = [command]
                dist.broadcast_object_list(payload, src=0, device=torch.device("cuda", tp.RANK))
                command = payload[0]
                if command.get("shutdown"):
                    break
                for index, request in enumerate(command["requests"]):
                    ids = torch.tensor([request["input_ids"]], dtype=torch.long)
                    sampler = AgreedSampler(temperature=request["temperature"], top_p=request["top_p"],
                                             pres_p=request["presence"], freq_p=request["frequency"])
                    job = Job(input_ids=ids, max_new_tokens=request["max_tokens"], sampler=sampler,
                              seed=request["seed"], stop_conditions=[self.tokenizer.eos_token_id] + request["stop"],
                              decode_special_tokens=request["tools"])
                    if tp.DSPARK_POLICY == "auto":
                        policy_class, policy_ema = content_seed.seed_acceptance(request["prompt"])
                        request["policy_ema"] = policy_ema
                        request["policy_off"] = policy_ema < content_seed.POLICY_THRESHOLD
                        request["policy_stats"] = 0
                        self.generator.num_draft_tokens = 0 if request["policy_off"] else tp.NUM_DRAFT
                        tp.log(f"POLICY class={policy_class} seed_al={policy_ema:.3f} speculative={not request['policy_off']}")
                    self.generator.enqueue(job)
                    active[job.serial_number] = (request, entries[index][1] if tp.RANK == 0 else None, job)
                if active:
                    for result in self.generator.iterate():
                        entry = active.get(result.get("serial"))
                        if entry is None:
                            continue
                        request, events, job = entry
                        if tp.DSPARK_POLICY == "auto" and not request["policy_off"] and len(job.draft_stats) > request["policy_stats"]:
                            for _, window, accepted in job.draft_stats[request["policy_stats"]:]:
                                predicted = 1 + window * self.generator.last_draft_confidence
                                observed = (1 - content_seed.POLICY_CONF_WEIGHT) * (1 + accepted) + content_seed.POLICY_CONF_WEIGHT * predicted
                                request["policy_ema"] += (observed - request["policy_ema"]) / content_seed.POLICY_EMA_WINDOW
                            request["policy_stats"] = len(job.draft_stats)
                            if request["policy_ema"] < content_seed.POLICY_THRESHOLD:
                                request["policy_off"] = True
                                self.generator.num_draft_tokens = 0
                                tp.log(f"POLICY_OFF generated={job.new_tokens} ema_al={request['policy_ema']:.3f}")
                        if tp.RANK == 0:
                            if "error" in result:
                                events.put({"error": str(result["error"])})
                            else:
                                item = {k: result[k] for k in ("text", "eos", "eos_reason", "new_tokens", "prompt_tokens", "cached_tokens") if k in result}
                                if "token_ids" in result:
                                    item["token_ids"] = result["token_ids"][0].tolist()
                                events.put(item)
                        if result.get("eos") or "error" in result:
                            del active[result["serial"]]
        except BaseException as exc:
            if tp.RANK == 0:
                with self.lock:
                    self.failure = str(exc) or type(exc).__name__
                    notified = set()
                    for _, events, *_ in list(active.values()) + entries:
                        if events is not None and id(events) not in notified:
                            events.put({"error": self.failure})
                            notified.add(id(events))
                    while True:
                        try:
                            _, events = self.pending.get_nowait()
                            events.put({"error": self.failure})
                        except queue.Empty:
                            break
                if self.on_failure is not None:
                    self.on_failure()
            raise
        finally:
            dist.destroy_process_group()


def usage(result):
    prompt = result.get("prompt_tokens", 0)
    completion = result.get("new_tokens", 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": result.get("cached_tokens", 0)}}


def message_for(text, tools):
    if tools and "<｜DSML｜ calls" in text:
        try:
            message = parse_message_from_completion_text(text + eos_token, "chat")
            for call in message["tool_calls"]:
                call["id"] = "call_" + uuid.uuid4().hex
            if message["tool_calls"]:
                message["content"] = message["content"] or None
                message.pop("reasoning_content", None)
                return message
        except (AssertionError, ValueError, KeyError, IndexError):
            pass
    return {"role": "assistant", "content": text}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        if not API_KEY or hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + API_KEY):
            return True
        self.send_json(401, {"error": {"message": "Invalid API key", "type": "authentication_error"}})
        self.close_connection = True
        return False

    def do_GET(self):
        if not self.authorized():
            return
        if self.path.rstrip("/") == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}]})
        else:
            self.send_json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})

    def do_POST(self):
        if not self.authorized():
            return
        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self.send_json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("invalid Content-Length")
            data = json.loads(self.rfile.read(length))
            request = prepare(data, path == "/v1/chat/completions")
            self.server.engine.validate_request(request)
        except Exception as exc:
            self.send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return
        events, error = self.server.engine.submit(request)
        if error is not None:
            self.send_json(503, {"error": {"message": error, "type": "server_error"}})
            return
        ident = ("chatcmpl-" if request["chat"] else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        base = {"id": ident, "object": "chat.completion" if request["chat"] else "text_completion", "created": created, "model": MODEL_ID}
        stream = request["stream"]
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
        def emit(obj):
            self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
            self.wfile.flush()
        def chunk(delta, finish=None):
            choice = {"index": 0, "finish_reason": finish}
            if request["chat"]:
                choice["delta"] = delta
            else:
                choice.update(text=delta, logprobs=None)
            emit({**base, "object": "chat.completion.chunk" if request["chat"] else "text_completion", "choices": [choice]})
        text, tokens = "", []
        try:
            if stream and request["chat"]:
                chunk({"role": "assistant"})
            while True:
                result = events.get()
                if "error" in result:
                    if stream:
                        emit({"error": {"message": result["error"], "type": "server_error"}})
                        self.wfile.write(b"data: [DONE]\n\n")
                    else:
                        self.send_json(500, {"error": {"message": result["error"], "type": "server_error"}})
                    return
                delta = result.get("text", "")
                text += delta
                tokens.extend(result.get("token_ids", []))
                if stream and delta and not request["tools"]:
                    chunk({"content": delta} if request["chat"] else delta)
                if result.get("eos"):
                    break
            message = message_for(text, request["tools"]) if request["chat"] else None
            finish = "tool_calls" if message and message.get("tool_calls") else ("length" if result.get("eos_reason") == "max_new_tokens" else "stop")
            if stream:
                if request["tools"]:
                    if message.get("tool_calls"):
                        calls = [{**call, "index": i} for i, call in enumerate(message["tool_calls"])]
                        if message.get("content"):
                            chunk({"content": message["content"]})
                        chunk({"tool_calls": calls})
                    elif text:
                        chunk({"content": text})
                chunk({} if request["chat"] else "", finish)
                if request["include_usage"]:
                    emit({**base, "object": "chat.completion.chunk" if request["chat"] else "text_completion", "choices": [], "usage": usage(result)})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                choice = {"index": 0, "finish_reason": finish, "token_ids": tokens}
                if request["chat"]:
                    choice["message"] = message
                    choice["logprobs"] = None
                else:
                    choice.update(text=text, logprobs=None)
                self.send_json(200, {**base, "choices": [choice], "usage": usage(result)})
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    generator, tokenizer = setup()
    engine = Engine(generator, tokenizer)
    if tp.RANK != 0:
        engine.run()
        return
    tp.log(f"SERVER listening on http://{HOST}:{PORT}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.engine = engine
    server.daemon_threads = True
    engine.on_failure = lambda: threading.Thread(target=server.shutdown, daemon=True).start()
    thread = threading.Thread(target=engine.run, name="tp2-scheduler")
    thread.start()
    def shutdown(signum, frame):
        engine.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever()
    finally:
        engine.stop()
        server.server_close()
        thread.join()
    if engine.failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

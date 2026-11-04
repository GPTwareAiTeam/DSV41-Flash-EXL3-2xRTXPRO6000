"""OpenAI-compatible HTTP front end for the P7-hot / GX-cold seam pair.

One process, one GPU: the pair runs a single model instance on the P7 card whose cold experts are
computed by the GX box over the RDMA seam. Every construct the 2xRTX reference needed for two ranks
-- the `RANK == 0` guards, the per-step object broadcast, the agreed sampler and the process-group
teardown -- has no counterpart here and is not ported. What is ported is the wire: request
validation, SSE with `data: [DONE]`, usage in a final choices=[] event when asked, the
finish_reason rules (length / stop / tool_calls), token_ids in the non-streaming choice, 400 on a
bad request, 503 on a submit failure and 401 with API_KEY.

Batch 1 is the measured configuration. max_batch_size=BATCH is a knob rather than a rewrite (the
scheduler below keeps the reference's pending/active structure), but the seam's batched path has
not been validated: any BATCH>1 has to be measured before it is quoted.
"""
from __future__ import annotations

import copy
import hmac
import json
import os
import queue
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Set before exllamav3 is imported, which reads them at import time. These are the pair's shipping
# choices (qmesh/release/pair_arm.sh's client list passes the same values); a launcher may still
# override any of them with an explicit -e.
os.environ.setdefault("DSPARK", "1")
os.environ.setdefault("NUM_DRAFT", "5")
os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
os.environ.setdefault("REMOTE_TRANSPORT", "seam")

# The chat encoder is not part of exllamav3: it ships with the checkpoint (same path and usage as
# qmesh/mem_budget.py:66 and :691-697).
ENC_DIR = os.environ.get("ENC_DIR", "/models/dsv41/DeepSeek-V4.1-Flash-1017-orig/encoding")
if ENC_DIR not in sys.path:
    sys.path.insert(0, ENC_DIR)
from encoding import encode_messages, eos_token, parse_message_from_completion_text

import torch
from exllamav3 import Cache, CacheLayer_quant, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import ComboSampler
from qmesh.prefill_limit import PREFILL_ADAPTIVE, PrefillChunkLimiter, resolve_max_chunk
from qmesh import placement as seam_placement
from qmesh import prefix_cache

CHECKPOINT = os.environ.get("CHECKPOINT", "/models/dsv41/exl3-2.8535-p7hot-hc1")
DRAFT_CHECKPOINT = os.environ.get("DRAFT_CHECKPOINT", CHECKPOINT)
SEAM_PEER = os.environ.get("SEAM_PEER", "10.0.100.2")
SEAM_PORT = int(os.environ.get("SEAM_PORT", "47311"))
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
MAX_CHUNK_RAW = os.environ.get("MAX_CHUNK", "8192")
DSPARK = os.environ.get("DSPARK") == "1"
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "5"))
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", "65536"))
BATCH = int(os.environ.get("BATCH", "1"))
WARM_TOKENS = int(os.environ.get("WARM_TOKENS", "8"))
# The first prefill of a fresh container is ~31 % slower than the second whatever the config, and an
# ablation over every on-disk cache the container owns (Triton kernels, exllamav3's autotune file,
# torch's NVRTC cubins) cut it from 12.9 s to ~6 s and no further: the first chunk of a *process*
# pays one-time buffer growth plus a gap between enqueue and the first chunk timer, and both are
# absent from every later request (notes/HalfWire.md, "Which cache the first arm is paying for").
# So startup does one chunk at the load-time row count and one short call per seam bucket, which
# moves that cost behind SERVER_READY. WARM_PREFILL=0 turns it off; WARM_PREFILL_ROWS=0 skips only
# the big chunk (the cheap buckets still run, since they cover the decode path).
WARM_PREFILL = os.environ.get("WARM_PREFILL", "1") != "0"
WARM_PREFILL_ROWS = int(os.environ.get("WARM_PREFILL_ROWS", "0"))  # 0: the load-time row count
# KV_BITS=4 (default) is DeepSeek's trained FP4 layout: whole 512-wide latent rows plus indexer
# keys, as the model was trained. 8 packs the nope part with CacheLayer_quant, 16 is the fp16 pool.
# Same three-way pick as the shipping client (qmesh/e2e_prompts.py:85-92); the pair ships 4.
KV_BITS = int(os.environ.get("KV_BITS", "4"))
assert KV_BITS in (4, 8, 16), "KV_BITS must be 4, 8 or 16"
KV_CACHE_KWARGS = ({} if KV_BITS == 16 else
                   {"kv_format": "fp4"} if KV_BITS == 4 else
                   {"layer_type": CacheLayer_quant, "k_bits": 8, "v_bits": 8})

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-v4.1-flash-pair")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
API_KEY = os.environ.get("API_KEY", "")


def log(message):
    print(message, flush=True)


def setup():
    # Knob typos raise here, not after a twelve-minute hot-pack load: generator_kwargs() below is the
    # first consumer of the prefix-cache settings and it runs past model.load().
    prefix_cache.check()
    placement, placement_path, placement_sha = seam_placement.load(CHECKPOINT)
    # Printed before any model work: the seam handshake refuses a server whose Hello advertises
    # another table, so this number either matches the server's or names the mismatch here.
    log(f"CLIENT_PLACEMENT path={placement_path} sha={placement_sha[:16]} ckpt={CHECKPOINT}")
    hot = seam_placement.hot(placement)
    config = Config.from_directory(CHECKPOINT)
    config.infer_params.moe_cpu_split = 1
    config.infer_params.moe_remote_hot = hot
    config.infer_params.moe_remote_warm = seam_placement.warm(placement)
    config.infer_params.remote_expert_host = {"host": SEAM_PEER, "port": SEAM_PORT}
    model = Model.from_config(config)
    # Recipe A sizes its hybrid expert pools during load, so the cache has to exist first. The KV
    # layout is the pair's shipped one (KV_BITS=4, DeepSeek's trained FP4 pools), picked above so an
    # arm that asks for another layout gets it instead of a server that quietly ignores the knob.
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_history=5, **KV_CACHE_KWARGS)
    t0 = time.time()
    model.load(device="cuda:0", progressbar=False)
    free, total = torch.cuda.mem_get_info(0)
    log(f"P7_LOAD_OK {time.time() - t0:.1f}s "
        f"vram={torch.cuda.memory_allocated(0) / 2**30:.2f}GiB "
        f"free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB")

    tokenizer = Tokenizer.from_config(config)
    draft_model = draft_cache = None
    if DSPARK:
        draft_config = Config.from_directory(DRAFT_CHECKPOINT)
        draft_model = Model.from_config(draft_config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=CACHE_TOKENS, max_history=5,
                            **KV_CACHE_KWARGS)
        t1 = time.time()
        draft_model.load(device="cuda:0", progressbar=False)
        log(f"DSPARK_LOADED {time.time() - t1:.1f}s")

    # The load-time chunk is sized once the weights are resident, against the room this card has
    # left (MAX_CHUNK=auto hands the pick to resolve_max_chunk instead of a fixed number).
    device = torch.device("cuda:0")
    rows = resolve_max_chunk(device, MAX_CHUNK_RAW)
    log(f"MAX_CHUNK_EFFECTIVE rows={rows} raw={MAX_CHUNK_RAW}")

    generator = Generator(model, cache, tokenizer, max_batch_size=BATCH, max_chunk_size=rows,
                          draft_model=draft_model, draft_cache=draft_cache,
                          num_draft_tokens=NUM_DRAFT if DSPARK else None,
                          **prefix_cache.generator_kwargs())
    prefix_cache.apply(generator)
    log(prefix_cache.summary())
    if PREFILL_ADAPTIVE:
        generator.prefill_chunk_limit = PrefillChunkLimiter(device)
    warm_text = encode_messages([{"role": "user", "content": "Hello"}], thinking_mode="chat")
    warm_ids = tokenizer.encode(warm_text, add_bos=False, encode_special_tokens=True)
    if WARM_TOKENS:
        for _ in range(BATCH):
            generator.enqueue(Job(input_ids=warm_ids, max_new_tokens=WARM_TOKENS,
                                  min_new_tokens=WARM_TOKENS, sampler=ComboSampler(temperature=0),
                                  stop_conditions=[]))
        while generator.num_remaining_jobs():
            for _ in generator.iterate():
                pass
    if WARM_PREFILL:
        # One call per seam bucket, so every expert-row width the wire will see is compiled and its
        # buffers grown here rather than inside the first request. The buckets are the placement's
        # (1, 6, 64, 8192 by default); WARM_PREFILL_ROWS names the big one, capped by what the
        # request cache could hold and by the chunk the generator was built with.
        big = min(WARM_PREFILL_ROWS or rows, rows, max(CACHE_TOKENS - MAX_NEW, 0))
        filler = warm_ids * (max(rows, 64) // max(len(warm_ids), 1) + 2)
        for n_rows in [n for n in (1, 6, 64, big) if n and n <= len(filler)]:
            t_warm = time.time()
            generator.enqueue(Job(input_ids=filler[:n_rows], max_new_tokens=1,
                                  min_new_tokens=1, sampler=ComboSampler(temperature=0),
                                  stop_conditions=[]))
            while generator.num_remaining_jobs():
                for _ in generator.iterate():
                    pass
            log(f"WARM_PREFILL_OK rows={n_rows} s={time.time() - t_warm:.2f}")
    log("SERVER_READY")
    return generator, tokenizer


def prepare(data, chat):
    if not isinstance(data, dict):
        raise ValueError("request must be an object")
    if data.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError("unknown model")
    if data.get("n", 1) != 1:
        raise ValueError("only n=1 is supported")
    max_tokens = data.get("max_completion_tokens", data.get("max_tokens", MAX_NEW))
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
        entries = []
        try:
            while True:
                entries = []
                # One scheduler thread owns the generator: with one process there is no second rank
                # to keep in lockstep, so the pending queue is read and the jobs enqueued here and
                # only here.
                if self.stopping.is_set() and not active and self.pending.empty():
                    break
                if not active:
                    try:
                        entries.append(self.pending.get(timeout=0.1))
                    except queue.Empty:
                        continue
                while len(entries) + len(active) < BATCH:
                    try:
                        entries.append(self.pending.get_nowait())
                    except queue.Empty:
                        break
                # pending carries (request, events) tuples -- the events queue rides along so the
                # failure path can tell a waiter its job will never run. Unpack once here: the first
                # cut indexed the tuple as if it were the request, so every request died with
                # "tuple indices must be integers or slices, not str" and the server answered 503.
                for entry in entries:
                    request, events = entry
                    ids = torch.tensor([request["input_ids"]], dtype=torch.long)
                    sampler = ComboSampler(temperature=request["temperature"], top_p=request["top_p"],
                                           pres_p=request["presence"], freq_p=request["frequency"])
                    job = Job(input_ids=ids, max_new_tokens=request["max_tokens"], sampler=sampler,
                              seed=request["seed"],
                              stop_conditions=[self.tokenizer.eos_token_id] + request["stop"],
                              decode_special_tokens=request["tools"])
                    self.generator.enqueue(job)
                    active[job.serial_number] = (request, events, job)
                if active:
                    for result in self.generator.iterate():
                        entry = active.get(result.get("serial"))
                        if entry is None:
                            continue
                        request, events, job = entry
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
            # A scheduler that dies leaves the HTTP server accepting requests nobody will serve, so
            # every waiter is told and the server is shut down rather than left limping.
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
    log(f"SERVER listening on http://{HOST}:{PORT}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.engine = engine
    server.daemon_threads = True
    engine.on_failure = lambda: threading.Thread(target=server.shutdown, daemon=True).start()
    thread = threading.Thread(target=engine.run, name="pair-scheduler")
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

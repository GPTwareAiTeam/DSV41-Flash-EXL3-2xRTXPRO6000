#!/usr/bin/env python3
"""Offline checks on the pair server's request contract: what it accepts, what it refuses, what it reports.

The server is the OpenAI-compatible surface shipped for item 3, and the failures that reach a GPU slot are
exactly the ones a pure function can catch here: a rejected-then-accepted field, a `usage` block that never
reports `cached_tokens` (which is what makes the prefix cache measurable through /v1), a tool-call message
builder that raises instead of falling back, and the scheduler's `(request, events)` enqueue contract -- the
tuple shape a fix at 751f47ed changed. All of these are decided before any token is generated, so none of
them needs a pair, a worker, or a GPU.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python3 -u qmesh/tests/serve_pair_contract.py
  # expect: PASS lines and exit 0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from qmesh import serve_pair as srv  # noqa: E402

FAILS = []


def check(name, cond, detail = ""):
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' ' + detail) if detail else ''}", flush = True)
    if not cond:
        FAILS.append(name)


def refuses(name, fn, needle):
    try:
        fn()
    except ValueError as exc:
        check(name, needle in str(exc), f"msg={str(exc)[:80]}")
        return
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"wrong exception {type(exc).__name__}: {exc}")
        return
    check(name, False, "no exception raised")


# ---- prepare(): the accepted shape and every documented refusal --------------------------------------
def prepare_accepts():
    req = srv.prepare({"prompt": "hello"}, chat = False)
    check("prepare.defaults",
          req["max_tokens"] == srv.MAX_NEW and req["temperature"] == 1.0 and req["top_p"] == 1.0 and
          req["stream"] is False and req["include_usage"] is False and req["tools"] is False and
          req["stop"] == [] and req["seed"] is None, f"={ {k: req[k] for k in ('max_tokens', 'temperature')} }")
    req = srv.prepare({"prompt": "x", "max_tokens": 8, "max_completion_tokens": 3, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "stop": "END", "seed": 7}, chat = False)
    check("prepare.completion_tokens_wins", req["max_tokens"] == 3, f"={req['max_tokens']}")
    check("prepare.stop_string_to_list", req["stop"] == ["END"], f"={req['stop']}")
    check("prepare.stream_flags", req["stream"] is True and req["include_usage"] is True)
    check("prepare.seed", req["seed"] == 7)


def prepare_refusals():
    refuses("prepare.refuse_non_object", lambda: srv.prepare([], chat = False), "must be an object")
    refuses("prepare.refuse_unknown_model",
            lambda: srv.prepare({"model": "other", "prompt": "x"}, chat = False), "unknown model")
    refuses("prepare.refuse_n2", lambda: srv.prepare({"prompt": "x", "n": 2}, chat = False), "only n=1")
    for bad, label in ((0, "zero"), (-1, "negative"), (True, "bool"), ("10", "string")):
        refuses(f"prepare.refuse_max_tokens_{label}",
                lambda bad = bad: srv.prepare({"prompt": "x", "max_tokens": bad}, chat = False),
                "positive integer")
    refuses("prepare.refuse_temperature",
            lambda: srv.prepare({"prompt": "x", "temperature": 3}, chat = False), "invalid sampling")
    refuses("prepare.refuse_top_p",
            lambda: srv.prepare({"prompt": "x", "top_p": 0}, chat = False), "invalid sampling")
    refuses("prepare.refuse_stop_five",
            lambda: srv.prepare({"prompt": "x", "stop": ["a", "b", "c", "d", "e"]}, chat = False),
            "up to four")
    refuses("prepare.refuse_stop_empty_string",
            lambda: srv.prepare({"prompt": "x", "stop": [""]}, chat = False), "up to four")
    refuses("prepare.refuse_prompt_type",
            lambda: srv.prepare({"prompt": 5}, chat = False), "prompt must be a string")
    refuses("prepare.refuse_seed_type",
            lambda: srv.prepare({"prompt": "x", "seed": "1"}, chat = False), "seed must be an integer")
    refuses("prepare.refuse_tools_on_completions",
            lambda: srv.prepare({"prompt": "x", "tools": [{"type": "function"}]}, chat = False),
            "only supported for chat")


def prepare_chat():
    req = srv.prepare({"messages": [{"role": "user", "content": "hi"}]}, chat = True)
    check("prepare.chat_prompt_built", isinstance(req["prompt"], str) and req["prompt"], "nonempty")
    refuses("prepare.refuse_no_messages", lambda: srv.prepare({"messages": []}, chat = True),
            "nonempty array")
    refuses("prepare.refuse_image",
            lambda: srv.prepare({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}, chat = True),
            "image input is not supported")
    tool = {"type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}
    req = srv.prepare({"messages": [{"role": "user", "content": "weather?"}], "tools": [tool]}, chat = True)
    check("prepare.chat_tools_flag", req["tools"] is True)
    check("prepare.chat_tools_in_prompt", "get_weather" in req["prompt"])
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    req = srv.prepare({"messages": messages, "tools": [tool]}, chat = True)
    check("prepare.chat_tools_into_system", "get_weather" in req["prompt"])
    # the tools block is attached to a copy: a retry of the same dict must not see the previous call's tools
    check("prepare.chat_does_not_mutate_caller",
          messages[0] == {"role": "system", "content": "be brief"} and len(messages) == 2, f"={messages[0]}")


# ---- usage()/message_for(): what a client actually reads back -----------------------------------------
def usage_block():
    u = srv.usage({"prompt_tokens": 8192, "new_tokens": 8, "cached_tokens": 7936})
    check("usage.cached_tokens_nested", u["prompt_tokens_details"]["cached_tokens"] == 7936, f"={u}")
    check("usage.total_is_sum", u["total_tokens"] == 8200, f"={u['total_tokens']}")
    u = srv.usage({"prompt_tokens": 4, "new_tokens": 1})
    check("usage.missing_cached_is_zero", u["prompt_tokens_details"]["cached_tokens"] == 0, f"={u}")
    u = srv.usage({})
    check("usage.empty_result", u["prompt_tokens"] == 0 and u["completion_tokens"] == 0, f"={u}")


def tool_messages():
    plain = srv.message_for("just text", tools = False)
    check("message.no_tools_plain", plain == {"role": "assistant", "content": "just text"}, f"={plain}")
    fallback = srv.message_for("a call_ but no tool block", tools = True)
    check("message.malformed_tool_text_falls_back",
          fallback["role"] == "assistant" and fallback["content"] == "a call_ but no tool block",
          f"={fallback}")
    fallback = srv.message_for("", tools = True)
    check("message.empty_text_plain", fallback == {"role": "assistant", "content": ""}, f"={fallback}")


# ---- Engine: enqueue contract and the 503 condition ----------------------------------------------------
class StubTokenizer:
    """encode() as validate_request uses it: one batched tensor, numel for the emptiness test."""
    def __init__(self, n):
        self.ids = torch.tensor([list(range(1, n + 1))]) if n else torch.zeros((1, 0), dtype = torch.long)

    def encode(self, prompt, add_bos = False, encode_special_tokens = True):
        return self.ids


class StubGenerator:
    max_draft_tokens = 2

    def __init__(self, total = 4096):
        self.max_total_tokens = total


def engine_contract():
    gen = StubGenerator()
    eng = srv.Engine(gen, StubTokenizer(8))
    request = {"prompt": "x", "max_tokens": 4}
    events, err = eng.submit(request)
    check("engine.submit_accepts", err is None and events is not None)
    entry = eng.pending.get_nowait()
    check("engine.pending_is_request_events_pair",
          isinstance(entry, tuple) and len(entry) == 2 and entry[0] is request and entry[1] is events,
          f"type={type(entry).__name__}")
    check("engine.pending_pair_unpack",
          (lambda request_, events_: request_ is request and events_ is events)(*entry),
          "the fix at 751f47ed")
    eng.stop()
    events, err = eng.submit({"prompt": "x", "max_tokens": 1})
    check("engine.stopped_returns_503_text",
          events is None and "shutting down" in str(err).lower(), f"err={err}")


def engine_validation():
    eng = srv.Engine(StubGenerator(), StubTokenizer(8))
    req = {"prompt": "x", "max_tokens": 4}
    eng.validate_request(req)
    check("validate.sets_input_ids", req["input_ids"] == [1, 2, 3, 4, 5, 6, 7, 8], f"={req['input_ids']}")
    empty = srv.Engine(StubGenerator(), StubTokenizer(0))
    refuses("validate.refuse_empty_prompt",
            lambda: empty.validate_request({"prompt": "", "max_tokens": 4}), "at least one token")
    # a real capacity refusal: long prompt, room in the cache for the tokens but not for the completion
    refuses("validate.refuse_over_capacity",
            lambda: srv.Engine(StubGenerator(total = 4096), StubTokenizer(4000)).validate_request(
                {"prompt": "x", "max_tokens": 256}), "exceed cache capacity")
    tight = srv.Engine(StubGenerator(total = 10), StubTokenizer(8))
    refuses("validate.refuse_too_long",
            lambda: tight.validate_request({"prompt": "x", "max_tokens": 4}), "exceed cache capacity")


def main():
    prepare_accepts()
    prepare_refusals()
    prepare_chat()
    usage_block()
    tool_messages()
    engine_contract()
    engine_validation()
    print(f"RESULT {'PASS' if not FAILS else 'FAIL'} fails={len(FAILS)}", flush = True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

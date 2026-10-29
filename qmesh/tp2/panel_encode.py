"""Panel rendering for the sudoingX/dgx-spark-deepseek-v41 quality panel (qmesh/tp2).

Vendored from the upstream scripts so the ids and the scored mask match the published panel exactly:
  scripts/quantize/caldata/build_panel.py  (row filter, --panel-max truncation, index fields)
  scripts/quantize/caldata/build.py::make_encoder  (encoder + assistant-only score mask)
The special token strings are taken from the checkpoint's own encoding module rather than retyped, so the
mask cannot drift from the encoder that renders the text.
Kept free of exllamav3 imports so it can be exercised on CPU without the JIT extension.
"""
from __future__ import annotations

import copy
import os
import sys
import urllib.request
import zipfile

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/dsv41/DeepSeek-V4.1-Flash-1017-orig")
WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKITEXT_ZIP = os.environ.get("WIKITEXT_ZIP", "/artifacts/quality/wikitext-2-raw-v1.zip")


def make_encoder(model_dir: str = MODEL_DIR):
    """(raw tokenizer, encode(conversation) -> (ids, score)) as the upstream build.py builds it."""
    sys.path.insert(0, os.path.join(model_dir, "encoding"))
    from encoding import ASSISTANT_SP_TOKEN, encode_messages, eos_token, SYSTEM_SP_TOKEN, USER_SP_TOKEN
    from tokenizers import Tokenizer as RawTokenizer
    tok = RawTokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    # build.py's sp entries used by the score mask: turn boundaries reset the mask, the assistant
    # header opens it (the header itself is a prompt token), and the message end closes it
    prompt_tokens = (tok.token_to_id(USER_SP_TOKEN), tok.token_to_id(SYSTEM_SP_TOKEN))
    assistant_start, msg_end = tok.token_to_id(ASSISTANT_SP_TOKEN), tok.token_to_id(eos_token)
    assert all(t is not None for t in prompt_tokens + (assistant_start, msg_end))

    def encode(r):
        msgs = copy.deepcopy(r["messages"])
        if r.get("tools"):
            if msgs[0]["role"] != "system":
                msgs.insert(0, {"role": "system", "content": ""})
            msgs[0]["tools"] = r["tools"]            # the encoder renders tools only on a system message
        mode = "thinking" if r["thinking"] else "chat"
        effort = {"low": 50, "high": 75}.get(r["effort"]) if r["thinking"] else None
        text = encode_messages(msgs, thinking_mode = mode, reasoning_effort = effort)
        ids = tok.encode(text, add_special_tokens = False).ids
        score, inside = [], False
        for t in ids:
            if t in prompt_tokens:
                inside = False
            score.append(1 if inside else 0)
            if t == assistant_start:
                inside = True                        # the header itself is a prompt token
            elif t == msg_end:
                inside = False
        return ids, score

    return tok, encode


def load_wikitext2_raw(zip_path: str = WIKITEXT_ZIP) -> str:
    """The raw wikitext-2-raw test split, byte-identical to llama.cpp's perplexity.c input."""
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            return zf.read("wikitext-2-raw/wiki.test.raw").decode("utf-8")
    # No local copy: fetch to a scratch path (only the test split is read)
    path = "/tmp/wikitext-2-raw-v1.zip"
    if not os.path.exists(path):
        urllib.request.urlretrieve(WIKITEXT_URL, path)
    with zipfile.ZipFile(path, "r") as zf:
        return zf.read("wikitext-2-raw/wiki.test.raw").decode("utf-8")

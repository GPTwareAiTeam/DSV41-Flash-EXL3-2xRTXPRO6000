"""Prompt-side lexical classification and tunables for the DSpark on/off policy.

Class-specific acceptance-length estimates seed the online policy; no model call is needed.
"""

import os
import re

POLICY_THRESHOLD = float(os.environ.get("DSPARK_POLICY_THRESHOLD", "2.55"))
POLICY_EMA_WINDOW = float(os.environ.get("DSPARK_POLICY_EMA_WINDOW", "8"))
POLICY_CONF_WEIGHT = float(os.environ.get("DSPARK_POLICY_CONF_WEIGHT", "0.25"))
POLICY_SEED_PROSE = float(os.environ.get("DSPARK_POLICY_SEED_PROSE", "2.6"))
POLICY_SEED_CODE = float(os.environ.get("DSPARK_POLICY_SEED_CODE", "4.0"))
POLICY_SEED_JSON = float(os.environ.get("DSPARK_POLICY_SEED_JSON", "2.4"))
POLICY_CODE_MIN_SIGNALS = float(os.environ.get("DSPARK_POLICY_CODE_MIN_SIGNALS", "2"))
POLICY_FENCE_WEIGHT = float(os.environ.get("DSPARK_POLICY_FENCE_WEIGHT", "1"))
POLICY_BRACE_WEIGHT = float(os.environ.get("DSPARK_POLICY_BRACE_WEIGHT", "1"))
POLICY_KEYWORD_WEIGHT = float(os.environ.get("DSPARK_POLICY_KEYWORD_WEIGHT", "1"))
POLICY_FEATURES = set(os.environ.get("DSPARK_POLICY_FEATURES", "fence,braces,keyword,json").split(","))

_FENCE = re.compile(r"```")
_BRACES = re.compile(r"[{}\[\]]")
_CODE_KW = re.compile(r"\b(def|return|class|import|function|const|let|var)\b", re.IGNORECASE)
_JSON_KEY = re.compile(r'"[A-Za-z_][\w ]*"\s*:')


def classify(prompt: str) -> str:
    """Return 'json', 'code', or 'prose' for a prompt string."""
    if "json" in POLICY_FEATURES and (_JSON_KEY.search(prompt) or re.search(r"\bJSON\b", prompt, re.IGNORECASE)):
        return "json"
    signals = (
        (POLICY_FENCE_WEIGHT if "fence" in POLICY_FEATURES and _FENCE.search(prompt) else 0)
        + (POLICY_BRACE_WEIGHT if "braces" in POLICY_FEATURES and len(_BRACES.findall(prompt)) >= 2 else 0)
        + (POLICY_KEYWORD_WEIGHT if "keyword" in POLICY_FEATURES and _CODE_KW.search(prompt) else 0)
    )
    return "code" if signals >= POLICY_CODE_MIN_SIGNALS else "prose"

def seed_acceptance(prompt: str) -> tuple[str, float]:
    cls = classify(prompt)
    return cls, {"prose": POLICY_SEED_PROSE, "code": POLICY_SEED_CODE, "json": POLICY_SEED_JSON}[cls]

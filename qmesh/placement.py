"""The one cold-expert placement resolver: every loader and the seam client read it here.

A pack ships `PACK_MANIFEST.json`, whose `placement` names the file the pack was cut against, and that
file's sha256 is the `placement_sha` the GX seam server puts in its Hello. So the file a client loads
must be the pack's, not a repo default: the repo default is the 2.8535 pack's table, and against the
4620 packs it makes the client translate picks with the wrong per-layer cold count (layer 0: 259 cold
there, 248 here), which the server correctly refuses as "cold ids out of range" and then fail-stops on
(2026-11-01 18:03, GX2 arm A prefill: 2 of 384 picks at 251 for 248 cold -- exactly the old table's
range, while the same run's identity phase, which resolved through qmesh/e2e_prompts.py, was clean).

Order: an explicit PLACEMENT wins (operators overriding a pack on purpose), then the pack's manifest,
then the repo default. `load()` also publishes its sha through `current_sha()`, so a client can refuse
a server whose Hello advertises a different table *before* it sends the first request.
"""
from __future__ import annotations

import hashlib
import json
import os

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placement.final.json")
MANIFEST = "PACK_MANIFEST.json"

_last = {"path": None, "sha": None}


def _manifest_placement(ckpt):
    """The placement path named by the pack's manifest, absolute, or None when the pack ships none."""
    mpath = os.path.join(ckpt, MANIFEST)
    try:
        with open(mpath) as fh:
            p = json.load(fh).get("placement")
    except (OSError, ValueError):
        return None
    if not isinstance(p, str) or not p:
        return None
    # Packs record it absolute; a relative entry is relative to the pack root, then to the manifest.
    for cand in (p, os.path.join(ckpt, p), os.path.join(os.path.dirname(mpath), p)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def resolve(ckpt = None):
    """Path of the placement this process must use for `ckpt` (default: $CHECKPOINT)."""
    explicit = os.environ.get("PLACEMENT")
    if explicit:
        return explicit
    ckpt = ckpt or os.environ.get("CHECKPOINT")
    if ckpt:
        p = _manifest_placement(ckpt)
        if p:
            return p
    return DEFAULT


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load(ckpt = None):
    """(placement dict, path, sha256 hex) for `ckpt`; remembers the sha for the seam handshake."""
    path = resolve(ckpt)
    digest = sha(path)
    with open(path) as fh:
        placement = json.load(fh)
    _last["path"], _last["sha"] = path, digest
    return placement, path, digest


def hot(placement):
    """The per-module hot list keyed the way the model's modules are: {"layers.N.ffn": [expert ids]}."""
    return {f"layers.{layer}.ffn": v["hot"] for layer, v in placement["layers"].items()}

def warm(placement):
    """The per-module cold list **in the file's order**, keyed like hot().

    The seam's cold-local ids are positions in the server's `layers.N.warm`, so the client must
    number its cold experts the same way. Ascending order happened to agree with the unranked
    file and silently stops agreeing with a reordered (ranked) one: every cold id then names a
    different expert on the two sides, which the transport cannot detect. block_sparse_mlp_cpu
    therefore prefers this list and asserts hot + warm is a permutation of 0..E-1.
    """
    return {f"layers.{layer}.ffn": v["warm"] for layer, v in placement["layers"].items()
            if "warm" in v}


def current_sha():
    """sha256 of the placement this process loaded, or None before any load."""
    return _last["sha"]


def current_path():
    return _last["path"]

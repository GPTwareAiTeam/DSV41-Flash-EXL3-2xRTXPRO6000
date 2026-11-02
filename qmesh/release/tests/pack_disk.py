#!/usr/bin/env python3
"""Battery for pair_pack_disk.py -- expert packs live on the drive of the box that serves them.

Owner rule 2026-11-02: P7 hot packs on the P7's own drive, GX cold packs on the GX's own disk, never
across the network. The checker is what refuses a release whose pack does not satisfy it, so its two
refusals (a mount that names the 12 TB drive, and a path or symlink that leaves the allowed tree) are
pinned here: one unit case per rule with probe stubbed, then one end-to-end case over a real
directory that has a link pointing outside the allowed root.

No container, GPU, ssh or queue is touched.
"""
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PAIR_QUEUE_ROOT", "/tmp/packdiskq")
_s = importlib.util.spec_from_file_location("pd", os.path.join(HERE, "pair_pack_disk.py"))
pd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pd)

fails = 0


def check(name, cond, detail=""):
    global fails
    print(("OK   " if cond else "FAIL ") + name + (" | " + detail if detail else ""))
    if not cond:
        fails += 1


def probe_d(real="/models/dsv41/pack", fstype="ext4", links=(), missing=None, failed=None):
    return {"real": real, "fstype": fstype, "links": list(links), "missing": missing,
            "real_failed": failed, "rc": 0, "err": ""}


def p7_spec(mounts):
    return {"mounts": mounts}


def gx_spec(mounts):
    return {"mounts": mounts}

GOOD_P7 = p7_spec([["__PAIR_ROOT__", "/pair"], ["/models", "/models"], ["/usr/local/cuda", "/opt/cuda"]])
GOOD_GX = gx_spec([["__PAIR_ROOT__", "/pair"], ["__GX_HOME__/exl3-cold-pack-4460", "/cold"]])


def run(p7spec, p7probe, gxspec=None, gxprobe=None, placement=None):
    """-> (fails, printed) with spec and probe stubbed."""
    pd.pc.spec = lambda cfg, side: p7spec if side == "p7" else (gxspec if gxspec else GOOD_GX)
    pd.manifest_refs = lambda pack: (placement, None)

    def fake_probe(host, path):
        return p7probe if host is None else (gxprobe or probe_d("/home/x/exl3-cold-pack-4460"))
    pd.probe = fake_probe
    out = []
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        pd.check_p7({"pair": "dev"}, out, True)
        pd.check_gx({"pair": "dev", "gx_host": "10.0.4.33"}, out, True)
    return out, buf.getvalue()


def reasons(fs):
    return " | ".join(f"{s}:{w}:{p}:{r}" for s, w, p, r in fs)


# 1. the spec must not hand the pair the 12 TB drive
fs, _ = run(p7_spec(GOOD_P7["mounts"] + [["/mnt/data12/models/exl3", "/mnt/data12/models/exl3"]]),
            probe_d(), placement="/models/dsv41/placement.json")
check("a P7 mount naming the 12 TB drive is refused",
      any(w == "mount" and p == "/mnt/data12/models/exl3" and "12 TB" in r for s, w, p, r in fs), reasons(fs))

# 2. /models must be the drive itself, not a bind of a subdirectory of someone else's tree
fs, _ = run(p7_spec([["__PAIR_ROOT__", "/pair"], ["/mnt/data12/models", "/models"]]), probe_d())
check("a /models mount from another disk is refused",
      any(w == "mount" and p == "/models" for s, w, p, r in fs), reasons(fs))

# 3. the pack itself, and everything a link inside it resolves to, stays under the P7 root
fs, _ = run(GOOD_P7, probe_d(real="/mnt/data12/models/pack"))
check("a pack that resolves outside the P7 drive is refused",
      any(w == "pack" and "/mnt/data12/models/pack" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7,
            probe_d(links=[("/models/dsv41/pack/model-00041.safetensors",
                            "/mnt/data12/models/exl3/DeepSeek/model-00041.safetensors")]))
check("a pack's link into the 12 TB drive is refused",
      any(w == "link" and "/mnt/data12" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(links=[("/models/dsv41/pack/x.safetensors", "/models/dsv41/serve/x.safetensors")]))
check("a link that stays on the P7 drive passes", not fs, reasons(fs))

# 4. a pack on a network filesystem is refused even when its path looks local
fs, _ = run(GOOD_P7, probe_d(fstype="nfs4"))
check("a pack on an NFS mount is refused", any("sits on nfs4" in r for s, w, p, r in fs), reasons(fs))

# 5. the manifest's placement profile is read at serving time and is a pack path too
fs, _ = run(GOOD_P7, probe_d(real="/mnt/x"), placement="/mnt/data12/packs/placement.json")
check("a placement profile outside the P7 drive is refused",
      any(w == "placement" for s, w, p, r in fs), reasons(fs))

# 6. the GX half: /cold on the box's own disk, never /mnt (P7 over NFS) and never NFS at all
fs, _ = run(GOOD_P7, probe_d(),
            gx_spec(GOOD_GX["mounts"] + [["/mnt/data12/models/exl3", "/mnt/data12/models/exl3"]]))
check("a GX mount naming /mnt is refused", any(s == "gx" and w == "mount" for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(),
            gxprobe=probe_d(real="/mnt/data12/models/exl3-cold-pack-4460"))
check("a GX pack under /mnt is refused",
      any(s == "gx" and w == "pack" and "under /mnt" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), gxprobe=probe_d(fstype="nfs"))
check("a GX pack on NFS is refused", any(s == "gx" and "sits on nfs" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), gxprobe=probe_d(missing="/home/x/pack"))
check("a GX pack that does not exist is refused", any(s == "gx" and w == "pack" for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d())
check("a local GX pack with no links passes", not fs, reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), gxprobe=probe_d(links=[("/home/x/pack/a", "/net/host/pack/a")]))
check("a GX link into /net is refused", any(s == "gx" and w == "link" for s, w, p, r in fs), reasons(fs))

# 7. end to end over a real tree: the shipped hot pack carries links, and a link into the 12 TB drive
#    (which is what the 4460 pack did through its overlay before 2026-11-02) must be caught by the
#    filesystem walk, not only by a stubbed probe.
W = tempfile.mkdtemp(prefix="packdisk.")
try:
    old_allow, old_pack = pd.P7_ALLOW, pd.P7_PACK
    root = os.path.join(W, "models", "dsv41")
    pack = os.path.join(root, "pack")
    outside = os.path.join(W, "data12", "models", "exl3")
    for d in (pack, outside):
        os.makedirs(d)
    open(os.path.join(pack, "model-00001-of-00041.safetensors"), "w").write("bytes")
    open(os.path.join(outside, "model-00041-of-00041.safetensors"), "w").write("bytes")
    os.symlink(os.path.join(outside, "model-00041-of-00041.safetensors"),
               os.path.join(pack, "model-00041-of-00041.safetensors"))
    import json
    open(os.path.join(pack, "PACK_MANIFEST.json"), "w").write(json.dumps(
        {"placement": os.path.join(root, "packs", "placement.json"), "overlay": outside}))
    os.makedirs(os.path.join(root, "packs"))
    open(os.path.join(root, "packs", "placement.json"), "w").write("{}")
    pd.P7_ALLOW, pd.P7_PACK = root, pack
    pd.pc.spec = lambda cfg, side: GOOD_P7 if side == "p7" else GOOD_GX
    pd.probe = lambda host, path: probe_d("/home/x/exl3-cold-pack-4460")
    real_probe = pd.probe
    # p7 paths are read locally by the checker itself, so restore the real probe for host None only.
    sys.path.insert(0, HERE)
    import pair_pack_disk as _real
    pd.probe = lambda host, path: _real.probe(None, path) if host is None else real_probe(None, path)
    fs = []
    with contextlib.redirect_stdout(io.StringIO()):
        pd.check_p7({"pair": "dev"}, fs, True)
    check("a real symlink out of the pack is refused by the walk",
          any(w == "link" and os.path.join(outside, "model-00041") in r for s, w, p, r in fs), reasons(fs))
    os.unlink(os.path.join(pack, "model-00041-of-00041.safetensors"))
    open(os.path.join(pack, "model-00041-of-00041.safetensors"), "w").write("bytes")
    fs = []
    with contextlib.redirect_stdout(io.StringIO()):
        pd.check_p7({"pair": "dev"}, fs, True)
    check("the same pack with its bytes local passes", not fs, reasons(fs))
    pd.P7_ALLOW, pd.P7_PACK = old_allow, old_pack
finally:
    shutil.rmtree(W, ignore_errors=True)

print("\nRESULT %s fails=%d" % ("PASS" if not fails else "FAIL", fails))
sys.exit(1 if fails else 0)

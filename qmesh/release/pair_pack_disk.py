#!/usr/bin/env python3
"""pair_pack_disk.py -- expert packs live on the drive of the box that serves them.

    pair_pack_disk.py <pair> [--quiet]

Owner rule (2026-11-02): "gx10 weights go on the 4TB NVMe of the GX10s, gpu0/gpu1 weights go on the
4TB drive of the P7." A pack served over the network makes every decode step wait for a filesystem,
and on this hardware it already did: GX1 mounts /mnt/data10 and /mnt/data12 from P7 over NFS, so a
pack under /mnt is P7's disk being read across the wire by the GX half.

Checked against the spec the pair actually runs (containers/<pair>-<side>.json), so this check can
not drift from what the containers mount:

  p7  no mount may name /mnt -- the 12 TB drive is not the pair's, and the containers stop mounting
      it (a mount in the spec that names it is refused here). /models must be the drive itself.
      Then the hot pack, the placement profile its PACK_MANIFEST.json names, and every symlink
      inside those must resolve under /models/dsv41, on a local filesystem. The manifest's `overlay`
      is provenance (the tree the pack was built from, not read by a server), so it is printed but
      not walked: a pack file that still points into it is caught by the link check, which is exact,
      where walking the overlay would flag files nothing loads.

Prints PACK_DISK lines, then PACK_DISK_OK, or PACK_DISK_FAIL side=.. what=.. path=.. reason=..
Exit 0 clean / 3 refused / 2 usage. PAIR_PACK_DISK=0 is for stub harnesses only (no packs to see).
"""
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pair_containers as pc  # config parser and spec loader: one source of truth

P7_ALLOW = os.environ.get("PAIR_PACK_P7_ROOT", "/models/dsv41")
P7_PACK = os.environ.get("PAIR_PACK_P7", os.path.join(P7_ALLOW, "exl3-p7hot-4460-hc1"))
# Filesystems that put the bytes somewhere other than the box's own disk. virtiofs is a VM's local
# disk handed through, so it is not in the list; an NFS mount of a local disk is.
REMOTE_FS = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "9p", "sshfs", "fuse.sshfs", "ceph", "glusterfs",
             "lustre", "gpfs", "afs", "ncpfs", "vboxsf"}

# One round trip per path: realpath, the filesystem it lands on, and every symlink below it (a link
# is how a pack names a file that lives somewhere else -- if that somewhere is NFS, the pack is not
# local). find -type l does not follow links, so the walk stays inside the tree.
PROBE = r"""
p=$1
[ -e "$p" ] || { printf 'MISSING %s\n' "$p"; exit 0; }
r=$(readlink -f -- "$p") || { printf 'REALFAIL %s\n' "$p"; exit 0; }
printf 'REAL %s\n' "$r"
ft=$(findmnt -T "$r" -no FSTYPE 2>/dev/null | head -1)
[ -n "$ft" ] || ft=$(stat -f -c %T "$r" 2>/dev/null)
printf 'FSTYPE %s\n' "${ft:-unknown}"
if [ -d "$p" ]; then
  find "$p" -type l -exec sh -c 'printf "LINK %s\t%s\n" "$1" "$(readlink -f -- "$1" 2>/dev/null || echo MISSING)"' _ {} \; 2>/dev/null
fi
"""


def ssh_argv():
    if os.environ.get("PAIR_DRY") == "1" and os.environ.get("PAIR_DRY_SSH"):
        return [os.environ["PAIR_DRY_SSH"]]
    return shlex.split(os.environ.get("PAIR_SSH") or "ssh")


def probe(host, path):
    """-> {real, fstype, links: [(name, target)]} or {missing|real_failed: path}."""
    if host is None:
        cmd = ["sh", "-c", PROBE, "_", path]
    else:
        cmd = ssh_argv() + pc.SSH_OPTS + [host, "sh -c " + shlex.quote(PROBE) + " _ " + shlex.quote(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = {"real": None, "fstype": "unknown", "links": [], "missing": None, "real_failed": None,
           "rc": p.returncode, "err": (p.stderr or "").strip()[:300]}
    for line in p.stdout.splitlines():
        if line.startswith("REAL "):
            out["real"] = line[5:]
        elif line.startswith("FSTYPE "):
            out["fstype"] = line[7:]
        elif line.startswith("LINK "):
            name, _, target = line[5:].partition("\t")
            out["links"].append((name, target))
        elif line.startswith("MISSING "):
            out["missing"] = line[8:]
        elif line.startswith("REALFAIL "):
            out["real_failed"] = line[9:]
    if out["rc"] != 0 and not p.stdout:
        out["real_failed"] = path
    return out


def remote_fs(fstype):
    f = (fstype or "").lower()
    return f in REMOTE_FS or f.startswith("nfs") or f.startswith("fuse.sshfs") or f.startswith("cifs")


def under(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


def manifest_refs(pack):
    """(placement, overlay) from the pack's manifest, both absolute paths or None.

    `placement` is read at serving time (the client resolves CLIENT_PLACEMENT through it), so it is a
    pack path like any other. `overlay` is provenance -- the tree the pack was built from -- and is
    not read by a server; a pack file that still points into it is caught by the link check instead,
    which is exact, where walking the whole overlay would flag files nothing loads.
    """
    man = os.path.join(pack, "PACK_MANIFEST.json")
    try:
        d = json.load(open(man))
    except (OSError, ValueError):
        return None, None
    placement = d.get("placement") if isinstance(d.get("placement"), str) else None
    overlay = d.get("overlay") if isinstance(d.get("overlay"), str) else None
    return (placement or None), (overlay or None)


def check_p7(cfg, fails, quiet):
    spec_ = pc.spec(cfg, "p7")
    models_src = None
    for src, dst in spec_.get("mounts", []):
        if src.startswith("/mnt/") or dst.startswith("/mnt/"):
            fails.append(("p7", "mount", dst,
                          f"the pair must not mount the 12 TB drive ({src}); packs live on the "
                          f"serving box's own disk (owner rule 2026-11-02)"))
        if dst == "/models":
            models_src = src
    if models_src is None:
        fails.append(("p7", "mount", "/models", "the spec mounts no /models: the hot pack is unreachable"))
    elif models_src != "/models":
        fails.append(("p7", "mount", "/models", f"mounted from {models_src}, not from the drive itself"))

    pr = probe(None, P7_PACK)
    placement, overlay = manifest_refs(P7_PACK)
    roots = [("pack", P7_PACK)] + ([("placement", placement)] if placement else [])
    nlinks = 0
    for what, path in roots:
        r = pr if path == P7_PACK else probe(None, path)
        if r["missing"]:
            fails.append(("p7", what, path, "does not exist on P7"))
            continue
        if r["real_failed"] or not r["real"]:
            fails.append(("p7", what, path, f"cannot be resolved ({r['err'] or 'readlink failed'})"))
            continue
        if not under(r["real"], P7_ALLOW):
            fails.append(("p7", what, path,
                          f"resolves to {r['real']}, outside {P7_ALLOW} (packs live on the P7 4 TB drive)"))
        if remote_fs(r["fstype"]):
            fails.append(("p7", what, path, f"sits on {r['fstype']}: a pack must not be served over the network"))
        for name, target in r["links"]:
            nlinks += 1
            if target == "MISSING":
                fails.append(("p7", "link", name, "is a dangling link (its target does not exist)"))
            elif not under(target, P7_ALLOW):
                fails.append(("p7", "link", name,
                              f"points at {target}, outside {P7_ALLOW} (a pack's bytes stay on the drive)"))
    if not quiet:
        print(f"PACK_DISK p7 pack={P7_PACK} real={pr['real'] or '-'} fstype={pr['fstype']} "
              f"placement={placement or '-'} overlay={overlay or '-'} links={nlinks}")


def check_gx(cfg, fails, quiet):
    spec_ = pc.spec(cfg, "gx")
    src = None
    for s, dst in spec_.get("mounts", []):
        if dst == "/cold":
            src = s
        elif s.startswith("/mnt/") or dst.startswith("/mnt/"):
            fails.append(("gx", "mount", dst,
                          f"the GX container must not mount /mnt ({s}): that is P7's disk over NFS"))
    if src is None:
        fails.append(("gx", "mount", "/cold", "the spec mounts no /cold: the cold pack is unreachable"))
        return
    r = probe(cfg["gx_host"], src)
    if r["missing"]:
        fails.append(("gx", "pack", src, f"does not exist on {cfg['gx_host']}"))
        return
    if r["real_failed"] or not r["real"]:
        fails.append(("gx", "pack", src, f"cannot be resolved on {cfg['gx_host']} ({r['err'] or 'readlink failed'})"))
        return
    for bad in ("/mnt", "/net"):
        if under(r["real"], bad):
            fails.append(("gx", "pack", src,
                          f"resolves to {r['real']} under {bad} (mounted from P7 over NFS; packs live on "
                          f"the GX box's own disk)"))
    if remote_fs(r["fstype"]):
        fails.append(("gx", "pack", src, f"sits on {r['fstype']}: a pack must not be served over the network"))
    for name, target in r["links"]:
        if target != "MISSING":
            for bad in ("/mnt", "/net"):
                if under(target, bad):
                    fails.append(("gx", "link", name, f"points at {target} under {bad} (network disk)"))
    if not quiet:
        print(f"PACK_DISK gx pack=/cold src={src} real={r['real']} fstype={r['fstype']} links={len(r['links'])}")


def main(argv):
    args = [a for a in argv[1:] if a != "--quiet"]
    quiet = "--quiet" in argv[1:]
    if len(args) != 1:
        print("usage: pair_pack_disk.py <pair> [--quiet]", file=sys.stderr)
        return 2
    if os.environ.get("PAIR_PACK_DISK") == "0":
        print(f"PACK_DISK_SKIPPED pair={args[0]} PAIR_PACK_DISK=0")
        return 0
    cfg = pc.load_conf(args[0])
    fails = []
    check_p7(cfg, fails, quiet)
    check_gx(cfg, fails, quiet)
    for side, what, path, reason in fails:
        print(f"PACK_DISK_FAIL side={side} what={what} path={path} reason={reason}")
    if fails:
        print(f"PACK_DISK_REFUSED pair={cfg['pair']} fails={len(fails)}")
        return 3
    print(f"PACK_DISK_OK pair={cfg['pair']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

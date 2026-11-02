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
      it (a mount in the spec that names it is refused here). The spec must mount the pool itself,
      /models/dsv41:/models/dsv41: /models is a directory on the 1.9 TB root disk and the 4 TB NVMe
      is a submount inside it, so a /models:/models bind exposes the pool only by accident and a
      string comparison cannot tell whether the bytes are on the pool. The check therefore compares
      st_dev of the hot pack with st_dev of /models/dsv41 -- the same device, not the same prefix.
      The pack, the placement profile its PACK_MANIFEST.json names, and every symlink inside those
      must also resolve under /models/dsv41, on a local filesystem. The manifest's `overlay`
      is provenance (the tree the pack was built from, not read by a server), so it is printed but
      not walked: a pack file that still points into it is caught by the link check, which is exact,
      where walking the overlay would flag files nothing loads.

  gx  /cold must be the cold pack on the GX's own root disk: same st_dev as the GX home directory,
      not a symlink that resolves into /mnt or /net, not a network filesystem.

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

# One round trip per path: realpath, the filesystem and the device it lands on, and every symlink
# below it (a link is how a pack names a file that lives somewhere else -- if that somewhere is NFS,
# the pack is not local). find -type l does not follow links, so the walk stays inside the tree.
# st_dev is what tells "my pack is on the pool" apart from "the pool happens to be bind-mounted
# inside a directory I mounted": on the P7 /models is the root NVMe and the pool is a submount.
PROBE = r"""
p=$1
[ -e "$p" ] || { printf 'MISSING %s\n' "$p"; exit 0; }
r=$(readlink -f -- "$p") || { printf 'REALFAIL %s\n' "$p"; exit 0; }
printf 'REAL %s\n' "$r"
ft=$(findmnt -T "$r" -no FSTYPE 2>/dev/null | head -1)
[ -n "$ft" ] || ft=$(stat -f -c %T "$r" 2>/dev/null)
printf 'FSTYPE %s\n' "${ft:-unknown}"
printf 'DEV %s\n' "$(stat -c %d -- "$r" 2>/dev/null || echo unknown)"
if [ -d "$p" ]; then
  find "$p" -type l -exec sh -c 'printf "LINK %s\t%s\n" "$1" "$(readlink -f -- "$1" 2>/dev/null || echo MISSING)"' _ {} \; 2>/dev/null
fi
"""


def ssh_argv():
    if os.environ.get("PAIR_DRY") == "1" and os.environ.get("PAIR_DRY_SSH"):
        return [os.environ["PAIR_DRY_SSH"]]
    return shlex.split(os.environ.get("PAIR_SSH") or "ssh")


def probe(host, path):
    """-> {real, fstype, dev, links: [(name, target)]} or {missing|real_failed: path}."""
    if host is None:
        cmd = ["sh", "-c", PROBE, "_", path]
    else:
        cmd = ssh_argv() + pc.SSH_OPTS + [host, "sh -c " + shlex.quote(PROBE) + " _ " + shlex.quote(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = {"real": None, "fstype": "unknown", "dev": None, "links": [], "missing": None,
           "real_failed": None, "rc": p.returncode, "err": (p.stderr or "").strip()[:300]}
    for line in p.stdout.splitlines():
        if line.startswith("REAL "):
            out["real"] = line[5:]
        elif line.startswith("FSTYPE "):
            out["fstype"] = line[7:]
        elif line.startswith("DEV "):
            out["dev"] = line[4:] if line[4:] not in ("", "unknown") else None
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


def spec_mounts(cfg, side):
    """[(src, dst)] from the container spec, read raw.

    The spec loader in pair_containers.py refuses a /mnt mount by exiting 2, which is the right
    behaviour for plan/apply but would turn this checker's report into an abort: the smoke wants one
    PACK_DISK_FAIL line per bad mount and rc 3, so read the file here and let check_p7/check_gx name
    the mount. A missing or malformed spec is the loader's business, not this file's.
    """
    path = os.path.join(pc.SPEC_DIR, f"{cfg['pair']}-{side}.json")
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return []
    return [tuple(m) for m in d.get("mounts") or [] if isinstance(m, list) and len(m) == 2]


def check_p7(cfg, fails, quiet):
    pool_src = None
    for src, dst in spec_mounts(cfg, "p7"):
        if src == "/mnt" or src.startswith("/mnt/") or dst == "/mnt" or dst.startswith("/mnt/"):
            fails.append(("p7", "mount", dst,
                          f"the pair must not mount the 12 TB drive ({src}); packs live on the "
                          f"serving box's own disk (owner rule 2026-11-02)"))
        if dst == P7_ALLOW:
            pool_src = src
    if pool_src is None:
        fails.append(("p7", "mount", P7_ALLOW, f"the spec mounts no {P7_ALLOW}: the hot pack is unreachable"))
    elif pool_src != P7_ALLOW:
        fails.append(("p7", "mount", P7_ALLOW,
                      f"mounted from {pool_src}, not from the pool itself (mount {P7_ALLOW}:{P7_ALLOW})"))

    poold = probe(None, P7_ALLOW)
    if poold["missing"] or poold["real_failed"] or not poold["real"]:
        fails.append(("p7", "pool", P7_ALLOW, f"cannot be resolved on P7 ({poold['err'] or 'missing'})"))
    elif remote_fs(poold["fstype"]):
        fails.append(("p7", "pool", P7_ALLOW, f"sits on {poold['fstype']}: the pool must be the P7's own NVMe"))
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
        if poold["dev"] and r["dev"] and r["dev"] != poold["dev"]:
            fails.append(("p7", what, path,
                          f"is on device {r['dev']}, but {P7_ALLOW} is device {poold['dev']}: the path is "
                          f"under the pool only as a prefix, not on the 4 TB NVMe"))
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
              f"dev={pr['dev'] or '-'} pool={P7_ALLOW} pool_dev={poold['dev'] or '-'} "
              f"placement={placement or '-'} overlay={overlay or '-'} links={nlinks}")


def check_gx(cfg, fails, quiet):
    src = None
    for s, dst in spec_mounts(cfg, "gx"):
        if dst == "/cold":
            src = s
        elif s == "/mnt" or s.startswith("/mnt/") or dst == "/mnt" or dst.startswith("/mnt/"):
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
    # Same device as the box's own disk, not merely a path that does not start with /mnt: a /cold
    # that is a symlink into the NFS mount passes a prefix test only until the link is resolved, and
    # `under(real, "/mnt")` above depends on the GX having /mnt mounted at all.
    home = os.environ.get("PAIR_PACK_GX_HOME", "/home/treadall")
    hd = probe(cfg["gx_host"], home)
    if hd["missing"] or hd["real_failed"] or not hd["real"]:
        fails.append(("gx", "home", home,
                      f"cannot be resolved on {cfg['gx_host']} ({hd['err'] or 'missing'}); cannot prove "
                      f"the pack sits on the box's own disk"))
    elif hd["dev"] and r["dev"] and r["dev"] != hd["dev"]:
        fails.append(("gx", "pack", src,
                      f"is on device {r['dev']}, but the box's own disk ({home}) is device {hd['dev']}: "
                      f"the cold pack must live on the GX's root NVMe (owner rule 2026-11-02)"))
    elif hd["dev"] and not r["dev"]:
        fails.append(("gx", "pack", src, "device cannot be read, so locality cannot be proven"))
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
        print(f"PACK_DISK gx pack=/cold src={src} real={r['real']} fstype={r['fstype']} "
              f"dev={r['dev'] or '-'} root_dev={hd['dev'] or '-'} links={len(r['links'])}")


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

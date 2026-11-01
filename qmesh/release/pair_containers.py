#!/usr/bin/env python3
"""pair_containers.py -- keep each pair's two containers on the release layout, by construction.

The containers never mount a tree that can go stale: they mount the pair root at /pair and use a
symlink inside the container, /src/exllamav3 -> /pair/current/src, plus TORCH_EXTENSIONS_DIR=/pair/
extcache. Flipping the host symlinks therefore changes what the *next* job runs without recreating
anything, and a running job keeps the release it started with.

    pair_containers.py plan <pair>              # live containers vs qmesh/release/containers/<pair>-<side>.json
    pair_containers.py apply <pair> <sha> [--ts T]   # recreate what plan says is stale; keep the old one as <name>.bak-<ts>
    pair_containers.py rollback <pair> <ts>     # undo one apply: put <name>.bak-<ts> back
    pair_containers.py link <pair> <sha>        # host symlinks: current -> releases/<sha>, extcache -> releases/<sha>/ext
    pair_containers.py inner <pair>             # inside the containers: /src/exllamav3 -> /pair/current/src
    pair_containers.py ls <pair>                # container state, for pair_status.sh

Exit: 0 ok / nothing to do, 1 plan says recreate (plan only), 2 error.
Old containers are renamed, never removed: rollback needs them and a removal is not reversible.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(HERE, "containers")
ART = os.environ.get("PAIR_ART", "/home/treadall/recipe-artifacts/gpu0-gx1")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def die(msg):
    print(f"pair_containers: {msg}", file=sys.stderr)
    sys.exit(2)


def load_conf(pair):
    conf = os.environ.get("PAIR_CONF", os.path.join(HERE, "pairs.conf"))
    for line in open(conf):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if f[0] != pair:
            continue
        keys = ["pair", "p7ct", "gpu", "seam_dev", "seam_gid", "gx_gid", "seam_peer", "gx_host",
                "gxct", "root", "port", "buckets", "p7_arch", "gx_arch"]
        d = dict(zip(keys, f))
        d["root"] = os.environ.get("PAIR_ROOT_OVERRIDE") or d["root"]
        return d
    die(f"pair {pair} not in {conf}")


def sh(cmd, stdin=None):
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def side_host(cfg, side):
    return None if side == "p7" else cfg["gx_host"]


def ssh_bin():
    """The dry-run harness passes PAIR_DRY_SSH=stub_ssh so no command leaves the box."""
    if os.environ.get("PAIR_DRY") == "1" and os.environ.get("PAIR_DRY_SSH"):
        return [os.environ["PAIR_DRY_SSH"]]
    return ["ssh"]


def docker(cfg, side, args, stdin=None):
    # PAIR_DRY only ever runs behind the stub-ssh harness (PAIR_DRY_SSH): without it the dry run
    # would talk to a real docker, and a "dry" apply that creates containers is not a dry run.
    if os.environ.get("PAIR_DRY") == "1" and not os.environ.get("PAIR_DRY_SSH"):
        die("PAIR_DRY=1 needs PAIR_DRY_SSH (stub harness); refusing to touch real docker")
    cmd = ["docker"] + args
    host = side_host(cfg, side)
    if host:
        return sh(ssh_bin() + SSH_OPTS + [host, " ".join(_q(a) for a in cmd)], stdin=stdin)
    return sh(cmd, stdin=stdin)


def _q(s):
    if s and all(c.isalnum() or c in "-_./:=,@" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def spec(cfg, side):
    ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
    path = os.path.join(SPEC_DIR, f"{cfg['pair']}-{side}.json")
    if not os.path.exists(path):
        die(f"missing container spec {path}")
    spec = json.load(open(path))
    if spec["name"] != ct:
        die(f"{path} says name={spec['name']} but pairs.conf says {ct}")
    spec = json.loads(json.dumps(spec).replace("__PAIR_ROOT__", cfg["root"]))
    return spec


def run_args(spec, name):
    argv = ["run", "-d", "--name", name, "--workdir", spec["workdir"]]
    for k, v in spec.get("env", {}).items():
        argv += ["-e", f"{k}={v}"]
    for src, dst in spec["mounts"]:
        argv += ["--mount", f"type=bind,source={src},target={dst}"]
    argv += spec.get("args", [])
    argv += ["--entrypoint", spec["entrypoint"], spec["image"]] + spec.get("cmd", [])
    return argv


def live(cfg, side):
    ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
    rc, out, err = docker(cfg, side, ["inspect", ct])
    if rc:
        return None
    d = json.loads(out)[0]
    mounts = {m["Destination"]: m["Source"] for m in (d.get("Mounts") or [])}
    env = {}
    for e in d["Config"].get("Env") or []:
        k, _, v = e.partition("=")
        env[k] = v
    ul = {u["Name"]: u.get("Soft", -1) for u in (d["HostConfig"].get("Ulimits") or [])}
    return {
        "running": bool(d["State"]["Running"]),
        "image": d["Config"]["Image"],
        "workdir": d["Config"].get("WorkingDir") or "",
        "entrypoint": d["Config"].get("Entrypoint") or [],
        "cmd": d["Config"].get("Cmd") or [],
        "mounts": mounts,
        "env": env,
        "network": d["HostConfig"].get("NetworkMode"),
        "ipc": d["HostConfig"].get("IpcMode"),
        "shm": d["HostConfig"].get("ShmSize"),
        "security": d["HostConfig"].get("SecurityOpt") or [],
        "ulimits": ul,
        "devices": [x.get("PathOnHost", "") for x in (d["HostConfig"].get("Devices") or [])],
        "gpus": bool(d["HostConfig"].get("DeviceRequests")),
        "restart": (d["HostConfig"].get("RestartPolicy") or {}).get("Name"),
    }


def plan_side(cfg, side):
    """List of reasons the live container does not match the spec (empty = fine)."""
    ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
    want = spec(cfg, side)
    got = live(cfg, side)
    if got is None:
        return [f"missing:{ct}"]
    why = []
    if got["image"] != want["image"]:
        why.append(f"image={got['image']} want={want['image']}")
    if got["workdir"] != want["workdir"]:
        why.append(f"workdir={got['workdir']}")
    if list(got["entrypoint"]) != [want["entrypoint"]]:
        why.append(f"entrypoint={got['entrypoint']}")
    if list(got["cmd"]) != list(want["cmd"]):
        why.append(f"cmd={got['cmd']}")
    for src, dst in want["mounts"]:
        if got["mounts"].get(dst) != src:
            why.append(f"mount:{dst}={got['mounts'].get(dst) or '-'} want={src}")
    for k, v in want["env"].items():
        if got["env"].get(k) != v:
            why.append(f"env:{k}={got['env'].get(k) or '-'} want={v}")
    if got["network"] != "host" or got["ipc"] != "host":
        why.append(f"net={got['network']}/{got['ipc']}")
    if got["shm"] < 64 * 1024 ** 3:
        why.append(f"shm={got['shm']}")
    if got["security"] != ["label=disable"]:
        why.append(f"security={got['security']}")
    if not got["gpus"]:
        why.append("gpus=none")
    if "/dev/infiniband" not in got["devices"]:
        why.append("devices=missing_infiniband")
    if got["ulimits"].get("memlock") != -1:
        why.append(f"ulimit:memlock={got['ulimits'].get('memlock')}")
    ulimit_want = {}
    args = want.get("args", [])
    for i, a in enumerate(args):
        if a == "--ulimit":
            k, _, v = args[i + 1].partition("=")
            ulimit_want[k] = int(v)
    for k, v in ulimit_want.items():
        if got["ulimits"].get(k) != v:
            why.append(f"ulimit:{k}={got['ulimits'].get(k)} want={v}")
    if got["restart"] != "no":
        why.append(f"restart={got['restart']}")
    return why


def cmd_plan(cfg, quiet=False):
    rc = 0
    for side in ("p7", "gx"):
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        why = plan_side(cfg, side)
        if why:
            rc = 1
            print(f"PLAN {cfg['pair']} {side} {ct} RECREATE {' '.join(why)}")
        elif not quiet:
            print(f"PLAN {cfg['pair']} {side} {ct} OK")
    return rc


def stop_our_server(cfg):
    """Stop only OUR pair's server; never touch another container's processes."""
    ct = cfg["gxct"]
    rc, out, _ = docker(cfg, "gx", ["exec", ct, "pgrep", "-c", "worker_seam_server.py"])
    try:
        n = int(out.strip() or 0)
    except ValueError:
        n = 0
    if n:
        print(f"STOP_SERVER {ct} pids={n}")
        docker(cfg, "gx", ["exec", ct, "pkill", "-TERM", "-f", "worker_seam_server.py"])
        for _ in range(30):
            rc, out, _ = docker(cfg, "gx", ["exec", ct, "pgrep", "-c", "worker_seam_server.py"])
            if (out.strip() or "0") == "0":
                break
            subprocess.run(["sleep", "1"])
        else:
            die(f"server still running in {ct}; refusing to recreate")


def cmd_apply(cfg, sha, ts, force=False):
    if not sha:
        die("apply needs a release sha")
    if not os.path.isdir(os.path.join(cfg["root"], "releases", sha, "src")):
        die(f"release {sha[:8]} is not cut at {cfg['root']}")
    todo = [(s, plan_side(cfg, s)) for s in ("p7", "gx")]
    todo = [(s, w) for s, w in todo if w]
    if not todo and not force:
        print(f"APPLY {cfg['pair']} no_recreate_needed")
        return 0
    stop_our_server(cfg)
    for side, why in todo:
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        print(f"RECREATE {cfg['pair']} {side} {ct} {' '.join(why)}")
        new = f"{ct}.new-{ts}"
        docker(cfg, side, ["rm", "-f", new])
        argv = run_args(spec(cfg, side), new)
        rc, out, err = docker(cfg, side, argv)
        if rc:
            die(f"docker run failed for {new}: {err.strip()[:400]}")
        docker(cfg, side, ["stop", "--time", "20", ct])
        docker(cfg, side, ["rename", ct, f"{ct}.bak-{ts}"])
        rc, out, err = docker(cfg, side, ["rename", new, ct])
        if rc:
            docker(cfg, side, ["rename", f"{ct}.bak-{ts}", ct])
            die(f"rename {new} -> {ct} failed: {err.strip()[:400]}")
        docker(cfg, side, ["start", ct])
    cmd_inner(cfg, sha, verify=False)
    left = [(s, w) for s, w in ((s, plan_side(cfg, s)) for s in ("p7", "gx")) if w]
    if left:
        die("after recreate the containers still differ: " +
            " ".join(f"{s}:{';'.join(w)}" for s, w in left))
    print(f"APPLY {cfg['pair']} containers_ok ts={ts}")
    return 0


def cmd_rollback(cfg, ts):
    stop_our_server(cfg)
    for side in ("p7", "gx"):
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        bak = f"{ct}.bak-{ts}"
        rc, out, _ = docker(cfg, side, ["inspect", "-f", "{{.State.Running}}", bak])
        if rc:
            print(f"ROLLBACK {cfg['pair']} {side} no {bak}")
            continue
        docker(cfg, side, ["stop", "--time", "20", ct])
        docker(cfg, side, ["rename", ct, f"{ct}.failed-{ts}"])
        docker(cfg, side, ["rename", bak, ct])
        docker(cfg, side, ["start", ct])
        print(f"ROLLBACK {cfg['pair']} {side} {ct} restored (new one kept as {ct}.failed-{ts})")
    return 0


def link_side_cmd(root, sha):
    """Atomic symlink flip: build the new link under a temp name, then rename over the old one."""
    rel = os.path.join(root, "releases", sha)
    return (f"set -e; mkdir -p {root}/releases; "
            f"ln -sfn {rel} {root}/.current.tmp && mv -T {root}/.current.tmp {root}/current; "
            f"ln -sfn {rel}/ext {root}/.extcache.tmp && mv -T {root}/.extcache.tmp {root}/extcache; "
            f"readlink -f {root}/current; readlink -f {root}/extcache")


def cmd_link(cfg, sha):
    if not sha:
        die("link needs a release sha")
    rel = os.path.join(cfg["root"], "releases", sha)
    if not os.path.isdir(os.path.join(rel, "src")):
        die(f"release {sha[:8]} is not cut at {cfg['root']}")
    cmd = link_side_cmd(cfg["root"], sha)
    rc, out, err = sh(["bash", "-c", cmd])
    if rc:
        die(f"P7 symlink flip failed: {err.strip()[:300]}")
    print(f"LINK {cfg['pair']} p7 {out.split()[0] if out.split() else '-'}")
    ssh = ssh_bin() + SSH_OPTS + [cfg["gx_host"], cmd]
    rc, out, err = sh(ssh)
    if rc:
        die(f"GX symlink flip failed: {err.strip()[:300]}")
    print(f"LINK {cfg['pair']} gx {out.split()[0] if out.split() else '-'} "
          f"ext={out.split()[1] if len(out.split()) > 1 else '-'}")
    return 0


def cmd_inner(cfg, sha=None, verify=True):
    """Inside each container: /src/exllamav3 must resolve into this release, via /pair/current.

    readlink -f resolves the whole chain, so the answer is /pair/releases/<sha>/src, not the
    intermediate /pair/current/src -- accept either, but it has to be THIS sha.
    """
    for side in ("p7", "gx"):
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        rc, out, err = docker(cfg, side, ["inspect", "-f", "{{.State.Running}}", ct])
        if rc or out.strip() != "true":
            print(f"INNER {cfg['pair']} {side} {ct} not_running")
            continue
        cmd = ("set -e; mkdir -p /src /pair; "
               "if [ ! -L /src/exllamav3 ]; then rm -rf /src/exllamav3; fi; "
               "ln -sfn /pair/current/src /src/exllamav3")
        # right after a recreate, /pair/current may not have moved yet: wire the symlink, verify later
        if verify:
            cmd += "; readlink -f /src/exllamav3"
        rc, out, err = docker(cfg, side, ["exec", ct, "bash", "-lc", cmd])
        if rc:
            die(f"inner symlink in {ct} failed: {err.strip()[:300]}")
        got = out.strip()
        if not verify:
            print(f"INNER {cfg['pair']} {side} {ct} linked")
            continue
        want_sha = sha
        if not want_sha:
            _, cur, _ = docker(cfg, side, ["exec", ct, "readlink", "-f", "/pair/current"])
            want_sha = os.path.basename(cur.strip())
        ok = {f"/pair/releases/{want_sha}/src", "/pair/current/src"}
        if got not in ok:
            die(f"inner symlink in {ct} resolved to {got!r}, want one of {sorted(ok)}")
        print(f"INNER {cfg['pair']} {side} {ct} {got}")
    return 0


def cmd_ls(cfg):
    for side in ("p7", "gx"):
        ct = cfg["p7ct"] if side == "p7" else cfg["gxct"]
        rc, out, _ = docker(cfg, side, ["inspect", "-f", "{{.State.Running}} {{.Config.Image}}", ct])
        print(f"CONTAINER {cfg['pair']} {side} {ct} "
              f"{'running ' + out.strip().split()[1] if rc == 0 else 'ABSENT'}")
    return 0


def main():
    if len(sys.argv) < 3:
        die(__doc__)
    what, pair = sys.argv[1], sys.argv[2]
    cfg = load_conf(pair)
    args = sys.argv[3:]
    if what == "plan":
        return cmd_plan(cfg, quiet="--quiet" in args)
    if what == "apply":
        ts = ""
        if "--ts" in args:
            i = args.index("--ts")
            ts = args[i + 1]
            del args[i:i + 2]
        ts = ts or subprocess.run(["date", "-u", "+%Y%m%dT%H%M%SZ"],
                                  capture_output=True, text=True).stdout.strip()
        return cmd_apply(cfg, args[0] if args else "", ts, force="--force" in args)
    if what == "rollback":
        return cmd_rollback(cfg, args[0] if args else "")
    if what == "link":
        return cmd_link(cfg, args[0] if args else "")
    if what == "inner":
        return cmd_inner(cfg)
    if what == "ls":
        return cmd_ls(cfg)
    die(f"unknown subcommand {what!r}")


if __name__ == "__main__":
    sys.exit(main())

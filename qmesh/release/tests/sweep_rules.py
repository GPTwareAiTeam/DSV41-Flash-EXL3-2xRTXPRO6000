#!/usr/bin/env python3
import importlib.util, os, sys, io, contextlib
os.environ["PAIR_QUEUE_ROOT"] = "/tmp/sweepq"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("pc", os.path.join(HERE, "pair_containers.py"))
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)

fails = []
def check(name, cond, detail=""):
    print(("OK   " if cond else "FAIL ") + name + (" | " + detail if detail else ""))
    if not cond: fails.append(name)

def die_msg(fn, *a):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            fn(*a)
        return ""
    except SystemExit:
        return err.getvalue().strip()

rows = pc._conf_rows()
names = [n for _, p, g in rows for n in (p, g)]
check("container names unique across pairs", len(names) == len(set(names)), str(rows))
d = dict((r[0], (r[1], r[2])) for r in rows)
check("dev owns seam-p7-dev / seam-gx2-dev", d["dev"] == ("seam-p7-dev", "seam-gx2-dev"), str(d["dev"]))
check("prod owns seam-p7-prod / seam-gx1-prod", d["prod"] == ("seam-p7-prod", "seam-gx1-prod"), str(d["prod"]))
check("the containers arms actually use belong to no pair",
      not any(n in names for n in ("seam-p7v2", "seam-gx1v2", "seam-gx1b", "seam-p7v2-robust")), str(names))

cfg = pc.load_conf("dev"); cfg["p7ct"] = "seam-p7-prod"
m = die_msg(pc.check_ownership, cfg)
check("foreign pair's container refused", "belongs to pair prod" in m, m)

import subprocess
os.makedirs("/tmp/sweepq/dev/gpu_queue", exist_ok=True)
h = subprocess.Popen(["bash", "-c", "sleep 60", "seam-p7-dev"])
t = f"/tmp/sweepq/dev/gpu_queue/01-1-{h.pid}-SomeAgent"; open(t, "w").write("x")
m = die_msg(pc.check_ownership, pc.load_conf("dev"))
check("container named by a live ticket refused", "named by a live ticket" in m, m)
h.kill(); os.remove(t)

# apply: existing container must not be recreated; absent ones are created under their final name
calls = []; state = {}
def fake_docker(cfg, side, args, stdin=None, **kw):
    calls.append((side, " ".join(args)))
    if args[0] == "run":
        state[("p7" if "seam-p7-dev" in " ".join(args) else "gx",
               "seam-p7-dev" if "seam-p7-dev" in " ".join(args) else "seam-gx2-dev")] = True
    if args[0] == "inspect":
        return (0, "true\n", "") if (side, args[-1]) in state else (1, "", "no such container")
    return (0, "", "")
pc.docker = fake_docker
pc.plan_side = lambda cfg, side: None if (side, "seam-p7-dev" if side == "p7" else "seam-gx2-dev") in state else ["absent"]
links = []; inners = []
pc.cmd_link = lambda cfg, sha: links.append("link")
pc.cmd_inner = lambda cfg, sha=None, verify=True: inners.append(verify)

root = "/tmp/sweeppair"; os.makedirs(f"{root}/releases/abc123/src", exist_ok=True)
cfg = pc.load_conf("dev"); cfg["root"] = root
state[("p7", "seam-p7-dev")] = True; state[("gx", "seam-gx2-dev")] = True
_old_plan = pc.plan_side
pc.plan_side = lambda cfg, side: ["spec-differs"]
m = die_msg(pc.cmd_apply, cfg, "abc123", "T")
check("existing container not recreated", "refusing to recreate an existing" in m, m)
check("no rename/stop/rm before the refusal", not any(a.split()[0] in ("rename", "stop", "rm") for _, a in calls))

pc.plan_side = _old_plan
calls.clear(); state.clear(); links.clear(); inners.clear()
msg = io.StringIO()
with contextlib.redirect_stdout(msg):
    pc.cmd_apply(cfg, "abc123", "T")
out = msg.getvalue()
runs = [a for _, a in calls if a.startswith("run ")]
check("creates seam-p7-dev and seam-gx2-dev", any("seam-p7-dev" in a for a in runs) and any("seam-gx2-dev" in a for a in runs), str(runs)[:200])
check("never renames an existing container", not any(a.startswith("rename") for _, a in calls))
check("bootstrap current before wiring", "LINK_BOOTSTRAP" in out and links == ["link"] and inners == [False], out.strip()[:300])
check("no PAIR_ALLOW_RECREATE needed for a fresh pair", True)

# inner without current
cfg2 = dict(cfg); cfg2["root"] = "/tmp/sweeppair2"; os.makedirs("/tmp/sweeppair2/releases/abc123/src", exist_ok=True)
m = die_msg(pc.cmd_inner_real if hasattr(pc, "cmd_inner_real") else None, None) if False else None
spec2 = importlib.util.spec_from_file_location("pc2", os.path.join(HERE, "pair_containers.py"))
pc2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(pc2)
m = die_msg(pc2.cmd_inner, cfg2, "abc123")
check("inner refuses a release root without current", "current is missing" in m, m)

# The ext cache is only reusable by a client if the builder spells the include path the way the
# client does; these greps keep the two halves of that invariant from drifting apart again.
def src(name):
    return open(os.path.join(HERE, name)).read()

be = src("pair_build_ext.sh")
check("ext built through the client path (/src/exllamav3)",
      "ln -sfn $rel_c/src $link" in be and "PYTHONPATH=$link" in be, "")
check("ext build restores the link afterwards", "trap 'ln -sfn $back $link' EXIT" in be, "")
check("client import is proven, not assumed",
      "EXT_CLIENT_HIT" in be and "EXT_CLIENT_MISS" in be and "EXT_CLIENT_TIMEOUT" in be, "")
check("client check uses the release's own cache",
      "cd $link && TORCH_EXTENSIONS_DIR=$rel_c/ext" in be, "")

sm = src("pair_smoke.sh")
check("smoke keeps the arm's bytes", "SMOKE_ARM_LOG" in sm, "")
check("smoke prints the server log tail", "SMOKE_SERVER_TAIL" in sm, "")

ur = src("pair_update_run.sh")
check("update records whether the pair was serving", "UPDATE_SERVER_BEFORE" in ur, "")
check("restore skips a server that never served", "RESTORE_SERVER_SKIP" in ur, "")

# The canary gate: prod is reachable only through pair_promote.sh, and only for a sha dev proved by
# generating token ids. These are source invariants because the alternative -- trusting the flags at
# run time -- is exactly the drift this tooling exists to stop.
check("prod update refuses without the promote flag", 'PAIR_PROMOTE:-0' in src("pair_update.sh"), "")
check("prod update names the only way in", "pair_promote.sh" in src("pair_update.sh"), "")
check("promote passes the flag and nothing else does",
      'PAIR_PROMOTE=1 "${cmd[@]}"' in src("pair_promote.sh")
      and 'PAIR_PROMOTE=1' not in src("pair_update_run.sh"), "")
check("promote requires per-sha dev evidence", "dev/evidence/$SHA.json" in src("pair_promote.sh"), "")
check("promote wants decode ids, not just a row", "pair_canary_gate" in src("pair_promote.sh"), "")
check("one definition of the canary gate",
      sum(s.count("no_decode_ids") for s in (src("pair_lib.sh"), src("pair_update.sh"),
                                            src("pair_promote.sh"), src("pair_update_run.sh"),
                                            src("pair_smoke.sh"))) == 1, "")
check("the script that flips checks the canary too",
      "pair_canary_gate" in ur, "")
check("update_run needs the sha for prod", '[ "$PAIR" = prod ]' in ur, "")
check("box memory refusals are classified, not blamed on the release",
      ur.count("SMOKE_FAIL memory ") == 1 and "memory_fail" in ur, "")
check("a memory failure keeps the code (no restore)",
      "rollback=none" in ur and "restore\n}" not in ur.split("memory_fail()")[1].split("\n}")[0], "")
check("smoke runs the arm with ids", "--ids" in sm, "")
check("missing ids fail the smoke", "arm_no_ids" in sm, "")
check("smoke totals the ids across arms", "SMOKE_ARM_IDS_TOTAL" in sm, "")
check("update records the ids total, not one arm",
      "SMOKE_ARM_IDS_TOTAL n=" in ur, "")
# the success line is the one that also claims the check; the memory-failure line above it says
# rollback=none and must not be mistaken for it
_ok = 'PAIR_UPDATED pair=$PAIR sha=$SHA build=$BUILD_ID check='
check("update writes evidence before announcing",
      ur.index("evidence ok") < ur.index(_ok), "")
check("failed update writes failed evidence", 'evidence failed "$*"' in ur, "")

def code_lines(text):
    return [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]

# The client must ask for its own superset and let the server narrow it. Handing the client the
# server's list is what capped every prefill chunk at 64 rows.
check("the client does not pin buckets",
      not any("SEAM_BUCKETS" in l for l in code_lines(src("pair_arm.sh")) + code_lines(src("pair_env.sh"))),
      "")
check("the server is asked for the wide grant",
      all(int(pc.load_conf(q)["buckets"].split(",")[-1]) >= 2048 for q in ("prod", "dev")),
      str([pc.load_conf(q)["buckets"] for q in ("prod", "dev")]))
check("a release prefix resolves to the commit before it is compared",
      'rev-parse --verify "$SHA_ARG^{commit}"' in src("pair_update_run.sh")
      and sum('rev-parse --verify "$SHA^{commit}"' in src(f)
              for f in ("pair_build_lib.sh", "pair_build_ext.sh")) == 2, "")
check("an update ticket carries a command",
      'cmd=("$HERE/pair_update_run.sh"' in src("pair_update.sh"), "")
check("an empty ticket is refused, not queued",
      "no command given" in src("pair_lib.sh"), "")
# The smoke is the canary for what ships, so its server gets the pair's own grant -- not less: with
# only 1,6,64 the 32k prefill would be served in 64-row chunks and the rate gate would measure a shape
# no pair runs. What it must not do is ask for a bucket it never fills, so the prefill phase has to be
# deep enough to reach the widest bucket of both pairs.
check("the smoke grants the pair's own buckets",
      "SMOKE_BUCKETS=${SMOKE_BUCKETS:-$BUCKETS}" in sm
      and "--buckets $SMOKE_BUCKETS" in sm, "")
_depth = [l for l in code_lines(sm) if l.startswith("SMOKE_PREFILL_DEPTH=")][0]
_depth = int(_depth.split(":-")[1].rstrip("}").rstrip('"'))
check("the smoke's prefill is deep enough to fill the widest bucket it grants",
      all(_depth >= int(pc.load_conf(q)["buckets"].split(",")[-1]) for q in ("prod", "dev")),
      f"depth={_depth} " + str([pc.load_conf(q)["buckets"] for q in ("prod", "dev")]))
# One payload slot is the shipping shape (seam_server_ensure.sh's own default, and the slab the validated
# server printed); a second is only memory. The rollback has to start the server the same way.
check("the smoke asks for the shipping slot count", "SMOKE_SLAB_SLOTS:-1" in sm, "")
check("smoke passes the slot count to the ensure", "SEAM_SLAB_SLOTS=$SMOKE_SLAB_SLOTS" in sm, "")
check("the rollback serves the same slots", "SEAM_SLAB_SLOTS=${SMOKE_SLAB_SLOTS:-1}" in ur, "")
check("nothing sets the slower wide-dispatch tier",
      not any("EXL3_MOE_FUSED_ROWS_WIDE" in l
              for f in ("pair_arm.sh", "pair_env.sh", "pair_smoke.sh", "pair_update_run.sh")
              for l in code_lines(src(f))), "")
check("a pair root that is not absolute is refused",
      "pair root must be absolute" in src("pair_lib.sh"), "")
def _tracked_release_trees():
    out = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(HERE)),
                          "ls-tree", "-r", "--name-only", "HEAD", "--", "qmesh/release"],
                         capture_output=True, text=True).stdout.split()
    return [f for f in out if f.endswith("/RELEASE.json") or "/src/" in f]


check("no extracted release tree is tracked inside the tooling tree",
      not _tracked_release_trees(), ",".join(_tracked_release_trees()[:3]))
check("a dry run cannot declare a free GPU for a real ticket",
      '[ "$PAIR_DRY" = 1 ] && [ -n "${PAIR_DRY_GPU:-}" ]' in src("pair_update_run.sh"), "")
check("a rehearsed cut keeps its seeded release stand-in",
      '[ "$PAIR_DRY" != 1 ]' in src("pair_cut.sh"), "")
check("a release is stamped once for both sides",
      "CUT_ADOPT" in src("pair_cut.sh") and "does not describe this commit" in src("pair_cut.sh"), "")
check("release dir resolves a sha prefix", 'releases/"$1"*' in src("pair_lib.sh"), "")

# The client's CPU-host worker is started with multiprocessing "spawn", which re-executes the arm
# driver by path. Fed on stdin the driver's main file is "<stdin>": the child dies on
# FileNotFoundError and the parent only ever sees EOFError, which is not a release fault.
check("the arm driver is run by path, not fed on stdin",
      "python3 -u -" not in code_lines(src("pair_arm.sh"))
      and "pair_ct_path" in src("pair_arm.sh"), "")
check("the ids driver runs only as the main module",
      'if __name__ == "__main__":' in src("pair_arm_ids.py"), "")
check("a path handed to the containers is translated to their mount",
      'echo "/pair/${p#"$PAIR_ROOT"/}"' in src("pair_lib.sh")
      and 'echo "/artifacts/${p#"$art"/}"' in src("pair_lib.sh"), "")
check("the smoke reads the worker's own refusal, not the wrapper's rc",
      "*SEAM_NO_MEMORY*" in src("pair_smoke.sh")
      and 'fail "memory server_no_memory' in src("pair_smoke.sh"), "")

check("a failed smoke stops the server it started",
      "RESTORE_SERVER_STOP" in src("pair_update_run.sh")
      and "pkill -f worker_seam_server.py" in src("pair_update_run.sh"), "")

# A released tree carries no .git and its container carries no SEAM_BUILD_ID, so qes.build_id()
# answered "unknown" and the session handshake accepted a mixed pair on proto alone (2026-11-01).
# Both halves must name the same build: the cut writes <tree>/qmesh/BUILD_ID, the arm exports it.
check("the cut writes the build id into the release tree",
      src("pair_cut.sh").count('$tmp/src/qmesh/BUILD_ID') == 2, "")
check("the client half is told which build it runs",
      'envs+=(-e "SEAM_BUILD_ID=$BUILD_ID")' in src("pair_arm.sh")
      and '"$PAIR_ROOT/current/RELEASE.json"' in src("pair_arm.sh"), "")

# No load may run on the worker's own 2048 MiB default: the pair states what the pack load must
# leave behind, and both places that start a server have to pass it (measured GX numbers).
check("the load margin is the pair's number, not the worker's default",
      pc.load_conf("dev")["load_margin"] == "2048"
      and pc.load_conf("prod")["load_margin"] == "2048", "")
# The server's environment is a release property in three parts -- the load margin, the MOE kernel with
# its memory margin, and the piece size with the picker off -- and it used to be written out at each
# start. Two copies is how the rollback once restarted a server with the worker's own defaults while the
# smoke used the pair's numbers; there is one builder (pair_server_env in pair_lib.sh) and both starts
# call it, so a knob cannot reach one and not the other.
check("the server environment has one definition",
      "pair_server_env() {" in src("pair_lib.sh")
      and '%s %s %s\' "$1" "$LOAD_MARGIN" "$MOE_ENV" "$WS_ENV"' in src("pair_lib.sh")
      and "SEAM_MOE_KERNEL=cudaexl3 SEAM_MOE_PRELOAD=1" in "\n".join(code_lines(src("pair_lib.sh")))
      and "SEAM_CX_MARGIN_MIB=$CX_MARGIN_MIB" in "\n".join(code_lines(src("pair_lib.sh"))), "")
check("both server starts use that definition",
      sm.count('SEAM_ENV="$(pair_server_env ') == 1
      and ur.count('SEAM_ENV="$(pair_server_env ') == 1
      and not any("SEAM_LOAD_MARGIN_MIB=" in l or "SEAM_WS_ROWS=" in l
                  for f in ("pair_smoke.sh", "pair_update_run.sh") for l in code_lines(src(f))), "")
check("the piece size is the pair's number, pinned",
      pc.load_conf("dev")["ws_rows"] == "1024"
      and pc.load_conf("prod")["ws_rows"] == "1024"
      and 'WS_ENV="SEAM_WS_ROWS=$WS_ROWS SEAM_WS_ROWS_AUTO=0"' in src("pair_lib.sh"), "")
# One pack per GX box. A build and a smoke both load or hammer that box, so both must claim it through
# the take helper (which reads a foreign claim and waits); writing the file directly would let two jobs
# believe they own the box, which is what collided on 2026-11-02.
check("every GX job claims the box through the helper",
      any("pair_gx_gate_take " in l for l in code_lines(src("pair_build_ext.sh")))
      and any("pair_gx_gate_take " in l for l in code_lines(src("pair_smoke.sh")))
      and not any("pair_gx_gate_up " in l
                  for f in ("pair_build_ext.sh", "pair_smoke.sh") for l in code_lines(src(f)))
      and "pair_gx_gate_state() {" in src("pair_lib.sh")
      and "pair_gx_gate_down() {" in src("pair_lib.sh"), "")
check("the gate test is in the battery",
      os.path.exists(os.path.join(HERE, "tests", "gx_gate.sh")), "")
# A GX command runs on the host unless it says docker exec, and the host has no /pair: a container path
# passed to a bare pair_gxsh lookup reports files missing that are there, and a bare chown does nothing.
# (2026-11-02: the cuda-exl3 build succeeded and the stamp said `no_so_in_release_tree`.)
check("GX host commands never take a container path",
      not any("rel_c" in l and "pair_gxsh" in l and "docker exec" not in l
              for l in code_lines(src("pair_build_ext.sh")))
      and not any("rel_c" in l and "pair_gxsh" in l and "docker exec" not in l
                  for f in ("pair_smoke.sh", "pair_update_run.sh") for l in code_lines(src(f))), "")
check("a glob in a docker exec goes through a shell",
      not any("docker exec" in l and "_C*" in l and "sh -c" not in l
              for f in ("pair_build_ext.sh", "pair_smoke.sh", "pair_update_run.sh", "pair_lib.sh")
              for l in code_lines(src(f))), "")



print("\nRESULT %s fails=%d" % ("PASS" if not fails else "FAIL", len(fails)))
sys.exit(1 if fails else 0)

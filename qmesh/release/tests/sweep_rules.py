#!/usr/bin/env python3
import importlib.util, os, re, sys, io, contextlib
os.environ["PAIR_QUEUE_ROOT"] = "/tmp/sweepq"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("pc", os.path.join(HERE, "pair_containers.py"))
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
_REAL_PLAN = pc.plan_side   # the apply test below stubs the module attribute; the arrow tests need the real one

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
# The owner's rule: a release reaches a pair's `current` only through a passed promote. prepare may
# arm the containers (they are idle, no current, nothing that resolves through /pair/current can
# import), but creating prod/current at 00:00 on 2026-11-02 before dev had passed was a promote.
check("prepare never creates current", "LINK_UNPROMOTED" in out and "LINK_BOOTSTRAP" not in out,
      out.strip()[:300])
check("prepare promotes nothing", links == [] and inners == [False], f"links={links} inners={inners}")
check("no PAIR_ALLOW_RECREATE needed for a fresh pair", True)

# inner without current
cfg2 = dict(cfg); cfg2["root"] = "/tmp/sweeppair2"; os.makedirs("/tmp/sweeppair2/releases/abc123/src", exist_ok=True)
m = die_msg(pc.cmd_inner_real if hasattr(pc, "cmd_inner_real") else None, None) if False else None
spec2 = importlib.util.spec_from_file_location("pc2", os.path.join(HERE, "pair_containers.py"))
pc2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(pc2)
m = die_msg(pc2.cmd_inner, cfg2, "abc123")
check("inner refuses a release root without current", "current is missing" in m, m)
execed = []
def fake_docker2(cfg, side, args, stdin=None, **kw):
    if args[0] == "inspect":
        return (0, "true\n", "")
    if args[0] == "exec":
        execed.append(args[-1])
    return (0, "", "")
pc2.docker = fake_docker2
msg = io.StringIO()
with contextlib.redirect_stdout(msg):
    pc2.cmd_inner(cfg2, "abc123", verify=False)
check("an unpromoted container is wired straight at the release",
      any("ln -sfn /pair/releases/abc123/src /src/exllamav3" in c for c in execed)
      and "target=/pair/releases/abc123/src" in msg.getvalue(), msg.getvalue().strip()[:200])

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
check("ext build runs the probe as a file, never on stdin",
      "python3 -u /tmp/pair_ext_probe.py" in be and "python3 -u -" not in be, "")

check("client check uses the release's own cache, from a neutral cwd",
      "cd /tmp && PYTHONPATH=$link TORCH_EXTENSIONS_DIR=$rel_c/ext" in be, "")

check("the smoke's ext probe runs as a client would (neutral cwd, image env)",
      "cd /tmp && timeout -k 5 ${SMOKE_EXT_PROBE_S:-60}" in src("pair_smoke.sh")
      and "TORCH_CUDA_ARCH_LIST" not in src("pair_smoke.sh"), "")

sm = src("pair_smoke.sh")
check("smoke keeps the arm's bytes", "SMOKE_ARM_LOG" in sm, "")
check("smoke prints the server log tail", "SMOKE_SERVER_TAIL" in sm, "")

ur = src("pair_update_run.sh")
check("update records whether the pair was serving", "UPDATE_SERVER_BEFORE" in ur, "")
check("a recreate refuses while any P7 GPU 0/1 has a foreign compute process",
      'gpu_busy p7_gpus=0,1' in ur
      and ur.index('pair_gpu_foreign_users 0 1 "$P7CT"') < ur.index('pair_containers.py" apply')
      and 'PAIR_GPU_WAIT_MIN' in ur, "")
check("the GPU probe covers every P7 GPU by UUID, and the pair's own container is exempt",
      'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name' in src("pair_lib.sh")
      and 'pair_pid_in_container' in src("pair_lib.sh")
      and 'not this recipe, not touched' in ur, "")

# The canary gate: prod is reachable only through pair_promote.sh, and only for a sha dev proved by
# generating token ids. These are source invariants because the alternative -- trusting the flags at
# run time -- is exactly the drift this tooling exists to stop.
check("prod update refuses without the promote flag", 'PAIR_PROMOTE:-0' in src("pair_update.sh"), "")
check("prod update names the only way in", "pair_promote.sh" in src("pair_update.sh"), "")
check("promote passes the flag and nothing else does",
      'PAIR_PROMOTE=1 "${cmd[@]}"' in src("pair_promote.sh")
      and 'PAIR_PROMOTE=1' not in src("pair_update_run.sh"), "")
check("promote queues a flip-only ticket (prepares out of band)",
      "cmd+=(--flip-only)" in src("pair_promote.sh"), "")
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
# The stamp names the artifact, but only an import can say whether the next client reuses it. A cache
# written under another include-path spelling has a full plan waiting; inside a smoke that plan runs
# under load and evicts the pack (2026-11-02: 565.2 tok/s). The probe must come before the load.
check("a smoke proves the client cache before it loads a pack",
      "SMOKE_EXT_HIT" in sm and "ext_client_miss" in sm
      and sm.index("ext_client_miss") < sm.index('pair_step "SMOKE_SERVER_START'), "")

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
# An admission is a historical fact about the sha, not a receipt for the latest attempt: a dev re-smoke
# that fails must not erase the ok file prod was promoted on (dev, 2026-11-02 03:21: a re-run of the live
# release wrote ok=false over the admission, and the record prod refuses without was gone).
check("a failed re-run never erases an admission",
      "admitted = bool(json.load(open(path)).get(\"ok\"))" in ur
      and '.failed.json' in ur and "admission=" in ur, "")


def code_lines(text):
    return [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]

# The client adopts the intersection of what it asks for and what the server grants, and its
# fallback default lists 1,6,64,256,512,1024,2048,4096 -- no 8192. So a client that asks for
# nothing gets 1,6,64 from an 8192-granting server and caps every prefill chunk at 64 rows (dev,
# 2026-11-02). The arm asks for the pair's list, which is the server's list; ask for a superset
# and the pair would still run, but on a width the release was never measured at.
check("the client asks for the pair's buckets",
      'SEAM_BUCKETS=$BUCKETS' in "".join(l.replace('"', '') for l in code_lines(src("pair_arm.sh"))), "")
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
check("a JIT lock left by a dead build is cleared, and only when nothing alive owns it",
      'SEAM_ENSURE stale torch build lock(s):' in src("seam_server_ensure.sh")
      and src("seam_server_ensure.sh").index("seam_gxct \"rm -f $SEAM_LOG\"")
          < src("seam_server_ensure.sh").index("stale torch build lock")
      and 'pair_torch_locks' in src("pair_smoke.sh")
      and 'pair_torch_build_live' in src("pair_smoke.sh"), "")
check("the busy-box probe neither counts stubs nor its own wrapper",
      '--stub' in src("pair_lib.sh")
      and 'worker_seam_server[.]py' in src("pair_lib.sh"), "")
check("a busy box is refused before anything is cut or re-created",
      'pair_gx_foreign_server' in src("pair_update_run.sh")
      and src("pair_update_run.sh").index("pair_gx_foreign_server")
          < src("pair_update_run.sh").index("phase containers"), "")
check("the smoke repairs a container that lost NVML, but only after it owns the box",
      'pair_ensure_nvml p7 "$P7CT"' in src("pair_smoke.sh")
      and 'pair_ensure_nvml gx "$GXCT"' in src("pair_smoke.sh")
      and src("pair_smoke.sh").index("pair_gx_gate_take") < src("pair_smoke.sh").index("pair_ensure_nvml p7")
      and 'PREPARE_NVML_WARN' in src("pair_prepare.sh")
      and "docker restart" not in src("pair_prepare.sh"), "")
check("a claim carries its host, and only a same-host pid may declare it dead",
      "host=%s" in src("pair_lib.sh")
      and 'hostname -s' in src("pair_lib.sh")
      and 'flock' in src("pair_lib.sh")
      and r'host=\([^ ]*\)' in src("seam_server_ensure.sh"), "")
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
      and "'^python3 -u qmesh/worker_seam_server.py'" in src("pair_update_run.sh"), "")

# 2026-11-02: the worker is launched as `bash -lc 'cd /src/exllamav3 && stdbuf … python3 -u
# qmesh/worker_seam_server.py …'`, so while the wrapper lives its cmdline mentions the script and an
# unanchored pgrep/pkill reaches it first. Recording that pid and TERMing it stops a shell and orphans
# the process holding 116 GiB of pack. Every site that records or stops a server must anchor on the
# worker's own argv; the counts that only ask "is anything serving" may over-match.
_stops = []
for _n in ("pair_lib.sh", "pair_containers.py", "pair_update_run.sh", "seam_server_ensure.sh",
           "pair_smoke.sh", "pair_build_ext.sh"):
    for _l in src(_n).splitlines():
        if "worker_seam_server" in _l and "pkill" in _l and "^python3" not in _l:
            _stops.append(_n)
        if "pair_server_record" in _l and "pgrep" in _l and "^python3" not in _l:
            _stops.append(_n)
check("a server stop or record is anchored on the worker's argv, never the launcher's",
      not _stops, ",".join(sorted(set(_stops))))

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
# believe they own the box, which is what collided on 2026-11-02. Serving is the third job that loads a
# pack, so it takes the claim the same way.
check("every GX job claims the box through the helper",
      all(any("pair_gx_gate_take " in l for l in code_lines(src(f)))
          for f in ("pair_build_ext.sh", "pair_build_lib.sh", "pair_smoke.sh", "pair_serve.sh"))
      and not any("pair_gx_gate_up " in l
                  for f in ("pair_build_ext.sh", "pair_build_lib.sh", "pair_smoke.sh", "pair_serve.sh")
                  for l in code_lines(src(f)))
      and "pair_gx_gate_state() {" in src("pair_lib.sh")
      and "pair_gx_gate_down() {" in src("pair_lib.sh"), "")
# "The pair is down" must be a state somebody chose, reached from the release `current` names and only
# after the sync check says the artifact on disk is the stamped one; and the worker it leaves behind has
# to be named in the pidfile, or the next ticket can only stop it by pattern.
check("serving starts from `current`, refuses drift, and records the worker",
      "pair_current_sha" in src("pair_serve.sh")
      and "pair_check.sh" in src("pair_serve.sh")
      and "PAIR_SERVE_DRIFT" in src("pair_serve.sh")
      and "pair_server_record" in src("pair_serve.sh")
      and "--fresh" in src("pair_serve.sh")
      and "pair_gx_foreign_server" in src("pair_serve.sh"), "")
check("the gate test is in the battery",
      os.path.exists(os.path.join(HERE, "tests", "gx_gate.sh")), "")
# The smoke's two pre-load guards (NVML on both ends, a JIT lock nobody owns) are the difference between
# a 30 s diagnosis and a lost ticket; the third job that loads a pack must not rediscover that.
check("serving checks NVML on both ends and the torch lock before it loads",
      "pair_ensure_nvml p7" in src("pair_serve.sh")
      and "pair_ensure_nvml gx" in src("pair_serve.sh")
      and "pair_torch_locks" in src("pair_serve.sh")
      and src("pair_serve.sh").index("pair_gx_gate_take") < src("pair_serve.sh").index("pair_ensure_nvml p7"), "")

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

# `python3 -` reads its *program* from stdin, so a heredoc plus a pipe (`... | python3 - <<'PY'`) leaves
# the parser an empty stream. That silently turned a complete 8-token arm row into "arm_no_tokens NONE"
# on dev (2026-11-02) and would have written empty references. The arm's rows are parsed by a helper the
# battery can test; no shell script may pipe data into a heredoc'd python again.
check("arm rows are parsed by a file, not through a heredoc's stdin",
      'pair_arm_rows.py' in src("pair_smoke.sh")
      and os.path.exists(os.path.join(HERE, "pair_arm_rows.py"))
      and not any(re.search(r"\|\s*python3\s+-\s+[^|]*<<", l)
                  for f in ("pair_smoke.sh", "pair_ref.sh", "pair_update_run.sh")
                  for l in code_lines(src(f))), "")
check("the arm-row parser is in the battery",
      os.path.exists(os.path.join(HERE, "tests", "arm_rows.sh")), "")
# The smoke asks a box that has just unloaded 113 GiB to prove it came back, before the worker's own
# refusal decides the ticket: a load started into a box still releasing memory reads as a bad release.
check("the smoke waits for the GX box before the load preflight",
      'SMOKE_MEM_READY' in src("pair_smoke.sh")
      and any("SMOKE_MEM_WAIT_S" in l for l in code_lines(src("pair_smoke.sh")))
      and "LOAD_MARGIN" in src("pair_smoke.sh"), "")
# The arm must ask for the pair's list. The client adopts the intersection of its request and the
# server's grant, so a request that misses the shipping width (8192) is a silent 64-row cap: on
# 2026-11-02 the dev arm negotiated accepted=1,6,64 against a server granting 1,6,64,8192 and every
# prefill chunk went 128 calls deep. Pairs.conf is where the list lives; the arm quotes it.
conf = src("pairs.conf")
conf_buckets = [l.split()[11] for l in conf.splitlines()
                if l.split()[:1] and l.split()[0] in ("prod", "dev") and len(l.split()) > 11]
check("every pair ships the prefill width its release is measured on",
      len(conf_buckets) == 2 and all(b == "1,6,64,8192" for b in conf_buckets),
      "pairs.conf buckets: %s" % conf_buckets)
check("the arm requests the pair's buckets, not a client fallback",
      'SEAM_BUCKETS=$BUCKETS' in "".join(l.replace('"', '') for l in code_lines(src("pair_arm.sh"))), "")
check("the smoke fails when the grant is not the pair's list",
      'bucket_grant' in src("pair_smoke.sh") and 'SMOKE_BUCKETS_GRANT' in src("pair_smoke.sh"), "")

check("the grant comparator is a tested file, not an inline comparison",
      'pair_bucket_grant.py' in src("pair_smoke.sh")
      and os.path.exists(os.path.join(HERE, "pair_bucket_grant.py"))
      and os.path.exists(os.path.join(HERE, "tests", "bucket_grant.sh")), "")
# The stub smoke must not take the live gate: it is a test of one failure path, and waiting out the
# real dev smoke turned the battery into a 20-minute hang.
check("the stub smoke keeps its own gate directory",
      any("PAIR_GATE_DIR" in l for l in code_lines(src("tests/smoke_memory.sh"))), "")
# An interrupted ticket used to leave the pack resident (23:32 on 2026-11-02) and the next smoke
# refused `pair_busy gx_procs=2`. The fix is ownership, not a pattern kill: record the worker's pid,
# stop only that pid, and reclaim it at the next ticket's preflight.
check("the smoke records the worker it started",
      'pair_server_record' in src("pair_smoke.sh"), "")
check("an interrupted smoke stops its own server, a verified one keeps it",
      'SMOKE_VERDICT' not in src("pair_smoke.sh")
      and 'PAIR_SMOKE_KEEP_SERVER' not in src("pair_smoke.sh")
      and 'PAIR_SMOKE_KEEP_SERVER' not in src("pair_update_run.sh")
      and 'pair_server_stop' in src("pair_smoke.sh")
      and src("pair_smoke.sh").index("pair_server_stop") < src("pair_smoke.sh").index("pair_gx_gate_down")
      and 'smoke_exit' in src("pair_smoke.sh"), "")
check("the interrupt path stops by recorded pid, not by pattern",
      'pair_server_stop' in src("pair_update_run.sh")
      and 'pair_client_stop' in src("pair_update_run.sh")
      and not any('pkill -f worker_seam_server' in l for l in src("pair_update_run.sh").split("on_term")[1].split("restore()")[0].splitlines()), "")
_upd = src("pair_update_run.sh")
check("the preflight reclaims this pair's own orphan before refusing",
      _upd.count("others=$(pair_others_busy)") == 1
      and 'UPDATE_ORPHAN' in _upd
      and _upd.index('pair_server_stop 2>&1') < _upd.index('others=$(pair_others_busy)'), "")
check("a verified flip leaves no server behind it",
      'PAIR_SMOKE_KEEP_SERVER' not in src("pair_update_run.sh")
      and 'rm -f "$(pair_server_pidfile)"' not in src("pair_update_run.sh"), "")
# A pair's first release has no containers to reuse, and the lib build runs inside them: prepare that
# answers "unchanged" to a CREATE plan leaves the release cut and the pair unable to start a server.
check("prepare creates a missing pair container instead of reporting it unchanged",
      '*CREATE*' in src("pair_prepare.sh") and 'pair_containers.py" apply' in src("pair_prepare.sh"), "")
check("prepare leaves a matching pair alone",
      'unchanged' in src("pair_prepare.sh"), "")
# A release's stamp must not travel on the gx ssh stdin: a redirect written outside
# `$(pair_gxsh …)` is not reliably applied inside the substitution, the remote `cat` reads the
# host's own stdin, and the release ships a 0-byte RELEASE.json that only verify notices -- after
# a green smoke has already moved `current` (gx2, 9fc3ca8a, 2026-11-02).
_cut = src("pair_cut.sh")
check("the gx stamp travels in the command, not on stdin",
      "base64 -w0" in _cut and "cat > '$tmp/RELEASE.json'" not in _cut, "")
check("a cut refuses a stamp it cannot read back",
      "STAMP_EMPTY" in _cut and "STAMP_WRITE_FAILED" in _cut, "")
check("a cut refuses stamps that differ between the sides", "STAMP_MISMATCH" in _cut, "")
check("an existing release with an empty stamp is repaired", "STAMP_REPAIR" in _cut, "")

# The stamp and the binary are two files. A build in place (a stray client import from another tree)
# rewrites the .so and leaves the stamp intact, so the pair reads IN_SYNC while the container serves
# an artifact no build produced: on 2026-11-02 a probe build's 27,513,952-byte .so became the client's
# 27,509,856-byte one and pair_check still printed OK. Both hashes are now compared against the
# stamp -- the remote probe stubbed, the comparison real.
_json = __import__("json")
_spec2 = importlib.util.spec_from_file_location("psc", os.path.join(HERE, "pair_sync_check.py"))
psc = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(psc)
_sha = "9fc3ca8aa8388340e324507aae237c40c1019a29"
_bid = "dev-9fc3ca8aa838-20261102T035748Z"
_root = "/pair"
_paths = ["a.py"]
_ref = {p: p for p in _paths}
_mf = {"pair": "dev", "release_sha": _sha, "build_id": _bid, "_paths": _paths,
       "p7": {"container": "seam-p7-dev", "arch": "x86_64", "root": _root, "sm": "sm_120",
              "cuda_arch": "12.0"},
       "gx": {"container": "seam-gx2-dev", "arch": "aarch64", "root": _root, "sm": "sm_121",
              "cuda_arch": "12.1"}}
def _state(side, **over):
    st = {"LIVE": _root + "/releases/" + _sha, "EXT_LIVE": _root + "/releases/" + _sha + "/ext",
          "RELJSON": _json.dumps({"sha": _sha, "build_id": _bid}),
          "LIBSTAMP": _json.dumps({"build_id": _bid, "source_sha": _sha, "sha256": "aa"}),
          "LIBSHA": "aa",
          "EXTSTAMP": _json.dumps({"build_id": _bid, "source_sha": _sha,
                                   "arch_list": [_mf[side]["cuda_arch"]], "sha256": "beef"}),
          "EXTSHA": "beef"}
    if side == "gx":
        st["EXTSTAMP"] = _json.dumps({"build_id": _bid, "source_sha": _sha,
                                      "arch_list": ["12.1"], "sha256": "beef",
                                      "cx": {"build_id": _bid, "source_sha": _sha,
                                             "arch_list": ["12.1"], "sha256": "cafe"}})
        st["CXSHA"] = "cafe"
    st.update(over)
    return st
def _check_side(side, st):
    psc.side_state = lambda m, s, remote, skip_ct=False, skip_srv=False: (st, dict(_ref), "")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = psc.check_side(_mf, side, dict(_ref), True, skip_ct=True, skip_srv=True)
    return ok, buf.getvalue()
_ok, _out = _check_side("p7", _state("p7"))
check("a cache whose .so is the stamp's .so verifies", _ok, _out.strip()[-120:])
_ok, _out = _check_side("p7", _state("p7", EXTSHA="26de612b"))
check("a cache rebuilt under its stamp is DRIFT, not IN_SYNC",
      not _ok and "ext_so=26de612b want=beef" in _out, _out.strip()[-160:])
_ok, _out = _check_side("p7", _state("p7", EXTSHA=""))
check("an unreadable ext .so is DRIFT", not _ok and "ext_so=unreadable" in _out, _out.strip()[-160:])
_ok, _out = _check_side("gx", _state("gx"))
check("a gx cache whose artifacts are the stamp's verifies", _ok, _out.strip()[-120:])
_ok, _out = _check_side("gx", _state("gx", CXSHA="0bad"))
check("a cuda-exl3 .so rebuilt under its stamp is DRIFT",
      not _ok and "cx_so=0bad want=cafe" in _out, _out.strip()[-160:])


# Owner rule 2026-11-02: an expert pack lives on the local drive of the box that serves it. The P7
# containers used to be handed /mnt/data12 (the 12 TB drive) so that the hot pack's links could be
# resolved; a pack reached that way makes serving speed depend on the network, and the GX half reads
# that same drive over NFS. The rule is two source invariants and one behaviour:
#   * no P7 spec mounts /mnt -- the mount is gone from both pairs' specs;
#   * prepare and smoke both run the checker (a check only one path remembers is not a gate);
#   * the spec change lands through a queued ticket (--recreate), never by hand.
import json as _json
for _pair, _ct in (("prod", "seam-p7-prod"), ("dev", "seam-p7-dev")):
    _sp = _json.load(open(os.path.join(HERE, "containers", f"{_pair}-p7.json")))
    _mt = _sp.get("mounts") or []
    check(f"{_pair}'s P7 spec mounts the pool itself at /models/dsv41",
          any(d == "/models/dsv41" and s == "/models/dsv41" for s, d in _mt), str(_mt))
    check(f"{_pair}'s P7 spec does not mount the /models directory (root disk)",
          not any(d == "/models" for s, d in _mt), str(_mt))
check("prepare gates on the pack's disk before it cuts",
      "pair_pack_disk.py" in src("pair_prepare.sh")
      and src("pair_prepare.sh").index("pair_pack_disk.py") < src("pair_prepare.sh").index("pair_cut.sh"),
      "")
check("the smoke gates on the pack's disk too",
      "pair_pack_disk.py" in src("pair_smoke.sh") and 'fail "pack_disk' in src("pair_smoke.sh"), "")
check("the pack check is a tested file",
      os.path.exists(os.path.join(HERE, "tests", "pack_disk.py")), "")
_pd = src("pair_pack_disk.py")
check("the pack check names the P7 drive as the allowed root",
      'PAIR_PACK_P7_ROOT", "/models/dsv41"' in _pd, "")
check("the pack check refuses /mnt and /net on the GX side",
      '("/mnt", "/net")' in _pd, "")
check("the pack check refuses a network filesystem",
      '"nfs"' in _pd and "remote_fs(" in _pd, "")
check("the smoke skips the pack probe when the harness has no filesystem",
      "PAIR_PACK_DISK=0" in src("tests/smoke_memory.sh") and "PAIR_PACK_DISK=0" in src("tests/memory_fail.sh"), "")

# A spec change cannot reach a running pair any other way: the flip-only ticket refuses a RECREATE
# plan unless the operator queued it with --recreate, and then the ticket itself re-creates from the
# spec inside the queue (old container renamed, never removed). Without this branch the mount could
# only be dropped by hand, which is exactly what the owner forbade.
_pu = src("pair_update.sh")
check("--recreate is the only thing that permits a recreate",
      _pu.count("PAIR_ALLOW_RECREATE=1") == 1 and "--recreate) FLIP_ONLY=1; RECREATE=1" in _pu, "")
check("the flip-only ticket refuses a spec change it was not asked to apply",
      'PAIR_ALLOW_RECREATE:-0' in ur and "--recreate" in ur, "")
check("the flip-only ticket re-creates through the containers tool",
      ur.count("pair_containers.py\" apply") >= 2 and "UPDATE_CONTAINERS pair=$PAIR recreated" in ur, "")
check("promote can carry the same permission",
      "--recreate) RECREATE=1" in src("pair_promote.sh") and 'cmd+=(--recreate)' in src("pair_promote.sh"), "")


# The gate is binding for every launch, not only for the smoke that takes it. seam_server_ensure.sh is
# the one program that starts a GX server, so the check lives there: on a missing, expired, stale or
# foreign claim it refuses with rc 3 and names the state, and it runs before anything on the box is
# touched. The dispatch must pass that status through -- the old `|| exit 2` turned every refusal into
# "server_timeout_or_died", which is how a foreign pack load hid for a whole 900 s timeout.
_ens = src("seam_server_ensure.sh")
_ensure_body = _ens[_ens.index("seam_server_ensure() {"):]
check("the ensure checks the gate before it touches the box",
      "seam_gate_check" in _ensure_body
      and _ensure_body.index("seam_gate_check") < _ensure_body.index("seam_pids"),
      "")
check("a missing or foreign claim is refused by name, with the reason",
      all(x in _ens for x in ("gate_free owner=none", "gate_expired owner=", "gate_stale owner=",
                              "gate_foreign owner=", "PREFLIGHT_FAIL gate_bad")), "")
_disp = _ens[_ens.rindex('if [ "${BASH_SOURCE[0]}" = "$0" ]'):]
check("the ensure exits with its real status",
      'seam_server_ensure "$@"' in _disp and "exit $?" in _disp and "|| exit 2" not in _disp,
      " | ".join(_disp.splitlines()))
check("only the ensure can start a server: it marks the launch SEAM_GATE_OK=1",
      'local denv="-e SEAM_GATE_OK=1 "' in _ens, "")
check("the ensure only checks the claim, it never takes or drops it",
      not re.search(r"^\s*pair_gx_gate_(take|down)\b", _ens, re.M), "")
check("the dead-holder rule lives in pair_lib too",
      "kill -0 \"$p\"" in src("pair_lib.sh"), "")
check("the smoke hands the ensure the gate directory it actually claims in",
      'SEAM_GATE_DIR="$(dirname "$(pair_gx_gate_file)")"' in src("pair_smoke.sh"), "")
check("a busy box is classified apart from a memory-tight one",
      '*gx-busy*) fail "gx_busy server_preflight $line"' in src("pair_smoke.sh")
      and "*gx-mem*)  fail \"memory server_preflight" in src("pair_smoke.sh"), "")

# Preflight order: the other container's server explains the missing memory, so it is reported first,
# and both refusals name the holder. On 2026-11-02 the memory line fired first and read
# "avail=0GiB need=115GiB" about a box whose pack (and port) belonged to seam-gx2-4460, pid 1083950.
_pf = _ens[_ens.index("seam_preflight() {"):_ens.index("seam_server_ensure() {")]
check("a foreign server is named with its container and pid",
      'echo "server_in=\\$c:\\$p"' in _pf, "")
check("the busy-box refusal comes before the memory refusal",
      _pf.index("gx-busy") < _pf.index("gx-mem avail="), "")
check("the memory refusal names the holder and the heaviest container",
      "held_by=${other:-none} top_mem=$top" in _pf and "memory.current" in _pf, "")

# Plan arrows: `mount:-` is a removal, `~` a wrong source, `+` a wanted mount that is absent. The
# removal used to print as `mount:+<dst>=<src>`, i.e. exactly like an add, on the very line that
# documents dropping the 12 TB mount from the P7 spec.
_cfg = pc.load_conf("dev")
_want = pc.spec(_cfg, "p7")
_got = {"running": True, "image": _want["image"], "workdir": _want["workdir"],
        "entrypoint": [_want["entrypoint"]], "cmd": list(_want["cmd"]),
        "mounts": {d: s for s, d in _want["mounts"]}, "env": dict(_want["env"]),
        "network": "host", "ipc": "host", "shm": 64 * 1024 ** 3, "security": ["label=disable"],
        "ulimits": {"memlock": -1}, "devices": ["/dev/infiniband"], "gpus": True, "restart": "no"}
pc.live = lambda cfg, side: _got
_base = _REAL_PLAN(_cfg, "p7") or []
check("an unchanged container plans nothing (baseline for the mount arrows)",
      not [w for w in _base if w.startswith("mount")], str(_base))
_got["mounts"]["/mnt/data12/models/exl3"] = "/mnt/data12/models/exl3"
_why = " ".join(_REAL_PLAN(_cfg, "p7") or [])
check("an extra live mount is planned as a removal, not an add",
      "mount:-/mnt/data12/models/exl3=/mnt/data12/models/exl3 (not in spec)" in _why
      and "mount:+/mnt/data12" not in _why, _why)
del _got["mounts"]["/mnt/data12/models/exl3"]
# The difference between "recreate this container" and "note this and keep running" is the
# difference between a refused prepare and a release that ships; dev's prepare died on it on
# 2026-11-02 because both live P7 containers mount /models where the spec names the pack directory.
check("a removable extra mount is an advisory, not a verdict",
      "advisory:mount:-/mnt/data12/models/exl3=/mnt/data12/models/exl3 (not in spec)" in _why
      and not pc.real_reasons(["advisory:mount:-/x=/x (not in spec)"])
      and pc.real_reasons(["mount:+/x=absent want=/x"]) == ["mount:+/x=absent want=/x"], _why)
del pc.live
_got2 = dict(_got)
_got2["mounts"] = dict(_got["mounts"], **{"/models": "/models"})
_got2["mounts"].pop("/models/dsv41", None)
pc.live = lambda cfg, side: _got2
_cov = " ".join(_REAL_PLAN(_cfg, "p7") or [])
check("a parent mount that carries the wanted path is covered, not absent",
      "mount:+/models/dsv41=absent" not in _cov and "covered_by /models=/models" in _cov
      and not [w for w in pc.real_reasons(_REAL_PLAN(_cfg, "p7") or []) if w.startswith("mount")], _cov)
del pc.live
pc.live = lambda cfg, side: _got
_d0 = sorted(_got["mounts"])[0]
_s0 = _got["mounts"][_d0]
_got["mounts"][_d0] = "/wrong/place"
_why = " ".join(_REAL_PLAN(_cfg, "p7") or [])
check("a mount from the wrong source is planned as ~", f"mount:~{_d0}=" in _why, _why)
_got["mounts"].pop(_d0)
_why = " ".join(_REAL_PLAN(_cfg, "p7") or [])
check("a wanted mount that is absent is planned as + … absent",
      f"mount:+{_d0}=absent want={_s0}" in _why, _why)
del pc.live

# A spec that names /mnt is refused by the loader (plan, apply, smoke) and reported, not aborted, by the
# pack check: an operator who pastes the 12 TB mount back must get a refusal that says which mount and
# why, not a traceback.
import tempfile as _tf
_W = _tf.mkdtemp(prefix="specmnt.")
_old_spec = pc.SPEC_DIR
try:
    pc.SPEC_DIR = _W
    _bad = {"name": "seam-p7-dev", "image": "img", "workdir": "/src", "entrypoint": "/bin/bash",
            "mounts": [["/mnt/data12/models/exl3", "/mnt/data12/models/exl3"]], "env": {}, "args": [],
            "cmd": []}
    _json.dump(_bad, open(os.path.join(_W, "dev-p7.json"), "w"))
    _m = die_msg(pc.spec, pc.load_conf("dev"), "p7")
    check("a P7 spec that mounts /mnt is refused by the loader",
          "SPEC_REFUSED" in _m and "own disk" in _m, _m)
    _bad["name"] = "seam-gx2-dev"
    _json.dump(_bad, open(os.path.join(_W, "dev-gx.json"), "w"))
    _m = die_msg(pc.spec, pc.load_conf("dev"), "gx")
    check("a GX spec that mounts /mnt is refused by the loader", "SPEC_REFUSED" in _m, _m)
finally:
    pc.SPEC_DIR = _old_spec
    import shutil as _sh
    _sh.rmtree(_W, ignore_errors=True)
check("the pack check reads the spec's mounts raw so a bad mount is a report, not an abort",
      "spec_mounts(" in src("pair_pack_disk.py") and "pc.spec(cfg, \"p7\")" not in src("pair_pack_disk.py")
      and "pc.spec(cfg, \"gx\")" not in src("pair_pack_disk.py"), "")
lib = src("pair_lib.sh")
check("a box claim outside a queue ticket is refused, not taken",
      "PAIR_GATE_REFUSED outside_ticket" in lib and "pair_ticket_hardcap" in src("pair_lib.sh"),
      "")
check("a claim never runs past the ticket's hard cap",
      "PAIR_GATE_REFUSED ticket_hardcap_passed" in lib and "PAIR_GATE_CLAMP" in lib, "")
# Both halves run for real: the same claim that a ticket may take is refused from a shell with no
# ticket in its ancestry, and inside one it is clamped to the cap the wrapper exported.
import subprocess as _sp, tempfile as _tf, shutil as _sh
_gd = _tf.mkdtemp(prefix="gategate")
_env = dict(os.environ, PAIR_GATE_DIR=_gd, PAIR_QUEUE_ROOT="/tmp/sweepq", PAIR_QUEUE="/tmp/sweepq",
            GXCT="seam-gx2-dev")
for _k in ("GPU_RUN_STATUS", "GPU_TICKET_HARDCAP"):
    _env.pop(_k, None)
_probe = ('. /home/treadall/recipe-pair-release/qmesh/release/pair_lib.sh 2>/dev/null || '
          '. "$(dirname "$0")/pair_lib.sh"; printf "%s\n" "claim_outside_$?"; exit 0')
_r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; pair_gx_gate_up outsider 30; echo "rc=$?"; '
              'ls "%s" 2>/dev/null' % (HERE, _gd)],
             env=_env, capture_output=True, text=True)
check("gate_up refuses outside a ticket and writes no claim",
      "rc=5" in _r.stdout and ".gate" not in _r.stdout and "PAIR_GATE_REFUSED outside_ticket" in _r.stderr,
      (_r.stdout + _r.stderr).strip()[:200])
_env["GPU_TICKET_HARDCAP"] = str(1)
_r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; pair_gx_gate_up pastcap 30; echo "rc=$?"' % HERE],
             env=_env, capture_output=True, text=True)
check("an expired ticket's cap refuses the claim",
      "rc=5" in _r.stdout and "ticket_hardcap_passed" in _r.stderr, (_r.stdout + _r.stderr).strip()[:200])
_env["GPU_TICKET_HARDCAP"] = str(int(__import__("time").time()) + 600)
_r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; pair_gx_gate_up clamps 120; echo "rc=$?"; '
              'cat "%s"/*.gate 2>/dev/null' % (HERE, _gd)],
             env=_env, capture_output=True, text=True)
_until = re.search(r"until=(\d+)", _r.stdout or "")
check("a claim longer than the ticket is clamped to its cap",
      "rc=0" in _r.stdout and _until and int(_until.group(1)) <= int(_env["GPU_TICKET_HARDCAP"]),
      (_r.stdout + _r.stderr).strip()[:200])

# A reader must never guess which box it is reading. With GXCT unset the path used to be
# queue/gx.gate -- a file that never exists -- so a claimed box answered "free", and on 2026-11-02
# 03:1x a hand-run check reported GX2 free while a smoke held it. An unresolvable gate is an error
# that reaches the caller as a refusal (rc 5), never as "free" and never as a wait.
for _label, _gx in (("unset", None), ("unparsable", "bogus")):
    _e = dict(_env)
    _e["GPU_TICKET_HARDCAP"] = str(int(__import__("time").time()) + 600)
    _e.pop("GXCT", None)
    if _gx:
        _e["GXCT"] = _gx
    _r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; pair_gx_gate_file 2>/dev/null >/dev/null; '
                  'echo "file_rc=$?"; echo "state=$(pair_gx_gate_state)"; '
                  'pair_gx_gate_up nobody 5 >/dev/null 2>&1; echo "up_rc=$?"' % HERE],
                 env=_e, capture_output=True, text=True)
    check("an unresolvable GXCT (%s) reads unresolved, never free, and refuses a claim" % _label,
          "file_rc=2" in _r.stdout and "state=gate_unresolved" in _r.stdout
          and "up_rc=5" in _r.stdout, (_r.stdout + _r.stderr).strip()[:200])
# ...and a resolvable one still names its own box, so the guard is not a blanket refusal. The claim
# is taken by the probing shell itself, so its pid is alive for the read that follows.
_e = dict(_env); _e["GPU_TICKET_HARDCAP"] = str(int(__import__("time").time()) + 600)
_e["GXCT"] = "seam-gx2-dev"
_r = _sp.run(["bash", "-c", '. "%s/pair_lib.sh"; pair_gx_gate_up keeper 5 >/dev/null 2>&1; '
              'echo "file=$(pair_gx_gate_file)"; echo "state=$(pair_gx_gate_state)"' % HERE],
             env=_e, capture_output=True, text=True)
check("a resolvable GXCT still reads its own box's claim",
      "file=%s/gx2.gate" % _gd in _r.stdout and "state=keeper until=" in _r.stdout,
      (_r.stdout + _r.stderr).strip()[:200])
check("the wrapper hands its hard cap to the ticket's children",
      "export GPU_TICKET_HARDCAP=$hard" in src("gpu_run_pair.sh")
      and "hardcap=%s" in src("gpu_run_pair.sh"), "")
check("the release guard only holds tickets that name a pair container",
      'check_release "$@"' in src("gpu_run_pair.sh")
      and 'PAIR_CONTAINERS="seam-p7-dev seam-p7-prod seam-gx2-dev seam-gx1-prod"' in src("gpu_run_pair.sh")
      and "PAIR_CHECK_FORCE" in src("gpu_run_pair.sh"), "")
_sh.rmtree(_gd, ignore_errors=True)
check("the ensure gate has its own battery",
      os.path.exists(os.path.join(HERE, "tests", "gx_gate_ensure.sh")), "")
print("\nRESULT %s fails=%d" % ("PASS" if not fails else "FAIL", len(fails)))
sys.exit(1 if fails else 0)

#!/bin/bash
# smoke_memory.sh -- the smoke tells a full box apart from a bad release.
#
#   ./tests/smoke_memory.sh
#
# The worker refuses a pack that does not fit in MemAvailable and exits 3, but the ensure wrapper
# reports "died" (rc 2) for that too. Reading only the exit code turned "GX2 has 800 MiB free"
# into "the release is broken" and rolled a good release back. pair_smoke.sh now reads the verdict
# from SEAM_NO_MEMORY / SEAM_NO_PACK in the ensure output, whatever the wrapper's rc was.
#
# Swept with stubs: no container, no GPU, no server. Only pair_smoke.sh runs; the ensure it would
# start is replaced by a script that prints the lines the real one printed tonight.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
SHA=$(git -C "$(dirname "$(dirname "$(dirname "$HERE")")")" rev-parse HEAD)
WORK=$(mktemp -d /tmp/pairsm.XXXXXX)
mkdir -p "$WORK/bin" "$WORK/state" "$WORK/pair/releases/$SHA/src" "$WORK/pairs/dev"
cp "$HERE/stub_ssh.sh" "$WORK/bin/stub_ssh"; chmod +x "$WORK/bin/stub_ssh"
printf '{"build_id":"sm-demo","seam":{"fingerprint":"fp:dry","proto":"3"}}' > "$WORK/pair/releases/$SHA/RELEASE.json"
cd "$REL"
export PAIR_ROOT_OVERRIDE="$WORK/pair" PAIR_STATE_OVERRIDE="$WORK/pairs" PAIR_SSH="$WORK/bin/stub_ssh" \
       PAIR_QUEUE="$WORK/noq"
unset PAIR_DRY PAIR_DRY_SSH
fails=0
want() {  # <case> <expected-substring> <text>
  if printf '%s' "$3" | grep -q -- "$2"; then echo "OK   $1"
  else echo "FAIL $1 (wanted '$2')"; printf '%s\n' "$3" | tail -6 | sed 's/^/     /'; fails=$((fails + 1)); fi
}
want_not() {
  if printf '%s' "$3" | grep -q -- "$2"; then echo "FAIL $1 (found '$2')"; fails=$((fails + 1)); else echo "OK   $1"; fi
}
ensure_stub() {  # <rc> <lines...>
  local rc=$1; shift
  { echo '#!/bin/bash'
    for l in "$@"; do printf 'echo %q\n' "$l"; done
    printf 'exit %s\n' "$rc"
  } > "$WORK/bin/ensure"; chmod +x "$WORK/bin/ensure"
}

# 1. The worker's own refusal, seen through a wrapper that calls it "died".
ensure_stub 2 \
  "SEAM_ENSURE state=died after=120s" \
  "SEAM_NO_MEMORY need_mib=116362+1024 avail_mib=116574 swapped_mib=834 - refusing to load"
out=$(SEAM_ENSURE="$WORK/bin/ensure" ./pair_smoke.sh dev "$SHA" 2>&1); rc=$?
want "a worker memory refusal is a memory verdict" "SMOKE_FAIL memory server_no_memory SEAM_NO_MEMORY need_mib=116362" "$out"
want_not "and is not reported as a dead server" "server_timeout_or_died" "$out"
[ "$rc" = 1 ] && echo "OK   smoke exits 1" || { echo "FAIL smoke exit=$rc"; fails=$((fails + 1)); }

# 2. A missing pack is a release fault, not a full box: it must not take the memory path.
ensure_stub 3 "SEAM_ENSURE state=died after=3s" "SEAM_NO_PACK dir=/pair/current/model - expected cold-L00.safetensors"
out=$(SEAM_ENSURE="$WORK/bin/ensure" ./pair_smoke.sh dev "$SHA" 2>&1); rc=$?
want "a missing pack is its own reason" "SMOKE_FAIL no_pack SEAM_NO_PACK" "$out"
want_not "and is not called memory" "SMOKE_FAIL memory" "$out"

# 3. A wrapper that dies without a verdict stays a timeout, and does not pretend to be memory.
ensure_stub 2 "SEAM_ENSURE state=died after=120s"
out=$(SEAM_ENSURE="$WORK/bin/ensure" ./pair_smoke.sh dev "$SHA" 2>&1); rc=$?
want "a silent death stays a timeout" "SMOKE_FAIL server_timeout_or_died" "$out"
want_not "with no memory verdict invented" "SMOKE_FAIL memory" "$out"

# 4. The grant is the smoke's, not the pair's: 2048 rows reserve 294 MiB the smoke never touches.
ensure_stub 0 "SEAM_ENSURE state=started port=1"
out=$(SEAM_ENSURE="$WORK/bin/ensure" ./pair_smoke.sh dev "$SHA" 2>&1)
want "the server is asked for the smoke's buckets" "buckets=1,6,64" "$out"

echo "RESULT $([ "$fails" = 0 ] && echo PASS || echo FAIL) fails=$fails"
rm -rf "$WORK"
[ "$fails" = 0 ]

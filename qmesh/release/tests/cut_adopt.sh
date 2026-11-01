#!/bin/bash
# A release is one object with two copies: whatever RELEASE.json one side already carries decides the
# stamp for both. Re-rendering it on the other side puts a different build_id/cut_at into the same
# release, and the smoke compares the build id the client reads with the one the server reads.
#   ./tests/cut_adopt.sh            stub ssh (real transport seam, not PAIR_DRY), no container, no queue, no GPU
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
REPO=$(cd "$REL/../.." && pwd)
SHA=$(git -C "$REPO" rev-parse HEAD)
[ -n "$SHA" ] || { echo "FAIL no sha"; exit 1; }
W=$(mktemp -d /tmp/cutadopt.XXXXXX)
P7=$W/p7; GX=$W/gx
mkdir -p "$P7/releases" "$GX/releases"

cat > "$W/stub_ssh" <<EOF
#!/bin/bash
while [ \$# -gt 0 ]; do case "\$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
shift
cmd="\$*"
cmd=\$(printf '%s' "\$cmd" | sed "s|$P7|$GX|g")
exec bash -c "\$cmd"
EOF
chmod +x "$W/stub_ssh"

fails=0
check() { if [ "$1" = 1 ]; then echo "OK   $2"; else echo "FAIL $2 ${3:-}"; fails=$((fails+1)); fi; }

run() {  # <extra env...> -- runs cut with the p7 override and the stub transport
  PAIR_SSH="$W/stub_ssh" PAIR_ROOT_OVERRIDE="$P7" "$REL/pair_cut.sh" dev "$SHA" "$@" 2>&1
}

out1=$(run --no-gx); rc1=$?
build1=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' \
          "$P7/releases/$SHA/RELEASE.json" 2>/dev/null)
check "$([ "$rc1" = 0 ] && [ -n "$build1" ] && echo 1)" "a first cut stamps a release" "rc=$rc1 build=$build1"

# move the release to the GX side only: the p7 half is now missing, the pair already has the stamp
mkdir -p "$GX/releases"
mv "$P7/releases/$SHA" "$GX/releases/$SHA"
cp "$GX/releases/$SHA/RELEASE.json" "$W/stamp.json"

out2=$(run); rc2=$?
build2=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' \
          "$P7/releases/$SHA/RELEASE.json" 2>/dev/null)
check "$(printf '%s' "$out2" | grep -q CUT_ADOPT && echo 1)" "a missing side adopts the stamp the pair already has" "$out2"
check "$([ "$rc2" = 0 ] && [ "$build2" = "$build1" ] && echo 1)" \
      "both sides end up with one build id" "rc=$rc2 p7=$build2 was=$build1"
check "$(cmp -s "$W/stamp.json" "$P7/releases/$SHA/RELEASE.json" && echo 1)" \
      "the adopted stamp is copied verbatim, not re-rendered"

# a stamp that describes a different tree must be refused, not silently adopted
rm -rf "$P7/releases/$SHA"
python3 - "$GX/releases/$SHA/RELEASE.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); d["digest"] = "0" * 64
json.dump(d, open(p, "w"), indent=2, sort_keys=True)
PY
out3=$(run); rc3=$?
check "$([ "$rc3" != 0 ] && printf '%s' "$out3" | grep -q "does not describe this commit" && echo 1)" \
      "a stamp from a different tree is refused" "rc=$rc3 $out3"
check "$([ ! -d "$P7/releases/$SHA" ] && echo 1)" "a refused cut leaves no half release behind"

# --force re-stamps both sides even when one already has the release
# (the stamp has one-second resolution: a cut in the same second is the same stamp)
sleep 1
out4=$(run --force); rc4=$?
build4=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' \
          "$P7/releases/$SHA/RELEASE.json" 2>/dev/null)
check "$([ "$rc4" = 0 ] && [ "$build4" != "$build1" ] && echo 1)" \
      "--force stamps a new release instead of adopting" "rc=$rc4 build=$build4"

rm -rf "$W"
echo "RESULT $([ "$fails" = 0 ] && echo PASS || echo FAIL) fails=$fails"
exit $([ "$fails" = 0 ] && echo 0 || echo 1)

#!/usr/bin/env bash
# pair_promote.sh -- move the OTHER pair onto the release dev is already serving.
#
#   pair_promote.sh <sha> [--force] [--max-min N]
#
# Rollout order is not a preference: dev is the pair whose failures are cheap, so a release reaches
# prod only after dev runs it and its check says IN_SYNC. This script refuses otherwise. It also
# refuses to promote a release whose seam fingerprint differs from what dev is serving -- a wire
# change has to be promoted as its own release, never mixed into a version bump.
#
# It then queues pair_update.sh prod <sha>, which lands behind whatever prod is running.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

SHA_ARG=${1:?usage: pair_promote.sh <sha> [--force] [--max-min N]}
shift
FORCE=0; MINS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1 ;;
    --max-min) MINS=$2; shift ;;
    *) pair_die "unknown flag $1" ;;
  esac
  shift
done

pair_load dev
SHA=$(git -C "$PAIR_REPO" rev-parse --verify "$SHA_ARG^{commit}" 2>/dev/null) \
  || pair_die "no such commit: $SHA_ARG"
dev_sha=$(pair_manifest_get release_sha)
dev_seam=$(pair_manifest_get seam fingerprint)
dev_build=$(pair_manifest_get build_id)
pair_step "PROMOTE_CHECK sha=${SHA:0:12} dev_manifest=${dev_sha:0:12} dev_build=${dev_build:-none}"

if [ "$dev_sha" != "$SHA" ]; then
  [ "$FORCE" = 1 ] || pair_die "dev is at ${dev_sha:0:12}, not ${SHA:0:12}; promote what dev ran (or --force)"
  echo "PROMOTE_WARN dev is at ${dev_sha:0:12}; promoting ${SHA:0:12} anyway (--force)"
fi
dev_check=$("$HERE/pair_check.sh" dev --quiet 2>&1)
crc=$?
printf '%s\n' "$dev_check"
[ "$crc" = 0 ] || { [ "$FORCE" = 1 ] || pair_die "dev does not check out (rc=$crc); not promoting"; }

rel=$(pair_release_dir "$SHA")
new_seam=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["seam"]["fingerprint"])' \
           "$rel/RELEASE.json" 2>/dev/null)
[ -n "$new_seam" ] || pair_die "release ${SHA:0:12} has no RELEASE.json at $rel (cut it on dev first)"
if [ "$new_seam" != "$dev_seam" ]; then
  echo "PROMOTE_WARN seam ${new_seam} differs from dev ${dev_seam}"
  [ "$FORCE" = 1 ] || pair_die "seam fingerprint changed; that is a different protocol, promote it deliberately (--force)"
fi
pair_step "PROMOTE_OK sha=${SHA:0:12} seam=$new_seam dev=IN_SYNC"

pair_load prod
cmd=("$HERE/pair_update.sh" prod "$SHA" ${MINS:+--max-min "$MINS"})
"${cmd[@]}"

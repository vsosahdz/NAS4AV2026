#!/usr/bin/env bash
# Run the whole campaign on the spot cluster, as four disjoint shards.
#
#   ./scripts/azure_fanout.sh            # submit and return
#   ./scripts/azure_fanout.sh --wait     # submit and wait for all of them
#
# Four shards because the cluster scales to four nodes, and one command job holds a whole
# node whether it uses sixteen cores or one. Each shard runs its (setting, strategy) pairs
# one per vCPU, so the campaign's 165 units land in roughly one pass rather than 165.
#
# The split is by setting and is disjoint, which is the only coordination the shards need:
# a unit's identity is what it measures, so disjoint settings produce disjoint filenames
# and no shard can overwrite another's work. They share one output path on purpose — a
# shard that restarts after a preemption reads every artifact already written and skips
# those units, so a mass eviction costs only what was in flight.
#
# Settings are ordered longest-first within each shard. `synBetter` carries the widest
# input and costs about twice what `sem` does per evaluation, so a split by count alone
# would leave one node running hours after the others had finished.
set -euo pipefail

cd "$(dirname "$0")/.."

WAIT=0
[[ "${1:-}" == "--wait" ]] && WAIT=1

RESOURCE_GROUP="${NAS4AV_RESOURCE_GROUP:-nas4av-rg}"
WORKSPACE="${NAS4AV_WORKSPACE:-nas4av-ws}"

# Eleven settings over four shards, balanced by cost rather than by count.
SHARDS=(
  "2020/synBetter 2013/semBert"
  "2014Novel2/synBetter 2015/synBetter"
  "2014Essay2/synBetter 2013/synBetter 2015/sem"
  "2020/sem 2014Novel2/sem 2014Essay2/sem 2013/sem"
)

./scripts/azure_assert_subscription.sh

REVISION="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to fan out from a dirty tree: every artifact would name ${REVISION}," >&2
  echo "which does not contain what would run. Commit or stash first." >&2
  exit 1
fi

echo "fanning out at ${REVISION}"
JOBS=()
for index in "${!SHARDS[@]}"; do
  settings="${SHARDS[$index]}"
  job="$(az ml job create \
    -f azure/sweep.yml \
    -g "${RESOURCE_GROUP}" \
    -w "${WORKSPACE}" \
    --set "display_name=campaign-shard-${index}" \
    --set "environment_variables.NAS4AV_SHARD_SETTINGS=${settings}" \
    --set "environment_variables.NAS4AV_GIT_REVISION=${REVISION}" \
    --query name -o tsv)"
  JOBS+=("${job}")
  echo "  shard ${index}: ${job}  <- ${settings}"
done

printf '%s\n' "${JOBS[@]}" > artifacts/azure-campaign-jobs.txt
echo "job names written to artifacts/azure-campaign-jobs.txt"

if (( ! WAIT )); then
  echo
  echo "watch with: ./scripts/azure_fanout_status.sh"
  exit 0
fi

while :; do
  pending=0
  line=""
  for job in "${JOBS[@]}"; do
    state="$(az ml job show -n "${job}" -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
      --query status -o tsv 2>/dev/null || echo "?")"
    line+=" ${state}"
    case "${state}" in
      Completed|Failed|Canceled) ;;
      *) pending=$((pending + 1)) ;;
    esac
  done
  printf '  %s %s\n' "$(date +%H:%M:%S)" "${line}"
  (( pending == 0 )) && break
  sleep 120
done

echo "all shards finished"

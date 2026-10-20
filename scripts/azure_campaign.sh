#!/usr/bin/env bash
# Run the campaign to completion on spot compute, resubmitting after evictions.
#
#   ./scripts/azure_campaign.sh
#
# Eviction is the normal case here, not the exception: the first fan-out lost three of four
# shards to `ReceivedTerminationSignal` within a few hours, having written 68 units and
# left 23 checkpoints behind. Everything needed to carry on was already in place — the
# shared output path, the per-generation checkpoints, the skip-if-complete resume — and the
# campaign still stopped, because resubmitting was a thing a person had to do.
#
# So this is the Azure counterpart of supervise-arm.sh, and it differs in one way that
# matters: it recomputes the shards from what is actually missing before each wave, rather
# than resubmitting the original four. After an eviction the remaining work is not spread
# the way it was at the start, and re-sending the old split would put a node on a shard
# that has nothing left to do while another node carries everything.
set -euo pipefail

cd "$(dirname "$0")/.."

RESOURCE_GROUP="${NAS4AV_RESOURCE_GROUP:-nas4av-rg}"
WORKSPACE="${NAS4AV_WORKSPACE:-nas4av-ws}"
MAX_WAVES="${NAS4AV_MAX_WAVES:-20}"
NODES="${NAS4AV_NODES:-4}"
TOTAL_UNITS=165

ACCOUNT="$(az ml datastore show -n artifacts -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
  --query account_name -o tsv)"

written() {
  az storage blob list --account-name "${ACCOUNT}" --container-name artifacts \
    --prefix "campaign/sweep/" --auth-mode login \
    --query "[?ends_with(name,'.json') && !contains(name,'/scores/')].name" -o tsv 2>/dev/null \
    | sed 's|.*/||; s|\.json$||'
}

active_jobs() {
  az ml job list -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
    --query "[?contains(display_name,'campaign-shard') && (status=='Running' || status=='Queued' || status=='Starting' || status=='Preparing')].name" \
    -o tsv 2>/dev/null || true
}

# Settings that a live shard is already working on, so a new wave can avoid them instead of
# waiting for them.
#
# Waiting was the first version and it was far too conservative: after three of four shards
# were evicted, the survivor held the cluster for fifty minutes while three nodes sat idle
# and the unit count did not move. Spot capacity that is available now may not be available
# later, so declining to use it is not the safe choice it looks like.
claimed_settings() {
  az ml job list -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
    --query "[?contains(display_name,'campaign-shard') && (status=='Running' || status=='Queued' || status=='Starting' || status=='Preparing')].environment_variables.NAS4AV_SHARD_SETTINGS" \
    -o tsv 2>/dev/null | tr '\t' '\n' | tr ' ' '\n' | sed '/^$/d' | sort -u || true
}

# Which settings still have unfinished units, packed into `NODES` shards of roughly equal
# cost. Done in Python because it is the planner that knows what a unit is called, and a
# shell reimplementation of that naming would drift from it silently.
remaining_shards() {
  written > /tmp/nas4av-written.txt
  claimed_settings > /tmp/nas4av-claimed.txt
  .venv/bin/python - "${NODES}" "$(active_jobs | wc -l | tr -d ' ')" <<'PY'
import pathlib, sys
sys.path.insert(0, "src")
from nas4av.sweep import SETTINGS, plan

# Free nodes, not total nodes. A live shard is holding one, and asking the cluster for more
# than it has simply queues jobs that then sit behind the running one -- which looks like
# progress in the job list and is not.
nodes = max(1, int(sys.argv[1]) - int(sys.argv[2]))
done = {line.strip() for line in pathlib.Path("/tmp/nas4av-written.txt").read_text().splitlines() if line.strip()}
claimed = {line.strip() for line in pathlib.Path("/tmp/nas4av-claimed.txt").read_text().splitlines() if line.strip()}
units = plan(settings=list(SETTINGS), strategies=["evolutionary-variant", "canonical-ga", "uniform-random"])

outstanding: dict[str, int] = {}
for unit in units:
    key = f"{unit.corpus}/{unit.features}"
    # A setting a live shard already holds is skipped rather than waited for. The split is
    # by setting, so excluding those is enough to guarantee no two jobs touch the same
    # unit, and the rest of the cluster stays busy.
    if unit.identifier not in done and key not in claimed:
        outstanding[key] = outstanding.get(key, 0) + 1
if not outstanding:
    raise SystemExit(0)


# Longest-first into the emptiest shard. synBetter carries the widest input and costs
# about twice what sem does per evaluation, so weight by that rather than by unit count.
weight = lambda s: 2.0 if s.endswith("synBetter") else 1.0
ordered = sorted(outstanding, key=lambda s: -outstanding[s] * weight(s))
shards: list[list[str]] = [[] for _ in range(nodes)]
load = [0.0] * nodes
for setting in ordered:
    index = min(range(nodes), key=lambda i: load[i])
    shards[index].append(setting)
    load[index] += outstanding[setting] * weight(setting)

for shard in shards:
    if shard:
        print(" ".join(shard))
PY
}

REVISION="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run from a dirty tree: artifacts would name ${REVISION}," >&2
  echo "which does not contain what would run." >&2
  exit 1
fi

for wave in $(seq 1 "${MAX_WAVES}"); do
  count="$(written | wc -l | tr -d ' ')"
  echo "=== wave ${wave}, $(date '+%H:%M:%S'), ${count}/${TOTAL_UNITS} units written ==="
  if (( count >= TOTAL_UNITS )); then
    echo "campaign complete"
    exit 0
  fi

  running="$(active_jobs | wc -l | tr -d ' ')"
  if (( running >= NODES )); then
    echo "  ${running} shard(s) active on ${NODES} node(s); nothing to add"
    sleep 300
    continue
  fi

  # Read line by line rather than with `mapfile`, which macOS does not have: the system
  # bash is 3.2, and `mapfile` arrived in bash 4. The failure is a bare "command not
  # found" in the middle of a wave, which leaves the shard list empty and reads like
  # having nothing left to do.
  SHARDS=()
  while IFS= read -r line; do
    [[ -n "${line}" ]] && SHARDS+=("${line}")
  done < <(remaining_shards)
  if (( ${#SHARDS[@]} == 0 )); then
    if (( running > 0 )); then
      echo "  ${running} shard(s) hold every outstanding setting; waiting"
      sleep 300
      continue
    fi
    echo "  nothing outstanding, but only ${count} units written -- stopping so this is visible" >&2
    exit 1
  fi

  for index in "${!SHARDS[@]}"; do
    job="$(az ml job create -f azure/sweep.yml -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
      --set "display_name=campaign-shard-w${wave}-${index}" \
      --set "environment_variables.NAS4AV_SHARD_SETTINGS=${SHARDS[$index]}" \
      --set "environment_variables.NAS4AV_GIT_REVISION=${REVISION}" \
      --query name -o tsv)"
    echo "  shard ${index}: ${job}  <- ${SHARDS[$index]}"
  done

  sleep 300
done

echo "reached ${MAX_WAVES} waves without completing" >&2
exit 1

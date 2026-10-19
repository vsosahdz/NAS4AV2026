#!/usr/bin/env bash
# Bring the whole campaign back up, from nothing.
#
#   ./scripts/campaign-boot.sh
#
# The supervisor restarts a shard whose process dies. It cannot restart a shard on a
# machine that is switched off, and that is the failure that has actually cost time here:
# the Mac mini went offline for a day with 39 units outstanding, and nothing resumed until
# somebody noticed. Two layers, two different failures — the supervisor handles a killed
# process, this handles a cold machine.
#
# Safe to run at any moment, including while the campaign is already running: it refuses to
# start a shard that is already up rather than racing a second copy against it. Two shards
# writing the same unit would not corrupt an artifact — a unit's identity is what it
# measures, and a truncated file counts as incomplete — but they would do the same work
# twice and halve the throughput of a machine that is already the schedule's bottleneck.
set -u
cd "$(dirname "$0")/.."

# Disjoint settings, so the shards cannot collide. Split to balance wall-clock rather than
# unit count: `synBetter` carries the widest input and costs roughly twice what `sem` does
# per evaluation, so shard B takes fewer settings.
SHARD_A_SETTINGS=(2013/sem 2013/semBert 2015/sem)
SHARD_B_SETTINGS=(2020/synBetter 2013/synBetter 2014Novel2/sem 2020/sem 2014Novel2/synBetter)
STRATEGIES=(evolutionary-variant canonical-ga uniform-random)

start() {
  local name="$1"; shift
  if screen -ls 2>/dev/null | grep -q "[.]${name}\b"; then
    echo "  ${name}: already running, left alone"
    return 0
  fi
  # A stale socket from a machine that lost power looks like a live session to `screen -ls`
  # until it is wiped, and then nothing starts because the name appears taken.
  screen -wipe >/dev/null 2>&1 || true
  screen -dmS "${name}" bash -c "./scripts/supervise-arm.sh --strategies ${STRATEGIES[*]} --settings $* > artifacts/${name}.log 2>&1"
  echo "  ${name}: started on $*"
}

mkdir -p artifacts
echo "campaign boot, $(date '+%Y-%m-%d %H:%M:%S')"
start nas4av-a "${SHARD_A_SETTINGS[@]}"
start nas4av-b "${SHARD_B_SETTINGS[@]}"

sleep 10
echo "  units complete: $(ls artifacts/sweep/*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "  sweep processes: $(pgrep -f 'nas4av.cli sweep' | wc -l | tr -d ' ')"

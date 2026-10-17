#!/usr/bin/env bash
# Keep a shard running until its work is actually done.
#
#   screen -dmS nas4av-a bash -c "./scripts/supervise-arm.sh --strategies ... --settings ..."
#
# A shard has now been killed twice mid-campaign, both times by SIGKILL with the screen
# session gone with it, which points at the machine rather than at the run. Checkpointing
# means almost nothing was lost either time — and the campaign still stopped, because the
# thing that noticed was a person looking, roughly a day later. Resumption that depends on
# somebody checking is not resumption.
#
# So the loop. `run-arm.sh` resumes by default and skips completed units, which is what
# makes restarting safe rather than wasteful: a restart re-reads the artifacts, picks up
# the two or three units that were partway, and carries on.
#
# What this deliberately does not do is retry forever on a run that cannot work. A unit
# failing instantly in a tight loop looks identical to progress from outside, so the loop
# requires evidence of progress — a new artifact, or at least a slow failure — before it
# will keep going.
set -u
cd "$(dirname "$0")/.."

ARTIFACTS="${NAS4AV_ARTIFACTS:-./artifacts}"
MAX_RESTARTS="${NAS4AV_MAX_RESTARTS:-50}"
# Below this, an exit is treated as a crash loop rather than as work. A unit takes tens of
# minutes; anything returning in under two is not making progress.
MIN_PROGRESS_SECONDS="${NAS4AV_MIN_PROGRESS_SECONDS:-120}"

count_artifacts() { ls "${ARTIFACTS}"/sweep/*.json 2>/dev/null | wc -l | tr -d ' '; }

# The same argument shape run-arm.sh takes, parsed again here because the completion test
# has to ask about this shard's work rather than about the campaign as a whole.
STRATEGIES=()
SETTINGS=()
target="settings"
for argument in "$@"; do
  case "$argument" in
    --strategies) target="strategies" ;;
    --settings)   target="settings" ;;
    *) if [ "$target" = "strategies" ]; then STRATEGIES+=("$argument"); else SETTINGS+=("$argument"); fi ;;
  esac
done
[ ${#STRATEGIES[@]} -gt 0 ] || STRATEGIES=(evolutionary-variant)

# How many of this shard's units still have no artifact.
#
# Asked of the planner rather than inferred from run-arm.sh's exit code, which cannot
# answer it: that script runs its strategies in sequence and does not stop when one of them
# dies, so it returns the *last* strategy's status and reports success for a campaign that
# lost its first strategy to a SIGKILL half an hour in. That is exactly what happened, and
# a supervisor trusting the exit code would have declared the shard finished.
outstanding() {
  .venv/bin/python - "${STRATEGIES[@]}" -- "${SETTINGS[@]}" <<'PY'
import pathlib, sys
sys.path.insert(0, "src")
from nas4av.sweep import SETTINGS as ALL, plan

divider = sys.argv.index("--")
strategies = sys.argv[1:divider]
chosen = [tuple(s.split("/", 1)) for s in sys.argv[divider + 1:]] or list(ALL)
done = {p.stem for p in pathlib.Path("artifacts/sweep").glob("*.json")}
units = plan(settings=chosen, strategies=strategies)
print(sum(1 for unit in units if unit.identifier not in done))
PY
}

restarts=0
while :; do
  before="$(count_artifacts)"
  started="$(date +%s)"
  echo "=== attempt $((restarts + 1)), $(date '+%Y-%m-%d %H:%M:%S'), ${before} units complete ==="

  ./scripts/run-arm.sh "$@"
  code=$?

  elapsed=$(( $(date +%s) - started ))
  after="$(count_artifacts)"
  left="$(outstanding 2>/dev/null || echo unknown)"

  if [ "${left}" = "0" ]; then
    echo "=== shard complete: every unit has an artifact, ${after} in total ==="
    exit 0
  fi

  echo "=== exited ${code} after ${elapsed}s; ${before} -> ${after} units, ${left} outstanding ==="

  if [ "${after}" -eq "${before}" ] && [ "${elapsed}" -lt "${MIN_PROGRESS_SECONDS}" ]; then
    echo "=== no progress and it returned in under ${MIN_PROGRESS_SECONDS}s: this is a crash"
    echo "    loop, not an interruption. Stopping so the cause is visible. ==="
    exit 1
  fi

  restarts=$((restarts + 1))
  if [ "${restarts}" -ge "${MAX_RESTARTS}" ]; then
    echo "=== ${MAX_RESTARTS} restarts reached; stopping ==="
    exit 1
  fi

  # Short, because the common case is a machine that has just come back and the work is
  # simply waiting.
  sleep 30
done

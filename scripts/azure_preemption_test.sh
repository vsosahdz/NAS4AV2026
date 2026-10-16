#!/usr/bin/env bash
# Cause a preemption, then check that it cost a generation rather than the unit.
#
#   ./scripts/azure_preemption_test.sh artifacts/sweep/<uninterrupted>.json
#
# The test the sweep is gated on. Three claims are made about interruption — the checkpoint
# is written only after `next_generation`, the generator state travels with it, and a
# truncated artifact counts as incomplete — and none of them is exercised by a job that
# runs to completion. The only way to test the boundary between a process and the storage
# that outlives it is to destroy the process.
#
# Cancelling is not an eviction, and the difference is worth being honest about: Azure ML
# retries an evicted job itself, in place, while a cancel ends it and a human resubmits.
# What both share is the thing under test — a second attempt reading a checkpoint written
# by a first attempt that no longer exists. The scheduling differs; the recovery does not.
set -euo pipefail

cd "$(dirname "$0")/.."

REFERENCE="${1:?usage: azure_preemption_test.sh <uninterrupted-artifact.json>}"
RESOURCE_GROUP="${NAS4AV_RESOURCE_GROUP:-nas4av-rg}"
WORKSPACE="${NAS4AV_WORKSPACE:-nas4av-ws}"
# How long to keep waiting for a checkpoint to appear before giving up on the attempt.
#
# Not how long to wait before killing. Two fixed delays were tried and both killed the job
# during setup — `Running` is reported well before the command starts, while the node is
# still pulling the image and installing the project — which left no checkpoint and made
# the test unable to say whether resume worked or whether there had been nothing to resume
# from. Guessing the delay is guessing at something observable, so the kill now waits on
# the checkpoint itself and happens the moment one lands.
DEADLINE="${NAS4AV_CHECKPOINT_DEADLINE:-900}"
JOB_FILE="azure/preemption-test.yml"
STORE="artifacts"
PREFIX="preemption-test"

az() { command az "$@" -g "${RESOURCE_GROUP}" -w "${WORKSPACE}"; }

# A checkpoint left by an earlier attempt would make the first run resume and the test
# prove nothing, so the pinned path is cleared first. This is the one destructive step and
# it touches only this prefix.
echo "clearing ${STORE}/${PREFIX}"
ACCOUNT="$(command az ml datastore show -n "${STORE}" -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
  --query account_name -o tsv)"
command az storage blob delete-batch --account-name "${ACCOUNT}" --source "${STORE}" \
  --pattern "${PREFIX}/*" --auth-mode login --only-show-errors >/dev/null 2>&1 || true

submit() {
  command az ml job create -f "${JOB_FILE}" -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
    --set "environment_variables.NAS4AV_GIT_REVISION=$(git rev-parse HEAD)" \
    --query name -o tsv
}

status() { az ml job show -n "$1" --query status -o tsv; }

wait_for_running() {
  local job="$1"
  while true; do
    local state; state="$(status "${job}")"
    case "${state}" in
      Running) return 0 ;;
      Completed|Failed|Canceled)
        echo "attempt reached ${state} before it could be interrupted" >&2; return 1 ;;
      *) printf '  %s %s\n' "$(date +%H:%M:%S)" "${state}" ; sleep 20 ;;
    esac
  done
}

wait_for_end() {
  local job="$1"
  while true; do
    local state; state="$(status "${job}")"
    case "${state}" in
      Completed|Failed|Canceled) echo "${state}"; return 0 ;;
      *) printf '  %s %s\n' "$(date +%H:%M:%S)" "${state}" >&2 ; sleep 30 ;;
    esac
  done
}

checkpoints() {
  command az storage blob list --account-name "${ACCOUNT}" --container-name "${STORE}" \
    --prefix "${PREFIX}/checkpoints/" --auth-mode login --query "[].name" -o tsv 2>/dev/null || true
}

echo "=== attempt 1: killed the moment it has something to lose ==="
FIRST="$(submit)"
echo "  ${FIRST}"
wait_for_running "${FIRST}"
echo "  running; waiting for the first checkpoint to reach blob storage"

# Waiting on the checkpoint rather than on the clock is what makes this a test of
# preemption instead of a test of startup latency. It also measures the thing the fan-out
# actually needs to know — whether a checkpoint is durable *while the job is alive* —
# separately from whether a retry can read it, and those fail differently.
DEADLINE_AT=$(( $(date +%s) + DEADLINE ))
LEFT=""
while [[ -z "${LEFT}" ]]; do
  if (( $(date +%s) > DEADLINE_AT )); then
    az ml job cancel -n "${FIRST}" >/dev/null 2>&1 || true
    cat >&2 <<EOF

No checkpoint reached blob storage within ${DEADLINE}s, while the job was still running.

A checkpoint is written after each completed generation, so either the unit is far slower
here than the canary's 120 evaluations in under seven minutes, or writes to the mounted
output are buffered and never reach storage until the job ends. The second is the finding
this test exists to surface, and it would mean spot compute cannot be used as configured:
a checkpoint that only becomes durable on a clean exit is not a checkpoint.
EOF
    exit 1
  fi
  STATE="$(status "${FIRST}")"
  if [[ "${STATE}" != "Running" ]]; then
    echo "attempt ended as ${STATE} before any checkpoint appeared" >&2
    exit 1
  fi
  LEFT="$(checkpoints)"
  [[ -z "${LEFT}" ]] && sleep 20
done
echo "  checkpoint is durable:"
echo "${LEFT}" | sed 's/^/    /'

az ml job cancel -n "${FIRST}" >/dev/null
echo "  cancelled mid-flight: $(wait_for_end "${FIRST}")"

SURVIVED="$(checkpoints)"
if [[ -z "${SURVIVED}" ]]; then
  echo "the checkpoint disappeared when the job was cancelled" >&2
  exit 1
fi

echo "=== attempt 2: must resume, not restart ==="
SECOND="$(submit)"
echo "  ${SECOND}"
FINAL="$(wait_for_end "${SECOND}")"
if [[ "${FINAL}" != "Completed" ]]; then
  echo "the resumed attempt did not complete: ${FINAL}" >&2
  exit 1
fi

OUT="$(mktemp -d)"
command az ml job download -n "${SECOND}" -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
  --download-path "${OUT}" --all --only-show-errors >/dev/null
RESUMED="$(find "${OUT}" -path '*/sweep/*.json' -not -path '*/scores/*' | head -1)"
[[ -n "${RESUMED}" ]] || { echo "the resumed attempt produced no artifact" >&2; exit 1; }

python3 - "${REFERENCE}" "${RESUMED}" <<'PY'
import json, sys

reference, resumed = (json.load(open(p)) for p in sys.argv[1:3])
a, b = reference["run"], resumed["run"]

def fail(message):
    print(f"MISMATCH: {message}", file=sys.stderr)
    sys.exit(1)

# That it resumed at all is the first thing to establish. A job that quietly restarted
# would produce a perfectly matching artifact and prove nothing, so an absent or zero
# resume point is a failure of the test rather than a pass.
generation = b.get("resumed_from_generation")
if not generation:
    fail("the second attempt did not resume; it restarted from scratch, so nothing was tested")

if len(a["records"]) != len(b["records"]):
    fail(f"evaluation count {len(a['records'])} vs {len(b['records'])}")

for index, (left, right) in enumerate(zip(a["records"], b["records"])):
    for field in ("genotype", "params", "failed"):
        if left[field] != right[field]:
            fail(f"record {index} differs on {field}: {left[field]} vs {right[field]}")
    for field in ("fitness", "test_roc_auc", "validation_roc_auc"):
        if field in left and left[field] != right[field]:
            fail(f"record {index} differs on {field}: {left[field]} vs {right[field]}")

# Timing is excluded from every comparison, here and in the records above. `seconds` is how
# long an evaluation took, not what it found: the same architecture on the same data took
# 2.3966 s before the interruption and 2.3570 s after, which is a property of the node and
# would fail a test of reproducibility that nothing can pass.
TIMING = {"seconds"}
left = {k: v for k, v in a["best"].items() if k not in TIMING}
right = {k: v for k, v in b["best"].items() if k not in TIMING}
if left != right:
    differing = {k for k in left.keys() | right.keys() if left.get(k) != right.get(k)}
    fail(f"best differs on {sorted(differing)}: {left} vs {right}")

print(f"resumed from generation {generation}")
print(f"{len(b['records'])} evaluations identical to the uninterrupted run, including the best")
print("PASS: an interruption cost a generation, not the unit")
PY

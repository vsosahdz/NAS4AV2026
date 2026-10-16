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
KILL_AFTER="${NAS4AV_KILL_AFTER:-150}"
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

echo "=== attempt 1: will be interrupted ==="
FIRST="$(submit)"
echo "  ${FIRST}"
wait_for_running "${FIRST}"
echo "  running; letting it work for ${KILL_AFTER}s before cancelling"
sleep "${KILL_AFTER}"
az ml job cancel -n "${FIRST}" >/dev/null
echo "  cancelled: $(wait_for_end "${FIRST}")"

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

if a["best"] != b["best"]:
    fail(f"best differs: {a['best']} vs {b['best']}")

print(f"resumed from generation {generation}")
print(f"{len(b['records'])} evaluations identical to the uninterrupted run, including the best")
print("PASS: an interruption cost a generation, not the unit")
PY

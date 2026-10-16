#!/usr/bin/env bash
# Submit a job to the provisioned workspace, with the things a YAML file cannot state.
#
#   ./scripts/azure_submit.sh azure/canary.yml
#   ./scripts/azure_submit.sh azure/canary.yml --wait
#
# Two of them, and both matter more than their size suggests.
#
# **The revision.** A job snapshot carries no .git directory, so `git rev-parse` on a
# compute node returns nothing and the artifact records a null revision. Locally that is
# survivable because the working tree is right there; on a node deallocated an hour ago it
# is the only route back to the code that produced the numbers. The submitter knows the
# revision, so the submitter states it.
#
# **The dirty check.** A revision declared from a dirty tree is worse than no revision: it
# names a commit that does not contain what ran. Refused rather than warned, because the
# artifact would look complete.
set -euo pipefail

cd "$(dirname "$0")/.."

JOB_FILE="${1:?usage: azure_submit.sh <job.yml> [--wait]}"
shift || true
WAIT=0
[[ "${1:-}" == "--wait" ]] && WAIT=1

RESOURCE_GROUP="${NAS4AV_RESOURCE_GROUP:-nas4av-rg}"
WORKSPACE="${NAS4AV_WORKSPACE:-nas4av-ws}"

./scripts/azure_assert_subscription.sh

REVISION="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  cat >&2 <<EOF
Refusing to submit from a dirty working tree.

  revision would be recorded as: ${REVISION}

That commit does not contain what would run, so every artifact from this job would
name the wrong code while looking fully provenanced. Commit or stash first.
EOF
  exit 1
fi

echo "submitting ${JOB_FILE} at ${REVISION}"
JOB="$(az ml job create \
  -f "${JOB_FILE}" \
  -g "${RESOURCE_GROUP}" \
  -w "${WORKSPACE}" \
  --set "environment_variables.NAS4AV_GIT_REVISION=${REVISION}" \
  --query name -o tsv)"
echo "${JOB}"

if (( WAIT )); then
  # Polled rather than streamed. `az ml job stream` holds a connection for the life of the
  # job, and a dropped connection on a two-hour job looks exactly like a failure.
  while true; do
    STATUS="$(az ml job show -n "${JOB}" -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
      --query status -o tsv)"
    case "${STATUS}" in
      Completed) echo "completed: ${JOB}"; exit 0 ;;
      Failed|Canceled) echo "${STATUS}: ${JOB}" >&2; exit 1 ;;
      *) printf '  %s %s\n' "$(date +%H:%M:%S)" "${STATUS}" ;;
    esac
    sleep 60
  done
fi

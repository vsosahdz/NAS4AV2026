#!/usr/bin/env bash
# What the fan-out has actually produced, as opposed to what its jobs report.
#
#   ./scripts/azure_fanout_status.sh
#
# Job status and campaign progress answer different questions, and on spot compute they
# disagree often enough to matter: a shard can be Failed because it was evicted and still
# have written most of its units, and a shard can be Running for an hour having written
# none. The artifact count is the one that says how much work exists.
set -euo pipefail

cd "$(dirname "$0")/.."

RESOURCE_GROUP="${NAS4AV_RESOURCE_GROUP:-nas4av-rg}"
WORKSPACE="${NAS4AV_WORKSPACE:-nas4av-ws}"
ACCOUNT="$(az ml datastore show -n artifacts -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
  --query account_name -o tsv)"

# Asked of the workspace rather than read from the job list the first fan-out wrote. That
# file names the original four shards only, so after the supervisor has resubmitted a few
# waves it reports four long-dead jobs as the state of the campaign -- every one of them
# Failed, while later shards are running fine.
echo "recent shards:"
az ml job list -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" --max-results 40 \
  --query "reverse(sort_by([?contains(display_name,'campaign-shard')],&creation_context.created_at))[:8].[name,status,display_name]" \
  -o tsv 2>/dev/null | while IFS=$'\t' read -r job status name; do
    printf '  %-28s %-12s %s\n' "${job}" "${status}" "${name}"
  done

units="$(az storage blob list --account-name "${ACCOUNT}" --container-name artifacts \
  --prefix "campaign/sweep/" --auth-mode login \
  --query "length([?ends_with(name, '.json') && !contains(name, '/scores/')])" -o tsv 2>/dev/null || echo 0)"
checkpoints="$(az storage blob list --account-name "${ACCOUNT}" --container-name artifacts \
  --prefix "campaign/checkpoints/" --auth-mode login --query "length(@)" -o tsv 2>/dev/null || echo 0)"

echo
echo "  units written:   ${units} of 165"
echo "  units in flight: ${checkpoints} (have a checkpoint, no artifact yet)"

nodes=0
for cluster in sweep-spot measure-dedicated; do
  count="$(az ml compute list-nodes -n "${cluster}" -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
    -o tsv 2>/dev/null | wc -l | tr -d ' ')"
  nodes=$((nodes + count))
  echo "  ${cluster}: ${count} node(s)"
done
(( nodes == 0 )) && echo "  nothing is allocated; the clusters are at zero"

#!/usr/bin/env bash
# Read the quota Azure ML actually enforces, and check the planned clusters against it.
#
# There are two counters for the same VM family and they are granted independently. The
# one `az vm list-usage` reports is Microsoft.Compute; Azure ML enforces its own under
# Microsoft.MachineLearningServices, validating each cluster's maximum at creation rather
# than against running cores. A plan that fits the first and not the second fails as
# ClusterMinNodesExceedCoreQuota while the family allowance still reads zero used.
#
#   ./scripts/azure_check_quota.sh

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/infra/target.tfvars"

read_var() {
  sed -n "s/^$1[[:space:]]*=[[:space:]]*\"\{0,1\}\([^\"]*\)\"\{0,1\}[[:space:]]*$/\1/p" "$TARGET" | head -1 | tr -d ' '
}

"$ROOT/scripts/azure_assert_subscription.sh" || exit 1

az extension show --name ml >/dev/null 2>&1 || {
  echo "  the az ml extension is required:  az extension add --name ml"
  exit 1
}

LOCATION="$(read_var location)"
SKU="$(read_var sweep_sku)"
MEASURE_SKU="$(read_var measurement_sku)"
MEASURE_PER="$(read_var measurement_units_per_node)"
SWEEP="$(read_var sweep_max_nodes)"
MEASURE="$(read_var measurement_max_nodes)"
PER_NODE="$(read_var units_per_node)"

echo
echo "  region      $LOCATION"
echo "  sweep sku   $SKU"
echo "  measure sku $MEASURE_SKU"
echo "  planned     spot      $SWEEP nodes x $PER_NODE vCPU = $(( SWEEP * PER_NODE ))"
echo "              dedicated $MEASURE node(s) x $MEASURE_PER vCPU = $(( MEASURE * MEASURE_PER ))"
echo

echo "  --- is the SKU supported by Azure ML at all ---"
echo "  Azure ML keeps its own allow-list of VM sizes, narrower than what the subscription"
echo "  permits. A size can be unrestricted for the subscription and still be refused at"
echo "  cluster creation as InvalidPropertyValue. This subscription has NO CPU size that"
echo "  is both ML-supported and available, in this region or in eastus2 — it was"
echo "  provisioned for GPU work, so the sweep runs on a T4 and uses its vCPUs."
echo

echo "  --- Azure ML quota, the counter that is enforced ---"
echo "  Dedicated and low-priority are granted SEPARATELY for the same family, and they"
echo "  are nothing alike here: 4 dedicated vCPU against an uncapped low-priority. Sizing"
echo "  both clusters against one number makes the dedicated one impossible."
echo
WORKSPACE="$(az ml workspace list -g nas4av-rg --query "[0].name" -o tsv 2>/dev/null)"
if [ -n "$WORKSPACE" ]; then
  az ml compute list-usage -g nas4av-rg -w "$WORKSPACE" -o json 2>/dev/null | python3 -c "
import json,sys
for row in json.load(sys.stdin):
    name = row.get('name') or {}
    label = name.get('localized_value') or name.get('localizedValue') or ''
    if 'NC' in label.upper() or 'Total' in label:
        print(f\"    {label:<48}{row.get('current_value', row.get('currentValue')):>5} / {row.get('limit')}\")
" || echo "  (could not read the workspace usage)"
else
  echo "  (no workspace yet — the enforced counter is per workspace; apply the platform first)"
fi

echo
echo "  --- Microsoft.Compute quota, for contrast, NOT the one that binds ---"
az vm list-usage --location "$LOCATION" \
  --query "[?contains(localName, 'Standard F') || contains(localName, 'Total Regional')].{family:localName, used:currentValue, limit:limit}" \
  -o table 2>/dev/null || echo "  (could not read)"

echo
echo "  --- is the SKU available here at all ---"
az vm list-skus --location "$LOCATION" --size "$SKU" --all \
  --query "[0].{name:name, restrictions:restrictions[].reasonCode}" -o json 2>/dev/null \
  || echo "  (could not read)"

echo
echo "  If the enforced limit is below the planned total, lower the maximums in"
echo "  target.tfvars or request more quota. Insufficient quota changes the schedule,"
echo "  not the experimental design."

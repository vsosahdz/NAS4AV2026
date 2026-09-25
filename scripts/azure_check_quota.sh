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
SKU="$(read_var cpu_sku)"
SWEEP="$(read_var sweep_max_nodes)"
MEASURE="$(read_var measurement_max_nodes)"
PER_NODE="$(read_var units_per_node)"
DECLARED="$(read_var family_quota_vcpu)"

echo
echo "  region      $LOCATION"
echo "  sku         $SKU"
echo "  planned     ($SWEEP + $MEASURE) nodes x $PER_NODE vCPU = $(( (SWEEP + MEASURE) * PER_NODE ))"
echo "  declared    $DECLARED vCPU in target.tfvars"
echo

echo "  --- Azure ML quota, the counter that is enforced ---"
az ml compute list-usage --location "$LOCATION" \
  --query "[?contains(name.localizedValue, 'Standard F') || contains(name.localizedValue, 'Total')].{family:name.localizedValue, used:currentValue, limit:limit}" \
  -o table 2>/dev/null || echo "  (could not read; check the extension and your permissions)"

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

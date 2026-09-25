#!/usr/bin/env bash
# Assert the active Azure subscription is the one this campaign is pinned to.
#
# Called by every script that touches Azure, and by the Terraform wrapper. The failure it
# prevents is quiet and expensive: resources created against whichever subscription
# happened to be active, billing a budget that did not authorise them.
#
#   ./scripts/azure_assert_subscription.sh          # check only
#   ./scripts/azure_assert_subscription.sh --set    # check, and switch if needed

set -uo pipefail
TARGET_FILE="$(dirname "$0")/../infra/target.tfvars"

expected="$(sed -n 's/^subscription_name[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "$TARGET_FILE")"
[ -n "$expected" ] || { echo "could not read subscription_name from $TARGET_FILE"; exit 1; }

command -v az >/dev/null 2>&1 || { echo "azure-cli is not installed."; exit 1; }
az account show >/dev/null 2>&1 || { echo "Not signed in. Run: az login"; exit 1; }

active="$(az account show --query name -o tsv)"

if [ "$active" = "$expected" ]; then
  echo "  subscription: $active  (matches the pinned target)"
  exit 0
fi

echo "  ACTIVE SUBSCRIPTION DOES NOT MATCH THE PINNED TARGET"
echo "    active:   $active"
echo "    expected: $expected"

if [ "${1:-}" = "--set" ]; then
  echo "  switching..."
  az account set --subscription "$expected" \
    && echo "  now active: $(az account show --query name -o tsv)" \
    && exit 0
  echo "  could not switch — is the name exact, and do you have access?"
  exit 1
fi

echo
echo "  Fix with:  az account set --subscription \"$expected\""
echo "  Or rerun:  $0 --set"
exit 1

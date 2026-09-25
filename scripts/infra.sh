#!/usr/bin/env bash
# Terraform, with the guards that make an apply safe to run without thinking.
#
#   ./scripts/infra.sh plan       what would change
#   ./scripts/infra.sh apply      create or update the platform
#   ./scripts/infra.sh output     what was created
#   ./scripts/infra.sh destroy    tear it down
#
# Every path asserts the pinned subscription first. Creating resources against the wrong
# one bills a budget that did not authorise them, and the mistake is invisible until the
# invoice.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/infra"

COMMAND="${1:-}"
shift || true
[ -n "$COMMAND" ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

"$ROOT/scripts/azure_assert_subscription.sh" || exit 1
command -v terraform >/dev/null 2>&1 || { echo "  terraform is not installed."; exit 1; }

[ -d .terraform ] || terraform init -input=false || exit 1

case "$COMMAND" in
  plan)
    terraform plan -input=false -var-file=target.tfvars "$@"
    ;;
  apply)
    echo
    echo "  Before applying, confirm the quota arithmetic holds:"
    echo "    ./scripts/azure_check_quota.sh"
    echo
    terraform apply -input=false -var-file=target.tfvars "$@"
    ;;
  output)
    terraform output "$@"
    ;;
  destroy)
    echo
    echo "  Teardown deletes the storage account, and with it anything not already"
    echo "  downloaded. Confirm results are collected and checksummed first."
    echo
    terraform destroy -input=false -var-file=target.tfvars "$@"
    ;;
  *)
    echo "  unknown command: $COMMAND"
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac

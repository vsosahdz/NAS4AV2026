#!/usr/bin/env bash
# Bring the campaign's artifacts down, and refuse to call them a campaign until they are.
#
#   ./scripts/azure_fetch_results.sh [destination]
#
# Downloading is the easy half. The half worth scripting is the check: a campaign assembled
# from several waves of spot jobs can be short of units, can contain two platforms, or can
# carry a checkpoint for a unit whose artifact never landed — and all three look like a
# full directory of JSON until something downstream quietly averages over them.
#
# So this refuses rather than warns. An analysis run against 162 of 165 units produces
# numbers, and nothing in them says which three are missing.
set -euo pipefail

cd "$(dirname "$0")/.."

DESTINATION="${1:-artifacts}"
RESOURCE_GROUP="${NAS4AV_RESOURCE_GROUP:-nas4av-rg}"
WORKSPACE="${NAS4AV_WORKSPACE:-nas4av-ws}"
EXPECTED_UNITS="${NAS4AV_EXPECTED_UNITS:-165}"

ACCOUNT="$(az ml datastore show -n artifacts -g "${RESOURCE_GROUP}" -w "${WORKSPACE}" \
  --query account_name -o tsv)"

echo "downloading campaign/ from ${ACCOUNT} into ${DESTINATION}/"
mkdir -p "${DESTINATION}"
az storage blob download-batch \
  --account-name "${ACCOUNT}" \
  --source artifacts \
  --pattern "campaign/*" \
  --destination "${DESTINATION}" \
  --auth-mode login \
  --no-progress \
  --overwrite >/dev/null

# The blobs land under campaign/, and the rest of the codebase expects artifacts/sweep/.
# Moved rather than symlinked so that a later analysis cannot read half of one layout.
if [[ -d "${DESTINATION}/campaign" ]]; then
  mkdir -p "${DESTINATION}/sweep"
  cp -R "${DESTINATION}/campaign/." "${DESTINATION}/"
  rm -rf "${DESTINATION}/campaign"
fi

NAS4AV_ARTIFACTS="${DESTINATION}" .venv/bin/python - "${EXPECTED_UNITS}" <<'PY'
import json, pathlib, sys
sys.path.insert(0, "src")
from nas4av.sweep import SETTINGS, plan

expected = int(sys.argv[1])
root = pathlib.Path(__import__("os").environ["NAS4AV_ARTIFACTS"])
sweep = root / "sweep"
present = {p.stem for p in sweep.glob("*.json")}
units = plan(settings=list(SETTINGS), strategies=["evolutionary-variant", "canonical-ga", "uniform-random"])
missing = [u.identifier for u in units if u.identifier not in present]

print(f"  units present: {len(present)} of {len(units)} planned (expected {expected})")

problems = []
if missing:
    problems.append(f"{len(missing)} unit(s) missing, first few: {missing[:5]}")

# A checkpoint beside a complete campaign means a unit stopped partway and its artifact
# never landed, even if the count happens to come out right.
stale = [p.stem for p in (root / "checkpoints").glob("*.json")] if (root / "checkpoints").is_dir() else []
orphaned = [s for s in stale if s not in present]
if orphaned:
    problems.append(f"{len(orphaned)} checkpoint(s) with no artifact: {orphaned[:5]}")

# Platform homogeneity. Two platforms in one comparison is the defect that cost seven
# units the first time, and it is invisible in a directory listing.
environments: dict[str, int] = {}
for path in sweep.glob("*.json"):
    provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
    key = json.dumps(
        {"platform": provenance.get("platform"), **provenance.get("packages", {})}, sort_keys=True
    )
    environments[key] = environments.get(key, 0) + 1

print(f"  distinct environments: {len(environments)}")
for key, count in sorted(environments.items(), key=lambda kv: -kv[1]):
    record = json.loads(key)
    print(f"    {count:>4} units  {record.get('platform')}  torch {record.get('torch')}")
if len(environments) > 1:
    problems.append(f"{len(environments)} platforms in one campaign")

if problems:
    print("\nREFUSING to treat this as a complete campaign:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    raise SystemExit(1)
print("\n  complete and homogeneous")
PY

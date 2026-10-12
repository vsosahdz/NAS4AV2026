#!/usr/bin/env bash
# One shard of a campaign, over one or more strategies.
#
# A full comparison is three strategies over eleven settings, which is tens of hours. The
# work is split two ways: by strategy, run in sequence so a failure in one does not take
# the others down; and by setting, run as concurrent shards on a machine with cores to
# spare. Shards cannot collide, because a unit's identity is what it measures and disjoint
# settings produce disjoint filenames.
#
#   ./scripts/run-arm.sh --strategies evolutionary-variant uniform-random \
#                        --settings 2015/sem 2020/sem
#
# Detached, for a long run:
#
#   screen -dmS arm-a bash -c "./scripts/run-arm.sh ... > artifacts/arm-a.log 2>&1"
#
# Resumption is automatic and is what makes this safe to re-run after a crash: a unit that
# already has a complete artifact is skipped, and a truncated one is not mistaken for
# complete.

set -u
cd "$(dirname "$0")/.."

export NAS4AV_ARTIFACTS=./artifacts
export NAS4AV_SEED=20260920
export PYTHONHASHSEED=0

# The extracted pipeline emits deprecation warnings on every batch. They are real and
# recorded in the design notes; repeating them a million times buries everything else.
export PYTHONWARNINGS=ignore

# Pinned for two reasons that happen to agree. float64 reduction order varies with the
# thread count, so an unpinned run is not reproducible; and one shard on one core is what
# keeps two shards from contending for the same machine.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

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
[ ${#SETTINGS[@]} -gt 0 ] || { echo "  no settings given"; exit 1; }

# Sequentially, not in parallel. Two strategies at once would double the memory on a
# machine where the widest setting has already taken a process down once, and the
# comparison needs all three finished rather than all three started.
for strategy in "${STRATEGIES[@]}"; do
  echo "=== ${strategy} ==="
  .venv/bin/python -m nas4av.cli sweep \
    --strategies "$strategy" \
    --runs 5 \
    --settings "${SETTINGS[@]}"
  echo
done

#!/usr/bin/env bash
# One shard of a campaign arm.
#
# A sweep is tens of hours, so it is split across processes by setting rather than run as
# one. The shards write into the same artifact directory and cannot collide: a unit's
# identity is what it measures, so disjoint settings produce disjoint filenames.
#
# Split by *measured* cost rather than by count, so the shards finish together. The
# per-setting rates come from `nas4av.cli rate`.
#
#   ./scripts/run-arm.sh 2014Essay2/synBetter 2015/sem 2020/sem
#
# For a long run, detach it:
#
#   screen -dmS nas4av-a bash -c "./scripts/run-arm.sh <settings> > artifacts/arm-a.log 2>&1"

set -u
cd "$(dirname "$0")/.."

export NAS4AV_ARTIFACTS=./artifacts
export NAS4AV_SEED=20260920
export PYTHONHASHSEED=0

# The extracted pipeline emits deprecation warnings on every batch. They are real and
# recorded in the design notes; repeating them a million times buries everything else.
export PYTHONWARNINGS=ignore

# Pinned for two reasons, and both matter. float64 reduction order varies with the thread
# count, so an unpinned run is not reproducible; and one shard on one core is what keeps
# two shards from contending for the same machine.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

exec .venv/bin/python -m nas4av.cli sweep \
  --strategies evolutionary-variant \
  --runs 5 \
  --settings "$@"

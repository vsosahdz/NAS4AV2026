# NAS4AV

Neural architecture search for authorship verification over the CLEF-PAN English
subsets, reporting predictive performance and parameter count together.

## Environment

Python 3.12 exactly.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv     # adjust the interpreter path
.venv/bin/python -m pip install --upgrade pip
```

Dependencies are split so that the analysis tier never requires a CUDA runtime.

```bash
# analysis only — CPU, no CUDA, installs on macOS and CPU-only Linux
.venv/bin/python -m pip install -e .

# add training — pulls a CUDA build of PyTorch on Linux
.venv/bin/python -m pip install -e ".[train]"

# add Azure job submission
.venv/bin/python -m pip install -e ".[azure]"

# contributors
.venv/bin/python -m pip install -e ".[dev]"
```

Always invoke `.venv/bin/python` explicitly. The system `python3` may resolve to an
unrelated interpreter; on some machines it is 3.9, which this package does not support.

The PyTorch index for CUDA wheels is declared in `pyproject.toml` rather than in these
instructions, so the environment resolves without any remembered command-line flag.

### Hardware notes

The models use float64 `BatchNorm1d`, which Apple MPS does not support, so training on
Apple Silicon runs on CPU. Artifacts record the device that produced them; two artifacts
differing only in that field are not expected to agree bit-for-bit.

## Running the checks

```bash
.venv/bin/python -m pytest tests/ -q          # unit tests
.venv/bin/python -m ruff check src tests      # lint
.venv/bin/python -m ruff format --check src tests
.venv/bin/python -m mypy src/nas4av           # type check
```

Tests marked `legacy` read the previous campaign's published tables and are skipped
unless that repository is present:

```bash
git clone https://github.com/luisferro2/NAS_4_AV.git reference/NAS_4_AV
git -C reference/NAS_4_AV checkout 65ee41046f8c3417e4159f8f9bc96ca1a7ddf84b
```

It is pinned to that commit because every figure derived from it is derived from that
state. `reference/` is not vendored into this repository: it belongs to its own authors,
so anything asserted about it stays checkable against the source.

Tests marked `gpu` need a CUDA device and are excluded from the analysis tier.

## Reproducibility

No number reported from this work is entered by hand. Every stage writes its output to
`artifacts/` with a provenance header recording the git revision and whether the tree
was dirty, the Python and platform versions, the versions of the packages whose
resolution can change a result, the seed, the compute device, and the command that ran.

`artifacts/` is not committed. Everything in it regenerates from this repository, and
tables that need a citable identifier are distributed as releases instead.

Determinism is controlled by environment variables that are recorded alongside the
results:

```bash
export NAS4AV_SEED=...            # recorded in every artifact
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1          # thread count changes float64 reduction order
export MKL_NUM_THREADS=1
export NAS4AV_ARTIFACTS=./artifacts
export NAS4AV_REFERENCE=./reference/NAS_4_AV
```

## Layout

```
src/nas4av/
  space/        genotype, feasibility constraints, exact parameter cost
  prior/        readers for the previous campaign's published tables
  artifacts.py  provenance-stamped artifact writing
tests/          unit tests, including the closed-form cost check against PyTorch
infra/          Terraform for the Azure platform
scripts/        subscription guard, quota check, Terraform wrapper
reference/      the previous campaign's repository, obtained not vendored
artifacts/      generated output
```

## Running a campaign

```bash
.venv/bin/python -m nas4av.cli plan      # enumerate and cost it
.venv/bin/python -m nas4av.cli rate      # measure seconds per evaluation, per setting
.venv/bin/python -m nas4av.cli sweep     # run, resuming by default
.venv/bin/python -m nas4av.cli status    # what is already on disk
```

The sweep is resumable: each run writes its own artifact when it finishes, and a
truncated file counts as incomplete rather than done.

On Azure, the platform is declared in `infra/` and applied through a wrapper that asserts
the pinned subscription first:

```bash
./scripts/azure_check_quota.sh     # read the allowance Azure ML actually enforces
./scripts/infra.sh plan
./scripts/infra.sh apply
```

The compute is CPU rather than GPU, which is a measurement and not a preference: these
architectures run about 2.5x faster on CPU than on the GPU the prior campaign used,
because at this size kernel launch overhead dominates the arithmetic.

## License

Not yet chosen.

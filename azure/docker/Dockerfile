# The job image, built to match the reference platform rather than to inherit one.
#
# The first three canaries ran on a curated Azure ML PyTorch image, which is the obvious
# choice and the wrong one here. That image ships Python 3.10 and torch 2.2; the Mac mini
# that is this project's reference platform runs Python 3.12 and torch 2.14.1. A result
# produced under the curated image could not be compared against one produced on the mini,
# which is the entire purpose of running anything on Azure.
#
# The failure was at least loud: the project declares `requires-python = "==3.12.*"`, so
# pip refused to install rather than running under the wrong interpreter. That refusal is
# the only reason the mismatch surfaced before a campaign instead of after one.
#
# So the image is built from uv.lock — the same lock the mini is pinned to — and the
# versions are a consequence of that file rather than of this one.

FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

# A base without CUDA, on purpose. A PyPI torch wheel for Linux carries its own CUDA
# runtime as nvidia-* dependencies and needs only the driver, which comes from the host.
# Layering those libraries over a CUDA base image would mean two copies of the runtime and
# an unresolved question about which one loads.

# Pinned by digest-equivalent tag rather than :latest. The lock file fixes what is
# installed; this fixes what does the installing, and a resolver that changes under us can
# produce a different environment from the same lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/

# Outside /opt/miniconda so the base image's own conda environment is left intact. Azure
# ML's runtime uses it for job bookkeeping, and replacing its interpreter breaks the job
# before any of this code runs.
ENV UV_PROJECT_ENVIRONMENT=/opt/nas4av \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /build
COPY pyproject.toml uv.lock README.md ./

# `--frozen` is the point of the whole file: it installs what the lock says and fails if
# the lock disagrees with pyproject.toml, rather than quietly re-resolving. Without it this
# image would drift from the mini the same way the two machines drifted from each other.
#
# `--no-install-project` leaves the project itself out. The code travels with each job
# snapshot and changes between jobs; the environment does not, so baking the code in would
# rebuild this image on every edit and make "which commit produced this" a question about
# the image rather than about the snapshot.
RUN uv python install 3.12.8 \
 && uv sync --frozen --no-install-project --extra train --extra azure

# The interpreter jobs get. A command in a job YAML runs `python`, so this has to resolve
# to the locked environment and not to the base image's conda Python.
ENV PATH=/opt/nas4av/bin:$PATH \
    VIRTUAL_ENV=/opt/nas4av

# Reduction order must not vary with the thread count, or two runs of the same unit on the
# same data disagree in the last digits and the verify tier cannot tell a real mismatch
# from a scheduling one. Set here as well as in the job, because a job that forgets them
# should still be reproducible.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0

# Fail the build rather than the campaign if the locked environment is not what the lock
# says. An image that builds and then turns out to hold the wrong torch costs a queue wait
# and a node to discover.
RUN python -c "import sys, torch, numpy, scipy; \
assert sys.version_info[:2] == (3, 12), sys.version; \
assert torch.__version__.split('+')[0] == '2.14.1', torch.__version__; \
assert numpy.__version__ == '2.5.3', numpy.__version__; \
assert scipy.__version__ == '1.18.1', scipy.__version__; \
print('environment matches the reference platform:', sys.version.split()[0], torch.__version__)"

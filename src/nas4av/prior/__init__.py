"""Reading and analysing the prior campaign's published artifacts.

Everything here reads ``reference/NAS_4_AV`` at the pinned commit and writes findings
into the artifact tree. None of it trains anything, so all of it belongs to the
``verify`` replication tier.

The distinction from ``nas4av.original`` matters: that package holds code extracted
from the published notebook and is kept as close to it as running outside Jupyter
allows, so a reproduction mismatch is a real finding. This package is our own.
"""

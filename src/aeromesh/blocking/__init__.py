"""Block decomposition.

``splits`` supplies the decomposition splits: curves of the medial coordinate
system, orthogonal to the wall by construction and ranked the way Fogg ranks
them. ``blocks`` turns a chosen set of them into blocks, constructively from the
medial graph rather than by computing a planar arrangement.
"""

from aeromesh.blocking.blocks import Block, BlockSystem, boundary_arc, decompose
from aeromesh.blocking.splits import (
    Split,
    best_split,
    candidate_splits,
    radius_split,
    split_rank,
)

__all__ = [
    "Split", "split_rank", "radius_split", "candidate_splits", "best_split",
    "Block", "BlockSystem", "decompose", "boundary_arc",
]

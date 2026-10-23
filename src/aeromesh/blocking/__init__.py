"""Block decomposition.

``splits`` supplies the decomposition splits: curves of the medial coordinate
system, orthogonal to the wall by construction and ranked the way Fogg ranks
them. ``blocks`` turns a chosen set of them into blocks, constructively from the
medial graph rather than by computing a planar arrangement. ``layers`` supplies
the other family of curves -- constant-rho offsets, the outer edge of a
near-wall block.

``collar`` is zone 1 of the zoned blocking and does not use any of the above:
the near-wall region is *built* rather than decomposed, because a certified
decomposition turned out not to be a meshable one there. See SZ in the tracker.
"""

from aeromesh.blocking.blocks import Block, BlockSystem, boundary_arc, decompose
from aeromesh.blocking.collar import (
    build_collar,
    cell_quality,
    curvature_spacing,
    mesh_collar,
    smooth_assembly,
    smooth_offset,
    smooth_patch,
    validate_mesh,
)
from aeromesh.blocking.layers import (
    medial_radius_on_boundary,
    outward_normals,
    peel,
    rho_offset,
)
from aeromesh.blocking.search import Placement, locate, place_singularities, select_splits
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
    "select_splits", "place_singularities", "locate", "Placement",
    "rho_offset", "peel", "medial_radius_on_boundary", "outward_normals",
    "build_collar", "mesh_collar", "validate_mesh", "cell_quality",
    "curvature_spacing", "smooth_offset", "smooth_patch", "smooth_assembly",
]

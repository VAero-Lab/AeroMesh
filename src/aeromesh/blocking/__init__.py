"""Block decomposition (S3, in progress).

``splits`` provides the decomposition splits: curves of the medial coordinate
system, orthogonal to the wall by construction, ranked the way Fogg ranks them.
The block builder that consumes them is still to come.
"""

from aeromesh.blocking.splits import (
    Split,
    best_split,
    candidate_splits,
    radius_split,
    split_rank,
)

__all__ = ["Split", "split_rank", "radius_split", "candidate_splits", "best_split"]

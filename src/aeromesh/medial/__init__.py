"""Medial axis computation.

``aeromesh.medial.axis`` is the current engine (S1): it computes both skeletons
with their full field set and types their vertices.

``aeromesh.medial.cdt`` and ``aeromesh.medial.graph`` are superseded and retained
only until the repository is under version control; see decision D7 in
PROJECT_TRACKER.md. Nothing on the current path imports them.
"""

from aeromesh.medial.axis import (
    EdgeKind,
    MedialAxis,
    VertexKind,
    exterior_axis,
    interior_axis,
)
from aeromesh.medial.fields import optimum_flow_index

__all__ = [
    "MedialAxis", "VertexKind", "EdgeKind",
    "exterior_axis", "interior_axis", "optimum_flow_index",
]

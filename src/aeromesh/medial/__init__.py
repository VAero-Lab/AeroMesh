"""Medial axis computation.

``aeromesh.medial.axis`` is the current engine (S1): it computes both skeletons
with their full field set and types their vertices.

``fields`` computes the geometric quantities on it: the medial radius, the
medial angle, the unit normals and the touch points.
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

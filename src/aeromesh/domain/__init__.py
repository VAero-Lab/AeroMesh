"""Where the domain is truncated.

Every constructor returns a plain ``Loop``. The shape is never recorded and
never reaches a later stage, so no stage can branch on which one was used --
but its corners do drive topology, which is why the standard shapes are offered
as standard shapes.
"""

from aeromesh.domain.farfield import (
    box_farfield,
    c_farfield,
    circle_farfield,
    offset_farfield,
    rectangle,
)

__all__ = [
    "circle_farfield", "c_farfield", "box_farfield", "offset_farfield", "rectangle",
]

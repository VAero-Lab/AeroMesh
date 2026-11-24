"""Mesh construction for a single body, marched from the wall.

``spacing``     graded node distributions; first cell height from y+
``hyperbolic``  the marching solver (orthogonality and cell area as equations)
``airfoil``     ``o_mesh`` and ``c_mesh``; the C topology is derived from the
                corners' element counts
``structured``  blocks, and assembly into one conformal mesh with exact checks
``farfield``    fit a supplied far field (circle, C-shape, box) by TFI
``export``      meshio (SU2, VTK, ...) and OpenFOAM polyMesh
"""

from aeromesh.mesh.airfoil import c_mesh, o_mesh, wall_segments
from aeromesh.mesh.export import write_meshio, write_openfoam
from aeromesh.mesh.farfield import c_domain, domain_mesh, fit_farfield
from aeromesh.mesh.hyperbolic import march
from aeromesh.mesh.spacing import (first_cell_height, graded_parameters,
                                   layer_heights)
from aeromesh.mesh.structured import Block, Mesh, MultiBlock

__all__ = ["c_domain", "domain_mesh", "fit_farfield", "c_mesh", "o_mesh", "wall_segments", "write_meshio",
           "write_openfoam", "march", "first_cell_height",
           "graded_parameters", "layer_heights", "Block", "Mesh",
           "MultiBlock"]

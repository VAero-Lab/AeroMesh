"""Topology: the singularity solver and the index budget.

``singularities`` is the current module (S2). ``classify`` and ``design`` are
superseded and retained only until the repository is under version control; see
decision D7 in PROJECT_TRACKER.md.
"""

from aeromesh.topology.singularities import (
    Corner,
    Singularity,
    SingularityField,
    critical_crossings,
    finite_contact_singularities,
    flux_residual,
    merge,
    solve_singularities,
)

__all__ = [
    "Corner", "Singularity", "SingularityField",
    "flux_residual", "critical_crossings", "finite_contact_singularities",
    "merge", "solve_singularities",
]

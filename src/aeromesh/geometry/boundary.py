"""AeroShape integration shim and Boundary data structure.

This is the **only** module that imports from AeroShape. Every other
module in AeroMesh works with plain numpy arrays via the ``Boundary``
class.

Coordinate convention
---------------------
AeroShape stores airfoil profiles in the XZ plane (X = chordwise,
Z = thickness, Y = span for 3D lofting).  AeroMesh operates in 2D and
maps (x, z) → (x, y) for the meshing plane.  This aligns with the
standard 2D CFD convention (x = streamwise, y = wall-normal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
#  AeroShape import — the only place in AeroMesh that touches AeroShape
# ---------------------------------------------------------------------------
from aeroshape.geometry.airfoils import AirfoilProfile


# ═══════════════════════════════════════════════════════════════════
#  Boundary dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Boundary:
    """A validated, closed 2D airfoil boundary ready for meshing.

    Attributes
    ----------
    points : np.ndarray, shape (N, 2)
        Ordered boundary points in the 2D meshing plane.
        Column 0 = chordwise (x), Column 1 = thickness (y).
        The boundary is closed: ``points[0] == points[-1]``.
        Winding is counter-clockwise (CCW), which for an airfoil means:
        TE → upper surface (y > 0) → LE → lower surface (y < 0) → TE.
    name : str
        Descriptive name of the airfoil.
    chord : float
        Reference chord length (after normalization, typically 1.0).
    le_index : int
        Index of the leading-edge point (minimum x coordinate).
    """

    points: np.ndarray
    name: str = ""
    chord: float = 1.0
    le_index: int = 0

    # ── Derived properties ─────────────────────────────────────────

    @property
    def n_points(self) -> int:
        """Number of boundary points (including the closing point)."""
        return len(self.points)

    @property
    def le(self) -> np.ndarray:
        """Leading-edge point (minimum x)."""
        return self.points[self.le_index]

    @property
    def te(self) -> np.ndarray:
        """Trailing-edge point (first/last point)."""
        return self.points[0]

    @property
    def upper_surface(self) -> np.ndarray:
        """Upper surface points (y > 0), from TE to LE.

        In CCW winding this is the first segment: points[0 : le_index+1].
        """
        return self.points[: self.le_index + 1]

    @property
    def lower_surface(self) -> np.ndarray:
        """Lower surface points (y < 0), from LE to TE.

        In CCW winding this is the second segment: points[le_index :].
        """
        return self.points[self.le_index:]


# ═══════════════════════════════════════════════════════════════════
#  Validation helpers
# ═══════════════════════════════════════════════════════════════════

def _signed_area(pts: np.ndarray) -> float:
    """Signed area of a 2D polygon via the shoelace formula.

    Positive → CCW, Negative → CW.
    """
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def _remove_consecutive_duplicates(pts: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Remove consecutive near-duplicate points."""
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], diffs > tol))
    return pts[keep]


# ═══════════════════════════════════════════════════════════════════
#  Factory functions
# ═══════════════════════════════════════════════════════════════════

def _profile_to_boundary(profile: AirfoilProfile) -> Boundary:
    """Convert an AeroShape AirfoilProfile to a validated Boundary.

    Steps
    -----
    1. Stack profile.x and profile.z → (N, 2) in the (x, y) meshing plane.
    2. Remove consecutive near-duplicate points.
    3. Ensure closure (first == last within tolerance).
    4. Enforce CCW winding via signed-area test.
    5. Normalize to unit chord with LE at the origin.
    6. Validate.

    Raises
    ------
    ValueError
        If the boundary has fewer than 20 points after cleaning, or
        if the trailing edge does not close within tolerance.
    """
    # 1. Stack (x, z) → (x, y) for 2D meshing plane
    pts = np.column_stack((profile.x, profile.z)).astype(np.float64)

    # 2. Remove consecutive duplicates
    pts = _remove_consecutive_duplicates(pts)

    # 3. Ensure closure
    # Tolerance: 1% chord. Dat files often have imperfect closure due to
    # limited decimal precision. We force-close if the gap is small.
    gap = np.linalg.norm(pts[0] - pts[-1])
    if gap > 0.01:
        raise ValueError(
            f"Airfoil trailing edge is not closed (gap = {gap:.2e}, "
            f">{0.01:.0e} chord). AeroMesh requires closed-TE airfoils."
        )
    # Force exact closure
    pts[-1] = pts[0].copy()

    # 4. Enforce CCW winding
    if _signed_area(pts) < 0:
        pts = pts[::-1]

    # 5. Normalize: LE at origin, unit chord
    le_idx = np.argmin(pts[:, 0])
    le_x = pts[le_idx, 0]
    chord = pts[:, 0].max() - le_x
    if chord < 1e-10:
        raise ValueError("Degenerate airfoil: zero chord length.")
    pts[:, 0] = (pts[:, 0] - le_x) / chord
    pts[:, 1] = pts[:, 1] / chord
    # Force exact closure again after normalization
    pts[-1] = pts[0].copy()

    # 6. Validate
    if len(pts) < 20:
        raise ValueError(
            f"Airfoil has only {len(pts)} unique points after cleaning. "
            f"Need at least 20 for a usable boundary."
        )

    # Recompute LE index after potential reversal
    le_idx = np.argmin(pts[:, 0])

    return Boundary(
        points=pts,
        name=profile.name,
        chord=1.0,
        le_index=le_idx,
    )


def load_airfoil(
    source: str | AirfoilProfile,
    num_points: int = 150,
    chord: float = 1.0,
) -> Boundary:
    """Load an airfoil from any supported source and return a Boundary.

    Parameters
    ----------
    source : str or AirfoilProfile
        - A NACA 4-digit code string (e.g. ``"0012"``, ``"2412"``).
        - A NACA 5-digit code string (e.g. ``"23012"``).
        - A path to a ``.dat`` file (Selig or Lednicer format).
        - An ``AirfoilProfile`` instance from AeroShape.
    num_points : int
        Number of points per surface for NACA generators.
        Ignored when loading from file or passing a profile directly.
    chord : float
        Chord length for NACA generators (normalized to 1.0 internally).

    Returns
    -------
    Boundary
        Validated, closed, CCW-ordered boundary with unit chord.

    Examples
    --------
    >>> bnd = load_airfoil("0012")
    >>> bnd = load_airfoil("23012")
    >>> bnd = load_airfoil("path/to/sd7037.dat")
    >>> bnd = load_airfoil(AirfoilProfile.from_cst(wl, wu))
    """
    if isinstance(source, AirfoilProfile):
        return _profile_to_boundary(source)

    source_str = str(source)

    # Check if it's a file path
    path = Path(source_str)
    if path.suffix.lower() == ".dat" or path.exists():
        profile = AirfoilProfile.from_dat_file(str(path), chord=chord)
        return _profile_to_boundary(profile)

    # Try as a NACA code
    digits = source_str.strip()
    if digits.isdigit():
        if len(digits) == 4:
            profile = AirfoilProfile.from_naca4(digits, num_points, chord)
        elif len(digits) == 5:
            profile = AirfoilProfile.from_naca5(digits, num_points, chord)
        else:
            raise ValueError(
                f"NACA code must be 4 or 5 digits, got {len(digits)}: '{digits}'"
            )
        return _profile_to_boundary(profile)

    raise ValueError(
        f"Cannot interpret source '{source_str}'. "
        f"Expected a NACA code, .dat file path, or AirfoilProfile."
    )

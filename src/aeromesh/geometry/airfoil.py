"""AeroShape intake.

The only module in AeroMesh that imports AeroShape. It turns a profile into a
``Loop`` and stops there; nothing downstream knows the loop came from an
airfoil rather than a fuselage section or a hand-drawn curve.

Coordinate convention: AeroShape stores profiles in the XZ plane (x chordwise,
z thickness, y span). AeroMesh works in 2D and maps (x, z) -> (x, y), matching
the usual 2D CFD convention of x streamwise and y wall-normal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aeroshape.geometry.airfoils import AirfoilProfile

from aeromesh.geometry.loop import Loop


def profile_to_loop(profile: AirfoilProfile, normalize: bool = True) -> Loop:
    """Convert an AeroShape ``AirfoilProfile`` to a validated ``Loop``.

    Parameters
    ----------
    profile : AirfoilProfile
        Any AeroShape profile: NACA 4/5-digit, CST, PARSEC or from a .dat file.
    normalize : bool
        Place the leading edge at the origin and scale to unit chord. Turn this
        off when positioning several bodies in a shared coordinate system.

    Notes
    -----
    No trailing-edge policy is applied here. A cusped or closed trailing edge is
    accepted as given; opening it into a finite face is the job of
    ``prepare_boundary``, which does it for any cusp on any loop.
    """
    pts = np.column_stack((profile.x, profile.z)).astype(np.float64)
    loop = Loop.from_points(pts, name=getattr(profile, "name", "") or "airfoil")
    if not normalize:
        return loop
    lo, hi = loop.bbox
    chord = hi[0] - lo[0]
    if chord < 1e-12:
        raise ValueError("Degenerate profile: zero chord length.")
    shifted = (loop.points - np.array([lo[0], 0.0])) / chord
    return Loop.from_points(shifted, name=loop.name)


def load_airfoil(
    source: str | Path | AirfoilProfile,
    n_points: int = 300,
    chord: float = 1.0,
    normalize: bool = True,
) -> Loop:
    """Load an airfoil from any supported source and return a ``Loop``.

    Parameters
    ----------
    source : str, Path or AirfoilProfile
        A NACA 4- or 5-digit code (``"0012"``, ``"23012"``), a path to a
        ``.dat`` file in Selig or Lednicer format, or an ``AirfoilProfile``.
    n_points : int
        Points per surface for the NACA generators. Ignored for files and for
        profiles passed directly; ``prepare_boundary`` resamples anyway.
    chord : float
        Chord passed to the generator or file reader.
    normalize : bool
        See ``profile_to_loop``.

    Returns
    -------
    Loop
        A closed, CCW, de-duplicated curve. No trailing-edge policy applied.
    """
    if isinstance(source, AirfoilProfile):
        return profile_to_loop(source, normalize)

    text = str(source).strip()
    path = Path(text)
    if path.suffix.lower() == ".dat" or path.exists():
        return profile_to_loop(AirfoilProfile.from_dat_file(str(path), chord=chord),
                               normalize)

    if text.isdigit():
        if len(text) == 4:
            prof = AirfoilProfile.from_naca4(text, n_points, chord)
        elif len(text) == 5:
            prof = AirfoilProfile.from_naca5(text, n_points, chord)
        else:
            raise ValueError(f"NACA code must be 4 or 5 digits, got {len(text)}: {text!r}")
        return profile_to_loop(prof, normalize)

    raise ValueError(
        f"Cannot interpret source {text!r}. Expected a NACA code, a .dat path, "
        f"or an AirfoilProfile."
    )

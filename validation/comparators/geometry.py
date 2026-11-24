"""The one geometry every mesher is given.

Each comparator reads the *same* closed loop that AeroMesh meshes -- the
region's body boundary after its trailing-edge policy has been applied -- so a
difference in the result is a difference in the mesher, not in the section.
"""
from pathlib import Path

import numpy as np

import aeromesh as am


def section(name: str, te: str, n_points: int = 1000, spacing: float = 5e-4):
    """The body loop AeroMesh meshes, counter-clockwise, starting at the
    trailing edge on the upper surface (Selig order)."""
    body = am.load_airfoil(name, n_points=n_points)
    region = am.build_region([body], farfield=20.0, te=te,
                             body_spacing=spacing)
    bnd = region.boundaries[1]
    p = np.asarray(bnd.loop.points, dtype=float)
    ci = np.asarray(bnd.corners.index, dtype=int)
    if te == "sharp":
        k = int(ci[0])
        p = np.roll(p, -k, axis=0)
        p = np.vstack([p, p[:1]])            # closed: last point = first
    else:
        # two corners; start at the upper one and stop at the lower one, so
        # the base face is the gap between first and last point
        up, lo = sorted(ci, key=lambda i: -p[i, 1])
        p = np.roll(p, -up, axis=0)
        n_keep = (lo - up) % len(p) + 1
        p = p[:n_keep]
    return p, bnd


def write_selig(points: np.ndarray, path, title="airfoil") -> Path:
    path = Path(path)
    path.write_text(title + "\n" + "\n".join(
        f"{x:.10f} {y:.10f}" for x, y in points) + "\n")
    return path

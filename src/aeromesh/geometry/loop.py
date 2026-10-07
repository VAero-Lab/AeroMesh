"""Loop — a single closed curve.

The one geometric primitive the whole pipeline is built on.  A ``Loop`` is a
closed, non-self-intersecting polyline stored **open** (the closing point is
implied, never duplicated) and always wound counter-clockwise.

A configuration is a set of loops; nothing in AeroMesh knows or cares whether
a loop is an airfoil, a fuselage section or a far-field boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def signed_area(pts: np.ndarray) -> float:
    """Signed area of a closed polygon given as open points. Positive = CCW."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def drop_duplicates(pts: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Drop consecutive near-duplicate points, wrapping around the close."""
    if len(pts) < 2:
        return pts
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], d > tol))
    pts = pts[keep]
    # the wrap-around pair
    while len(pts) > 3 and np.linalg.norm(pts[-1] - pts[0]) <= tol:
        pts = pts[:-1]
    return pts


@dataclass(frozen=True)
class Loop:
    """A closed curve, stored open and wound counter-clockwise.

    Attributes
    ----------
    points : np.ndarray, shape (N, 2)
        Ordered vertices. ``points[-1]`` connects back to ``points[0]``; the
        closing point is *not* repeated.
    name : str
        Free-form label, used only for reporting.
    scale : float
        Reference length for this loop (its bounding-box diagonal at
        construction). Used to make tolerances dimensionless. Preserved
        through resampling so derived tolerances stay stable.
    """

    points: np.ndarray
    name: str = ""
    scale: float = 0.0

    # ── construction ────────────────────────────────────────────────

    @classmethod
    def from_points(cls, pts, name: str = "", tol: float = 1e-12) -> Loop:
        """Build a validated Loop: de-duplicated, closed, CCW."""
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Loop needs an (N, 2) array, got {pts.shape}.")
        pts = drop_duplicates(pts, tol)
        if len(pts) < 3:
            raise ValueError(f"Loop needs at least 3 distinct points, got {len(pts)}.")
        if signed_area(pts) < 0:
            pts = pts[::-1].copy()
        span = pts.max(axis=0) - pts.min(axis=0)
        return cls(points=pts, name=name, scale=float(np.hypot(*span)))

    # ── basic geometry ──────────────────────────────────────────────

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def closed_points(self) -> np.ndarray:
        """Points with the first vertex repeated at the end, for plotting."""
        return np.vstack([self.points, self.points[:1]])

    @property
    def area(self) -> float:
        return signed_area(self.points)

    @property
    def perimeter(self) -> float:
        return float(np.sum(self.edge_lengths))

    @property
    def edge_lengths(self) -> np.ndarray:
        """Length of edge i, from points[i] to points[i+1] (wrapping)."""
        return np.linalg.norm(np.roll(self.points, -1, axis=0) - self.points, axis=1)

    @property
    def arclength(self) -> np.ndarray:
        """Cumulative arc length at each vertex, starting at 0."""
        return np.concatenate(([0.0], np.cumsum(self.edge_lengths)[:-1]))

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    @property
    def centroid(self) -> np.ndarray:
        """Area centroid (not the mean of the samples)."""
        p = self.points
        q = np.roll(p, -1, axis=0)
        cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
        a = 0.5 * float(np.sum(cross))
        if abs(a) < 1e-300:
            return p.mean(axis=0)
        c = np.sum((p + q) * cross[:, None], axis=0) / (6.0 * a)
        return c

    # ── sampling ────────────────────────────────────────────────────

    def interpolate(self, s: np.ndarray) -> np.ndarray:
        """Point(s) at arc length ``s``, measured from points[0], wrapping."""
        L = self.perimeter
        knots = np.concatenate((self.arclength, [L]))
        ring = self.closed_points
        s = np.mod(np.asarray(s, dtype=np.float64), L)
        return np.column_stack([
            np.interp(s, knots, ring[:, 0]),
            np.interp(s, knots, ring[:, 1]),
        ])

    def sharp_vertices(self, angle_tol: float = np.deg2rad(20.0)) -> np.ndarray:
        """Indices of vertices whose single-vertex turn exceeds ``angle_tol``.

        On a densely sampled smooth curve this is empty. On a polygon it is
        exactly the corners. Used to keep resampling from chamfering them.
        """
        p = self.points
        v_out = np.roll(p, -1, axis=0) - p
        v_in = p - np.roll(p, 1, axis=0)
        turn = np.arctan2(
            v_in[:, 0] * v_out[:, 1] - v_in[:, 1] * v_out[:, 0],
            v_in[:, 0] * v_out[:, 0] + v_in[:, 1] * v_out[:, 1],
        )
        return np.flatnonzero(np.abs(turn) > angle_tol)

    def resample(
        self,
        n: int | None = None,
        spacing: float | None = None,
        preserve_corners: bool = True,
        angle_tol: float = np.deg2rad(20.0),
    ) -> Loop:
        """Uniform arc-length resampling.

        Exactly one of ``n`` (point count) or ``spacing`` (target edge length)
        must be given. ``scale`` is preserved so tolerances stay stable.

        With ``preserve_corners`` the arc positions of any sharp input vertices
        are forced into the output, so a polygon keeps its corners and its
        perimeter exactly instead of being chamfered by up to half a spacing.
        The returned point count is then approximately, not exactly, ``n``.
        """
        if (n is None) == (spacing is None):
            raise ValueError("Give exactly one of n= or spacing=.")
        L = self.perimeter
        if n is None:
            n = max(8, int(np.ceil(L / float(spacing))))
        if n < 3:
            raise ValueError(f"Resampling needs n >= 3, got {n}.")

        s = np.linspace(0.0, L, n, endpoint=False)
        if preserve_corners:
            forced = self.arclength[self.sharp_vertices(angle_tol)]
            if forced.size:
                step = L / n
                gap = np.min(np.abs(s[:, None] - forced[None, :]), axis=1)
                s = np.sort(np.concatenate([s[gap > 0.5 * step], forced]))
        return replace(self, points=self.interpolate(s))

    def transform(self, scale: float = 1.0, angle: float = 0.0,
                  dx: float = 0.0, dy: float = 0.0,
                  name: str | None = None) -> Loop:
        """Scale about the origin, rotate by ``angle`` radians, then translate.

        The operation order is fixed so that building a configuration -- a
        flap scaled, deflected and positioned relative to a main element --
        reads the same way every time.
        """
        c, s_ = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s_], [s_, c]])
        pts = (self.points * float(scale)) @ R.T + np.array([dx, dy], dtype=np.float64)
        span = pts.max(axis=0) - pts.min(axis=0)
        return replace(self, points=pts, scale=float(np.hypot(*span)),
                       name=self.name if name is None else name)

    def reversed(self) -> Loop:
        """Same curve, opposite winding. Breaks the CCW invariant by design."""
        return replace(self, points=self.points[::-1].copy())

    # ── containment ─────────────────────────────────────────────────

    def contains(self, query: np.ndarray) -> np.ndarray:
        """Even-odd (crossing-number) point-in-polygon test.

        Vectorised over query points, looping over edges. Points exactly on
        the boundary are not guaranteed either way — never rely on that.
        """
        q = np.atleast_2d(np.asarray(query, dtype=np.float64))
        px, py = q[:, 0], q[:, 1]
        a = self.points
        b = np.roll(a, -1, axis=0)
        inside = np.zeros(len(q), dtype=bool)
        for (ax, ay), (bx, by) in zip(a, b):
            if ay == by:
                continue  # horizontal edge contributes no crossing
            straddles = (ay > py) != (by > py)
            if not straddles.any():
                continue
            x_cross = ax + (py - ay) * (bx - ax) / (by - ay)
            inside ^= straddles & (px < x_cross)
        return inside

    def __repr__(self) -> str:
        lo, hi = self.bbox
        return (f"Loop(name={self.name!r}, n={self.n_points}, "
                f"bbox=[{lo[0]:.3g},{lo[1]:.3g}]..[{hi[0]:.3g},{hi[1]:.3g}], "
                f"area={self.area:.4g})")

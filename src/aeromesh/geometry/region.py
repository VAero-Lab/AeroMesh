"""Region — a set of closed loops, and the only input the pipeline takes.

A ``Region`` is one outer loop and any number of inner loops (holes). The fluid
occupies the space inside the outer loop and outside every hole.

That is the whole data model. It carries no notion of a "C domain" or an
"O domain", no body type and no body count: a single airfoil, a three-element
high-lift configuration and a wing-body crossflow cut are all just loop sets,
and every stage downstream sees them identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Voronoi, cKDTree

from aeromesh.geometry.corners import Corners, detect_corners, fluid_interior_angle
from aeromesh.geometry.cusp import find_cusps, open_cusp
from aeromesh.geometry.loop import Loop


# ═══════════════════════════════════════════════════════════════════
#  Largest inscribed circle — a robust interior point, and the seed of
#  the interior medial axis used in S1.
# ═══════════════════════════════════════════════════════════════════

def largest_inscribed_circle(loop: Loop) -> tuple[np.ndarray, float]:
    """Centre and radius of the largest circle that fits inside ``loop``.

    Found as the interior Voronoi vertex furthest from the boundary, which is
    by definition the maximum of the medial radius over the interior medial
    axis. Robust for any simple polygon, including strongly cambered and
    non-convex ones, where a sample centroid can fall outside the shape.
    """
    pts = loop.points
    vor = Voronoi(pts)
    v = vor.vertices
    inside = loop.contains(v)
    if not inside.any():
        # Degenerate (very coarse) input: fall back to the area centroid.
        c = loop.centroid
        if not bool(loop.contains(c[None, :])[0]):
            raise ValueError(f"Could not find an interior point for {loop!r}.")
        return c, float(cKDTree(pts).query(c)[0])
    v = v[inside]
    d = cKDTree(pts).query(v)[0]
    k = int(np.argmax(d))
    return v[k], float(d[k])


# ═══════════════════════════════════════════════════════════════════
#  BoundaryLoop — a loop plus the corner information the topology
#  stage needs, computed once, at the final sampling.
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BoundaryLoop:
    """One boundary of the region, with its corners resolved.

    Attributes
    ----------
    loop : Loop
        The curve, uniformly resampled.
    is_hole : bool
        True when the fluid lies outside this loop (a solid body), False for
        the outer boundary.
    corners : Corners
        Every C0 corner on the loop: those detected from the geometry, plus
        any created by opening a cusp, which are known exactly rather than
        detected and so survive being closer together than the detector's
        smallest window.
    """

    loop: Loop
    is_hole: bool
    corners: Corners

    @property
    def fluid_angles(self) -> np.ndarray:
        """Interior angle of the *fluid* at each corner, in radians."""
        return fluid_interior_angle(self.corners.turn, self.is_hole)

    @property
    def corner_elements(self) -> np.ndarray:
        """Fogg's n_c at each corner: the number of elements that meet there.

        Chosen to minimise the strength of the point source the corner implies
        in the phi-field, which reduces to rounding the fluid interior angle to
        the nearest multiple of pi/2.
        """
        return np.rint(self.fluid_angles / (np.pi / 2)).astype(int)

    def __repr__(self) -> str:
        kind = "hole" if self.is_hole else "outer"
        return (f"BoundaryLoop({kind}, name={self.loop.name!r}, "
                f"n={self.loop.n_points}, corners={len(self.corners)})")


def _merge_corners(loop: Loop, known: np.ndarray) -> Corners:
    """Detected corners plus exactly-known ones, de-duplicated by arc length.

    Known corners win: they were constructed, so their positions are exact and
    two of them may sit closer together than the detector can separate.
    """
    detected = detect_corners(loop)
    from aeromesh.geometry.corners import vertex_turns

    turns = vertex_turns(loop)
    s = loop.arclength
    L = loop.perimeter
    delta = L / loop.n_points
    guard = 12.0 * delta  # the detector's widest window

    idx = list(int(i) for i in known)
    turn = list(float(turns[i]) for i in known)
    resid = [0.0] * len(idx)
    for i, t, r in zip(detected.index, detected.turn, detected.residual):
        if idx:
            ds = np.abs(s[i] - s[np.array(idx)])
            if np.min(np.minimum(ds, L - ds)) < guard:
                continue
        idx.append(int(i))
        turn.append(float(t))
        resid.append(float(r))

    order = np.argsort(idx)
    return Corners(
        index=np.array(idx, dtype=int)[order],
        turn=np.array(turn)[order],
        residual=np.array(resid)[order],
    )


def prepare_boundary(
    loop: Loop,
    is_hole: bool,
    spacing: float | None = None,
    te: str = "blunt",
    te_thickness: float = 0.002,
    max_cusps: int = 8,
) -> BoundaryLoop:
    """Resample a loop, resolve its corners, and apply the cusp policy.

    Parameters
    ----------
    loop : Loop
        Raw input curve, at whatever sampling it arrived with.
    is_hole : bool
        True for a solid body, False for the outer boundary.
    spacing : float, optional
        Target point spacing. Defaults to ``loop.scale / 500``, so a unit-chord
        section gets roughly 1000 points and a far-field boundary is sampled in
        proportion to its own size.
    te : {'blunt', 'sharp'}
        ``'blunt'`` opens every cusp into a finite face of length
        ``te_thickness`` -- the well-posed case for a medial axis.
        ``'sharp'`` keeps cusps as they are; their corners are still detected
        and typed, and the medial branch they imply must be seeded explicitly
        downstream because a Voronoi will not produce it.
    te_thickness : float
        Face length used by the ``'blunt'`` policy, in the loop's own units.
    max_cusps : int
        Safety limit on how many cusps are opened.

    Returns
    -------
    BoundaryLoop
    """
    if te not in ("blunt", "sharp"):
        raise ValueError(f"te must be 'blunt' or 'sharp', got {te!r}.")
    if spacing is None:
        spacing = loop.scale / 500.0
    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    work = loop.resample(spacing=spacing)
    face_points: list[np.ndarray] = []

    if te == "blunt":
        for _ in range(max_cusps):
            # Faces already opened are excluded by position. A face narrower
            # than the detector's smallest window reads as a single corner
            # whose turn is the sum of its two -- indistinguishable from a
            # cusp -- so this loop, which knows what it built, says so.
            done = np.vstack(face_points) if face_points else None
            cusps = find_cusps(work, exclude=done,
                               exclude_radius=2.0 * te_thickness)
            if len(cusps) == 0:
                break
            # Positions are carried, not indices: every cut rebuilds the loop
            # and renumbers its vertices.
            work, face = open_cusp(work, int(cusps[0]), te_thickness,
                                   n=work.n_points)
            face_points.append(work.points[face].copy())
        else:
            raise ValueError(f"More than {max_cusps} cusps on {loop!r}; "
                             "check the input geometry.")

    if face_points:
        stacked = np.vstack(face_points)
        d = np.linalg.norm(work.points[:, None, :] - stacked[None, :, :], axis=2)
        known = np.unique(np.argmin(d, axis=0))
    else:
        known = np.empty(0, dtype=int)

    return BoundaryLoop(loop=work, is_hole=is_hole,
                        corners=_merge_corners(work, known))


# ═══════════════════════════════════════════════════════════════════
#  Region
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Region:
    """The fluid region: inside ``outer``, outside every loop in ``holes``."""

    outer: BoundaryLoop
    holes: tuple[BoundaryLoop, ...]

    # ── structure ───────────────────────────────────────────────────

    @property
    def n_bodies(self) -> int:
        return len(self.holes)

    @property
    def boundaries(self) -> tuple[BoundaryLoop, ...]:
        return (self.outer,) + self.holes

    @property
    def euler_characteristic(self) -> int:
        """chi = 1 - h. The medial graph's cycle count must equal ``h``, and
        the singularity budget is fixed by this number."""
        return 1 - self.n_bodies

    # ── geometry ────────────────────────────────────────────────────

    def contains(self, query: np.ndarray) -> np.ndarray:
        """True where a point lies in the fluid."""
        q = np.atleast_2d(np.asarray(query, dtype=np.float64))
        inside = self.outer.loop.contains(q)
        for h in self.holes:
            inside &= ~h.loop.contains(q)
        return inside

    def hole_points(self) -> np.ndarray:
        """One point strictly inside each body, for a triangulator.

        Taken as the centre of each body's largest inscribed circle, which is
        interior for any simple polygon -- unlike a sample centroid.
        """
        if not self.holes:
            return np.empty((0, 2))
        return np.array([largest_inscribed_circle(h.loop)[0] for h in self.holes])

    def segment_ids(self) -> np.ndarray:
        """A distinct id per *smooth boundary element*, one entry per sample.

        A C0 corner divides a loop into separate elements. This is the
        segmentation the medial test needs: two generators on different elements
        are always a genuine pair, however close together they are, while two on
        the same element have to be far apart before their bisector means
        anything. It is what lets a medial branch terminate at a trailing edge.

        Derived entirely from detected geometry. The first iteration split each
        airfoil at its minimum-x sample instead, which is a parameterisation
        artefact, and so invented a medial branch at every leading edge.
        """
        out, next_id = [], 0
        for b in self.boundaries:
            n = b.loop.n_points
            ids = np.zeros(n, dtype=int)
            idx = np.sort(b.corners.index) if len(b.corners) else np.empty(0, dtype=int)
            if len(idx) == 0:
                ids[:] = next_id
                next_id += 1
            else:
                # Roll so the loop starts at a corner, segment, then roll back.
                shift = int(idx[0])
                cuts = (idx - shift) % n
                seg = np.searchsorted(np.sort(cuts), np.arange(n), side="right") - 1
                ids = np.roll(seg, shift) + next_id
                next_id += len(idx)
            out.append(ids)
        return np.concatenate(out)

    def segment_is_closed(self) -> np.ndarray:
        """True for each segment that wraps, i.e. belongs to a corner-free loop.

        Index distance within a closed segment is circular; within an open one it
        is linear, so the two samples either side of a corner come out maximally
        far apart and their ridge is kept.
        """
        flags = []
        for b in self.boundaries:
            k = len(b.corners)
            flags.extend([True] if k == 0 else [False] * k)
        return np.array(flags, dtype=bool)

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All boundary points, with a loop id per point and per-loop offsets.

        This is the exact input the medial engine consumes: it never asks which
        loop is a body and which is the far field, only whether two Voronoi
        generators belong to the same loop and how far apart they are on it.
        """
        pts = np.vstack([b.loop.points for b in self.boundaries])
        counts = [b.loop.n_points for b in self.boundaries]
        loop_id = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
        offsets = np.concatenate(([0], np.cumsum(counts)))
        return pts, loop_id, offsets

    # ── validation ──────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise if the loop set is not a valid region.

        Checks that every loop is simple, that every body lies strictly inside
        the outer boundary, and that no two bodies overlap.
        """
        from shapely.geometry import Polygon
        from shapely.geometry.polygon import LinearRing

        for b in self.boundaries:
            if not LinearRing(b.loop.closed_points).is_simple:
                raise ValueError(f"{b!r} is self-intersecting.")

        outer_poly = Polygon(self.outer.loop.closed_points)
        polys = [Polygon(h.loop.closed_points) for h in self.holes]
        for h, p in zip(self.holes, polys):
            if not outer_poly.contains(p):
                raise ValueError(f"{h!r} is not strictly inside the outer boundary.")
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polys[i].intersects(polys[j]):
                    raise ValueError(
                        f"{self.holes[i]!r} and {self.holes[j]!r} overlap or touch."
                    )

    # ── construction ────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        outer: Loop,
        bodies: list[Loop] | tuple[Loop, ...] = (),
        spacing: float | None = None,
        body_spacing: float | None = None,
        te: str = "blunt",
        te_thickness: float = 0.002,
        validate: bool = True,
    ) -> Region:
        """Prepare every loop and assemble a validated Region.

        ``spacing`` applies to the outer boundary, ``body_spacing`` to the
        bodies; either defaults to that loop's own scale divided by 500, so a
        far field ten chords across is not forced to the sampling of a section.
        """
        o = prepare_boundary(outer, is_hole=False, spacing=spacing, te="sharp")
        hs = tuple(
            prepare_boundary(b, is_hole=True, spacing=body_spacing,
                             te=te, te_thickness=te_thickness)
            for b in bodies
        )
        region = cls(outer=o, holes=hs)
        if validate:
            region.validate()
        return region

    def __repr__(self) -> str:
        n = sum(b.loop.n_points for b in self.boundaries)
        nc = sum(len(b.corners) for b in self.boundaries)
        return (f"Region(bodies={self.n_bodies}, chi={self.euler_characteristic}, "
                f"points={n}, corners={nc})")

"""Fit a marched mesh to a supplied far-field curve.

The march stops where it stops; a far field is a modelling input (a circle, a
C-shape, a box). Between the marched mesh's outer boundary and the supplied
curve lies a region bounded by two closed curves, so it is an annulus: one
periodic block, forced by topology, with no wake cut and no irregular node.

It is filled by **transfinite interpolation** -- straight lines from each node
of the inner loop to a node of the far field, spaced geometrically so that the
first outer cell continues the marched growth -- and then optionally relaxed by
a few Laplacian sweeps away from both boundaries. This is the zone where TFI is
enough: there is no wall, no first-cell height to protect and no orthogonality
requirement beyond keeping cells convex.

Corners of the far field (a box has four; a C-shape two) are put on nodes
exactly. A corner of the domain asks for one element (its fluid angle is 90
degrees), which is a node of the boundary with two edges meeting there; a
corner falling mid-edge would be cut off.
"""

from __future__ import annotations

import numpy as np

from aeromesh.mesh.spacing import arclength
from aeromesh.mesh.structured import Block, MultiBlock


def outer_loop(mesh) -> np.ndarray:
    """Node ids of the mesh's non-wall boundary, as one closed loop traversed
    counter-clockwise (domain on the left), first node not repeated."""
    edges = np.concatenate([e for name, e in mesh.patches.items()
                            if name != "wall"])
    nxt = {int(a): int(b) for a, b in edges}
    if len(nxt) != len(edges):
        raise ValueError("outer boundary is not a simple loop")
    start = int(edges[0, 0])
    loop = [start]
    while True:
        k = nxt[loop[-1]]
        if k == start:
            break
        loop.append(k)
        if len(loop) > len(edges):
            raise ValueError("outer boundary does not close")
    return np.array(loop)


def _corners(curve: np.ndarray, tol_deg: float = 20.0) -> np.ndarray:
    """Indices of the C0 corners of a closed polyline (turn above tol)."""
    a = curve - np.roll(curve, 1, axis=0)
    b = np.roll(curve, -1, axis=0) - curve
    ang = np.degrees(np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
                                       np.einsum("ij,ij->i", a, b))))
    return np.where(ang > tol_deg)[0]


def fit_farfield(multiblock: MultiBlock, farfield: np.ndarray,
                 growth: float = 1.15, max_growth: float = 1.3,
                 n_layers: int | None = None, smooth: int = 200,
                 tol: float = 1e-11) -> MultiBlock:
    """Add the annulus between ``multiblock``'s outer boundary and the closed
    curve ``farfield`` (any orientation, first point not repeated).

    The inner loop's node on the wake axis downstream (largest x among its
    nodes nearest y = its centroid's y) is matched to the far field's
    rightmost point on the same horizontal line, and the remaining nodes are
    matched by normalised arc length, with far-field corners snapped onto
    nodes.
    """
    mesh = multiblock.assemble(tol)
    ids = outer_loop(mesh)[::-1]                  # clockwise: region outside
    inner = mesh.points[ids]
    n = len(inner)

    # anchor: the inner node farthest downstream close to the wake axis
    yc = inner[:, 1].mean()
    near = np.abs(inner[:, 1] - yc) < 0.05 * np.ptp(inner[:, 1])
    k0 = int(np.flatnonzero(near)[np.argmax(inner[near, 0])])
    inner = np.roll(inner, -k0, axis=0)

    ff = np.asarray(farfield, dtype=float)
    if np.allclose(ff[0], ff[-1]):
        ff = ff[:-1]
    sa = 0.5 * np.sum(ff[:, 0] * np.roll(ff[:, 1], -1)
                      - np.roll(ff[:, 0], -1) * ff[:, 1])
    if sa > 0:                                    # make it clockwise
        ff = ff[::-1]
    # anchor on the far field: the point straight downstream of inner[0]
    seg_a, seg_b = ff, np.roll(ff, -1, axis=0)
    best, k_ff, t_ff = np.inf, 0, 0.0
    for k, (p, q) in enumerate(zip(seg_a, seg_b)):
        d = q - p
        if abs(d[1]) < 1e-15:
            continue
        t = (inner[0, 1] - p[1]) / d[1]
        if 0.0 <= t <= 1.0:
            x = p[0] + t * d[0]
            if x > inner[0, 0] and -x < best:
                best, k_ff, t_ff = -x, k, t
    start = seg_a[k_ff] + t_ff * (seg_b[k_ff] - seg_a[k_ff])
    ff = np.vstack([start, np.roll(ff, -(k_ff + 1), axis=0)])
    closed_ff = np.vstack([ff, ff[:1]])
    s_ff = arclength(closed_ff)
    L_ff = s_ff[-1]

    s_in = arclength(np.vstack([inner, inner[:1]]))
    t_in = s_in[:-1] / s_in[-1]

    # first outer height continues the last marched layer at each node, and
    # the line leaving each loop node continues the marched line into it
    loop_ids = np.roll(ids, -k0)
    pos = {int(v): i for i, v in enumerate(loop_ids)}
    last = np.zeros(n)
    inward = np.zeros((n, 2))
    for q in mesh.quads:
        for a in range(4):
            u, w = int(q[a]), int(q[(a + 1) % 4])
            for p_, o_ in ((u, w), (w, u)):
                if p_ in pos and o_ not in pos:
                    d = mesh.points[p_] - mesh.points[o_]
                    L = np.linalg.norm(d)
                    if L > last[pos[p_]]:
                        last[pos[p_]] = L
                        inward[pos[p_]] = d / L
    ok = last > 0
    last = np.where(ok, last, np.median(last[ok]))
    # nodes matched by normalised arc length, far-field corners on nodes
    corner_s = s_ff[_corners(ff)] / L_ff
    anchors_t, anchors_s = [0.0], [0.0]
    for cs in sorted(corner_s):
        if cs <= 0.0 or cs >= 1.0:
            continue
        i = int(np.argmin(np.abs(t_in - cs)))
        if t_in[i] > anchors_t[-1]:
            anchors_t.append(t_in[i])
            anchors_s.append(cs)
    anchors_t.append(1.0)
    anchors_s.append(1.0)
    s_nodes = np.interp(t_in, anchors_t, anchors_s) * L_ff
    outer = np.column_stack([np.interp(s_nodes, s_ff, closed_ff[:, 0]),
                             np.interp(s_nodes, s_ff, closed_ff[:, 1])])

    dist = np.linalg.norm(outer - inner, axis=1)
    h1 = last * growth
    if n_layers is None:
        # layers needed at the mean distance with the requested growth
        D, h = float(dist.mean()), float(np.median(h1))
        n_layers = max(4, int(np.ceil(np.log(1 + D * (growth - 1) / h)
                                      / np.log(growth))))
    # per line: first height h1, growth r chosen so the n heights reach dist
    frac = np.zeros((n, n_layers + 1))
    for i in range(n):
        lo, hi = 1.0 + 1e-9, max_growth * 4
        target = dist[i] / h1[i]
        if target <= n_layers:                   # shorter than uniform cells
            r = 1.0
            hs = np.full(n_layers, dist[i] / n_layers)
        else:
            for _ in range(80):
                r = 0.5 * (lo + hi)
                if (r ** n_layers - 1) / (r - 1) < target:
                    lo = r
                else:
                    hi = r
            hs = h1[i] * r ** np.arange(n_layers)
            hs *= dist[i] / hs.sum()
        frac[i, 1:] = np.cumsum(hs) / dist[i]
    grid = inner[:, None, :] + frac[:, :, None] * (outer - inner)[:, None, :]

    # Laplacian relaxation of the interior, weighted to vanish at both
    # boundaries so neither the marched spacing nor the far field moves
    if smooth:
        jj = np.arange(n_layers + 1) / n_layers
        wgt = (np.sin(np.pi * jj) ** 2)[None, :, None] * 0.25
        for _ in range(int(smooth)):
            lap = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0)
                   - 2.0 * grid)
            grid = grid + wgt * lap
    grid = np.concatenate([grid, grid[:1]], axis=0)       # close the seam

    blocks = []
    for b in multiblock.blocks:
        sides = {}
        for side, tag in b.sides.items():
            if isinstance(tag, str):
                sides[side] = None if tag in ("farfield", "outflow") else tag
            else:
                sides[side] = [None if t in ("farfield", "outflow") else t
                               for t in tag]
        blocks.append(Block(b.grid, sides, b.name))
    blocks.append(Block(grid, {"jmax": "farfield"}, name="outer"))
    return MultiBlock(blocks)


# ----------------------------------------------------------------------------
# conventional far-field curves, as point loops
# ----------------------------------------------------------------------------

def circle(radius: float, centre=(0.5, 0.0), n: int = 2000) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([centre[0] + radius * np.cos(t),
                            centre[1] + radius * np.sin(t)])


def box(upstream: float, downstream: float, lateral: float,
        n_side: int = 500) -> np.ndarray:
    x0, x1, y0, y1 = -upstream, 1.0 + downstream, -lateral, lateral
    c = [(x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    pts = []
    for a, b in zip(c, c[1:] + c[:1]):
        t = np.linspace(0, 1, n_side, endpoint=False)[:, None]
        pts.append(np.array(a) + t * (np.array(b) - np.array(a)))
    return np.vstack(pts)


def c_shape(radius: float, downstream: float, centre=(0.5, 0.0),
            n: int = 2000) -> np.ndarray:
    """Semicircle upstream, straight sides to an outflow plane."""
    cx, cy = centre
    x1 = 1.0 + downstream
    t = np.linspace(np.pi / 2, 3 * np.pi / 2, n // 2)
    arc = np.column_stack([cx + radius * np.cos(t), cy + radius * np.sin(t)])
    m = n // 6
    bottom = np.column_stack([np.linspace(cx, x1, m, endpoint=False),
                              np.full(m, cy - radius)])
    right = np.column_stack([np.full(m, x1),
                             np.linspace(cy - radius, cy + radius, m,
                                         endpoint=False)])
    top = np.column_stack([np.linspace(x1, cx, m, endpoint=False),
                           np.full(m, cy + radius)])
    return np.vstack([arc[1:], bottom, right, top])


# ----------------------------------------------------------------------------
# C-topology meshes in a supplied domain
# ----------------------------------------------------------------------------

def _fill(inner, outer, h1, growth=1.15, max_growth=1.3, n_layers=None):
    """Straight lines inner[i] -> outer[i], geometric spacing starting at
    h1[i]; returns ``(len(inner), n_layers + 1, 2)``."""
    dist = np.linalg.norm(outer - inner, axis=1)
    if n_layers is None:
        D, h = float(np.median(dist)), float(np.median(h1))
        n_layers = max(4, int(np.ceil(np.log(1 + D * (growth - 1) / h)
                                      / np.log(growth))))
    frac = np.zeros((len(inner), n_layers + 1))
    for i in range(len(inner)):
        target = dist[i] / h1[i]
        if target <= n_layers:
            hs = np.full(n_layers, dist[i] / n_layers)
        else:
            lo, hi = 1.0 + 1e-9, 4.0 * max_growth
            for _ in range(80):
                r = 0.5 * (lo + hi)
                if (r ** n_layers - 1) / (r - 1) < target:
                    lo = r
                else:
                    hi = r
            hs = h1[i] * r ** np.arange(n_layers)
            hs *= dist[i] / hs.sum()
        frac[i, 1:] = np.cumsum(hs) / dist[i]
    return inner[:, None, :] + frac[:, :, None] * (outer - inner)[:, None, :]


def far_curve(kind: str, xo: float, size: float, upstream: float | None = None
              ) -> np.ndarray:
    """The far field upstream of the outflow plane ``x = xo``, as an open
    polyline from its lower end on the plane to its upper end.

    ``'box'``: an H-type rectangle, half-height ``size``, front at
    ``-upstream`` (default ``size``). ``'c'``: a semicircle of radius ``size``
    about the mid-chord with straight sides to the plane. ``'circle'``: a
    circle of radius ``size`` about the mid-chord, truncated by the plane.
    """
    up = size if upstream is None else upstream
    cx = 0.5
    if kind == "box":
        pts = [(xo, -size), (-up, -size), (-up, size), (xo, size)]
        out = []
        for a, b in zip(pts[:-1], pts[1:]):
            L = np.hypot(b[0] - a[0], b[1] - a[1])
            m = max(20, int(L / size * 300))
            t = np.linspace(0, 1, m, endpoint=False)[:, None]
            out.append(np.array(a) + t * (np.array(b) - np.array(a)))
        out.append(np.array([pts[-1]]))
        return np.vstack(out)
    if kind == "c":
        th = np.linspace(-np.pi / 2, -3 * np.pi / 2, 1200)
        arc = np.column_stack([cx + size * np.cos(th), size * np.sin(th)])
        lo = np.column_stack([np.linspace(xo, cx, 300, endpoint=False),
                              np.full(300, -size)])
        hi = np.column_stack([np.linspace(cx, xo, 300)[1:],
                              np.full(299, size)])
        return np.vstack([lo, arc, hi])
    if kind == "circle":
        c = np.clip((xo - cx) / size, -0.999, 0.999)
        th0 = np.arccos(c)
        th = np.linspace(-th0, -(2 * np.pi - th0), 2400)
        return np.column_stack([cx + size * np.cos(th), size * np.sin(th)])
    raise ValueError(kind)


def c_domain(c_multiblock: MultiBlock, kind: str, size: float,
             upstream: float | None = None, growth: float = 1.15,
             smooth: int = 300, project: float = 1.0,
             n_layers: int | None = None) -> MultiBlock:
    """Extend a C-mesh whose outflow plane is the far field's outflow plane
    with one outer block reaching the far-field curve.

    The outer block's inner side is the C block's outer row; its two other
    sides continue the outflow plane straight to the far field. Far-field
    corners (two for a box) land on nodes.

    Each node of the far field is where the marched grid line through it,
    continued straight, meets the far field (``project = 1``), rather than the
    node at the same fraction of arc length (``project = 0``). Measured on
    NACA 4412 in a box: minimum angle 24.0 degrees by arc length, 44.8 by
    projection; in a C-shape 22.0 and 76.6. Arc length sends the nodes where
    the marched row turns from the body onto the wake far upstream along the
    far field, and the lines slant.
    """
    C = c_multiblock.blocks[0]
    g = C.grid
    row = g[:, -1]
    xo = float(0.5 * (g[0, 0, 0] + g[-1, 0, 0]))
    curve = far_curve(kind, xo, size, upstream)
    if row[0, 1] > row[-1, 1]:              # orient: curve from the i=0 side
        curve = curve[::-1]
    s_row = arclength(row)
    s_cur = arclength(curve)
    t_row = s_row / s_row[-1]
    if project > 0.0:
        # where each marched line, continued straight, meets the far field
        d = g[:, -1] - g[:, -2]
        d /= np.linalg.norm(d, axis=1)[:, None]
        a_pts, e = curve[:-1], np.diff(curve, axis=0)
        t_hit = t_row.copy()
        for i in range(1, len(row) - 1):
            den = d[i, 0] * e[:, 1] - d[i, 1] * e[:, 0]
            w_ = a_pts - row[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = (w_[:, 0] * e[:, 1] - w_[:, 1] * e[:, 0]) / den
                uu = (w_[:, 0] * d[i, 1] - w_[:, 1] * d[i, 0]) / den
            hit = np.flatnonzero((tt > 0) & (uu >= 0) & (uu <= 1))
            if len(hit):
                k = hit[np.argmin(tt[hit])]
                t_hit[i] = (s_cur[k] + uu[k] * (s_cur[k + 1] - s_cur[k])) \
                    / s_cur[-1]
        t_mix = (1.0 - project) * t_row + project * t_hit
        floor = 0.1 / len(row)
        for i in range(1, len(row)):
            t_mix[i] = max(t_mix[i], t_mix[i - 1] + floor * (t_row[i] - t_row[i - 1]))
        t_mix = (t_mix - t_mix[0]) / (t_mix[-1] - t_mix[0])
    else:
        t_mix = t_row
    corner_s = s_cur[_corners(curve[1:-1]) + 1] / s_cur[-1] \
        if len(curve) > 3 else np.array([])
    anchors_t, anchors_s = [0.0], [0.0]
    for cs in sorted(corner_s):
        i = int(np.argmin(np.abs(t_mix - cs)))
        if 0.0 < t_mix[i] < 1.0 and t_mix[i] > anchors_t[-1]:
            anchors_t.append(t_mix[i])
            anchors_s.append(cs)
    anchors_t.append(1.0)
    anchors_s.append(1.0)
    s_nodes = np.interp(t_mix, anchors_t, anchors_s) * s_cur[-1]
    outer = np.column_stack([np.interp(s_nodes, s_cur, curve[:, 0]),
                             np.interp(s_nodes, s_cur, curve[:, 1])])
    h1 = np.linalg.norm(g[:, -1] - g[:, -2], axis=1) * growth
    grid = _fill(row, outer, h1, growth, n_layers=n_layers)
    if smooth:
        nj = grid.shape[1]
        jj = np.arange(nj) / (nj - 1)
        wgt = (np.sin(np.pi * jj) ** 2)[None, :, None] * 0.25
        for _ in range(int(smooth)):
            lap = np.zeros_like(grid)
            lap[1:-1] = grid[2:] - 2.0 * grid[1:-1] + grid[:-2]
            grid = grid + wgt * lap
    blocks = []
    for k, b in enumerate(c_multiblock.blocks):
        sides = dict(b.sides)
        if k == 0:
            sides["jmax"] = None
        blocks.append(Block(b.grid, sides, b.name))
    blocks.append(Block(grid, {"jmax": "farfield", "imin": "outflow",
                               "imax": "outflow"}, name="outer"))
    return MultiBlock(blocks)


def domain_mesh(boundary, first_cell: float, domain: str = "c",
                size: float = 50.0, upstream: float | None = None,
                downstream: float | None = None, inner: float = 3.0,
                inner_layers: int = 70, **kw) -> MultiBlock:
    """A mesh in a supplied far field.

    ``'c'`` (C-shape) and ``'box'`` (H-type rectangle) have an outflow plane,
    so the wake line the trailing edge emits can end on it: a C-mesh, marched
    to ``inner`` chords, then the TFI outer block. ``'circle'`` has no
    outflow plane and no corner: the band around the body is then an annulus
    and so is the zone outside it -- an O-mesh fitted to the circle.

    A C-mesh in a circle was tried and rejected on measurement. The outflow
    plane has to cut the circle, and the corner it leaves is as sharp as the
    plane is close to the edge: 27 degrees of minimum angle with the plane at
    0.9 of the radius, 15.5 at 0.97. Pulling the plane back to the centre
    removes the corner and makes the circle a C-shape.
    """
    from aeromesh.mesh.airfoil import c_mesh, o_mesh, wall_segments
    if domain == "circle":
        _, elems = wall_segments(boundary)
        if len(elems) and int(np.max(elems)) >= 4:
            raise ValueError(
                "domain_mesh: this body has a corner asking for %d elements "
                "(a sharp trailing edge), so it needs a wake line and an "
                "outflow boundary; a circle has neither. Use domain='c' (a "
                "semicircle with an outflow plane) or a blunt trailing edge."
                % np.max(elems))
        om = o_mesh(boundary, first_cell, radius=inner,
                    n_layers=inner_layers, **kw)
        return fit_farfield(om, circle(size))
    wake = size if downstream is None else downstream
    cm = c_mesh(boundary, first_cell, radius=inner, wake_length=wake,
                n_layers=inner_layers, **kw)
    return c_domain(cm, domain, size, upstream)


def snap_farfield(c_multiblock: MultiBlock, kind: str, size: float,
                  n_blend: int = 20, upstream: float | None = None
                  ) -> MultiBlock:
    """Put a marched C-mesh's outer row exactly on a supplied far field by
    sliding each node **along its own grid line**, the correction blended
    over the outer ``n_blend`` rows.

    The march already ends close to the far field (within 0.5 % at 500
    chords), so the move is small. Every node keeps its grid line, and the
    blend changes only cell lengths along it, not directions, so quality is
    the marched mesh's. Two constructions that replaced the outer part by TFI
    were measured first and rejected: marching to 3--25 chords and filling to
    500 inverts cells or reaches 73--90 degrees of non-orthogonality, because
    the wake still runs to 500 chords in slivers; a thin TFI band over the
    last 5--20 rows of a full march costs 22--80 degrees against 12.4.
    """
    C = c_multiblock.blocks[0]
    g = C.grid.copy()
    xo = float(0.5 * (g[0, 0, 0] + g[-1, 0, 0]))
    curve = far_curve(kind, xo, size, upstream)
    a_pts, e = curve[:-1], np.diff(curve, axis=0)
    nj = g.shape[1]
    jb = max(1, nj - 1 - int(n_blend))
    for i in range(g.shape[0]):
        p, d = g[i, -1], g[i, -1] - g[i, -2]
        d = d / np.linalg.norm(d)
        den = d[0] * e[:, 1] - d[1] * e[:, 0]
        w_ = a_pts - p
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (w_[:, 0] * e[:, 1] - w_[:, 1] * e[:, 0]) / den
            uu = (w_[:, 0] * d[1] - w_[:, 1] * d[0]) / den
        hit = np.flatnonzero((uu >= -1e-9) & (uu <= 1 + 1e-9)
                             & np.isfinite(tt))
        if not len(hit):
            raise ValueError(f"grid line {i} misses the far field")
        k = hit[np.argmin(np.abs(tt[hit]))]
        t_end = float(tt[k])
        # arc length along the line over the blended rows
        seg = np.linalg.norm(np.diff(g[i, jb:], axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        frac = s / s[-1]
        w = frac * frac * (3.0 - 2.0 * frac)
        # slide each blended node along the local line direction
        g[i, jb:] = g[i, jb:] + (w * t_end)[:, None] * d[None, :]
    blocks = [Block(g, dict(C.sides), C.name)] + list(c_multiblock.blocks[1:])
    return MultiBlock(blocks)

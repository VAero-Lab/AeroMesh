"""Geometric fields on the medial axis.

Every quantity Fogg's flux balance consumes is computed here: the medial radius
r_m, the medial angle theta_m, the unit vectors to the touching boundary points,
and the touch points themselves. The first iteration of AeroMesh computed none
of them, which is why its topology stage had only node degree to work with.

Touch grouping
--------------
A Voronoi vertex of a sampled boundary is equidistant from three *samples*, but
those samples may be neighbours on one smooth stretch of boundary, in which case
they are one *touch*, not three. Grouping contiguous same-loop runs into a single
touch is what makes ``theta_m`` mean what Fogg means by it, and it gives the
vertex classification for free: two touch groups is a point on a medial edge,
three or more is a medial vertex, one is curvature contact.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def segment_distance(query: np.ndarray, pts: np.ndarray, loop_id: np.ndarray,
                     offsets: np.ndarray, tree: cKDTree | None = None,
                     k: int = 6) -> np.ndarray:
    """Distance from each query point to the boundary *polyline*, not its samples.

    Sampling a curve and measuring to the nearest sample overestimates the
    distance by O(delta^2 / r). Measuring to the nearest segment removes that,
    which matters when r_m is checked against an analytic value.

    Parameters
    ----------
    query : np.ndarray, shape (Q, 2)
    pts : np.ndarray, shape (P, 2)
        All boundary samples, loops concatenated.
    loop_id : np.ndarray, shape (P,)
    offsets : np.ndarray, shape (n_loops + 1,)
        Start index of each loop in ``pts``; ``offsets[-1] == len(pts)``.
    tree : cKDTree, optional
        Prebuilt tree over ``pts``.
    k : int
        Nearest samples considered per query; their incident segments are the
        candidates. Six is ample for a uniformly sampled boundary.
    """
    query = np.atleast_2d(np.asarray(query, dtype=np.float64))
    if tree is None:
        tree = cKDTree(pts)
    k = min(k, len(pts))
    _, idx = tree.query(query, k=k)
    idx = np.atleast_2d(idx)

    # Successor of sample i within its own loop, wrapping at the loop end.
    start = offsets[loop_id]
    stop = offsets[loop_id + 1]
    nxt = np.where(np.arange(len(pts)) + 1 < stop, np.arange(len(pts)) + 1, start)
    prv = np.where(np.arange(len(pts)) - 1 >= start, np.arange(len(pts)) - 1, stop - 1)

    best = np.full(len(query), np.inf)
    for col in range(idx.shape[1]):
        i = idx[:, col]
        for j in (nxt[i], prv[i]):
            a, b = pts[i], pts[j]
            ab = b - a
            denom = np.einsum("ij,ij->i", ab, ab)
            t = np.where(denom > 0,
                         np.einsum("ij,ij->i", query - a, ab) / np.where(denom > 0, denom, 1.0),
                         0.0)
            t = np.clip(t, 0.0, 1.0)
            foot = a + t[:, None] * ab
            best = np.minimum(best, np.linalg.norm(query - foot, axis=1))
    return best


def touch_groups(generators: np.ndarray, loop_id: np.ndarray, offsets: np.ndarray,
                 merge_gap: int = 2) -> list[np.ndarray]:
    """Split a vertex's generator samples into distinct boundary touches.

    Samples on the same loop within ``merge_gap`` indices of each other are the
    same touch. Wrap-around within a loop is handled.
    """
    groups: list[np.ndarray] = []
    order = np.argsort(generators)
    g = generators[order]
    for lid in np.unique(loop_id[g]):
        sel = g[loop_id[g] == lid]
        n = offsets[lid + 1] - offsets[lid]
        local = np.sort(sel - offsets[lid])
        runs, cur = [], [local[0]]
        for a in local[1:]:
            if a - cur[-1] <= merge_gap:
                cur.append(a)
            else:
                runs.append(cur)
                cur = [a]
        runs.append(cur)
        # Merge the first and last runs if they touch across the wrap.
        if len(runs) > 1 and (local[0] + n) - runs[-1][-1] <= merge_gap:
            runs[0] = runs[-1] + runs[0]
            runs.pop()
        groups.extend(np.asarray(r, dtype=int) + offsets[lid] for r in runs)
    return groups


def group_foot(position: np.ndarray, group: np.ndarray, pts: np.ndarray,
               loop_id: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Exact closest point on the boundary polyline within one touch group.

    Using the nearest *sample* as the touch point puts an O(delta) error into
    every direction vector, and therefore into theta_m -- worst near a corner,
    where the two touches are only a few samples apart and the error is a large
    fraction of the angle between them. Projecting onto the segments instead
    makes theta_m second order, which is what keeps it invariant under scaling.
    """
    lid = loop_id[group[0]]
    start, stop = offsets[lid], offsets[lid + 1]
    n = stop - start
    local = np.unique(group - start)
    # The segments spanned by the group, plus one either side.
    lo, hi = local.min() - 1, local.max() + 1
    idx = (np.arange(lo, hi + 1) % n) + start
    a = pts[idx[:-1]]
    b = pts[idx[1:]]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.where(denom > 0, np.einsum("ij,ij->i", position - a, ab) /
                 np.where(denom > 0, denom, 1.0), 0.0)
    feet = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return feet[int(np.argmin(np.linalg.norm(feet - position, axis=1)))]


def vertex_fields(position: np.ndarray, generators: np.ndarray, pts: np.ndarray,
                  loop_id: np.ndarray, offsets: np.ndarray,
                  r_m: float | None = None) -> dict:
    """Medial fields at one medial point.

    Returns
    -------
    dict with keys
        ``r_m``       radius of the maximal inscribed circle
        ``theta_m``   largest angle subtended by two distinct touches (radians);
                      0 when there is a single touch (curvature contact)
        ``n_touch``   number of distinct touches
        ``touch``     (n_touch, 2) the touch points, ordered by (loop, index)
        ``touch_index`` (n_touch,) index of the nearest boundary sample to each
        ``normals``   (n_touch, 2) unit vectors from the medial point to each
    """
    groups = touch_groups(generators, loop_id, offsets)
    touch = np.array([group_foot(position, g, pts, loop_id, offsets) for g in groups])
    # The nearest sample of each group, so a block can walk the boundary between
    # two touches. The exact foot gives the geometry; the index gives the arc.
    index = np.array([int(g[int(np.argmin(np.linalg.norm(pts[g] - position, axis=1)))])
                      for g in groups], dtype=int)
    vec = touch - position
    dist = np.linalg.norm(vec, axis=1)
    # Order the touches by (loop, sample index), not by distance. A medial point
    # is equidistant from its touches by definition, so sorting on distance is a
    # coin flip that reorders t1 and t2 arbitrarily from one sample to the next
    # -- which would scramble the boundary arcs a block is built from.
    order = np.lexsort((index, loop_id[index]))
    touch, vec, dist, index = touch[order], vec[order], dist[order], index[order]
    normals = vec / np.maximum(dist, 1e-300)[:, None]

    theta = 0.0
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            theta = max(theta, float(np.arccos(np.clip(normals[i] @ normals[j], -1.0, 1.0))))

    return {
        "r_m": float(dist[0]) if r_m is None else float(r_m),
        "theta_m": theta,
        "n_touch": len(groups),
        "touch": touch,
        "touch_index": index,
        "normals": normals,
    }


def optimum_flow_index(theta_m: np.ndarray | float) -> np.ndarray:
    """Fogg's optimum mesh-flow index ``n = round((pi - theta_m) / (pi / 2))``.

    The number of quarter-turns a cross makes crossing the medial axis. Zero
    means boundary-aligned flow on both sides; one means the medial axis runs
    diagonally through the cross-field; two means the flow reverses, which is
    what a thin slot demands. Non-zero values are where singularities live.
    """
    return np.rint((np.pi - np.asarray(theta_m, dtype=np.float64)) / (np.pi / 2)).astype(int)

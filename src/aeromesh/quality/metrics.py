"""Mesh metrics, measured on a mesh that exists.

Every number is taken at the extremum that can fail, per the standing practice
written after the October audit: the first cell height is reported at *every*
wall node (its worst ratio to the request, not its minimum), orthogonality at
every wall face, and the expansion ratios as maxima.

Two scopes are reported for the cell metrics: the whole mesh, and the
**boundary-layer region** -- the layers whose distance from the wall is below a
given thickness -- because that is where the project claims invariance.
"""

from __future__ import annotations

import numpy as np


def cell_angles(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smallest and largest interior angle of every cell, in degrees."""
    c = np.stack([grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]],
                 axis=2)                                   # (ni-1, nj-1, 4, 2)
    nxt = np.roll(c, -1, axis=2) - c
    prv = np.roll(c, 1, axis=2) - c
    cosang = np.einsum("...k,...k->...", nxt, prv) / np.maximum(
        np.linalg.norm(nxt, axis=-1) * np.linalg.norm(prv, axis=-1), 1e-300)
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return ang.min(axis=2), ang.max(axis=2)


def cell_area(grid: np.ndarray) -> np.ndarray:
    a, b, c, d = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
    return 0.5 * ((c[..., 0] - a[..., 0]) * (d[..., 1] - b[..., 1])
                  - (d[..., 0] - b[..., 0]) * (c[..., 1] - a[..., 1]))


def cell_aspect(grid: np.ndarray) -> np.ndarray:
    di = 0.5 * (np.linalg.norm(grid[1:, :-1] - grid[:-1, :-1], axis=-1)
                + np.linalg.norm(grid[1:, 1:] - grid[:-1, 1:], axis=-1))
    dj = 0.5 * (np.linalg.norm(grid[:-1, 1:] - grid[:-1, :-1], axis=-1)
                + np.linalg.norm(grid[1:, 1:] - grid[1:, :-1], axis=-1))
    return np.maximum(di, dj) / np.maximum(np.minimum(di, dj), 1e-300)


def wall_report(grid: np.ndarray, wall: np.ndarray, first_cell: float,
                bl_thickness: float = 0.02) -> dict:
    """Metrics of a block marched from ``jmin``, restricted to its wall.

    ``wall`` is a boolean per ``jmin`` *edge*: True where the edge is wall.
    """
    wall = np.asarray(wall, dtype=bool)
    node = np.zeros(grid.shape[0], dtype=bool)
    node[:-1] |= wall
    node[1:] |= wall
    ii = np.where(node)[0]

    w0, w1 = grid[:, 0], grid[:, 1]
    # wall tangent at each wall node, from the wall edges around it
    tang = np.zeros_like(w0)
    e = w0[1:] - w0[:-1]
    e_w = np.where(wall[:, None], e, 0.0)
    tang[:-1] += e_w
    tang[1:] += e_w
    tang /= np.maximum(np.linalg.norm(tang, axis=1), 1e-300)[:, None]
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])

    step = w1 - w0
    y1 = np.einsum("ij,ij->i", step, nrm)[ii]            # normal height
    length = np.linalg.norm(step, axis=1)[ii]
    off = np.degrees(np.arccos(np.clip(y1 / np.maximum(length, 1e-300),
                                       -1.0, 1.0)))

    # wall-normal growth, along each wall line, inside the boundary layer
    seg = np.linalg.norm(np.diff(grid[ii], axis=1), axis=-1)   # (nw, nj-1)
    dist = np.cumsum(seg, axis=1)
    in_bl = dist[:, 1:] <= bl_thickness
    growth = seg[:, 1:] / np.maximum(seg[:, :-1], 1e-300)
    g_bl = np.where(in_bl, growth, np.nan)

    # along-wall spacing ratio between neighbouring wall edges
    ds = np.linalg.norm(e, axis=1)
    pair = wall[:-1] & wall[1:]
    r = ds[1:] / np.maximum(ds[:-1], 1e-300)
    r = np.maximum(r, 1.0 / r)[pair]

    # Face-based wall non-orthogonality, as a finite-volume solver sees it:
    # the angle between a wall face's normal and the line from the face centre
    # to the first cell centre. The node-based figure above uses the same
    # central difference the marching scheme does, so it is zero by
    # construction and proves nothing; this one shares no stencil with it.
    fi = np.where(wall)[0]
    fmid = 0.5 * (w0[fi] + w0[fi + 1])
    ft = w0[fi + 1] - w0[fi]
    fn = np.column_stack([-ft[:, 1], ft[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-300)[:, None]
    cc = 0.25 * (w0[fi] + w0[fi + 1] + w1[fi] + w1[fi + 1]) - fmid
    cc /= np.maximum(np.linalg.norm(cc, axis=1), 1e-300)[:, None]
    face_off = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", cc, fn),
                                            -1.0, 1.0)))

    ratio = y1 / first_cell
    return {
        "n_wall_nodes": int(len(ii)),
        "y1_min_ratio": float(ratio.min()),
        "y1_max_ratio": float(ratio.max()),
        "y1_within_2pct": float(np.mean(np.abs(ratio - 1.0) <= 0.02)),
        "wall_orth_median": float(np.median(off)),
        "wall_orth_p95": float(np.percentile(off, 95)),
        "wall_orth_max": float(off.max()),
        "wall_face_nonorth_median": float(np.median(face_off)),
        "wall_face_nonorth_p95": float(np.percentile(face_off, 95)),
        "wall_face_nonorth_max": float(face_off.max()),
        "bl_growth_max": float(np.nanmax(g_bl)),
        "bl_growth_median": float(np.nanmedian(g_bl)),
        "along_wall_ratio_max": float(r.max()) if len(r) else 1.0,
        "_orth": off, "_face_orth": face_off, "_face_x": fmid[:, 0], "_y1": y1, "_x": w0[ii, 0],
    }


def bl_mask(grid: np.ndarray, wall: np.ndarray, bl_thickness: float):
    """Cells of a wall-marched block lying over wall and within the layer."""
    seg = np.linalg.norm(np.diff(grid, axis=1), axis=-1)
    dist = np.cumsum(seg, axis=1)                            # (ni, nj-1)
    d_cell = 0.5 * (dist[:-1] + dist[1:])
    return (d_cell <= bl_thickness) & np.asarray(wall, bool)[:, None]


def scope_report(grid: np.ndarray, mask: np.ndarray | None = None) -> dict:
    lo, hi = cell_angles(grid)
    ar = cell_aspect(grid)
    area = cell_area(grid)
    skew = np.maximum(90.0 - lo, hi - 90.0)
    ri = area[1:] / np.maximum(area[:-1], 1e-300)
    ri = np.maximum(ri, 1.0 / np.maximum(ri, 1e-300))
    rj = area[:, 1:] / np.maximum(area[:, :-1], 1e-300)
    rj = np.maximum(rj, 1.0 / np.maximum(rj, 1e-300))
    if mask is None:
        mask = np.ones(lo.shape, bool)
    mi = mask[1:] & mask[:-1]
    mj = mask[:, 1:] & mask[:, :-1]
    return {
        "cells": int(mask.sum()),
        "min_angle": float(lo[mask].min()),
        "max_angle": float(hi[mask].max()),
        "skew_median": float(np.median(skew[mask])),
        "skew_p99": float(np.percentile(skew[mask], 99)),
        "skew_max": float(skew[mask].max()),
        "aspect_max": float(ar[mask].max()),
        "area_ratio_i_max": float(ri[mi].max()) if mi.any() else 1.0,
        "area_ratio_j_max": float(rj[mj].max()) if mj.any() else 1.0,
        "negative_cells": int(np.sum(area[mask] <= 0)),
    }


def mesh_report(multiblock, first_cell: float,
                bl_thickness: float = 0.02) -> dict:
    """Whole-mesh, boundary-layer and wall metrics of a :class:`MultiBlock`."""
    out = {"blocks": [], "wall": None, "bl": None}
    lo_all, hi_all, cells = [], [], 0
    for b in multiblock.blocks:
        rep = scope_report(b.grid)
        rep["name"] = b.name
        rep["shape"] = tuple(int(v) for v in b.shape)
        out["blocks"].append(rep)
        cells += rep["cells"]
        tag = b.sides.get("jmin")
        if tag is None:
            continue
        n_e = b.shape[0] - 1
        wall = (np.full(n_e, tag == "wall") if isinstance(tag, str)
                else np.array([t == "wall" for t in tag]))
        if wall.any():
            out["wall"] = wall_report(b.grid, wall, first_cell, bl_thickness)
            out["bl"] = scope_report(b.grid, bl_mask(b.grid, wall,
                                                     bl_thickness))
    out["cells"] = cells
    out["min_angle"] = min(r["min_angle"] for r in out["blocks"])
    out["max_angle"] = max(r["max_angle"] for r in out["blocks"])
    out["aspect_max"] = max(r["aspect_max"] for r in out["blocks"])
    return out


def cell_centroid(grid: np.ndarray) -> np.ndarray:
    """Area centroid of every cell (not the mean of its corners)."""
    c = np.stack([grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]],
                 axis=2)
    n = np.roll(c, -1, axis=2)
    cr = c[..., 0] * n[..., 1] - n[..., 0] * c[..., 1]
    a = 0.5 * cr.sum(axis=2)
    return ((c + n) * cr[..., None]).sum(axis=2) / (6.0 * a[..., None])


def fv_nonorthogonality(grid: np.ndarray):
    """Finite-volume non-orthogonality of the interior faces, in degrees.

    The angle between a face's normal and the line joining the centroids of
    the two cells it separates -- the quantity OpenFOAM's ``checkMesh``
    reports, and the one the corner-angle skew cannot see: a cell with four
    right angles and a slight taper has its centroid displaced *along* it, by
    more than its own height once the aspect ratio is large.

    Returns ``(across_i, across_j)``: faces between ``i`` neighbours, shape
    ``(ni-2, nj-1)``, and between ``j`` neighbours, shape ``(ni-1, nj-2)``.
    """
    cc = cell_centroid(grid)

    def ang(d, e):
        nrm = np.stack([e[..., 1], -e[..., 0]], axis=-1)
        cosv = np.abs(np.einsum("...k,...k->...", d, nrm)) / np.maximum(
            np.linalg.norm(d, axis=-1) * np.linalg.norm(nrm, axis=-1), 1e-300)
        return np.degrees(np.arccos(np.clip(cosv, 0.0, 1.0)))

    a_i = ang(cc[1:] - cc[:-1], grid[1:-1, 1:] - grid[1:-1, :-1])
    a_j = ang(cc[:, 1:] - cc[:, :-1], grid[1:, 1:-1] - grid[:-1, 1:-1])
    return a_i, a_j


def assembled_nonorthogonality(mesh) -> np.ndarray:
    """Finite-volume non-orthogonality of every interior face of an assembled
    :class:`~aeromesh.mesh.structured.Mesh`, block interfaces included.

    ``fv_nonorthogonality`` works block by block and so cannot see a face
    shared by two blocks; the 35 degrees across a wake line between the C
    block and its aft strip was only visible here and in ``checkMesh``.
    """
    P, q = mesh.points, mesh.quads
    c = P[q]
    n = np.roll(c, -1, axis=1)
    cr = c[..., 0] * n[..., 1] - n[..., 0] * c[..., 1]
    area = 0.5 * cr.sum(axis=1)
    cc = ((c + n) * cr[..., None]).sum(axis=1) / (6.0 * area[:, None])
    e = np.stack([q, np.roll(q, -1, axis=1)], axis=-1).reshape(-1, 2)
    cell = np.repeat(np.arange(len(q)), 4)
    key = np.sort(e, axis=1)
    o = np.lexsort((cell, key[:, 1], key[:, 0]))
    ks, cs = key[o], cell[o]
    same = np.where(np.all(ks[1:] == ks[:-1], axis=1))[0]
    d = cc[cs[same + 1]] - cc[cs[same]]
    t = P[ks[same, 1]] - P[ks[same, 0]]
    nrm = np.column_stack([t[:, 1], -t[:, 0]])
    cosv = np.abs(np.einsum("ij,ij->i", d, nrm)) / np.maximum(
        np.linalg.norm(d, axis=1) * np.linalg.norm(nrm, axis=1), 1e-300)
    return np.degrees(np.arccos(np.clip(cosv, 0.0, 1.0)))

"""Where is the worst finite-volume non-orthogonality of a standard C-mesh?

    python validation/locate_nonorth.py [level ...]

For each level, lists the worst faces with their position, block-grid indices
(i along the C, j off the wall) and the two cells' aspect ratios, and the
fraction of faces over 30 degrees by region. Writes
``results/nonorth_location.json``.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from study_next import am_c  # noqa: E402


def faces(mesh):
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
    a, b = cs[same], cs[same + 1]
    d = cc[b] - cc[a]
    t = P[ks[same, 1]] - P[ks[same, 0]]
    nrm = np.column_stack([t[:, 1], -t[:, 0]])
    cosv = np.abs(np.einsum("ij,ij->i", d, nrm)) / np.maximum(
        np.linalg.norm(d, axis=1) * np.linalg.norm(nrm, axis=1), 1e-300)
    ang = np.degrees(np.arccos(np.clip(cosv, 0.0, 1.0)))
    mid = 0.5 * (P[ks[same, 0]] + P[ks[same, 1]])
    side = np.linalg.norm(n - c, axis=2)
    asp = side.max(axis=1) / side.min(axis=1)
    return ang, mid, a, b, asp, area


def region(x, y):
    r = np.hypot(x - 0.5, y)
    if r > 3.0:
        return "outer (r > 3c)"
    if x > 1.0 and abs(y) < 0.5:
        return "near wake (x > 1, |y| < 0.5)"
    if r < 0.6:
        return "near body"
    return "inner field"


def main(levels):
    out = {}
    for L in levels:
        mb, y1 = am_c(L)
        mesh = mb.assemble()
        ang, mid, a, b, asp, area = faces(mesh)
        # cell -> (block, i, j)
        idx = []
        for k, blk in enumerate(mb.blocks):
            ni, nj = blk.grid.shape[0] - 1, blk.grid.shape[1] - 1
            ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
            idx += list(zip([k] * ni * nj, ii.ravel(), jj.ravel()))
        top = np.argsort(ang)[::-1][:12]
        rows = []
        for f in top:
            rows.append({"deg": float(ang[f]), "x": float(mid[f, 0]),
                         "y": float(mid[f, 1]),
                         "cell_a": [int(v) for v in idx[a[f]]],
                         "cell_b": [int(v) for v in idx[b[f]]],
                         "aspect_a": float(asp[a[f]]),
                         "aspect_b": float(asp[b[f]]),
                         "area_ratio": float(max(area[a[f]], area[b[f]])
                                             / min(area[a[f]], area[b[f]]))})
        reg = {}
        for f in np.flatnonzero(ang > 30.0):
            k = region(*mid[f])
            reg[k] = reg.get(k, 0) + 1
        nj_c = mb.blocks[0].grid.shape[1]
        out[str(L)] = {"max": float(ang.max()), "n_over_30": int(
            (ang > 30).sum()), "regions_over_30": reg, "worst": rows,
            "nj": int(nj_c)}
        print(f"L{L}: max {ang.max():.1f}  >30: {(ang > 30).sum()}  {reg}")
        for r in rows[:6]:
            print(f"   {r['deg']:5.1f} at ({r['x']:9.3f},{r['y']:9.3f}) "
                  f"cells {r['cell_a']} {r['cell_b']} AR {r['aspect_a']:.0f}"
                  f"/{r['aspect_b']:.0f} area x{r['area_ratio']:.2f}")
    Path(__file__).parent.joinpath("results", "nonorth_location.json") \
        .write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main([float(a) for a in sys.argv[1:]] or [1.0, 2.0])

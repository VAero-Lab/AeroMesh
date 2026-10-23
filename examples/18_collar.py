"""Example 18 — the near-wall collar, zone 1 of a zoned blocking.

The general medial decomposition was being asked to produce a boundary-layer
mesh, and it is not a boundary-layer method: it is block-structured meshing for
mechanical CAD parts, with no Delta_y1 in it, no wall normality requirement and
no anisotropy. Asked for a near-wall band it returned four blocks of lengths
0.27, 0.39, 0.69 and 0.80 with a 262 degree interior angle and an aspect ratio
of 402, and interior block edges that turned through 374 degrees.

The collar is built instead of decomposed. Nothing about it is selected:

* a thin band around a simple closed curve is an **annulus**, and an annulus is
  an O-grid because there is nothing else it could be -- no choice is made, so
  none can be wrong;
* its thickness is ``min(delta_target, rho_max * r_m)``, the flow setting it and
  the medial radius capping it to the room available;
* its cuts are at the **boundaries of each corner's normal cone** -- the
  perpendiculars to the two wall edges that form it. Fogg's own clamp.

Why the cone boundaries and not the interior cross directions
``alpha * i / n_c``: at a blunt trailing edge those point downstream from
*both* corners and converge. They cross 0.021 chord behind the trailing edge,
and a layer 0.02 thick puts its outer curve essentially at that crossing, so the
base face -- 0.002 long -- maps to 0.0001 of outer curve. A band that contracts
twentyfold is a block joining an edge to a point, and no amount of resampling
fixes it: it stays 0.0001 at four times the resolution, because the rays really
do meet there.

Perpendiculars to a straight edge are parallel. So the two rays bounding the
base face never converge, it propagates to its own offset one for one, each wall
segment maps to exactly its own offset, and each corner owns exactly its
rounding arc of ``alpha - pi``. A blunt trailing edge then gives two bands and
two 81.8 degree corner wedges; a sharp one gives a single band and a single
163.2 degree wedge. Nothing in the code names a trailing edge, and nothing
branches on which kind it is.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import aeromesh as am
from aeromesh.blocking.collar import (build_collar, cell_quality, mesh_collar,
                                      nearest_on_polyline, smooth_patch)

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

DELTA, RHO_MAX = 0.02, 0.05
N_ACROSS, SPACING, FIRST = 25, 0.01, 1e-4
N_OUTER, SWEEPS = 800, 0

# NACA 6409 is in the battery deliberately: 6 per cent camber on 9 per cent
# thickness, and the configuration that exposed both defects of 2026-10-21. Its
# leading edge was detected as a false 20.5 degree corner, which added three
# rays and two wedges and folded the mesh (minimum angle 0.5 degrees, aspect
# ratio 4779); and one degenerate medial sample out of 2075 set r_m = 0 on the
# wall near x = 0.38, which took the thickness cap to zero and put the outer
# curve on the wall (aspect ratio 8715). Neither showed on any 0012/2412/4415
# section, which is the whole reason it is plotted here.
CASES = (("2412", "blunt"), ("2412", "sharp"), ("6409", "blunt"))

fig, axes = plt.subplots(len(CASES), 3, figsize=(17.5, 10.5),
                         gridspec_kw={"width_ratios": [1.45, 1.45, 1.0]})

def _min_angles(g):
    a, b, c, d = g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]
    q = np.stack([a, b, c, d], axis=2)
    e = np.roll(q, -1, axis=2) - q
    L = np.linalg.norm(e, axis=3)
    pr = np.roll(e, 1, axis=2)
    cos = (np.einsum("ijkl,ijkl->ijk", -pr, e)
           / np.maximum(np.linalg.norm(pr, axis=3) * L, 1e-300))
    return np.degrees(np.arccos(np.clip(cos, -1, 1))).min(axis=2).ravel()


print(f"{'case':<16}{'blocks':>8}{'patches':>9}{'cells':>8}{'min ang':>9}"
      f"{'max ang':>9}{'max AR':>8}{'J>0':>7}{'y1':>10}{'<45deg':>10}{'median':>9}")
print("-" * 103)

for row, (name, te) in enumerate(CASES):
    body = am.load_airfoil(name, n_points=400).transform(name="main")
    region = am.build_region([body], outer=am.circle_farfield([body], 8.0),
                             te=te)
    collar = build_collar(region, delta_target=DELTA, rho_max=RHO_MAX,
                          n_outer=N_OUTER)
    grids = [smooth_patch(g, sweeps=SWEEPS)
             for g in mesh_collar(collar, n_across=N_ACROSS,
                                  target_spacing=SPACING, first_cell=FIRST)]
    q = [cell_quality(g) for g in grids]
    wall = collar["wall"]
    y1 = min(float(np.min(nearest_on_polyline(g[:, 1, :], wall)[1]))
             for g in grids if g.shape[1] > 2)
    ang = np.concatenate([_min_angles(x) for x in grids])
    print(f"{name + ' ' + te:<16}{len(collar['blocks']):>8}{len(grids):>9}"
          f"{sum(x['cells'] for x in q):>8}{min(x['min_angle'] for x in q):>9.1f}"
          f"{max(x['max_angle'] for x in q):>9.1f}"
          f"{max(x['max_aspect'] for x in q):>8.0f}"
          f"{str(all(x['all_positive'] for x in q)):>7}{y1:>10.1e}"
          f"{100 * (ang < 45).sum() / ang.size:>9.2f}%{np.median(ang):>9.1f}")

    # Limits from the geometry, not typed in: a cambered section does not sit
    # where a symmetric one does, and the trailing edge is wherever the corners
    # say it is.
    outer = collar["outer"]
    pad = 0.06 * float(np.ptp(outer[:, 0]))
    full = ((outer[:, 0].min() - pad, outer[:, 0].max() + pad),
            (outer[:, 1].min() - pad, outer[:, 1].max() + pad))
    tip = wall[region.boundaries[1].corners.index].mean(axis=0)
    half = 2.2 * DELTA
    close = ((tip[0] - 1.6 * half, tip[0] + 1.0 * half),
             (tip[1] - half, tip[1] + half))

    FILL = {"band": "#BFD8D2", "wedge": "#F2C14E"}
    for col, (what, xlim, ylim) in enumerate((
            ("blocking", full[0], full[1]),
            ("mesh", full[0], full[1]),
            ("mesh at the trailing edge", close[0], close[1]))):
        ax = axes[row, col]
        if what == "blocking":
            for sides, kind in zip(collar["blocks"], collar["kinds"]):
                o = np.vstack([sides[0]] + [s[1:] for s in sides[1:]])
                ax.fill(o[:, 0], o[:, 1], color=FILL[kind], ec="#204B57",
                        lw=1.0, zorder=3)
            for r in collar["rays"]:
                ax.plot([wall[r["ci"]][0], r["hit"][0]],
                        [wall[r["ci"]][1], r["hit"][1]],
                        color="#A62463", lw=1.8, zorder=5)
        else:
            for g in grids:
                ax.plot(g[:, :, 0], g[:, :, 1], color="#2B6A72", lw=0.35,
                        zorder=3)
                ax.plot(g[:, :, 0].T, g[:, :, 1].T, color="#2B6A72", lw=0.35,
                        zorder=3)
        ax.fill(*np.vstack([wall, wall[:1]]).T, color="#14181C", zorder=6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        n_c = region.boundaries[1].corner_elements
        ax.set_title(
            f"NACA {name}, {te} TE — {what}\n"
            f"{len(collar['blocks'])} blocks "
            f"({collar['kinds'].count('band')} band, "
            f"{collar['kinds'].count('wedge')} wedge) from "
            f"{len(region.boundaries[1].corners)} corner(s), n_c = "
            f"{n_c.tolist()}\n"
            f"{sum(x['cells'] for x in q)} cells · angles "
            f"{min(x['min_angle'] for x in q):.0f}-"
            f"{max(x['max_angle'] for x in q):.0f}° · "
            f"y1 = {y1:.1e} · all Jacobians positive", fontsize=8.5,
            linespacing=1.5)

fig.suptitle(
    "AeroMesh — zone 1, the near-wall collar, built rather than decomposed\n"
    "band (pale) · trailing-edge wedge (amber) · corner rays (magenta)   ·   "
    "the wake structure falls out of the corner's element count, not from a "
    "template", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "18_collar.png", dpi=125)
print(f"\nSaved {OUT / '18_collar.png'}")

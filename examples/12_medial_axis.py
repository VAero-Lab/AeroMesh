"""Example 12 — both medial axes, with their fields and the retract certificate (S1).

The exterior axis (teal) supplies everything between and around the bodies: the
ring against each wall, the flares into the corners of the truncation curve, the
branches through the gaps of a multi-element configuration.

The interior axis of each body (magenta) supplies that body's own shape. It is
scale-free, which is what makes it survive a far field at fifty chords, where the
exterior axis is featureless.

Vertices are typed by Rigby's rules, all of it derived: NORMAL where three or more
distinct boundary touches meet, CORNER where a flare reaches a C0 corner at
r_m = 0, DANGLE at curvature contact, FINITE_CONTACT where the touching circle
meets the boundary along an arc.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import aeromesh as am
from aeromesh.geometry.region import prepare_boundary

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

KIND_COLOUR = {"NORMAL": "#0C6F79", "CORNER": "#A62463",
               "DANGLE": "#8C5A0F", "FINITE_CONTACT": "#3B6F3A"}


def ellipse(a, b, cx=0.0, cy=0.0, n=600, name="ellipse"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return am.Loop.from_points(np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)]), name)


main = am.load_airfoil("2412", n_points=400).transform(name="main")
flap = main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02, dy=-0.10, name="flap")
slat = main.transform(scale=0.22, angle=np.deg2rad(22), dx=-0.20, dy=0.02, name="slat")
fus = ellipse(0.55, 0.85, 0.45, -1.55, name="fuselage")

CASES = [
    ("1 body, circle far field", [main], lambda b: am.circle_farfield(b, 4.0)),
    ("1 body, C-shape", [main], lambda b: am.c_farfield(b, 4.0, 6.0)),
    ("1 body, box", [main], lambda b: am.box_farfield(b, 4.0, 6.0, 4.0)),
    ("main + flap, circle", [main, flap], lambda b: am.circle_farfield(b, 4.0)),
    ("slat + main + flap, box", [main, flap, slat], lambda b: am.box_farfield(b, 4., 6., 4.)),
    ("airfoil + fuselage, circle", [main, fus], lambda b: am.circle_farfield(b, 4.0)),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
print(f"{'configuration':<28}{'V':>4}{'E':>4}{'b1':>4}{'h':>3}{'cert':>7}   "
      f"{'theta_m range':>18}   vertices")
print("-" * 118)

for ax, (name, bodies, make_outer) in zip(axes.ravel(), CASES):
    region = am.build_region(bodies, outer=make_outer(bodies))
    ext = am.exterior_axis(region)
    cert = ext.certificate()
    lo, hi = ext.theta_range()
    print(f"{name:<28}{ext.n_vertices:>4}{ext.n_edges:>4}{cert['betti_1']:>4}"
          f"{cert['n_bodies']:>3}{'OK' if cert['ok'] else 'FAIL':>7}   "
          f"{np.degrees(lo):7.1f} .. {np.degrees(hi):5.1f}   {ext.kinds()}")

    o = region.outer.loop.closed_points
    ax.plot(o[:, 0], o[:, 1], color="#5D6771", lw=1.0)
    for body in region.holes:
        p = body.loop.closed_points
        ax.fill(p[:, 0], p[:, 1], color="#DCE2E6", ec="#14181C", lw=0.9, zorder=3)

    for _, _, d in ext.graph.edges(data=True):
        ax.plot(d["polyline"][:, 0], d["polyline"][:, 1], color="#0C6F79", lw=1.6, zorder=4)
    for _, d in ext.graph.nodes(data=True):
        ax.plot(*d["pos"], "o", ms=7, color=KIND_COLOUR[d["kind"].name], zorder=6)

    # each body's own interior axis, at the same scale
    for body in bodies:
        b = prepare_boundary(body, is_hole=True, te="blunt")
        ia = am.interior_axis(b.loop, corners=b.corners.positions(b.loop),
                              corner_idx=b.corners.index, corner_angles=b.fluid_angles)
        for _, _, d in ia.graph.edges(data=True):
            ax.plot(d["polyline"][:, 0], d["polyline"][:, 1],
                    color="#A62463", lw=1.8, zorder=5)

    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}\n$b_1$={cert['betti_1']}  bodies={cert['n_bodies']}  "
                 f"{'certified' if cert['ok'] else 'FAILS'}   "
                 f"{ext.edge_kinds()}", fontsize=10)

fig.suptitle("AeroMesh S1 — exterior axis (teal), interior axes (magenta); "
             "vertices: NORMAL teal · CORNER magenta · DANGLE amber", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "12_medial_axis.png", dpi=130)
print(f"\nSaved {OUT / '12_medial_axis.png'}")

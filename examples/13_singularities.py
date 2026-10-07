"""Example 13 — the singularity field and its certificates (S2).

Singularities are solved, not looked up. Fogg's flux balance on the medial angle
decides where the cross field must break and by how much:

    class 1  theta_m crossing pi/4 or 3pi/4 on a medial edge   k = -|dn|
    class 2  flux balance at a medial vertex                   k = sum(2 - n_j) - 4
    class 3  a concave corner's reference direction switching  k = -1 per switch
             finite contact over an arc                        k = -floor(extent/(pi/2))

Two certificates then run before anything is built:

    retract   b_1 of the medial graph must equal the number of bodies
    budget    sum k = sum_corners (2 - n_c) - 4*chi,  chi = 1 - h

The budget comes from discrete Gauss-Bonnet for quad meshes, independently of
the flux balance, and is what turns "this looks reasonable" into "this is
admissible".
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import aeromesh as am

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)


def ellipse(a, b, cx=0.0, cy=0.0, n=600, name="ellipse"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return am.Loop.from_points(np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)]), name)


main = am.load_airfoil("2412", n_points=400).transform(name="main")
flap = main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02, dy=-0.10, name="flap")
slat = main.transform(scale=0.22, angle=np.deg2rad(22), dx=-0.20, dy=0.02, name="slat")
fus = ellipse(0.55, 0.85, 0.45, -1.55, name="fuselage")

CASES = [
    ("bluff body, circle", [fus], lambda b: am.circle_farfield(b, 4.0)),
    ("airfoil, circle", [main], lambda b: am.circle_farfield(b, 4.0)),
    ("airfoil, box", [main], lambda b: am.box_farfield(b, 4.0, 6.0, 4.0)),
    ("airfoil, C-shape", [main], lambda b: am.c_farfield(b, 4.0, 6.0)),
    ("main + flap, circle", [main, flap], lambda b: am.circle_farfield(b, 4.0)),
    ("slat + main + flap, circle", [main, flap, slat],
     lambda b: am.circle_farfield(b, 4.0)),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
hdr = f"{'configuration':<28}{'h':>3}{'b1':>4}{'chi':>5}{'Σk':>5}{'req':>5}{'res':>5}  certificates"
print(hdr)
print("-" * len(hdr))

for ax, (name, bodies, make_outer) in zip(axes.ravel(), CASES):
    region = am.build_region(bodies, outer=make_outer(bodies))
    axis = am.exterior_axis(region)
    field = am.solve_singularities(axis, region)
    retract, budget = axis.certificate(), field.budget()

    marks = ("retract " + ("OK" if retract["ok"] else "FAIL")
             + " | budget " + ("OK" if budget["ok"] else f"{budget['residual']:+d}"))
    print(f"{name:<28}{region.n_bodies:>3}{retract['betti_1']:>4}{budget['chi']:>5}"
          f"{budget['k_total']:>+5d}{budget['k_required']:>+5d}{budget['residual']:>+5d}"
          f"  {marks}")

    o = region.outer.loop.closed_points
    ax.plot(o[:, 0], o[:, 1], color="#5D6771", lw=1.0)
    for body in region.holes:
        p = body.loop.closed_points
        ax.fill(p[:, 0], p[:, 1], color="#DCE2E6", ec="#14181C", lw=0.9, zorder=3)
    for _, _, d in axis.graph.edges(data=True):
        ax.plot(d["polyline"][:, 0], d["polyline"][:, 1], color="#B9C4CB", lw=1.2, zorder=2)
    for c in field.corners:
        ax.plot(*c.position, "s", ms=5, color="#8C5A0F", zorder=5)
    for s in field.singularities:
        ax.plot(*s.position, "o", ms=9 + 2 * abs(s.k),
                color="#A62463" if s.k < 0 else "#0C6F79", zorder=6)
        ax.annotate(f"{s.k:+d}", s.position, textcoords="offset points",
                    xytext=(9, 6), fontsize=10, weight="bold",
                    color="#A62463" if s.k < 0 else "#0C6F79")

    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}\n$b_1$={retract['betti_1']}=h  $\\chi$={budget['chi']}  "
                 f"$\\Sigma k$={budget['k_total']:+d} / {budget['k_required']:+d} "
                 + ("  certified" if budget["ok"]
                    else f"  residual {budget['residual']:+d}"),
                 fontsize=10)

fig.suptitle("AeroMesh S2 — solved singularities (magenta k<0, teal k>0), "
             "corners (amber), medial axis (grey)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "13_singularities.png", dpi=130)
print(f"\nSaved {OUT / '13_singularities.png'}")

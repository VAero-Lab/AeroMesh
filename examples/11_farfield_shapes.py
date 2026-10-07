"""Example 11 — the far field is a supplied curve, and the shape matters (S0).

Where the domain is truncated is a modelling choice, so the standard shapes are
offered as standard shapes. What the choice is *not* is a topology selection:
the curve is passed as a curve and stored as a curve, and no later stage can
tell which constructor made it.

What it does influence is how many corners the fluid boundary has — and a corner
generates a medial flare, and a flare generates a block. That is the whole
reason the choice is real:

    circle   0 corners  ->  0 flares   (O-type outer boundary)
    C-shape  2 corners  ->  2 flares   (C-type)
    box      4 corners  ->  4 flares   (H-type / tunnel walls)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import aeromesh as am

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

R, WAKE = 6.0, 10.0          # modest, so the section stays visible in the plot

main = am.load_airfoil("2412", n_points=400).transform(name="main")
flap = main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02, dy=-0.10, name="flap")

SHAPES = [
    ("circle — O-type\n(the default)", lambda b: am.circle_farfield(b, R)),
    ("C-shape — C-type", lambda b: am.c_farfield(b, R, WAKE)),
    ("box — H-type / tunnel", lambda b: am.box_farfield(b, R, WAKE, R)),
    ("distance level set\n(tight or spread bodies)", lambda b: am.offset_farfield(b, R)),
]
GROUPS = [("single airfoil", [main]), ("main + flap", [main, flap])]

fig, axes = plt.subplots(2, 4, figsize=(21, 10))
print(f"{'bodies':<14}{'far field':<26}{'outer corners':>14}   fluid angles")
print("-" * 76)

for row, (gname, bodies) in enumerate(GROUPS):
    for col, (label, make) in enumerate(SHAPES):
        ax = axes[row, col]
        region = am.build_region(bodies, outer=make(bodies))
        ang = np.degrees(region.outer.fluid_angles).round(1).tolist()
        print(f"{gname:<14}{label.splitlines()[0]:<26}"
              f"{len(region.outer.corners):>14}   {ang}")

        o = region.outer.loop.closed_points
        ax.plot(o[:, 0], o[:, 1], color="#5D6771", lw=1.2)
        for body in region.holes:
            p = body.loop.closed_points
            ax.fill(p[:, 0], p[:, 1], color="#DCE2E6", ec="#14181C", lw=1.0)
        if len(region.outer.corners):
            cp = region.outer.corners.positions(region.outer.loop)
            ax.plot(cp[:, 0], cp[:, 1], "o", ms=8, color="#A62463", zorder=5)

        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label}\n{len(region.outer.corners)} outer corners "
                     f"→ {len(region.outer.corners)} flares", fontsize=10)
        if col == 0:
            ax.set_ylabel(gname, fontsize=12)

fig.suptitle("AeroMesh — the far field is a curve you supply; its corners drive "
             "topology, its name does not", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "11_farfield_shapes.png", dpi=130)
print(f"\nSaved {OUT / '11_farfield_shapes.png'}")

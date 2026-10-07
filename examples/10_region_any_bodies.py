"""Example 10 — one code path, any set of bodies (S0).

Builds a validated Region for seven configurations: a single airfoil, a
two-element and a three-element high-lift arrangement, an airfoil beside a
fuselage cross-section, a bluff body alone, a circle, and an airfoil inside a
prescribed box.

No configuration is named anywhere in the library. There is no domain type to
pick and no body count to declare -- the input is a set of closed loops, and
that is all the pipeline is told.

The outer boundaries below are the conventional CFD truncation curves, built
with ``circle_farfield``, ``c_farfield`` and ``box_farfield``. Choosing one is a
modelling decision, like choosing how far out to truncate; it is passed as a
curve and stored as a curve, and no later stage can tell which constructor made
it. Their corner counts differ -- 0, 2 and 4 -- and that is what will drive the
different blockings, once the medial engine exists to derive them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import aeromesh as am

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)


def ellipse(a, b, cx=0.0, cy=0.0, n=500, name="ellipse"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return am.Loop.from_points(
        np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)]), name)


main = am.load_airfoil("2412", n_points=400).transform(name="main")
flap = main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02, dy=-0.10, name="flap")
slat = main.transform(scale=0.22, angle=np.deg2rad(22), dx=-0.20, dy=0.02, name="slat")
fuselage = ellipse(0.55, 0.85, 0.45, -1.55, name="fuselage")
circle = ellipse(0.5, 0.5, name="circle")

# The far field is a supplied curve, built with the conventional constructors.
# Which one is a modelling choice; it is never a topology selection.
CASES = [
    ("single airfoil\ncircle (O-type)", [main],
     lambda b: am.circle_farfield(b, 4.0)),
    ("main + flap\nC-shape (C-type)", [main, flap],
     lambda b: am.c_farfield(b, 4.0, 6.0)),
    ("slat + main + flap\ncircle", [main, flap, slat],
     lambda b: am.circle_farfield(b, 4.0)),
    ("airfoil + fuselage\ncircle", [main, fuselage],
     lambda b: am.circle_farfield(b, 4.0)),
    ("fuselage alone\ncircle", [fuselage],
     lambda b: am.circle_farfield(b, 4.0)),
    ("circle body\nC-shape", [circle],
     lambda b: am.c_farfield(b, 4.0, 6.0)),
    ("airfoil in a box\n(H-type / tunnel)", [main],
     lambda b: am.box_farfield(b, 4.0, 6.0, 4.0)),
]

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
print(f"{'configuration / far field':<34}{'bodies':>7}{'chi':>5}"
      f"{'points':>8}{'corners':>9}   detail")
print("-" * 100)

for ax, (name, bodies, make_outer) in zip(axes.ravel(), CASES):
    region = am.build_region(bodies, outer=make_outer(bodies),
                             te="blunt", te_thickness=0.002)
    pts, _, _ = region.sample()
    detail = "  ".join(
        f"{'outer' if not b.is_hole else b.loop.name}:{len(b.corners)}"
        for b in region.boundaries)
    label = name.replace("\n", " / ")
    print(f"{label:<34}{region.n_bodies:>7}{region.euler_characteristic:>5}"
          f"{len(pts):>8}{sum(len(b.corners) for b in region.boundaries):>9}   {detail}")

    o = region.outer.loop.closed_points
    ax.plot(o[:, 0], o[:, 1], color="#5D6771", lw=1.0)
    for body in region.holes:
        p = body.loop.closed_points
        ax.fill(p[:, 0], p[:, 1], color="#DCE2E6", ec="#14181C", lw=1.0)
        if len(body.corners):
            cp = body.corners.positions(body.loop)
            ax.plot(cp[:, 0], cp[:, 1], "o", ms=5, color="#A62463", zorder=5)
    if len(region.outer.corners):
        cp = region.outer.corners.positions(region.outer.loop)
        ax.plot(cp[:, 0], cp[:, 1], "o", ms=5, color="#A62463", zorder=5)
    hp = region.hole_points()
    ax.plot(hp[:, 0], hp[:, 1], "+", ms=9, color="#0C6F79", mew=1.6, zorder=6)

    ax.set_aspect("equal")
    ax.set_title(f"{name}\nbodies={region.n_bodies}  "
                 f"$\\chi$={region.euler_characteristic}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

axes.ravel()[-1].axis("off")
axes.ravel()[-1].text(
    0.5, 0.5,
    "magenta  detected C0 corners\n"
    "teal +    hole points\n"
    "          (largest inscribed circle)\n\n"
    "One call builds all seven.\n"
    "No domain type, no body count.\n"
    "The far field is a curve you\n"
    "supply, not a letter you pick.",
    ha="center", va="center", fontsize=12, family="monospace")

fig.suptitle("AeroMesh S0 — the input is a set of closed loops", fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "10_region_any_bodies.png", dpi=130)
print(f"\nSaved {OUT / '10_region_any_bodies.png'}")

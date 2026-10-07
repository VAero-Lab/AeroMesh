"""Example 14 — the block decomposition (S3).

Blocks come straight off the medial graph, so no planar arrangement has to be
computed:

    a span between two cuts on a medial edge  -> a four-sided block
    a medial vertex                           -> an m-sided region
    a flare ending at a corner                -> a three-sided wedge
    a medial loop with no vertices at all     -> one ring, wrapping on itself

Every side is a curve of the medial coordinate system or a boundary arc, and the
cuts meet the wall orthogonally by construction. The decisive check is printed
with each case: the blocks' areas must sum to the fluid area.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import aeromesh as am

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

FILL = {"span": "#CDE3E6", "vertex": "#F2D8E4", "corner": "#F6E6CC", "ring": "#D8E8D4"}
EDGE = {"span": "#0C6F79", "vertex": "#A62463", "corner": "#8C5A0F", "ring": "#3B6F3A"}


def ellipse(a, b, cx=0.0, cy=0.0, n=600, name="ellipse"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return am.Loop.from_points(np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)]), name)


main = am.load_airfoil("2412", n_points=400).transform(name="main")
flap = main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02, dy=-0.10, name="flap")
slat = main.transform(scale=0.22, angle=np.deg2rad(22), dx=-0.20, dy=0.02, name="slat")
fus = ellipse(0.55, 0.85, 0.45, -1.55, name="fuselage")

CASES = [
    ("1 body, smooth far field", [main], lambda b: am.circle_farfield(b, 4.0), 1),
    ("1 body, C-shape", [main], lambda b: am.c_farfield(b, 4.0, 6.0), 1),
    ("1 body, box", [main], lambda b: am.box_farfield(b, 4.0, 6.0, 4.0), 1),
    ("2 bodies, box", [main, flap], lambda b: am.box_farfield(b, 4.0, 6.0, 4.0), 1),
    ("3 bodies, box", [main, flap, slat], lambda b: am.box_farfield(b, 4.0, 6.0, 4.0), 1),
    ("1 body, box, refined", [main], lambda b: am.box_farfield(b, 4.0, 6.0, 4.0), 4),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
print(f"{'configuration':<26}{'blocks':>7}  {'composition':<40}{'area error':>12}  tiles")
print("-" * 98)

for ax, (name, bodies, make_outer, splits) in zip(axes.ravel(), CASES):
    region = am.build_region(bodies, outer=make_outer(bodies))
    system = am.decompose(am.exterior_axis(region), region, splits_per_edge=splits)
    v = system.validate()
    print(f"{name:<26}{system.n_blocks:>7}  {str(system.kinds()):<40}"
          f"{v['relative_error']:>12.2e}  {'yes' if v['tiles'] else 'NO'}")

    for b in system.blocks:
        if b.kind == "ring":
            o, i = b.sides[0], b.sides[1]
            ax.fill(np.r_[o[:, 0], o[:1, 0], i[::-1, 0], i[-1:, 0]],
                    np.r_[o[:, 1], o[:1, 1], i[::-1, 1], i[-1:, 1]],
                    color=FILL[b.kind], ec=EDGE[b.kind], lw=1.2, zorder=2)
            continue
        p = b.outline()
        ax.fill(p[:, 0], p[:, 1], color=FILL[b.kind], ec=EDGE[b.kind], lw=1.2, zorder=2)

    for body in region.holes:
        q = body.loop.closed_points
        ax.fill(q[:, 0], q[:, 1], color="#14181C", zorder=4)

    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    comp = " ".join(f"{k[:4]}×{n}" for k, n in sorted(system.kinds().items()))
    ax.set_title(f"{name}\n{system.n_blocks} blocks   {comp}\n"
                 f"area error {v['relative_error']:.1e}", fontsize=10)

fig.suptitle("AeroMesh S3 — blocks from the medial graph: "
             "span (teal) · vertex (magenta) · corner wedge (amber) · ring (green)",
             fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "14_blocks.png", dpi=130)
print(f"\nSaved {OUT / '14_blocks.png'}")

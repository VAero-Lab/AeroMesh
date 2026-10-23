"""Example 14 — the block decomposition (S3).

Blocks come straight off the medial graph, so no planar arrangement has to be
computed:

    a span between two cuts on a medial edge  -> a four-sided block
    a medial vertex                           -> an m-sided region
    a flare ending at a corner                -> a three-sided wedge
    a medial loop with no vertices at all     -> one ring, wrapping on itself

Where the cuts go is decided by the ranked search: on each medial edge, one cut
per gap between consecutive singularities, taking the best-ranked candidate in
the gap (n = 0 before n = 1, and within a class the medial angle nearer the
ideal). The effect is that each block holds at most one singularity, with no cut
spent anywhere it is not needed.

A flare carrying no singularity is left uncut altogether. Along a flare theta_m
is exactly pi - alpha at every sample and r_m is exactly linear in arc length, so
every candidate ranks the same and the wedge a cut leaves is a scaled copy of
itself wherever it goes -- no criterion can choose. Fogg marks such a corner as
resolved, so the wedge simply merges into the neighbouring vertex region.

Blocks also carry connectivity: every side is either on the region boundary or
shared with exactly one other block, and the two traverse it in opposite
directions. That is what lets S4 match point counts across an interface.

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
    ("1 body, smooth far field", [main], lambda b: am.circle_farfield(b, 4.0)),
    ("1 body, C-shape", [main], lambda b: am.c_farfield(b, 4.0, 6.0)),
    ("1 body, box", [main], lambda b: am.box_farfield(b, 4.0, 6.0, 4.0)),
    ("2 bodies, box", [main, flap], lambda b: am.box_farfield(b, 4.0, 6.0, 4.0)),
    ("3 bodies, box", [main, flap, slat], lambda b: am.box_farfield(b, 4.0, 6.0, 4.0)),
    ("airfoil + fuselage, box", [main, fus], lambda b: am.box_farfield(b, 4.0, 6.0, 4.0)),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
print(f"{'configuration':<26}{'even':>6}{'ranked':>8}{'sing':>6}{'iface':>7}"
      f"  {'composition':<34}{'area err':>10}  ok")
print("-" * 108)

for ax, (name, bodies, make_outer) in zip(axes.ravel(), CASES):
    region = am.build_region(bodies, outer=make_outer(bodies))
    axis = am.exterior_axis(region)
    field = am.solve_singularities(axis, region)
    even = am.decompose(axis, region)
    system = am.decompose(axis, region, field=field)
    v = system.validate()
    c = system.interface_report()
    print(f"{name:<26}{even.n_blocks:>6}{system.n_blocks:>8}"
          f"{len(field.singularities):>6}{c['n_interfaces']:>7}"
          f"  {str(system.kinds()):<34}{v['relative_error']:>10.1e}"
          f"  {'yes' if v['tiles'] and c['conformal'] else 'NO'}")

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
    for sg in field.singularities:
        ax.plot(*sg.position, "o", ms=8, zorder=6,
                color="#A62463" if sg.k < 0 else "#0C6F79",
                markeredgecolor="white", markeredgewidth=1.0)

    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    comp = " ".join(f"{k[:4]}×{n}" for k, n in sorted(system.kinds().items()))
    ax.set_title(f"{name}\n{system.n_blocks} blocks (even spacing: {even.n_blocks})"
                 f"   {comp}\n{len(field.singularities)} singularities, each alone"
                 f" in its block · {c['n_interfaces']} interfaces, conformal",
                 fontsize=9)

fig.suptitle("AeroMesh S3 — blocks from the medial graph; cuts placed by the ranked "
             "search, flares left uncut\nspan (teal) · vertex (magenta) · "
             "ring (green) · singularities (dots)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "14_blocks.png", dpi=130)
print(f"\nSaved {OUT / '14_blocks.png'}")

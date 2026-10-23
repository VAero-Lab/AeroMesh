"""Example 15 — the near-wall layer and quadrilateralisation (S3 items C and D).

Two stages sit between a block decomposition and a mesh, and each one is a
*split of the blocking* rather than a new problem.

**The near-wall layer.** Without it a span runs wall to wall through the medial
axis, so for a body in a far field one block reaches from the surface to the
truncation boundary -- an aspect ratio no boundary layer survives. The layer is
a band at distance

    delta(s) = min( delta_target , rho_max * r_m(s) )

from the wall: the flow sets the thickness and the medial radius caps it to the
space available. Constant rho alone is the medial coordinate working correctly
and the wrong law -- near-uniform along a lone body, and collapsing by 37x in a
flap slot, because a boundary layer's thickness is not a property of how much
room there happens to be.

Inserting it needs no new topology. A cut is a medial radius and a medial radius
meets the wall at a right angle; the layer's outer edge is parallel to the wall.
So the two curves cross at the point reached by travelling delta along the cut
from its wall foot -- an arc-length step, not a search, and with no tolerance
anywhere in it. Four blocks meet at each crossing, so the node is regular and
the index budget is untouched.

**Quadrilateralisation.** Transfinite interpolation needs four sides and the
decomposition does not produce four-sided blocks. Midpoint subdivision is the
bridge, and it is the one prescribed construction this method allows: an m-gon
with corners V_i, side midpoints M_i and a centre C becomes the m quadrilaterals
(V_i, M_i, C, M_{i-1}).

It has to be applied to every block or none -- subdivision splits each side at
its midpoint, so a quad left whole beside a subdivided neighbour carries a node
in the middle of a side the neighbour runs straight past. And the spokes to C
cannot be straight lines: measured, 5 of 8 blocks fold at any fixed choice of
centre, including a four-sided span. They are built instead by a harmonic map
onto a disc, where Tutte's theorem guarantees injectivity, so the images are
non-crossing curves meeting at one interior point.

Printed with each case is what actually matters: that every stage is still a
sound decomposition, and that the index certificate has not moved.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import aeromesh as am
from aeromesh.blocking.layers import medial_radius_on_boundary, wall_distance

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)


def ellipse(a, b, n=400, name="body"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return am.Loop.from_points(
        np.column_stack([a * np.cos(t), b * np.sin(t)]), name=name)


main = am.load_airfoil("2412", n_points=400).transform(name="main")
flap = main.transform(scale=0.30, angle=np.deg2rad(-25), dx=1.05, dy=-0.08,
                      name="flap")
fuse = ellipse(1.6, 0.55, name="fuselage")

CASES = [
    ("aerofoil, box far field", [main], am.box_farfield([main], 4., 6., 4.)),
    ("aerofoil, circular far field", [main], am.circle_farfield([main], 8.)),
    ("two elements, box", [main, flap],
     am.box_farfield([main, flap], 4., 6., 4.)),
    ("bluff body, circular", [fuse], am.circle_farfield([fuse], 8.)),
]

DELTA, RHO_MAX = 0.02, 0.05

fig, axes = plt.subplots(3, len(CASES), figsize=(4.6 * len(CASES), 13.2))

print(f"{'case':<30}{'blocks':>8}{'+layer':>8}{'+quads':>8}"
      f"{'cert':>6}{'sound':>7}{'all quad':>10}")
print("-" * 77)

for col, (name, bodies, outer) in enumerate(CASES):
    region = am.build_region(bodies, outer=outer)
    axis = am.exterior_axis(region)
    field = am.solve_singularities(axis, region)

    blocked = am.decompose(axis, region, field=field)
    layered = am.insert_layer(blocked, delta_target=DELTA, rho_max=RHO_MAX)
    quads = am.to_quads(layered)

    all_quad = {b.n_sides for b in quads.blocks if not b.wraps} <= {4}
    sound = all(s.is_sound() for s in (blocked, layered, quads))
    print(f"{name:<30}{blocked.n_blocks:>8}{layered.n_blocks:>8}"
          f"{quads.n_blocks:>8}{quads.index_certificate()['residual']:>6}"
          f"{str(sound):>7}{str(all_quad):>10}")

    # how much the medial radius varies, and how much the capped layer does
    r_m = medial_radius_on_boundary(axis, region)
    delta = wall_distance(axis, region, delta_target=DELTA, rho_max=RHO_MAX)
    for li in sorted(delta):
        rm, dd = r_m[li], delta[li]
        print(f"{'':<30}  body {li}: r_m {rm.min():.3f}-{rm.max():.3f} "
              f"({rm.max() / rm.min():.0f}x)   delta "
              f"{dd.min():.4f}-{dd.max():.4f} ({dd.max() / dd.min():.1f}x)")

    for row, (system, label) in enumerate(
            ((blocked, "blocks"), (layered, "+ near-wall layer"),
             (quads, "+ quadrilateralised"))):
        ax = axes[row, col]
        for b in system.blocks:
            if b.wraps:
                ax.plot(*b.sides[0].T, color="#0C6F79", lw=1.0, zorder=3)
                ax.plot(*b.sides[1].T, color="#0C6F79", lw=1.0, zorder=3)
                continue
            p = b.outline()
            fill = {"layer": "#F2C14E", "quad": "#BFD8D2",
                    "ring": "#8FBF9F"}.get(b.kind, "#D9E5E8")
            ax.fill(p[:, 0], p[:, 1], color=fill, ec="#204B57", lw=0.6, zorder=2)
        for body in region.holes:
            q = body.loop.closed_points
            ax.fill(q[:, 0], q[:, 1], color="#14181C", zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        q = system.quality()
        extra = ("" if np.isnan(q["min_angle"]) else
                 f"   angles {q['min_angle']:.0f}-{q['max_angle']:.0f}°")
        ax.set_title(f"{name}\n{label}: {system.n_blocks} blocks, "
                     f"m = {q['m_gons']}{extra}", fontsize=8)
        # zoom the layer row onto the body, where the band is
        if row and region.holes:
            lo = np.min([h.loop.bbox[0] for h in region.holes], axis=0)
            hi = np.max([h.loop.bbox[1] for h in region.holes], axis=0)
            pad = 0.35 * float(np.max(hi - lo))
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)

fig.suptitle(
    "AeroMesh S3 items C and D — the near-wall layer inserted by splitting the "
    "blocking, then midpoint subdivision\n"
    f"delta = min({DELTA}, {RHO_MAX}·r_m)   ·   layer (amber) · quad (pale) · "
    "ring (green)   ·   rows 2-3 zoomed on the bodies", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "15_layer_and_quads.png", dpi=130)
print(f"\nSaved {OUT / '15_layer_and_quads.png'}")

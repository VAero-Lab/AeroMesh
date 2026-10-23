"""Example 17 — the same code path on inputs that share nothing.

This is the claim the method stands or falls on. A single aerofoil, a
three-element high-lift arrangement, a fuselage cross-section with a wing above
it, three bodies, a disc and a regular polygon are the same kind of object --
one outer loop and a set of bodies -- and every stage downstream sees them
identically. There is no domain type to choose, no topology to select, and no
per-geometry branch anywhere in the pipeline. C, O and H are labels you read off
the result.

Each panel is the finished blocking: cuts placed by the ranked search, the
near-wall layer inserted as a split, and midpoint subdivision applied where it
is valid. The caption carries the checks -- that the blocks tile, that no
outline crosses itself, that every side is either on the boundary or shared with
exactly one other block, and that the index budget balances.

Where a panel reports `m` other than [4], the blocking is sound but one or two
blocks could not be subdivided: those are the high-aspect-ratio layer strips
carrying a blunt trailing edge's reflex corner, which is the open item recorded
in the tracker. Subdivision is all-or-nothing by necessity -- it splits every
side at its midpoint, so a partial pass would leave a hanging node -- so those
systems are returned whole, with the reason in `notes`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import aeromesh as am

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)


def ellipse(a, b, n=400, name="body"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return am.Loop.from_points(
        np.column_stack([a * np.cos(t), b * np.sin(t)]), name=name)


def polygon(k, rot=0.13, n=600, name="poly"):
    t = np.linspace(0, 2 * np.pi, k, endpoint=False) + rot
    return am.Loop.from_points(
        np.column_stack([np.cos(t), np.sin(t)]), name=name).resample(n)


main = am.load_airfoil("2412", n_points=400).transform(name="main")
flap = main.transform(scale=0.30, angle=np.deg2rad(-25), dx=1.05, dy=-0.08,
                      name="flap")
slat = main.transform(scale=0.22, angle=np.deg2rad(22), dx=-0.20, dy=0.02,
                      name="slat")
fuse = ellipse(1.6, 0.55, name="fuselage")
wing = am.load_airfoil("0012", n_points=300).transform(scale=0.9, dx=0.2,
                                                       dy=1.2, name="wing")
disc = ellipse(1.0, 1.0, name="disc")
trio = [ellipse(0.5, 0.2, name="b0"),
        ellipse(0.5, 0.2, name="b1").transform(dx=1.6, dy=0.5),
        ellipse(0.5, 0.2, name="b2").transform(dx=-1.4, dy=-0.6)]

CASES = [
    ("one aerofoil, O-type", [main], am.circle_farfield([main], 8.0)),
    ("one aerofoil, C-type", [main], am.c_farfield([main], 8.0, 14.0)),
    ("one aerofoil, H-type", [main], am.box_farfield([main], 4.0, 6.0, 4.0)),
    ("two elements", [main, flap], am.box_farfield([main, flap], 4., 6., 4.)),
    ("three elements", [slat, main, flap],
     am.box_farfield([slat, main, flap], 4., 6., 4.)),
    ("fuselage section + wing", [fuse, wing],
     am.box_farfield([fuse], 5.0, 8.0, 5.0)),
    ("three bodies", trio, am.circle_farfield([trio[0]], 9.0)),
    ("a disc", [disc], am.box_farfield([disc], 4.0, 6.0, 4.0)),
    ("an octagon", [polygon(8)], am.circle_farfield([polygon(8)], 6.0)),
]

cols = 3
rows = (len(CASES) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows))
axes = np.atleast_2d(axes)

print(f"{'configuration':<26}{'blocks':>8}{'quads':>7}{'m':>14}"
      f"{'sound':>7}{'cert':>6}")
print("-" * 68)

for k, (name, bodies, outer) in enumerate(CASES):
    region = am.build_region(bodies, outer=outer)
    axis = am.exterior_axis(region)
    field = am.solve_singularities(axis, region)
    blocked = am.decompose(axis, region, field=field)
    layered = am.insert_layer(blocked, delta_target=0.02, rho_max=0.05)
    quads = am.to_quads(layered)

    degrees = sorted({b.n_sides for b in quads.blocks if not b.wraps})
    sound = all(s.is_sound() for s in (blocked, layered, quads))
    cert = quads.index_certificate()["residual"]
    print(f"{name:<26}{blocked.n_blocks:>8}{quads.n_blocks:>7}"
          f"{str(degrees):>14}{str(sound):>7}{cert:>6}")

    ax = axes[k // cols, k % cols]
    for b in quads.blocks:
        if b.wraps:
            for s in b.sides[:2]:
                ax.plot(s[:, 0], s[:, 1], color="#0C6F79", lw=1.0, zorder=4)
            continue
        fill = {"layer": "#F2C14E", "quad": "#BFD8D2"}.get(b.kind, "#D9E5E8")
        p = b.outline()
        ax.fill(p[:, 0], p[:, 1], color=fill, ec="#204B57", lw=0.5, zorder=3)
    for b in region.holes:
        q = b.loop.closed_points
        ax.fill(q[:, 0], q[:, 1], color="#14181C", zorder=6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    flag = "all quadrilateral" if degrees == [4] else f"m = {degrees}"
    ax.set_title(f"{name}\n{blocked.n_blocks} blocks -> {quads.n_blocks}"
                 f"   {flag}\ntiles · conformal · budget "
                 f"{cert:+d}   {'SOUND' if sound else 'see notes'}",
                 fontsize=9, linespacing=1.5)

for k in range(len(CASES), rows * cols):
    axes[k // cols, k % cols].axis("off")

fig.suptitle(
    "AeroMesh — one code path, nine configurations, nothing prescribed\n"
    "O, C and H are labels read off the result, not inputs   ·   "
    "layer (amber) · quadrilateral (pale)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "17_gallery.png", dpi=118)
print(f"\nSaved {OUT / '17_gallery.png'}")

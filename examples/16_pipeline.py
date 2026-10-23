"""Example 16 — the whole pipeline, one configuration, six panels.

Read left to right, top to bottom. Nothing in any panel is chosen: the number of
blocks, their connectivity, where the singularities sit, and the topology class
itself are all derived from the geometry. The bodies, where the domain is
truncated, and the layer thickness are supplied.

1. **Region.** One outer boundary and any number of bodies. The far-field curve
   is passed as a curve and stored as a curve; no later stage can tell which
   constructor made it. What it does influence is how many corners the fluid
   boundary has, and a corner generates a medial flare, and a flare generates a
   block.

2. **Medial axis** of the fluid. Primary edges run between branch points; flares
   run out to a convex corner of the fluid where r_m goes to zero. The retract
   certificate is checked here: the graph's cycle count must equal the body
   count.

3. **Singularities**, solved by flux balance on the medial angle rather than
   looked up in a rule table.

4. **Blocks**, built constructively from the medial graph. Every cut is a medial
   radius, so it meets the wall at a right angle by construction.

5. **Near-wall layer**, inserted by splitting the blocking rather than by
   decomposing a new region, so the topology is inherited instead of recomputed.

6. **Quadrilateralised** by midpoint subdivision, with the spokes from a
   transfinite map for a four-sided block and a harmonic map onto a disc
   otherwise.

The numbers under each panel are the checks, not decoration.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import aeromesh as am

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

BODY = "#14181C"
WALL = "#2B3A42"
TEAL = "#0C6F79"
MAGENTA = "#A62463"
AMBER = "#F2C14E"
PALE = "#BFD8D2"
SAND = "#D9E5E8"

body = am.load_airfoil("2412", n_points=400).transform(name="main")
region = am.build_region([body], outer=am.c_farfield([body], 8.0, 14.0))
axis = am.exterior_axis(region)
field = am.solve_singularities(axis, region)
blocked = am.decompose(axis, region, field=field)
layered = am.insert_layer(blocked, delta_target=0.02, rho_max=0.05)
quads = am.to_quads(layered)

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))


def frame(ax, title, zoom=False):
    for b in region.holes:
        q = b.loop.closed_points
        ax.fill(q[:, 0], q[:, 1], color=BODY, zorder=8)
    p = region.outer.loop.closed_points
    ax.plot(p[:, 0], p[:, 1], color=WALL, lw=1.4, zorder=7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, linespacing=1.5)
    if zoom:
        lo = np.min([h.loop.bbox[0] for h in region.holes], axis=0)
        hi = np.max([h.loop.bbox[1] for h in region.holes], axis=0)
        pad = 0.55 * float(np.max(hi - lo))
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)


def draw_blocks(ax, system, colour_by_kind=True):
    for b in system.blocks:
        if b.wraps:
            for s in b.sides[:2]:
                ax.plot(s[:, 0], s[:, 1], color=TEAL, lw=1.0, zorder=5)
            continue
        fill = SAND
        if colour_by_kind:
            fill = {"layer": AMBER, "quad": PALE, "ring": "#8FBF9F"}.get(b.kind, SAND)
        p = b.outline()
        ax.fill(p[:, 0], p[:, 1], color=fill, ec="#204B57", lw=0.55, zorder=3)


# ── 1. the region ───────────────────────────────────────────────────
ax = axes[0, 0]
for b in region.boundaries:
    if len(b.corners):
        pos = b.corners.positions(b.loop)
        ax.plot(pos[:, 0], pos[:, 1], "s", ms=6, color="#8C5A0F", zorder=9)
frame(ax, f"1 · region — {region.n_bodies} body, "
          f"{len(region.outer.corners)} far-field corners, "
          f"{len(region.holes[0].corners)} body corners\n"
          f"chi = {region.euler_characteristic}   "
          f"(corners marked; a C0 corner makes a flare, a flare makes a block)")

# ── 2. the medial axis ──────────────────────────────────────────────
ax = axes[0, 1]
for _, _, d in axis.graph.edges(data=True):
    colour = {"FLARE": MAGENTA, "LOOP": "#8FBF9F"}.get(d["kind"].name, TEAL)
    ax.plot(d["polyline"][:, 0], d["polyline"][:, 1], color=colour, lw=1.5, zorder=4)
for _, d in axis.graph.nodes(data=True):
    if d.get("anchor"):
        continue
    ax.plot(*d["pos"], "o", ms=7, color="white", mec=TEAL, mew=1.6, zorder=6)
cert = axis.certificate()
frame(ax, f"2 · exterior medial axis — primary (teal), flare (magenta)\n"
          f"retract certificate: b1 = {axis.betti_1} = h = {region.n_bodies}  "
          f"{'PASS' if cert['ok'] else 'FAIL'}")

# ── 3. the singularities ────────────────────────────────────────────
ax = axes[0, 2]
for _, _, d in axis.graph.edges(data=True):
    ax.plot(d["polyline"][:, 0], d["polyline"][:, 1], color="#AFC6CC", lw=1.0, zorder=3)
for c in field.corners:
    ax.plot(*c.position, "s", ms=5, color="#8C5A0F", zorder=5)
for s in field.singularities:
    ax.plot(*s.position, "o", ms=8 + 2 * abs(s.k), zorder=6,
            color=MAGENTA if s.k < 0 else TEAL, mec="white", mew=1.0)
    ax.annotate(f"{s.k:+d}", s.position, textcoords="offset points",
                xytext=(9, 6), fontsize=8, weight="bold",
                color=MAGENTA if s.k < 0 else TEAL)
b = field.budget()
frame(ax, f"3 · singularities by flux balance — {len(field.singularities)} found\n"
          f"corner term {b['corner_term']:+d},  chi {b['chi']:+d},  "
          f"required {b['k_required']:+d},  solved {b['k_total']:+d}")

# ── 4, 5, 6. blocks, layer, quads ───────────────────────────────────
for col, (system, label, zoom) in enumerate((
        (blocked, "4 · blocks", False),
        (layered, "5 · + near-wall layer", True),
        (quads, "6 · + quadrilateralised", True))):
    ax = axes[1, col]
    draw_blocks(ax, system)
    cert = system.index_certificate()
    q = system.quality()
    degrees = sorted({x.n_sides for x in system.blocks if not x.wraps})
    frame(ax, f"{label} — {system.n_blocks} blocks, m = {degrees}\n"
              f"tiles · conformal · index budget {cert['residual']:+d}   "
              f"{'SOUND' if system.is_sound() else 'see notes'}\n"
              f"angles {q['min_angle']:.0f}-{q['max_angle']:.0f}deg", zoom=zoom)

fig.suptitle(
    "AeroMesh — the whole pipeline on one configuration (NACA 2412, C-type far "
    "field)\nnothing prescribed: block count, connectivity, singularities and "
    "topology class are all derived   ·   bottom row right two panels zoomed on "
    "the body", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "16_pipeline.png", dpi=125)
print(f"blocked {blocked.n_blocks} -> layered {layered.n_blocks} -> quads {quads.n_blocks}")
print(f"all four checks: {[s.is_sound() for s in (blocked, layered, quads)]}")
print(f"Saved {OUT / '16_pipeline.png'}")

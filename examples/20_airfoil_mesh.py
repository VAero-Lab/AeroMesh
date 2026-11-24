"""Example 20 -- a complete single-airfoil mesh, marched from the wall.

Two sections at the two ends of the camber range and with the two trailing-edge
conventions, through one call each. Nothing is tuned per section: the corners
are detected, their element counts decide how many wake lines leave the body,
and the first cell height comes from Reynolds number and y+.

Writes ``output/20_airfoil_mesh.png`` and prints the metrics.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import aeromesh as am
from aeromesh.mesh import c_mesh, first_cell_height
from aeromesh.quality.metrics import assembled_nonorthogonality, mesh_report

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)
RE = 6e6
CASES = (("0012", "sharp"), ("8412", "blunt"))
INK, LINE, AFT = "#14181C", "#2B6A72", "#A62463"


def draw(ax, grid, color, lw, step=1):
    ax.plot(grid[::step, :, 0].T, grid[::step, :, 1].T, color=color, lw=lw)
    ax.plot(grid[:, ::step, 0], grid[:, ::step, 1], color=color, lw=lw)


if __name__ == "__main__":
    y1 = first_cell_height(RE, 1.0)
    fig, axes = plt.subplots(len(CASES), 4, figsize=(18, 8.2))
    for row, (name, te) in enumerate(CASES):
        body = am.load_airfoil(name, n_points=1000)
        region = am.build_region([body], farfield=20.0, te=te,
                                 body_spacing=5e-4)
        bnd = region.boundaries[1]
        mb = c_mesh(bnd, y1)
        mesh = mb.assemble()
        rep = mesh_report(mb, y1)
        w, bl = rep["wall"], rep["bl"]
        print(f"NACA {name} {te}: {mesh.n_cells} cells, blocks "
              f"{[b.shape for b in mb.blocks]}\n"
              f"  y1 {w['y1_min_ratio']:.4f}-{w['y1_max_ratio']:.4f} of the "
              f"request at every wall node\n"
              f"  boundary layer: skew max {bl['skew_max']:.2f} deg, growth "
              f"max {w['bl_growth_max']:.3f}\n"
              f"  whole mesh: angles {rep['min_angle']:.1f}-"
              f"{rep['max_angle']:.1f} deg, finite-volume non-orthogonality "
              f"max {assembled_nonorthogonality(mesh).max():.1f} deg")
        wall = np.asarray(bnd.loop.points)
        te_x = wall[:, 0].max()
        te_y = wall[np.argmax(wall[:, 0]), 1]
        views = [((-45, 55), (-55, 55), 4, 0.25, "whole domain"),
                 ((-0.4, 1.6), (-0.75, 0.75), 1, 0.3, "section"),
                 ((-0.03, 0.09), (-0.045, 0.045), 1, 0.35, "leading edge"),
                 ((te_x - 0.012, te_x + 0.012), (te_y - 0.009, te_y + 0.009),
                  1, 0.4, "trailing edge")]
        for ax, (xl, yl, step, lw, title) in zip(axes[row], views):
            for b in mb.blocks:
                draw(ax, b.grid, AFT if b.name == "aft" else LINE, lw, step)
            ax.fill(wall[:, 0], wall[:, 1], color=INK, zorder=3)
            ax.set_xlim(xl); ax.set_ylim(yl); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"NACA {name}, {te} trailing edge - {title}",
                         fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "20_airfoil_mesh.png", dpi=110)
    print("wrote", OUT / "20_airfoil_mesh.png")

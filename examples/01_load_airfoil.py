"""Example 01 — Load and plot airfoil boundaries.

Stage 0 deliverable: load NACA 0012 and SD7037, plot both, confirm
they are clean, closed, and properly ordered.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from aeromesh.geometry import load_airfoil
from aeromesh._viz import plot_boundary

# ── Load airfoils ─────────────────────────────────────────────────
naca0012 = load_airfoil("0012", num_points=150)
sd7037 = load_airfoil(
    str(Path(__file__).resolve().parent.parent / "data" / "airfoils" / "sd7037.dat")
)

print(f"NACA 0012: {naca0012.n_points} points, chord = {naca0012.chord}")
print(f"  LE at ({naca0012.le[0]:.4f}, {naca0012.le[1]:.4f})")
print(f"  TE at ({naca0012.te[0]:.4f}, {naca0012.te[1]:.4f})")

print(f"\nSD7037:   {sd7037.n_points} points, chord = {sd7037.chord}")
print(f"  LE at ({sd7037.le[0]:.4f}, {sd7037.le[1]:.4f})")
print(f"  TE at ({sd7037.te[0]:.4f}, {sd7037.te[1]:.4f})")

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

plot_boundary(naca0012, ax=axes[0])
axes[0].set_title(f"{naca0012.name} ({naca0012.n_points} pts)")
axes[0].grid(True, alpha=0.3)
axes[0].plot(*naca0012.le, "go", markersize=8, label="LE")
axes[0].plot(*naca0012.te, "rs", markersize=8, label="TE")
axes[0].legend()

plot_boundary(sd7037, ax=axes[1])
axes[1].set_title(f"{sd7037.name} ({sd7037.n_points} pts)")
axes[1].grid(True, alpha=0.3)
axes[1].plot(*sd7037.le, "go", markersize=8, label="LE")
axes[1].plot(*sd7037.te, "rs", markersize=8, label="TE")
axes[1].legend()

fig.suptitle("AeroMesh — Airfoil Boundaries (Stage 0)", fontsize=14)
fig.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "output"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "01_airfoil_boundaries.png", dpi=150)
print(f"\nFigure saved to {out_dir / '01_airfoil_boundaries.png'}")
plt.show()

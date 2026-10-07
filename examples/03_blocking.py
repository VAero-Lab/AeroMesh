"""Example 03 — Topological Block Decomposition.

Extracts the Medial Graph and uses its generator mappings to explicitly
extract the physical boundaries and interfaces for each structural block.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from aeromesh.geometry import load_airfoil
from aeromesh.domain import build_c_domain
from aeromesh.medial.cdt import triangulate_domain, extract_medial_axis
from aeromesh.topology.classify import classify_singularities
from aeromesh.topology.design import build_design_vector
from aeromesh.blocking.blocks import build_block_system

# ── Load airfoil and build domain ─────────────────────────────────
airfoil = load_airfoil("0012", num_points=150)
domain = build_c_domain(airfoil, farfield=5.0, wake_length=10.0)

# ── Triangulate and extract medial axis ───────────────────────────
tri_result = triangulate_domain(domain)
mg = extract_medial_axis(tri_result, domain)

# ── Topology & Blocking ───────────────────────────────────────────
topology = classify_singularities(mg)
dv = build_design_vector(topology)
system = build_block_system(mg, dv, domain)

# ── Plot Blocks ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

# Draw the underlying domain in faint grey
air = domain.airfoil.points
ax.plot(air[:, 0], air[:, 1], "k-", alpha=0.3)
outer_closed = np.vstack([domain.outer, domain.outer[:1]])
ax.plot(outer_closed[:, 0], outer_closed[:, 1], color="k", linestyle="--", alpha=0.3)

colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f1c40f", "#1abc9c", "#e67e22", "#34495e"]

for block in system.blocks:
    color = colors[block.id % len(colors)]
    
    # Plot South boundary
    ax.plot(block.south[:, 0], block.south[:, 1], color=color, linewidth=2)
    # Plot North boundary
    ax.plot(block.north[:, 0], block.north[:, 1], color=color, linewidth=2)
    
    # Plot West interface
    ax.plot(block.west[:, 0], block.west[:, 1], color=color, linestyle=":")
    # Plot East interface
    ax.plot(block.east[:, 0], block.east[:, 1], color=color, linestyle=":")

ax.set_aspect("equal")
ax.set_title(f"Topological Blocking (NACA 0012 C-Domain) - {len(system.blocks)} Blocks")

out_dir = Path(__file__).resolve().parent.parent / "output"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "03_blocking_test.png", dpi=150)
print("Saved 03_blocking_test.png")

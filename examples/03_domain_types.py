"""Example 03 — Domain topologies (C, O, H).

Demonstrates the construction of all three standard computational domains
around an airfoil.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from aeromesh.geometry import load_airfoil
from aeromesh.domain import build_domain, DomainType
from aeromesh._viz import plot_domain

# Configuration
FARFIELD = 5.0  # Kept small for visualization purposes
WAKE = 10.0

airfoil = load_airfoil("0012", num_points=100)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. C-Domain
domain_c = build_domain(airfoil, domain_type=DomainType.C, farfield=FARFIELD, wake_length=WAKE)
plot_domain(domain_c, ax=axes[0])

# 2. O-Domain
domain_o = build_domain(airfoil, domain_type=DomainType.O, farfield=FARFIELD)
plot_domain(domain_o, ax=axes[1])

# 3. H-Domain
domain_h = build_domain(airfoil, domain_type=DomainType.H, farfield=FARFIELD, wake_length=WAKE)
plot_domain(domain_h, ax=axes[2])

fig.suptitle("AeroMesh Domain Topologies", fontsize=16)
fig.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "output"
out_dir.mkdir(exist_ok=True)
fname_base = "03_domain_types"
fig.savefig(out_dir / f"{fname_base}.png", dpi=150)
fig.savefig(out_dir / f"{fname_base}.pdf")
print(f"Saved: {out_dir / fname_base}.[png, pdf]")

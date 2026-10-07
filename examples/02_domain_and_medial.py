"""Example 02 — Domain construction, CDT, and raw medial axis.

Stage 1 starter deliverable: build a C-domain around each airfoil,
triangulate, extract the raw medial axis, and plot everything.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from aeromesh.geometry import load_airfoil
from aeromesh.domain import build_c_domain, build_o_domain, build_h_domain
from aeromesh.medial.cdt import triangulate_domain, extract_medial_axis
from aeromesh._viz import plot_domain, plot_triangulation, plot_medial_graph

# ── Configuration ─────────────────────────────────────────────────
FARFIELD = 10.0
WAKE = 25.0  # Typical C-domain: wake length >= semicircle diameter (2*FARFIELD)

# ── Load airfoils ─────────────────────────────────────────────────
naca0012 = load_airfoil("0012", num_points=150)
naca2412 = load_airfoil("2412", num_points=150)
naca8412 = load_airfoil("8412", num_points=150)

airfoils = [("NACA 0012", naca0012), ("NACA 2412", naca2412), ("NACA 8412", naca8412)]
domain_types = [
    ("C", lambda a: build_c_domain(a, farfield=FARFIELD, wake_length=WAKE)),
    ("O", lambda a: build_o_domain(a, farfield=FARFIELD)),
    ("H", lambda a: build_h_domain(a, farfield=FARFIELD, wake_length=WAKE))
]

out_dir = Path(__file__).resolve().parent.parent / "output"
out_dir.mkdir(exist_ok=True)

for name, airfoil in airfoils:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    for dtype, builder in domain_types:
        print(f"\n  -- {dtype}-Domain --")
        domain = builder(airfoil)
        print(f"  Vertices: {len(domain.vertices)}, Segments: {len(domain.segments)}")

        # ── Triangulate ───────────────────────────────────────────────
        tri_result = triangulate_domain(domain)
        n_tri = len(tri_result["triangles"])
        print(f"  CDT: {n_tri} triangles, {len(tri_result['vertices'])} vertices")

        # ── Extract medial axis skeleton ──────────────────────────────
        mg = extract_medial_axis(tri_result, domain)
        print(f"  Medial axis: {mg.n_nodes} nodes, {mg.n_edges} edges")

        # ── Plot ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        plot_domain(domain, ax=axes[0])
        axes[0].set_title(f"{name} — {dtype}-Domain")

        plot_triangulation(tri_result, domain, ax=axes[1])
        axes[1].set_title(f"CDT ({n_tri} triangles)")

        plot_medial_graph(mg, domain, ax=axes[2])
        axes[2].set_title(f"Medial Skeleton ({mg.n_nodes} nodes, {mg.n_edges} edges)")

        fig.suptitle(f"AeroMesh — {name} ({dtype}-Domain)", fontsize=14)
        fig.tight_layout()

        fname_base = f"02_{name.lower().replace(' ', '_')}_{dtype.lower()}_medial"
        fig.savefig(out_dir / f"{fname_base}.png", dpi=150)
        print(f"  Saved: {fname_base}.png")
        plt.close(fig)

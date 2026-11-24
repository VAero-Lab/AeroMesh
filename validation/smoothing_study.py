"""Does elliptic smoothing improve the marched C-mesh? Measured.

    python validation/smoothing_study.py

Each variant smooths the C block with every boundary node held (the wall,
the wake cut, the outflow and far field), and is measured by the same code
as everything else. Results: ``results/smoothing.json``.
"""
import json
import sys
from pathlib import Path

import numpy as np

import aeromesh as am
from aeromesh.mesh import c_mesh, first_cell_height
from aeromesh.mesh.elliptic import smooth
from aeromesh.mesh.structured import Block, MultiBlock
from aeromesh.quality.metrics import assembled_nonorthogonality, mesh_report

Y1 = first_cell_height(6e6, 1.0)
VARIANTS = {
    "marched (none)": None,
    "Winslow, whole block": dict(sources=False, iters=300),
    "TTM + grid sources, whole block": dict(sources=True, iters=300),
    "Winslow, TE region": dict(sources=False, iters=300,
                               region=((0.9, 1.15), (-0.12, 0.12))),
    "TTM + sources, TE region": dict(sources=True, iters=300,
                                     region=((0.9, 1.15), (-0.12, 0.12))),
    "Winslow, away from wall (j > 40)": dict(sources=False, iters=300,
                                            hold_rows=41),
}

if __name__ == "__main__":
    out = {}
    for name, te in (("0012", "sharp"), ("0012", "blunt"), ("4412", "blunt")):
        bnd = am.build_region([am.load_airfoil(name, n_points=1000)],
                              farfield=20.0, te=te,
                              body_spacing=5e-4).boundaries[1]
        base = c_mesh(bnd, Y1, radius=500.0, wake_length=500.0, n_layers=140)
        for vname, kw in VARIANTS.items():
            if kw is None:
                mb = base
            else:
                C = base.blocks[0]
                g = smooth(C.grid, **kw)
                mb = MultiBlock([Block(g, C.sides, C.name)] + base.blocks[1:])
            key = f"{name}_{te}_{vname}"
            try:
                m = mb.assemble()
                rep = mesh_report(mb, Y1)
                fv = assembled_nonorthogonality(m)
                out[key] = {
                    "valid": True, "bl_skew_max": rep["bl"]["skew_max"],
                    "bl_skew_p99": rep["bl"]["skew_p99"],
                    "min_angle": rep["min_angle"],
                    "y1_min": rep["wall"]["y1_min_ratio"],
                    "y1_max": rep["wall"]["y1_max_ratio"],
                    "wall_face_nonorth_max":
                        rep["wall"]["wall_face_nonorth_max"],
                    "bl_growth_max": rep["wall"]["bl_growth_max"],
                    "fv_max": float(fv.max()),
                    "fv_p99": float(np.percentile(fv, 99))}
            except ValueError as err:
                out[key] = {"valid": False, "error": str(err)}
            r = out[key]
            print(f"{key:48s}", (f"BL skew {r['bl_skew_max']:6.2f} p99 "
                                 f"{r['bl_skew_p99']:5.2f} | y1 "
                                 f"{r['y1_min']:.3f}-{r['y1_max']:.3f} | "
                                 f"wall-face {r['wall_face_nonorth_max']:5.2f}"
                                 f" | growth {r['bl_growth_max']:.3f} | fv "
                                 f"{r['fv_max']:5.1f}") if r["valid"]
                  else r["error"], flush=True)
    p = Path(__file__).parent / "results" / "smoothing.json"
    p.write_text(json.dumps(out, indent=1))

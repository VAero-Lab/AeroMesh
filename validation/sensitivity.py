"""One-at-a-time sensitivity of the C-mesh metrics to its continuous
parameters, around the defaults: the search space of the optimisation layer.

    python validation/sensitivity.py
"""
import json
from pathlib import Path

import numpy as np

import aeromesh as am
from aeromesh.mesh import c_mesh, first_cell_height
from aeromesh.quality.metrics import assembled_nonorthogonality, mesh_report

Y1 = first_cell_height(6e6, 1.0)
PARAMS = {
    "eps": [0.1, 0.25, 0.5, 1.0, 2.0],
    "volume_smoothing": [0.0, 0.15, 0.3, 0.6, 1.0],
    "te_aspect": [40.0, 80.0, 135.0, 250.0, 500.0],
    "turn_per_cell": list(np.deg2rad([2.5, 3.5, 5.0, 7.5, 10.0])),
    "wake_relax": [0.25, 0.5, 1.0, 2.0, 4.0],
    "growth": [1.08, 1.12, 1.15, 1.2, 1.25],
    "h_wall": [0.006, 0.009, 0.012, 0.018, 0.024],
}

if __name__ == "__main__":
    out = {}
    for name, te in (("4412", "sharp"), ("4412", "blunt")):
        bnd = am.build_region([am.load_airfoil(name, n_points=1000)],
                              farfield=20.0, te=te,
                              body_spacing=5e-4).boundaries[1]
        for p, vals in PARAMS.items():
            for v in vals:
                key = f"{name}_{te}_{p}_{v:.4g}"
                try:
                    mb = c_mesh(bnd, Y1, radius=500.0, wake_length=500.0,
                                n_layers=140, **{p: v})
                    m = mb.assemble()
                    rep = mesh_report(mb, Y1)
                    fv = assembled_nonorthogonality(m)
                    out[key] = dict(valid=True, param=p, value=float(v),
                                    te=te, cells=m.n_cells,
                                    bl_skew=rep["bl"]["skew_max"],
                                    bl_skew_p99=rep["bl"]["skew_p99"],
                                    fv_max=float(fv.max()),
                                    fv_p99=float(np.percentile(fv, 99)),
                                    min_angle=rep["min_angle"],
                                    wall_face=rep["wall"]["wall_face_nonorth_max"],
                                    ds_ratio=rep["wall"]["along_wall_ratio_max"])
                except ValueError as err:
                    out[key] = dict(valid=False, param=p, value=float(v),
                                    te=te, error=str(err)[:120])
                r = out[key]
                print(key, {k: round(v_, 3) for k, v_ in r.items()
                            if isinstance(v_, float)} if r["valid"]
                      else r["error"], flush=True)
    (Path(__file__).parent / "results" / "sensitivity.json").write_text(
        json.dumps(out, indent=1))

"""NACA 0012 at Re = 6e6: build the meshes, write the cases, start the solver.

    python validation/run_naca0012.py <work_dir>

Cases are written under ``<work_dir>/cases`` and each ``foamRun`` is started
in the background. Collect with ``validation/collect_naca0012.py`` once they
have finished.

The reference is the NASA Turbulence Modeling Resource 2-D NACA 0012 case
(SA model, M = 0.15, Re = 6e6, outer boundary about 500 chords away). This
solver is incompressible, so lift is expected about one per cent below the
reference (Prandtl-Glauert at M = 0.15 is 1.0114).
"""

import json
import sys
from pathlib import Path

import numpy as np

import aeromesh as am
from aeromesh.mesh import c_mesh, first_cell_height, o_mesh
from aeromesh.quality.metrics import assembled_nonorthogonality, mesh_report

sys.path.insert(0, str(Path(__file__).parent))
from foam_case import _HEAD, check_mesh, foam, write_case  # noqa: E402

RE = 6e6
FAR = dict(radius=500.0, n_layers=140)
FINE = dict(radius=500.0, wake_length=500.0, n_layers=260, growth=1.08,
            h_wall=0.006, h_te=3e-4, turn_per_cell=np.deg2rad(2.5),
            ratio=1.08)
CASES = [  # label, builder, trailing edge, alpha, mesh options, y1 factor, cores
    ("C_sharp_a00", "C", "sharp", 0.0, dict(wake_length=500.0, **FAR), 1.0, 1),
    ("C_sharp_a10", "C", "sharp", 10.0, dict(wake_length=500.0, **FAR), 1.0, 1),
    ("C_sharp_a15", "C", "sharp", 15.0, dict(wake_length=500.0, **FAR), 1.0, 1),
    ("C_blunt_a10", "C", "blunt", 10.0, dict(wake_length=500.0, **FAR), 1.0, 1),
    ("O_blunt_a10", "O", "blunt", 10.0, dict(**FAR), 1.0, 1),
    ("C_sharp_a10_R50", "C", "sharp", 10.0, dict(), 1.0, 1),
    ("C_sharp_a10_fine", "C", "sharp", 10.0, FINE, 0.5, 4),
]

if __name__ == "__main__":
    work = Path(sys.argv[1])
    y1 = first_cell_height(RE, 1.0)
    body = am.load_airfoil("0012", n_points=1000)
    summary = {}
    for label, kind, te, alpha, kw, f, cores in CASES:
        region = am.build_region([body], farfield=20.0, te=te,
                                 body_spacing=5e-4)
        mb = (o_mesh if kind == "O" else c_mesh)(region.boundaries[1],
                                                 y1 * f, **kw)
        mesh = mb.assemble()
        case = work / "cases" / label
        write_case(mesh, case, reynolds=RE, alpha_deg=alpha, n_iter=2500)
        rep = mesh_report(mb, y1 * f)
        chk = check_mesh(case)
        summary[label] = {
            "kind": kind, "te": te, "alpha": alpha, "cells": mesh.n_cells,
            "first_cell": y1 * f,
            "radius": kw.get("radius", 50.0),
            "blocks": [list(b.shape) for b in mb.blocks],
            "checkMesh": chk,
            "own_nonorth_max": float(assembled_nonorthogonality(mesh).max()),
            "wall": {k: v for k, v in rep["wall"].items()
                     if not k.startswith("_")},
            "bl": rep["bl"], "min_angle": rep["min_angle"],
            "max_angle": rep["max_angle"],
        }
        if cores > 1:
            (case / "system" / "decomposeParDict").write_text(
                _HEAD.format(cls="dictionary", obj="decomposeParDict")
                + f"numberOfSubdomains {cores};\nmethod scotch;\n")
            cmd = (f"decomposePar > log.decompose 2>&1 && "
                   f"mpirun -np {cores} foamRun -parallel")
        else:
            cmd = "foamRun"
        foam(cmd, case, log="log.foamRun", background=True)
        print(label, mesh.n_cells, "cells; non-orth max: checkMesh",
              chk["nonorth_max"], "ours",
              round(summary[label]["own_nonorth_max"], 3), "- started",
              flush=True)
    (work / "cases" / "summary.json").write_text(json.dumps(summary, indent=1))

"""AeroMesh against pyHyp and Construct2D: metrics, time, and drag.

    python validation/compare_meshers.py <work_dir> [--solve]

Every mesher meshes the *same* closed loop (the region's body after its
trailing-edge policy), with the same first cell height and a far field about
500 chords away, and every mesh is measured by the same code and by
OpenFOAM's ``checkMesh``. With ``--solve`` the NACA 0012 meshes are run in
OpenFOAM at alpha = 0 and 10 degrees; collect with
``python validation/collect_naca0012.py <work_dir> cmp_cases comparison_solver``.

Each comparator runs with the settings its documentation recommends for the
topology (Construct2D's own NACA 0012 sample settings for a C-grid; pyHyp's
NACA 0012 RANS example), and pyHyp is given AeroMesh's own wall nodes, so the
two O-grids differ only in how they march.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import aeromesh as am
from aeromesh.mesh import c_mesh, first_cell_height, o_mesh
from aeromesh.quality.metrics import assembled_nonorthogonality, mesh_report

sys.path.insert(0, str(Path(__file__).parent))
from comparators import construct2d as c2d  # noqa: E402
from comparators import pyhyp_driver as phd  # noqa: E402
from foam_case import _HEAD, check_mesh, foam, write_case  # noqa: E402

RE = 6e6
Y1 = first_cell_height(RE, 1.0)
R = 500.0
SECTIONS = ["0012", "4412", "8412"]


def boundary(name, te):
    body = am.load_airfoil(name, n_points=1000)
    return am.build_region([body], farfield=20.0, te=te,
                           body_spacing=5e-4).boundaries[1]


def ccw(wall):
    a = 0.5 * np.sum(wall[:, 0] * np.roll(wall[:, 1], -1)
                     - np.roll(wall[:, 0], -1) * wall[:, 1])
    return wall if a > 0 else wall[::-1]


def build_all(name, te, work):
    """``{mesher: (MultiBlock, seconds)}`` for one section."""
    bnd = boundary(name, te)
    out = {}
    t0 = time.time()
    out["AeroMesh C"] = (c_mesh(bnd, Y1, radius=R, wake_length=R,
                                n_layers=140), None)
    out["AeroMesh C"] = (out["AeroMesh C"][0], time.time() - t0)
    t0 = time.time()
    om = o_mesh(bnd, Y1, radius=R, n_layers=140)
    out["AeroMesh O"] = (om, time.time() - t0)
    wall = ccw(om.blocks[0].grid[:-1, 0])
    out["pyHyp O"] = phd.mesh(wall, Y1, n_layers=141, march_dist=R,
                              workdir=work / f"pyhyp_{name}_{te}")
    topo = "CGRD" if te == "sharp" else "OGRD"
    kw = (dict(nsrf=250, nwke=75, lesp=7.5e-4, tesp=3e-3) if topo == "CGRD"
          else dict(nsrf=250, lesp=4e-3, tesp=6e-4))
    mb, dt, _ = c2d.mesh(name, te, topo=topo, jmax=121, ypls=2.0, recd=RE,
                         radi=R, workdir=work / f"c2d_{name}_{te}", **kw)
    out[f"Construct2D {topo[0]}"] = (mb, dt)
    return out


def measure(mb, case_dir):
    mesh = mb.assemble()
    rep = mesh_report(mb, Y1)
    fv = assembled_nonorthogonality(mesh)
    write_case(mesh, case_dir, reynolds=RE, alpha_deg=0.0, n_iter=1)
    chk = check_mesh(case_dir)
    w = {k: v for k, v in rep["wall"].items() if not k.startswith("_")}
    pts = mesh.points[np.unique(mesh.patches["farfield"])]
    return {
        "cells": mesh.n_cells, "blocks": [list(b.shape) for b in mb.blocks],
        "wall": w, "bl": rep["bl"], "min_angle": rep["min_angle"],
        "max_angle": rep["max_angle"], "aspect_max": rep["aspect_max"],
        "fv_nonorth_max": float(fv.max()),
        "fv_nonorth_p99": float(np.percentile(fv, 99)),
        "fv_nonorth_mean": float(fv.mean()),
        "farfield_min_dist": float(np.min(np.linalg.norm(pts - [0.5, 0.0],
                                                          axis=1))),
        "checkMesh": chk,
        "wall_orth_profile": {"x": rep["wall"]["_x"].tolist(),
                              "orth": rep["wall"]["_orth"].tolist(),
                              "y1": rep["wall"]["_y1"].tolist()},
    }


if __name__ == "__main__":
    work = Path(sys.argv[1])
    solve = "--solve" in sys.argv
    meshes_dir = work / "cmp_meshes"
    results = {"y1": Y1, "radius": R, "cases": {}}
    solver_summary = {}
    for name in SECTIONS:
        for te in ("sharp", "blunt"):
            built = build_all(name, te, meshes_dir)
            for mesher, (mb, dt) in built.items():
                key = f"{name}_{te}_{mesher}"
                try:
                    m = measure(mb, meshes_dir / key.replace(" ", "_"))
                    m["seconds"] = dt
                    m["valid"] = True
                except ValueError as err:
                    m = {"valid": False, "error": str(err), "seconds": dt}
                results["cases"][key] = m
                if m["valid"]:
                    print(f"{key:30s} {m['cells']:6d} cells {dt:5.2f} s | "
                          f"BL skew {m['bl']['skew_max']:5.1f} | min ang "
                          f"{m['min_angle']:5.1f} | fv {m['fv_nonorth_max']:5.1f}"
                          f" | foam {m['checkMesh']['nonorth_max']} | y1 "
                          f"{m['wall']['y1_min_ratio']:.3f}-"
                          f"{m['wall']['y1_max_ratio']:.3f}", flush=True)
                else:
                    print(key, "INVALID", m["error"], flush=True)
                if solve and name == "0012" and m["valid"]:
                    for alpha in (0.0, 10.0):
                        label = (f"{te}_{mesher.replace(' ', '_')}"
                                 f"_a{int(alpha):02d}")
                        case = work / "cmp_cases" / label
                        write_case(mb.assemble(), case, reynolds=RE,
                                   alpha_deg=alpha, n_iter=3000)
                        foam("foamRun", case, log="log.foamRun",
                             background=True)
                        solver_summary[label] = {
                            "kind": mesher, "te": te, "alpha": alpha,
                            "cells": m["cells"], "radius": R,
                            "checkMesh": m["checkMesh"],
                            "own_nonorth_max": m["fv_nonorth_max"],
                            "wall": m["wall"], "bl": m["bl"],
                            "min_angle": m["min_angle"],
                            "max_angle": m["max_angle"],
                            "blocks": m["blocks"], "first_cell": Y1}
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(results, indent=1))
    if solve:
        (work / "cmp_cases").mkdir(parents=True, exist_ok=True)
        (work / "cmp_cases" / "summary.json").write_text(
            json.dumps(solver_summary, indent=1))
    print("written", out / "comparison.json")

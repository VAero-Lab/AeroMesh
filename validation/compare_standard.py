"""Standardised comparison: every mesher at the same grid dimensions and the
same measured far-field distance, at three refinement levels.

    python validation/compare_standard.py <work_dir> [--solve]

Protocol
--------
* **Geometry**: the same closed loop for every mesher (``comparators.geometry``).
* **Far field**: each tool's own distance setting is tuned (secant on the
  ratio) until the *measured* distance from mid-chord to the nearest
  far-field node is 500 chords, to 0.5 %. The shape stays the tool's own -- a
  C-grid's outer boundary is C-shaped, an O-grid's round.
* **Grid dimensions** per level ``L``: ``N_i`` nodes round the body-and-wake
  line and ``N_j`` nodes off it, identical for every mesher; C-grids put
  ``N_wake`` of the ``N_i`` on each side of the wake cut and the rest on the
  wall, O-grids put all ``N_i`` on the wall.
* **First cell** ``y1 = 4.46e-6 * 2**(2 - L)``: halved with each level, as in
  NASA's grid families, so the levels form a refinement sequence (factor 2 in
  every direction).
* Outer-layer growth: AeroMesh's default cap (1.3) cannot reach 500 chords
  in the 75 layers of level 1; it is raised to 1.8 here, as the other tools
  raise theirs. The boundary layer keeps its 1.15.
* Every tool runs with the settings its documentation recommends for the
  topology; within that, only the counts and distances above are imposed.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import aeromesh as am
from aeromesh.mesh import c_mesh, first_cell_height, o_mesh
from aeromesh.quality.metrics import (assembled_nonorthogonality,
                                      cell_angles, cell_aspect, mesh_report)

sys.path.insert(0, str(Path(__file__).parent))
from comparators import construct2d as c2d  # noqa: E402
from comparators import pyhyp_driver as phd  # noqa: E402
from foam_case import _HEAD, check_mesh, foam, write_case  # noqa: E402

RE = 6e6
R_TARGET = 500.0
LEVELS = {1: dict(ni=192, nj=76, wake=40),
          2: dict(ni=384, nj=151, wake=80),
          3: dict(ni=768, nj=301, wake=160)}
CONFIGS = [("sharp", "AeroMesh C"), ("sharp", "pyHyp O"),
           ("sharp", "Construct2D C"),
           ("blunt", "AeroMesh C"), ("blunt", "AeroMesh O"),
           ("blunt", "pyHyp O"), ("blunt", "Construct2D O")]


def y1_of(level):
    return first_cell_height(RE, 1.0) * 2.0 ** (2 - level)


def boundary(name, te):
    return am.build_region([am.load_airfoil(name, n_points=1000)],
                           farfield=20.0, te=te,
                           body_spacing=5e-4).boundaries[1]


def far_distance(mb):
    mesh = mb.assemble()
    pts = mesh.points[np.unique(mesh.patches["farfield"])]
    return float(np.min(np.linalg.norm(pts - [0.5, 0.0], axis=1)))


def tuned(build, start):
    """Call ``build(d)`` with the distance parameter ``d`` adjusted until the
    measured far-field distance is R_TARGET to 0.5 %."""
    d, out = start, None
    for _ in range(6):
        out = build(d)
        mb = out[0] if isinstance(out, tuple) else out
        got = far_distance(mb)
        if abs(got / R_TARGET - 1.0) < 0.005:
            break
        d *= R_TARGET / got
    return out, d, got


def build(name, te, mesher, level, work):
    L = LEVELS[level]
    y1 = y1_of(level)
    ni, nj, nw = L["ni"], L["nj"], L["wake"]
    bnd = boundary(name, te)
    tag = f"{name}_{te}_{mesher.replace(' ', '_')}_L{level}"
    if mesher == "AeroMesh C":
        def f(d):
            t0 = time.perf_counter()
            mb = c_mesh(bnd, y1, radius=d, wake_length=R_TARGET,
                        n_layers=nj - 1, n_wall=ni - 2 * nw, n_wake=nw,
                        max_growth=1.8)
            return mb, time.perf_counter() - t0
        (mb, dt), d, got = tuned(f, R_TARGET)
    elif mesher == "AeroMesh O":
        def f(d):
            t0 = time.perf_counter()
            mb = o_mesh(bnd, y1, radius=d, n_layers=nj - 1, n_wall=ni - 1,
                        max_growth=1.8)
            return mb, time.perf_counter() - t0
        (mb, dt), d, got = tuned(f, R_TARGET)
    elif mesher == "pyHyp O":
        # pyHyp marches from AeroMesh's O-mesh wall nodes (same count)
        om = o_mesh(bnd, y1, radius=50.0, n_layers=20, n_wall=ni - 1,
                    allow_cusp=True)
        wall = om.blocks[0].grid[:-1, 0]
        a = 0.5 * np.sum(wall[:, 0] * np.roll(wall[:, 1], -1)
                         - np.roll(wall[:, 0], -1) * wall[:, 1])
        wall = wall if a > 0 else wall[::-1]

        def f(d):
            return phd.mesh(wall, y1, n_layers=nj, march_dist=d,
                            workdir=work / tag)
        (mb, dt), d, got = tuned(f, R_TARGET)
    else:
        topo = "CGRD" if mesher.endswith("C") else "OGRD"
        scale = 2.0 ** (2 - level)
        kw = (dict(nsrf=ni - 2 * nw, nwke=nw, lesp=1e-3 * scale,
                   tesp=3e-3 * scale) if topo == "CGRD" else
              dict(nsrf=ni, lesp=1e-3 * scale, tesp=6e-4 * scale))

        def f(d):
            mb, dt, _ = c2d.mesh(name, te, topo=topo, jmax=nj,
                                 ypls=2.0 * scale, recd=RE, radi=d,
                                 workdir=work / tag, **kw)
            return mb, dt
        (mb, dt), d, got = tuned(f, R_TARGET)
    return mb, dt, d, got, y1


def measure(mb, y1):
    mesh = mb.assemble()
    rep = mesh_report(mb, y1)
    fv = assembled_nonorthogonality(mesh)
    lo = np.concatenate([cell_angles(b.grid)[0].ravel() for b in mb.blocks])
    hi = np.concatenate([cell_angles(b.grid)[1].ravel() for b in mb.blocks])
    skew = np.maximum(90.0 - lo, hi - 90.0)
    ar = np.concatenate([cell_aspect(b.grid).ravel() for b in mb.blocks])
    w = {k: v for k, v in rep["wall"].items() if not k.startswith("_")}
    hist = lambda v, bins: np.histogram(v, bins=bins)[0].tolist()  # noqa
    return mesh, {
        "cells": mesh.n_cells, "blocks": [list(b.shape) for b in mb.blocks],
        "wall_nodes": int(len(np.unique(mesh.patches["wall"]))),
        "wall": w, "bl": rep["bl"], "min_angle": rep["min_angle"],
        "max_angle": rep["max_angle"], "aspect_max": float(ar.max()),
        "skew_p50": float(np.median(skew)), "skew_p99": float(
            np.percentile(skew, 99)), "skew_max": float(skew.max()),
        "fv_max": float(fv.max()), "fv_p99": float(np.percentile(fv, 99)),
        "fv_mean": float(fv.mean()),
        "frac_fv_over_30": float(np.mean(fv > 30.0)),
        "frac_skew_over_45": float(np.mean(skew > 45.0)),
        "hist_skew": hist(skew, np.arange(0, 91, 2.5)),
        "hist_fv": hist(fv, np.arange(0, 91, 2.5)),
        "hist_ar_log10": hist(np.log10(ar), np.arange(0, 8.01, 0.25)),
        "profile": {"x": rep["wall"]["_x"].tolist(),
                    "y1": rep["wall"]["_y1"].tolist(),
                    "orth": rep["wall"]["_orth"].tolist(),
                    "face_x": rep["wall"]["_face_x"].tolist(),
                    "face_orth": rep["wall"]["_face_orth"].tolist()},
    }


if __name__ == "__main__":
    work = Path(sys.argv[1])
    solve = "--solve" in sys.argv
    mdir = work / "std_meshes"
    res_path = Path(__file__).parent / "results" / "standard.json"
    results = (json.loads(res_path.read_text()) if res_path.exists()
               else {"protocol": {"R": R_TARGET, "levels": LEVELS},
                     "cases": {}})
    summary_path = work / "std_cases" / "summary.json"
    summary = (json.loads(summary_path.read_text())
               if summary_path.exists() else {})
    sections = [a for a in sys.argv[2:] if a.isdigit()] or ["0012"]
    levels = [int(a[1:]) for a in sys.argv[2:] if a.startswith("L")] or [1, 2, 3]
    only = [a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--only=")]
    for name in sections:
        for level in levels:
            for te, mesher in CONFIGS:
                if only and not any(o in mesher for o in only):
                    continue
                key = f"{name}_{te}_{mesher}_L{level}"
                try:
                    mb, dt, d, got, y1 = build(name, te, mesher, level, mdir)
                    mesh, m = measure(mb, y1)
                    case = mdir / key.replace(" ", "_")
                    write_case(mesh, case, reynolds=RE, alpha_deg=0.0,
                               n_iter=1)
                    m["checkMesh"] = check_mesh(case)
                    m.update(valid=True, seconds=dt, distance_param=d,
                             far_distance=got, y1=y1, level=level)
                except Exception as err:  # noqa: BLE001
                    m = {"valid": False, "error": str(err)[:300],
                         "level": level}
                results["cases"][key] = m
                if m["valid"]:
                    print(f"{key:32s} cells {m['cells']:7d} dims "
                          f"{m['blocks']} R {m['far_distance']:.1f} | BLskew "
                          f"{m['bl']['skew_max']:5.1f} fv {m['fv_max']:5.1f} "
                          f"foam {m['checkMesh']['nonorth_max']} y1 "
                          f"{m['wall']['y1_min_ratio']:.3f}-"
                          f"{m['wall']['y1_max_ratio']:.3f}", flush=True)
                else:
                    print(key, "INVALID", m["error"][:120], flush=True)
                    continue
                if not solve or name != "0012":
                    continue
                alphas = [0.0, 10.0] if level == 2 else [10.0]
                if level == 3 and te == "blunt":
                    alphas = []
                for alpha in alphas:
                    label = f"{key.replace(' ', '_')}_a{int(alpha):02d}"
                    case = work / "std_cases" / label
                    n_iter = {1: 3000, 2: 3000, 3: 4000}[level]
                    write_case(mesh, case, reynolds=RE, alpha_deg=alpha,
                               n_iter=n_iter)
                    if level == 3:
                        (case / "system" / "decomposeParDict").write_text(
                            _HEAD.format(cls="dictionary",
                                         obj="decomposeParDict")
                            + "numberOfSubdomains 2;\nmethod scotch;\n")
                        cmd = ("decomposePar > log.decompose 2>&1 && "
                               "mpirun -np 2 foamRun -parallel")
                    else:
                        cmd = "foamRun"
                    foam(cmd, case, log="log.foamRun", background=True)
                    summary[label] = {
                        "kind": mesher, "te": te, "alpha": alpha,
                        "level": level, "cells": m["cells"],
                        "radius": m["far_distance"], "first_cell": y1,
                        "checkMesh": m["checkMesh"], "wall": m["wall"],
                        "bl": m["bl"], "min_angle": m["min_angle"],
                        "max_angle": m["max_angle"], "blocks": m["blocks"]}
            res_path.write_text(json.dumps(results, indent=1))
    if solve:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=1))
    print("written", res_path)

"""Level 3 of the standardised comparison, started from level 2.

    python validation/run_level3.py <work_dir>

The 230-thousand-cell runs are initialised by interpolating each mesher's
converged level-2 solution onto its level-3 mesh (OpenFOAM ``mapFields``), and
run on 4 cores each. Only the sharp trailing edge is run at level 3 (it is
NASA's geometry). The AeroMesh level-3 mesh is rebuilt here: the first build
used a layer law that put its far field at 889 chords (fixed in
``spacing.layer_heights``).
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_standard import RE, build, measure  # noqa: E402
from foam_case import _HEAD, check_mesh, foam, write_case  # noqa: E402

N_ITER = 1500
CORES = 4

if __name__ == "__main__":
    work = Path(sys.argv[1])
    cases = work / "std_cases"
    res_path = Path(__file__).parent / "results" / "standard.json"
    results = json.loads(res_path.read_text())
    summary = json.loads((cases / "summary.json").read_text())

    # rebuild the AeroMesh level-3 meshes with the fixed layer law
    for te in ("sharp", "blunt"):
        key = f"0012_{te}_AeroMesh C_L3"
        mb, dt, d, got, y1 = build("0012", te, "AeroMesh C", 3,
                                   work / "std_meshes")
        mesh, m = measure(mb, y1)
        tmp = work / "std_meshes" / key.replace(" ", "_")
        write_case(mesh, tmp, reynolds=RE, alpha_deg=0.0, n_iter=1)
        m["checkMesh"] = check_mesh(tmp)
        m.update(valid=True, seconds=dt, distance_param=d, far_distance=got,
                 y1=y1, level=3)
        results["cases"][key] = m
        print(key, "R", round(got, 1), "BL skew",
              round(m["bl"]["skew_max"], 2), "fv", round(m["fv_max"], 1),
              flush=True)
        if te == "sharp":
            lab = "0012_sharp_AeroMesh_C_L3_a10"
            shutil.rmtree(cases / lab, ignore_errors=True)
            write_case(mesh, cases / lab, reynolds=RE, alpha_deg=10.0,
                       n_iter=N_ITER)
            summary[lab] = {
                "kind": "AeroMesh C", "te": "sharp", "alpha": 10.0,
                "level": 3, "cells": m["cells"], "radius": got,
                "first_cell": y1, "checkMesh": m["checkMesh"],
                "wall": m["wall"], "bl": m["bl"],
                "min_angle": m["min_angle"], "max_angle": m["max_angle"],
                "blocks": m["blocks"]}
    res_path.write_text(json.dumps(results, indent=1))
    (cases / "summary.json").write_text(json.dumps(summary, indent=1))

    for ms in ("AeroMesh_C", "Construct2D_C", "pyHyp_O"):
        lab = f"0012_sharp_{ms}_L3_a10"
        src = f"0012_sharp_{ms}_L2_a10"
        case = cases / lab
        for d in case.iterdir():
            if d.name.startswith("processor") or d.name == "postProcessing" \
                    or (d.name.replace(".", "").isdigit() and d.name != "0"):
                shutil.rmtree(d, ignore_errors=True)
        ctl = case / "system" / "controlDict"
        txt = ctl.read_text()
        import re
        txt = re.sub(r"endTime\s+\d+;", f"endTime         {N_ITER};", txt)
        txt = re.sub(r"writeInterval\s+\d+;", f"writeInterval   {N_ITER};",
                     txt)
        txt = txt.replace("startFrom       latestTime;",
                          "startFrom       startTime;")
        ctl.write_text(txt)
        (case / "system" / "decomposeParDict").write_text(
            _HEAD.format(cls="dictionary", obj="decomposeParDict")
            + f"numberOfSubdomains {CORES};\nmethod scotch;\n")
        cmd = (f"mapFields ../{src} -consistent -sourceTime latestTime "
               f"> log.mapFields 2>&1 && decomposePar -force > log.decompose "
               f"2>&1 && mpirun -np {CORES} foamRun -parallel")
        foam(cmd, case, log="log.foamRun", background=True)
        print("started", lab, "from", src, flush=True)

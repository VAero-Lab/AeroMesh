"""Collect the NACA 0012 runs into ``validation/results/naca0012.json``.

    python validation/collect_naca0012.py <work_dir> [cases_subdir] [out_name]

Forces are the mean over the last 200 iterations; ``drift`` is how far that
mean moved from the 200 before it, so a number that has not settled says so.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from foam_case import read_coeffs  # noqa: E402

# NASA Turbulence Modeling Resource, 2-D NACA 0012, SA, CFL3D, 897x257 grid,
# M = 0.15, Re = 6e6.  https://tmbwg.github.io/turbmodels/naca0012_val_sa.html
REFERENCE = {0.0: {"cl": 0.0, "cd": 0.00819},
             10.0: {"cl": 1.0909, "cd": 0.01231},
             15.0: {"cl": 1.5461, "cd": 0.02124}}

if __name__ == "__main__":
    work = Path(sys.argv[1]) / (sys.argv[2] if len(sys.argv) > 2 else "cases")
    out_name = sys.argv[3] if len(sys.argv) > 3 else "naca0012"
    summary = json.loads((work / "summary.json").read_text())
    out = {"reference": {str(k): v for k, v in REFERENCE.items()},
           "cases": {}}
    for label, info in summary.items():
        cols, h = read_coeffs(work / label)
        if h.size == 0:
            continue
        ic, il = cols.index("Cd"), cols.index("Cl")
        last, prev = h[-200:], h[-400:-200]
        cd, cl = last[:, ic].mean(), last[:, il].mean()
        ref = REFERENCE[info["alpha"]]
        log = (work / label / "log.foamRun").read_text()
        yp = []
        for yf in sorted((work / label / "postProcessing" / "yPlus").glob(
                "*/yPlus.dat")):
            rows = [ln.split() for ln in yf.read_text().splitlines()
                    if ln and not ln.startswith("#")]
            rows = [r for r in rows if float(r[0]) > 0]   # not the initial field
            if rows:                       # time, patch, min, max, average
                yp.append(rows[-1][2:5])
        finished = bool(re.search(r"^End", log, re.M))
        out["cases"][label] = {
            **info, "iterations": int(h[-1, 0]), "finished": finished,
            "cl": cl, "cd": cd,
            "cl_drift": abs(cl - prev[:, il].mean()),
            "cd_drift": abs(cd - prev[:, ic].mean()),
            "cd_counts_vs_ref": (cd - ref["cd"]) * 1e4,
            "cd_pct_vs_ref": 100 * (cd / ref["cd"] - 1),
            "cl_pct_vs_ref": (100 * (cl / ref["cl"] - 1) if ref["cl"] else None),
            "yplus": ([float(yp[-1][0]), float(yp[-1][1]), float(yp[-1][2])]
                      if yp else None),
            "history": h[::20, [0, ic, il]].tolist(),
        }
        c = out["cases"][label]
        print(f"{label:18s} it {c['iterations']:5d} {'done' if finished else 'RUNNING'}"
              f"  Cl {cl:.4f}  Cd {cd:.5f}  ({c['cd_counts_vs_ref']:+.1f} counts)"
              f"  drift Cd {c['cd_drift']:.1e}  y+ {c['yplus']}")
    res = Path(__file__).parent / "results"
    res.mkdir(exist_ok=True)
    (res / f"{out_name}.json").write_text(json.dumps(out, indent=1))

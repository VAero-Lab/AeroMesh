"""Collect the standardised runs: forces, convergence, y+, and Cp / Cf.

    python validation/collect_standard.py <work_dir>

Writes ``results/standard_solver.json``. Forces are the mean of the last 200
iterations; ``drift`` is how far that mean moved over the 200 before. Grid
convergence (levels 1-3, factor 2) is evaluated by Richardson extrapolation
where three levels exist, with the observed order and Roache's grid
convergence index (safety factor 1.25).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from foam_case import read_coeffs  # noqa: E402
from surface import read_surface, reference  # noqa: E402

REF = {0.0: {"cl": 0.0, "cd": 0.00819}, 10.0: {"cl": 1.0909, "cd": 0.01231},
       15.0: {"cl": 1.5461, "cd": 0.02124}}


def richardson(f1, f2, f3, r=2.0):
    """f1 coarse ... f3 fine. Returns (order, extrapolated, GCI_fine)."""
    e21, e32 = f2 - f1, f3 - f2
    if e32 == 0 or e21 == 0 or np.sign(e21) != np.sign(e32):
        return None, None, None          # oscillatory or converged
    p = np.log(abs(e21 / e32)) / np.log(r)
    ext = f3 + (f3 - f2) / (r ** p - 1.0)
    gci = 1.25 * abs((f3 - f2) / f3) / (r ** p - 1.0)
    return float(p), float(ext), float(gci)


def curve_error(x, v, xr, vr, lo=0.02, hi=0.98):
    """RMS difference to the reference over ``lo < x < hi`` (interpolated
    onto the reference abscissae)."""
    o = np.argsort(x)
    m = (xr > lo) & (xr < hi)
    vi = np.interp(xr[m], x[o], v[o])
    return float(np.sqrt(np.mean((vi - vr[m]) ** 2)))



def _split(xr, v, upper):
    """The reference Cp runs TE -> lower -> LE -> upper -> TE (checked: the
    suction peak, Cp ~ -5.5 at alpha 10, lies in the second half)."""
    k = int(np.argmin(xr))
    return (xr[k:], v[k:]) if upper else (xr[:k + 1], v[:k + 1])


if __name__ == "__main__":
    work = Path(sys.argv[1]) / "std_cases"
    summary = json.loads((work / "summary.json").read_text())
    out = {"reference": {str(k): v for k, v in REF.items()}, "cases": {},
           "convergence": {}}
    for label, info in summary.items():
        case = work / label
        cols, h = read_coeffs(case)
        if h.size == 0:
            continue
        ic, il = cols.index("Cd"), cols.index("Cl")
        last, prev = h[-200:], h[-400:-200]
        cd, cl = float(last[:, ic].mean()), float(last[:, il].mean())
        log = (case / "log.foamRun").read_text()
        finished = bool(re.search(r"^End", log, re.M))
        yp = None
        for yf in sorted((case / "postProcessing" / "yPlus").glob(
                "*/yPlus.dat")):
            rows = [ln.split() for ln in yf.read_text().splitlines()
                    if ln and not ln.startswith("#")]
            rows = [r for r in rows if float(r[0]) > 0]
            if rows:
                yp = [float(v) for v in rows[-1][2:5]]
        a = info["alpha"]
        rec = {**info, "iterations": int(h[-1, 0]), "finished": finished,
               "cl": cl, "cd": cd,
               "cd_drift": float(abs(cd - prev[:, ic].mean())),
               "cl_drift": float(abs(cl - prev[:, il].mean())),
               "cd_counts_vs_ref": (cd - REF[a]["cd"]) * 1e4,
               "cl_pct_vs_ref": (100 * (cl / REF[a]["cl"] - 1)
                                 if REF[a]["cl"] else None),
               "yplus": yp, "history": h[::20, [0, ic, il]].tolist()}
        try:
            s = read_surface(case)
            rec["surface"] = {k: (None if v is None else v.tolist())
                              for k, v in s.items()}
            xr, cpr = reference("cp", a)
            up = np.array(s["y"]) >= 0
            # the reference lists the whole section: split it at x = 0
            rec["cp_rms_upper"] = curve_error(
                np.asarray(s["x"])[up], np.asarray(s["cp"])[up],
                *_split(xr, cpr, upper=True)) if a else None
            rec["cp_rms_lower"] = curve_error(
                np.asarray(s["x"])[~up], np.asarray(s["cp"])[~up],
                *_split(xr, cpr, upper=False)) if a else None
            if s["cf"] is not None:
                xf, cfr = reference("cf", a)
                rec["cf_rms_upper"] = curve_error(
                    np.asarray(s["x"])[up], np.asarray(s["cf"])[up], xf, cfr)
        except Exception as err:  # noqa: BLE001
            rec["surface_error"] = str(err)[:200]
        out["cases"][label] = rec
        print(f"{label:38s} it {rec['iterations']:5d} "
              f"{'done' if finished else 'RUN '} Cl {cl:.4f} Cd {cd:.5f} "
              f"({rec['cd_counts_vs_ref']:+.1f} cts) drift "
              f"{rec['cd_drift'] * 1e4:.2f} y+ {yp}", flush=True)

    # grid convergence per mesher and alpha
    groups = {}
    for lab, r in out["cases"].items():
        key = f"{r['te']}|{r['kind']}|{r['alpha']}"
        groups.setdefault(key, {})[r["level"]] = r
    for key, lv in groups.items():
        conv = {"levels": sorted(lv), "cells": [lv[k]["cells"] for k in sorted(lv)],
                "cd": [lv[k]["cd"] for k in sorted(lv)],
                "cl": [lv[k]["cl"] for k in sorted(lv)]}
        if all(k in lv for k in (1, 2, 3)):
            for q in ("cd", "cl"):
                p, ext, gci = richardson(lv[1][q], lv[2][q], lv[3][q])
                conv[f"{q}_order"], conv[f"{q}_ext"], conv[f"{q}_gci"] = \
                    p, ext, gci
        out["convergence"][key] = conv
    p = Path(__file__).parent / "results" / "standard_solver.json"
    p.write_text(json.dumps(out))
    print("written", p)


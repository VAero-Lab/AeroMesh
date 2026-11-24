"""Numbers for report section 14.1 from ``results/study_next.json`` (and the
standardised runs in ``results/standard_solver.json``).

    python validation/analyse_next.py

Grid convergence uses the procedure of Celik et al. (2008, J. Fluids Eng.
130, 078001), which allows unequal refinement ratios: the levels at ratio
sqrt 2 are rounded to whole node counts, so r is computed from the cell
counts, r = (N_fine / N_coarse)^(1/2), not assumed.
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).parent / "results"


def celik(n, f):
    """``n``: cells coarse -> fine (3), ``f`` the values. Returns order p,
    extrapolated value and fine-grid GCI (fraction), or None if the triplet
    is not monotone."""
    n1, n2, n3 = n[2], n[1], n[0]            # 1 = fine (Celik's numbering)
    f1, f2, f3 = f[2], f[1], f[0]
    r21, r32 = np.sqrt(n1 / n2), np.sqrt(n2 / n3)
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0 or np.sign(e21) != np.sign(e32):
        return None
    s = np.sign(e32 / e21)
    p = 2.0
    for _ in range(200):
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    ext = (r21 ** p * f1 - f2) / (r21 ** p - 1.0)
    gci = 1.25 * abs((f1 - f2) / f1) / (r21 ** p - 1.0)
    return {"order": float(p), "ext": float(ext), "gci": float(gci),
            "r21": float(r21), "r32": float(r32)}


def main():
    s = json.loads((RES / "study_next.json").read_text())
    out = {}

    # ---- item 3: AeroMesh C sharp, alpha 10, five levels
    lv = {}
    for lab, r in s.items():
        if lab.startswith("base_L") and lab.endswith("_a10") \
                and r.get("done"):
            lv[float(r["level"])] = (r["cells"], r["cl_free"], r["cd_free"])
    std = json.loads((RES / "standard_solver.json").read_text())["cases"]
    for L in (1, 3):                       # reuse the standardised runs
        k = f"0012_sharp_AeroMesh_C_L{L}_a10"
        if float(L) not in lv and k in std:
            lv[float(L)] = (std[k]["cells"], std[k]["cl"], std[k]["cd"])
    levels = sorted(lv)
    out["levels"] = {str(L): {"cells": lv[L][0], "cl": lv[L][1],
                              "cd": lv[L][2]} for L in levels}
    trip = {}
    for i in range(len(levels) - 2):
        for j in range(i + 1, len(levels) - 1):
            for k in range(j + 1, len(levels)):
                a, b, c = levels[i], levels[j], levels[k]
                # only constant nominal ratio
                if not np.isclose(b - a, c - b):
                    continue
                n = [lv[a][0], lv[b][0], lv[c][0]]
                key = f"L{a:g},L{b:g},L{c:g}"
                trip[key] = {q: celik(n, [lv[a][x], lv[b][x], lv[c][x]])
                             for q, x in (("cl", 1), ("cd", 2))}
    out["triplets"] = trip
    for k, v in trip.items():
        cd, cl = v["cd"], v["cl"]
        print(f"{k:16s} Cd " + (f"p {cd['order']:5.2f} ext {cd['ext']*1e4:7.2f}"
                                f" GCI {cd['gci']*100:6.3f}%" if cd else
                                "not monotone") +
              " | Cl " + (f"p {cl['order']:5.2f} ext {cl['ext']:.4f} GCI "
                          f"{cl['gci']*100:6.3f}%" if cl else "not monotone"))
    # ---- surface errors against NASA CFL3D (items 1, 6)
    from collect_standard import REF, _split, curve_error
    from surface import reference
    surf = {}
    for lab, r in s.items():
        if "surface" not in r or not r.get("done"):
            continue
        a = float(r["alpha"])
        x, y = np.asarray(r["surface"]["x"]), np.asarray(r["surface"]["y"])
        cp = np.asarray(r["surface"]["cp"])
        up = y >= 0
        xr, cpr = reference("cp", a)
        e = {"cp_rms_upper": curve_error(x[up], cp[up],
                                         *_split(xr, cpr, upper=True)),
             "cp_rms_lower": curve_error(x[~up], cp[~up],
                                         *_split(xr, cpr, upper=False))}
        if r["surface"]["cf"] is not None:
            xf, cfr = reference("cf", a)
            e["cf_rms_upper"] = curve_error(
                x[up], np.asarray(r["surface"]["cf"])[up], xf, cfr)
        ref = REF.get(a)
        e.update(cl=r["cl_free"], cd=r["cd_free"],
                 cl_pct=100 * (r["cl_free"] / ref["cl"] - 1) if ref else None,
                 cd_counts=(r["cd_free"] - ref["cd"]) * 1e4 if ref else None)
        if "cl_vortex" in r:
            e.update(cl_vortex=r["cl_vortex"], cd_vortex=r["cd_vortex"])
        if "yplus_regions" in r:
            e["yplus"] = r["yplus_regions"]
        surf[lab] = e
        print(f"{lab:18s} Cl {e['cl']:.4f} ({e['cl_pct']:+.2f}%) Cd "
              f"{e['cd'] * 1e4:6.1f} ({e['cd_counts']:+.1f}) Cp rms "
              f"{e['cp_rms_upper']:.4f}/{e['cp_rms_lower']:.4f} Cf rms "
              f"{e.get('cf_rms_upper', float('nan')) * 1e4:.2f}e-4")
    out["cases"] = surf
    (RES / "analyse_next.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()

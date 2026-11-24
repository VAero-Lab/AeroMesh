"""Report section 14.1 -- the work still before the optimisation layer.

    python validation/study_next.py <work_dir> build <name> [<name> ...]
    python validation/study_next.py <work_dir> vortex <label> [n_iter]
    python validation/study_next.py <work_dir> collect

``build`` meshes one named variant, measures it exactly as the standardised
comparison does (``compare_standard.measure`` and ``checkMesh``), writes the
OpenFOAM case and starts it in the background. ``vortex`` restarts a finished
case with the point-vortex far field (below). ``collect`` writes
``results/study_next.json``.

Every variant is the standard NACA 0012 protocol (``compare_standard``) with
one thing changed, named in ``VARIANTS``.

Point-vortex far field
----------------------
A lifting section at finite distance sees, to leading order, the freestream
plus the velocity of a point vortex of circulation Gamma = Cl c U / 2 at the
quarter chord. Imposing that on the far field instead of the bare freestream
removes the leading domain-size error (Thomas and Salas 1986; incompressible
form). OpenFOAM has no such condition, so it is applied by restarting: the
far-field ``freestreamValue`` of U is set per face from the current Cl, and
the pressure from Bernoulli, ``p = (1 - |u|^2) / 2``. Repeated until Cl stops
moving (two to three updates).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_standard import (LEVELS, R_TARGET, RE, boundary, build,  # noqa
                              far_distance, measure, tuned)
from foam_case import _HEAD, check_mesh, foam, read_coeffs, write_case  # noqa
from surface import _list, read_surface, reference  # noqa: E402

from aeromesh.mesh import c_mesh, first_cell_height  # noqa: E402
from aeromesh.mesh.farfield import (c_domain, circle, fit_farfield,  # noqa
                                    snap_farfield)

HERE = Path(__file__).parent


def dims(level):
    """Grid dimensions at a (possibly fractional) level: every count scales
    by 2**(level - 2) from level 2, the first cell by 2**(2 - level)."""
    s = 2.0 ** (level - 2.0)
    L2 = LEVELS[2]
    ni = int(2 * round(L2["ni"] * s / 2))
    nj = int(round((L2["nj"] - 1) * s)) + 1
    nw = int(round(L2["wake"] * s))
    return ni, nj, nw, first_cell_height(RE, 1.0) * 2.0 ** (2 - level)


def am_c(level, te="sharp", **kw):
    ni, nj, nw, y1 = dims(level)
    bnd = boundary("0012", te)

    def f(d):
        return c_mesh(bnd, y1, radius=d, wake_length=R_TARGET,
                      n_layers=nj - 1, n_wall=ni - 2 * nw, n_wake=nw,
                      max_growth=1.8, **kw)
    mb, d, got = tuned(f, R_TARGET)
    return mb, y1


def am_fitted(level, kind, te="sharp", inner=3.0, inner_frac=0.5, **kw):
    """C-mesh marched to ``inner`` chords, then one TFI block to the supplied
    far field at exactly R_TARGET (``'c'``: C-shape, ``'circle'``: circle
    round the whole C, wake cut kept inside)."""
    ni, nj, nw, y1 = dims(level)
    bnd = boundary("0012", te)
    n_in = int(round((nj - 1) * inner_frac))
    n_out = nj - 1 - n_in
    if kind == "c":
        cm = c_mesh(bnd, y1, radius=inner, wake_length=R_TARGET,
                    n_layers=n_in, n_wall=ni - 2 * nw, n_wake=nw,
                    max_growth=1.8, **kw)
        return c_domain(cm, "c", R_TARGET, growth=1.15), y1
    # circle: the C (and its wake) stays local, the annulus closes round it
    nw_in = max(8, nw // 2)
    cm = c_mesh(bnd, y1, radius=inner, wake_length=2.0 * inner,
                n_layers=n_in, n_wall=ni - 2 * nw, n_wake=nw_in,
                max_growth=1.8, **kw)
    return fit_farfield(cm, circle(R_TARGET), n_layers=n_out), y1


def am_snap(level, te="sharp", n_blend=5, **kw):
    """Standard C-mesh with its outer row slid onto the C-shape far field at
    exactly R_TARGET (``farfield.snap_farfield``)."""
    mb, y1 = am_c(level, te, **kw)
    nb = max(3, int(round(n_blend * 2.0 ** (level - 2))))
    return snap_farfield(mb, "c", R_TARGET, n_blend=nb), y1


def am_o(level, te="blunt"):
    ni, nj, nw, y1 = dims(level)
    bnd = boundary("0012", te)
    from aeromesh.mesh import o_mesh

    def f(d):
        return o_mesh(bnd, y1, radius=d, n_layers=nj - 1, n_wall=ni - 1,
                      max_growth=1.8)
    mb, d, got = tuned(f, R_TARGET)
    return mb, y1


def std(mesher, level, te="sharp"):
    mb, dt, d, got, y1 = build("0012", te, mesher, level,
                               WORK / "meshes")
    return mb, y1


A10 = (np.cos(np.deg2rad(10.0)), np.sin(np.deg2rad(10.0)))
A15 = (np.cos(np.deg2rad(15.0)), np.sin(np.deg2rad(15.0)))

# name: (builder, alpha, level, note)
VARIANTS = {
    # 14.1 item 1 -- the C/O lift gap
    "base_L2_a10": (lambda: am_c(2), 10, 2, "control: standard C, L2"),
    "wake10_L2_a10": (lambda: am_c(2, wake_axis=A10), 10, 2,
                      "C with the wake line along the freestream"),
    "snap_L2_a10": (lambda: am_snap(2), 10, 2,
                    "C, outer row slid onto the C-shape at exactly 500c"),
    "bluntC_L2_a10": (lambda: am_c(2, "blunt"), 10, 2, "AeroMesh C, blunt"),
    "bluntO_L2_a10": (lambda: am_o(2, "blunt"), 10, 2, "AeroMesh O, blunt"),
    "bluntO_L3_a10": (lambda: am_o(3, "blunt"), 10, 3, "AeroMesh O, blunt"),
    "bluntC_L3_a10": (lambda: am_c(3, "blunt"), 10, 3, "AeroMesh C, blunt"),
    "coarsewake_L2_a10": (lambda: am_c(2, aft_divergence=1.0), 10, 2,
                          "C, wake first cell grows at the full rate"),
    # item 4 -- the blunt base: wall graded into the corner, base refined,
    # wake widening measured against the default TE spacing
    "baseA_L2_a10": (lambda: am_c(2, "blunt", te_aspect=8, base_aspect=3,
                                  spread_ref="te", volume_smoothing=1.0),
                     10, 2, "blunt C, TE 8 y1, base 3 y1"),
    "baseB_L2_a10": (lambda: am_c(2, "blunt", te_aspect=4, base_aspect=1.5,
                                  spread_ref="te", volume_smoothing=2.0),
                     10, 2, "blunt C, TE 4 y1, base 1.5 y1"),
    "baseC_L2_a10": (lambda: am_c(2, "blunt", te_aspect=6, base_aspect=3,
                                  spread_ref="te", volume_smoothing=1.0),
                     10, 2, "blunt C, TE 6 y1, base 3 y1"),
    # item 2 -- soft cap on the wake's first-cell growth
    "cap_L1_a10": (lambda: am_c(1, wake_soft_cap=3.0), 10, 1,
                   "C, wake_soft_cap 3"),
    "cap_L2_a10": (lambda: am_c(2, wake_soft_cap=3.0), 10, 2,
                   "C, wake_soft_cap 3"),
    "pyhyp_L2_a10": (lambda: std("pyHyp O", 2), 10, 2, "pyHyp O, L2"),
    "c2d_L2_a10": (lambda: std("Construct2D C", 2), 10, 2,
                   "Construct2D C, L2"),
    # item 3 -- asymptotic range, ratio sqrt 2
    "base_L1.5_a10": (lambda: am_c(1.5), 10, 1.5, "standard C, L1.5"),
    "base_L2.5_a10": (lambda: am_c(2.5), 10, 2.5, "standard C, L2.5"),
    "base_L1_a10": (lambda: am_c(1), 10, 1, "standard C, L1"),
    "base_L3_a10": (lambda: am_c(3), 10, 3, "standard C, L3"),
    # item 6 -- 15 degrees
    "base_L2_a15": (lambda: am_c(2), 15, 2, "standard C, L2"),
    "base_L3_a15": (lambda: am_c(3), 15, 3, "standard C, L3"),
    "pyhyp_L2_a15": (lambda: std("pyHyp O", 2), 15, 2, "pyHyp O, L2"),
    "c2d_L2_a15": (lambda: std("Construct2D C", 2), 15, 2,
                   "Construct2D C, L2"),
    "snap_L2_a15": (lambda: am_snap(2), 15, 2, "snapped C, L2"),
}


def cores_for(level):
    return 1 if level <= 2.5 else 4


def launch(name):
    builder, alpha, level, note = VARIANTS[name]
    mb, y1 = builder()
    mesh, m = measure(mb, y1)
    case = WORK / "cases" / name
    n_iter = {1: 3000, 1.5: 3000, 2: 4000, 2.5: 5000, 3: 6000}[level]
    write_case(mesh, case, reynolds=RE, alpha_deg=alpha, n_iter=n_iter)
    m["checkMesh"] = check_mesh(case)
    m.update(far_distance=far_distance(mb), y1=y1, level=level,
             alpha=alpha, note=note)
    m.pop("profile", None)
    nc = cores_for(level)
    if nc > 1:
        (case / "system" / "decomposeParDict").write_text(
            _HEAD.format(cls="dictionary", obj="decomposeParDict")
            + f"numberOfSubdomains {nc};\nmethod scotch;\n")
        cmd = (f"decomposePar -force > log.decompose 2>&1 && mpirun -np {nc} "
               f"foamRun -parallel && reconstructPar -latestTime "
               f"> log.reconstruct 2>&1")
    else:
        cmd = "foamRun"
    (case / "meta.json").write_text(json.dumps(m, indent=1))
    print(f"{name:16s} cells {m['cells']:7d} R {m['far_distance']:.1f} "
          f"BLskew {m['bl']['skew_max']:.2f} fv {m['fv_max']:.1f} foam "
          f"{m['checkMesh']['nonorth_max']} ok {m['checkMesh']['ok']}",
          flush=True)
    return foam(f"( {cmd} ) > log.foamRun 2>&1 ; touch DONE", case,
                background=True)


# ---------------------------------------------------------------- vortex
def _faces(case):
    poly = Path(case) / "constant" / "polyMesh"
    pts = _list((poly / "points").read_text())
    faces = np.array([list(map(int, f.split())) for f in
                      re.findall(r"4\(([^)]*)\)", (poly / "faces")
                                 .read_text())])
    btxt = (poly / "boundary").read_text()
    out = {}
    for name, nf, sf in re.findall(
            r"\n\s*(\w+)\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);",
            btxt):
        f = faces[int(sf):int(sf) + int(nf)]
        out[name] = pts[f].mean(axis=1)[:, :2]
    return out


def vortex_values(xy, alpha, cl):
    a = np.deg2rad(alpha)
    gam = 0.5 * cl
    r = xy - np.array([0.25, 0.0])
    r2 = np.einsum("ij,ij->i", r, r)
    u = np.cos(a) + gam * r[:, 1] / (2 * np.pi * r2)
    v = np.sin(a) - gam * r[:, 0] / (2 * np.pi * r2)
    return np.column_stack([u, v]), 0.5 * (1.0 - u * u - v * v)


def _set_patch(text, patch, key, values):
    i = text.index(f"\n    {patch}\n")
    j = text.index("\n    }", i)
    blk = text[i:j]
    if values.ndim == 2:
        body = "\n".join(f"({a:.12g} {b:.12g} 0)" for a, b in values)
        lst = f"nonuniform List<vector>\n{len(values)}\n(\n{body}\n)"
    else:
        body = "\n".join(f"{a:.12g}" for a in values)
        lst = f"nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)"
    blk = re.sub(rf"{key}\s+(uniform[^;]*|nonuniform.*?\n\))\s*;",
                 lambda _: f"{key} {lst};", blk, count=1, flags=re.S)
    return text[:i] + blk + text[j:]


def vortex(label, n_iter=1500):
    case = WORK / "cases" / label
    meta = json.loads((case / "meta.json").read_text())
    cols, h = read_coeffs(case)
    cl = float(h[-200:, cols.index("Cl")].mean())
    tdirs = [d for d in case.iterdir() if d.is_dir() and re.fullmatch(
        r"[0-9.]+", d.name) and float(d.name) > 0]
    t = max(tdirs, key=lambda d: float(d.name))
    fc = _faces(case)
    far = [p for p in fc if p not in ("wall", "frontAndBack")]
    for fld in ("U", "p"):
        txt = (t / fld).read_text()
        for p in far:
            uv, pp = vortex_values(fc[p], meta["alpha"], cl)
            txt = _set_patch(txt, p, "freestreamValue",
                             uv if fld == "U" else pp)
        (t / fld).write_text(txt)
    ctl = case / "system" / "controlDict"
    txt = ctl.read_text()
    end = int(float(t.name)) + int(n_iter)
    txt = re.sub(r"endTime\s+\d+;", f"endTime         {end};", txt)
    # the first match only: the forceCoeffs function has its own interval
    txt = re.sub(r"writeInterval\s+\d+;", f"writeInterval   {n_iter};", txt,
                 count=1)
    ctl.write_text(txt)
    hist = meta.setdefault("vortex_updates", [])
    hist.append({"from_iter": int(float(t.name)), "cl_used": cl})
    (case / "meta.json").write_text(json.dumps(meta, indent=1))
    (case / "DONE").unlink(missing_ok=True)
    print(label, "vortex from Cl", round(cl, 5), "at", t.name, flush=True)
    return foam(f"foamRun > log.vortex{len(hist)} 2>&1 ; touch DONE", case,
                background=True)


def yplus_regions(case):
    """y+ of the wall faces, split into the base (faces whose outward normal
    points downstream, x > 0.99) and the rest."""
    from surface import _latest, _patch_field
    t = _latest(case)
    yp = _patch_field((t / "yPlus").read_text(), "wall")
    poly = Path(case) / "constant" / "polyMesh"
    pts = _list((poly / "points").read_text())
    faces = np.array([list(map(int, f.split())) for f in re.findall(
        r"4\(([^)]*)\)", (poly / "faces").read_text())])
    m = re.search(r"\bwall\s*\{[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);",
                  (poly / "boundary").read_text())
    nf, sf = int(m.group(1)), int(m.group(2))
    f = faces[sf:sf + nf]
    fc = pts[f].mean(axis=1)[:, :2]
    t_ = pts[f[:, 1], :2] - pts[f[:, 0], :2]
    base = (fc[:, 0] > 0.99) & (np.abs(t_[:, 1]) > 3 * np.abs(t_[:, 0]))
    out = {"wall_mean": float(yp[~base].mean()), "wall_max": float(
        yp[~base].max()), "n_base": int(base.sum())}
    if base.any():
        out.update(base_mean=float(yp[base].mean()),
                   base_max=float(yp[base].max()))
    return out


# ---------------------------------------------------------------- collect
def collect():
    out = {}
    for case in sorted((WORK / "cases").iterdir()):
        mp = case / "meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        cols, h = read_coeffs(case)
        if h.size == 0:
            continue
        ic, il = cols.index("Cd"), cols.index("Cl")
        rec = {k: meta[k] for k in ("note", "level", "alpha", "cells", "y1",
                                    "far_distance", "fv_max", "skew_max",
                                    "checkMesh", "bl", "wall")
               if k in meta}
        rec["iterations"] = int(h[-1, 0])
        rec["done"] = (case / "DONE").exists()
        upd = meta.get("vortex_updates", [])
        # forces before the first vortex update, and at the end
        i0 = (np.searchsorted(h[:, 0], upd[0]["from_iter"]) if upd
              else len(h))
        pre = h[max(0, i0 - 200):i0]
        rec["cl_free"] = float(pre[:, il].mean())
        rec["cd_free"] = float(pre[:, ic].mean())
        rec["drift_free"] = float(abs(pre[:, ic].mean() - h[max(
            0, i0 - 400):max(1, i0 - 200), ic].mean()))
        if upd:
            post = h[i0 + 1:]
            tail = post[-200:] if len(post) >= 200 else post[-1:]
            rec["cl_vortex"] = float(tail[:, il].mean())
            rec["cd_vortex"] = float(tail[:, ic].mean())
            rec["vortex_updates"] = upd
        try:
            rec["yplus_regions"] = yplus_regions(case)
        except Exception as err:  # noqa: BLE001
            rec["yplus_error"] = str(err)[:200]
        try:
            s = read_surface(case)
            rec["surface"] = {k: (None if v is None else np.asarray(v)
                                  .tolist()) for k, v in s.items()}
        except Exception as err:  # noqa: BLE001
            rec["surface_error"] = str(err)[:200]
        out[case.name] = rec
        print(f"{case.name:16s} it {rec['iterations']:5d} "
              f"{'done' if rec['done'] else 'RUN '} Cl {rec['cl_free']:.4f} "
              f"Cd {rec['cd_free'] * 1e4:.1f}"
              + (f" | vortex Cl {rec['cl_vortex']:.4f} Cd "
                 f"{rec['cd_vortex'] * 1e4:.1f}" if upd else ""), flush=True)
    (HERE / "results" / "study_next.json").write_text(json.dumps(out))


if __name__ == "__main__":
    WORK = Path(sys.argv[1]).expanduser()
    (WORK / "cases").mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[2]
    if cmd == "build":
        procs = [launch(n) for n in sys.argv[3:]]
        print("started", len(procs), flush=True)
    elif cmd == "vortex":
        vortex(sys.argv[3], *(int(a) for a in sys.argv[4:5]))
    elif cmd == "collect":
        collect()

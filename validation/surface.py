"""Wall pressure and skin-friction coefficients from a finished OpenFOAM case.

Read straight from the ASCII result files: ``p`` from the cells next to the
wall (their centres sit half a first-cell height off it, a few millionths of
a chord), the wall shear stress from the ``wallShearStress`` function object's
boundary values. Freestream speed 1, density 1, pressure 0, chord 1, so
``Cp = 2 p`` and ``Cf = 2 |tau| * sign``, the sign negative where the stress
points upstream along the local surface direction (reversed flow).

Reference data: NASA TMR, CFL3D, SA, 897 x 257 grid
(``validation/reference/n0012c{p,f}_cfl3d_sa.dat``).
"""
import re
from pathlib import Path

import numpy as np

REF = Path(__file__).parent / "reference"


def _list(text, kind=float):
    m = re.search(r"\n(\d+)\s*\n\(\s*\n", text)
    n = int(m.group(1))
    body = text[m.end():].split("\n)", 1)[0]
    if kind is int:
        return np.array(body.split()[:n], dtype=int)
    rows = re.findall(r"\(([^()]*)\)", body)
    if rows:
        return np.array([r.split() for r in rows[:n]], dtype=float)
    return np.array(body.split()[:n], dtype=float)


def _latest(case):
    times = [d for d in Path(case).iterdir()
             if d.is_dir() and re.fullmatch(r"[0-9.e+-]+", d.name)
             and float(d.name) > 0]
    return max(times, key=lambda d: float(d.name))


def _patch_field(text, patch):
    i = text.index(f"\n    {patch}\n")
    blk = text[i:text.index("\n    }", i)]
    m = re.search(r"value\s+nonuniform\s+List<(\w+)>\s*\n?(\d+)\s*\n?\(", blk)
    if m is None:
        return None
    rest = blk[m.end():]
    if m.group(1) == "vector":
        rows = re.findall(r"\(([^()]*)\)", rest)
        return np.array([r.split() for r in rows[:int(m.group(2))]],
                        dtype=float)
    return np.array(rest.split(")")[0].split()[:int(m.group(2))],
                    dtype=float)


def read_surface(case):
    case = Path(case)
    poly = case / "constant" / "polyMesh"
    pts = _list((poly / "points").read_text())
    faces_txt = (poly / "faces").read_text()
    faces = np.array([list(map(int, f.split())) for f in
                      re.findall(r"4\(([^)]*)\)", faces_txt)])
    owner = _list((poly / "owner").read_text(), int)
    btxt = (poly / "boundary").read_text()
    m = re.search(r"\bwall\s*\{[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);",
                  btxt)
    nf, sf = int(m.group(1)), int(m.group(2))
    wf = faces[sf:sf + nf]
    fc = pts[wf].mean(axis=1)[:, :2]
    p0, p1 = pts[wf[:, 0], :2], pts[wf[:, 1], :2]
    t_dir = p1 - p0
    t_dir /= np.linalg.norm(t_dir, axis=1)[:, None]
    cells = owner[sf:sf + nf]
    tdir = _latest(case)
    p = _list((tdir / "p").read_text().split("internalField", 1)[1]
              .replace("nonuniform List<scalar>", "", 1).join(["\n", ""]))
    cp = 2.0 * p[cells]
    tau = _patch_field((tdir / "wallShearStress").read_text(), "wall")
    cf = None
    if tau is not None:
        mag = np.linalg.norm(tau[:, :2], axis=1)
        # attached flow drags the wall downstream along the surface: on the
        # upper surface (traversed upstream by an outward-on-the-left face
        # orientation) that is -t, on the lower +t; reversed flow flips it
        along = np.einsum("ij,ij->i", -tau[:, :2], -t_dir)
        upper = fc[:, 1] >= 0.0
        sign = np.where(upper, np.sign(-along), np.sign(along))
        cf = 2.0 * mag * np.where(sign == 0, 1.0, sign)
    o = np.argsort(np.where(fc[:, 1] >= 0, -fc[:, 0], fc[:, 0] + 10.0))
    return {"x": fc[o, 0], "y": fc[o, 1], "cp": cp[o],
            "cf": None if cf is None else cf[o]}


def reference(kind, alpha):
    """``(x, value)`` of the NASA CFL3D curve; Cf is upper surface only."""
    text = (REF / f"n0012{kind}_cfl3d_sa.dat").read_text()
    zones = re.split(r'zone, t="', text)[1:]
    for z in zones:
        head, body = z.split('"', 1)
        if head.startswith(f"alpha={int(alpha)}"):
            v = np.array([ln.split() for ln in body.strip().splitlines()
                          if ln.strip() and not ln.startswith("#")],
                         dtype=float)
            return v[:, 0], v[:, 1]
    raise KeyError(alpha)

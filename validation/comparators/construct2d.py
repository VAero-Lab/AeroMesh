"""Construct2D (Prosser 2013, Furst 2026, GPL-3): hyperbolic or elliptic
single-block C- or O-grids for airfoils.

Driven in batch through its ``grid_options.in`` namelist and a piped menu,
then read back from its Plot3D file into an AeroMesh ``MultiBlock`` so every
mesh is measured by the same code.
"""
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from aeromesh.mesh.structured import Block, MultiBlock

from .geometry import section, write_selig

EXE = Path.home() / "MDOLab-Packages" / "construct2d" / "construct2d"


def _options(name, topo, nsrf, jmax, ypls, recd, radi, nwke, slvr, lesp,
             tesp):
    return f"""&SOPT
  nsrf = {nsrf}
  lesp = {lesp}
  tesp = {tesp}
  radi = {radi}
  nwke = {nwke}
  fdst = 1.0
  fwkl = 1.0
  fwki = 10.0
/
&VOPT
  name = '{name}'
  jmax = {jmax}
  slvr = '{slvr}'
  topo = '{topo}'
  ypls = {ypls}
  recd = {recd}
  stp1 = 1000
  stp2 = 20
  nrmt = 1
  nrmb = 1
  alfa = 1.0
  epsi = {15.0 if topo == 'CGRD' else 5.0}
  epse = 0.0
  funi = {0.0 if topo == 'CGRD' else 0.2}
  asmt = {20 if topo == 'CGRD' else 10}
/
&OOPT
  gdim = 2
  npln = 2
  dpln = 1.0
/
"""


def read_p3d(path):
    tok = Path(path).read_text().split()
    ni, nj = int(tok[0]), int(tok[1])
    v = np.array(tok[2:2 + 2 * ni * nj], dtype=float)
    x = v[:ni * nj].reshape(nj, ni).T
    y = v[ni * nj:].reshape(nj, ni).T
    return np.stack([x, y], axis=-1)


def mesh(name, te, topo="CGRD", nsrf=250, jmax=121, ypls=1.0, recd=6e6,
         radi=500.0, nwke=50, slvr="HYPR", lesp=4e-3, tesp=6e-4,
         workdir=None):
    """Returns ``(MultiBlock, seconds, log)``."""
    pts, _ = section(name, te)
    work = Path(workdir or tempfile.mkdtemp(prefix="c2d_"))
    work.mkdir(parents=True, exist_ok=True)
    write_selig(pts, work / "foil.dat", f"NACA {name}")
    (work / "grid_options.in").write_text(
        _options("foil", topo, nsrf, jmax, ypls, recd, radi, nwke, slvr,
                 lesp, tesp))
    t0 = time.time()
    run = subprocess.run([str(EXE), "foil.dat"], cwd=work, text=True,
                         input="GRID\nSMTH\nQUIT\n", capture_output=True,
                         timeout=600)
    dt = time.time() - t0
    (work / "log.txt").write_text(run.stdout + run.stderr)
    g = read_p3d(work / "foil.p3d")
    # Construct2D runs i over the surface (and wake) and j outward; make the
    # cells counter-clockwise for AeroMesh
    e1, e2 = g[1, 0] - g[0, 0], g[0, 1] - g[0, 0]
    if e1[0] * e2[1] - e1[1] * e2[0] < 0:
        g = g[::-1]
    ni = g.shape[0]
    if topo == "CGRD":
        # wake points on the cut coincide pairwise: i and ni-1-i
        w = 0
        while w < ni // 2 and np.allclose(g[w, 0], g[ni - 1 - w, 0],
                                          atol=1e-10):
            w += 1
        n_wake_edges = max(w - 1, 0)
        jmin = ([None] * n_wake_edges + ["wall"] * (ni - 1 - 2 * n_wake_edges)
                + [None] * n_wake_edges)
        blk = Block(g, {"jmin": jmin, "jmax": "farfield",
                        "imin": "outflow", "imax": "outflow"}, name="C2D-C")
    else:
        if not np.allclose(g[0], g[-1], atol=1e-10):
            g = np.concatenate([g, g[:1]], axis=0)
        if te == "blunt":
            # the base face is meshed as part of the O-grid wall
            pass
        blk = Block(g, {"jmin": "wall", "jmax": "farfield"}, name="C2D-O")
    return MultiBlock([blk]), dt, run.stdout

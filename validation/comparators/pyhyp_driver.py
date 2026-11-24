"""pyHyp (Secco et al. 2021, MDO Lab): hyperbolic O-grids from a supplied
surface. Run in a subprocess, because its Fortran module needs CGNS and PETSc
on ``LD_LIBRARY_PATH`` at process start.

pyHyp is given AeroMesh's own wall node distribution, so the two O-grids
differ only in how they march.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from aeromesh.mesh.structured import Block, MultiBlock

_PKG = Path.home() / "MDOLab-Packages"


def _env():
    import petsc
    env = dict(os.environ)
    libs = [str(_PKG / "cgns" / "lib"),
            str(Path(petsc.get_petsc_dir()) / "lib")]
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    return env


_RUNNER = r'''
import json, sys
import numpy as np
from pyhyp import pyHyp
cfg = json.load(open(sys.argv[1]))
cfg["options"]["BC"] = {int(k): v for k, v in cfg["options"]["BC"].items()}
hyp = pyHyp(options=cfg["options"])
hyp.run()
hyp.writePlot3D(cfg["out"])
'''


def write_surface(wall, path):
    """Plot3D surface: the closed wall loop extruded to two z planes."""
    x, y = wall[:, 0], wall[:, 1]
    with open(path, "w") as f:
        f.write("1\n%d %d %d\n" % (len(x), 2, 1))
        for d in range(3):
            for j in range(2):
                for i in range(len(x)):
                    v = x[i] if d == 0 else y[i] if d == 1 else float(j)
                    f.write("%.15g\n" % v)


def read_plot3d(path):
    tok = Path(path).read_text().split()
    nb = int(tok[0])
    ni, nj, nk = (int(v) for v in tok[1:4])
    v = np.array(tok[1 + 3 * nb:1 + 3 * nb + 3 * ni * nj * nk], dtype=float)
    n = ni * nj * nk
    x = v[:n].reshape(nk, nj, ni)
    y = v[n:2 * n].reshape(nk, nj, ni)
    z = v[2 * n:].reshape(nk, nj, ni)
    k0 = np.argmin(np.abs(z[0, :, 0]))           # the z = 0 plane
    return np.stack([x[:, k0, :].T, y[:, k0, :].T], axis=-1)   # (ni, N, 2)


def mesh(wall_ccw, first_cell, n_layers=121, march_dist=500.0,
         workdir=None, **extra):
    """``wall_ccw``: closed wall loop, counter-clockwise, first point not
    repeated. Returns ``(MultiBlock, seconds)``."""
    work = Path(workdir or tempfile.mkdtemp(prefix="pyhyp_"))
    work.mkdir(parents=True, exist_ok=True)
    wall = np.vstack([wall_ccw, wall_ccw[:1]])
    write_surface(wall, work / "surface.fmt")
    options = {
        "inputFile": str(work / "surface.fmt"),
        "unattachedEdgesAreSymmetry": False,
        "outerFaceBC": "farfield",
        "autoConnect": True,
        "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
        "families": "wall",
        "N": int(n_layers),
        "s0": float(first_cell),
        "marchDist": float(march_dist),
        "ps0": -1.0, "pGridRatio": -1.0, "cMax": 3.0,
        "epsE": 1.0, "epsI": 2.0, "theta": 3.0,
        "volCoef": 0.25, "volBlend": 0.0001, "volSmoothIter": 100,
    }
    options.update(extra)
    cfg = {"options": options, "out": str(work / "volume.xyz")}
    (work / "cfg.json").write_text(json.dumps(cfg))
    (work / "run.py").write_text(_RUNNER)
    t0 = time.time()
    r = subprocess.run([sys.executable, "run.py", "cfg.json"], cwd=work,
                       env=_env(), capture_output=True, text=True,
                       timeout=1800)
    dt = time.time() - t0
    (work / "log.txt").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"pyHyp failed:\n{r.stderr[-2000:]}")
    g = read_plot3d(work / "volume.xyz")
    e1, e2 = g[1, 0] - g[0, 0], g[0, 1] - g[0, 0]
    if e1[0] * e2[1] - e1[1] * e2[0] < 0:
        g = g[::-1]
    if not np.allclose(g[0], g[-1], atol=1e-10):
        g = np.concatenate([g, g[:1]], axis=0)
    return MultiBlock([Block(g, {"jmin": "wall", "jmax": "farfield"},
                             name="pyHyp-O")]), dt

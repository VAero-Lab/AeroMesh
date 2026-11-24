"""Mesh-generation time on an idle machine, and metrics of every domain shape.

    python validation/timing_domains.py <work_dir>

Each mesher is timed five times on NACA 0012 (sharp and blunt) with the same
settings as ``compare_meshers.py``; the median is reported. pyHyp and
Construct2D run as separate processes, so their times include process start-up
and file I/O; AeroMesh's include neither, and its I/O-free time is reported as
such. Run it with nothing else on the machine.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import aeromesh as am
from aeromesh.mesh import (c_mesh, domain_mesh, first_cell_height, o_mesh,
                           write_openfoam)
from aeromesh.quality.metrics import (assembled_nonorthogonality,
                                      cell_angles, mesh_report)

sys.path.insert(0, str(Path(__file__).parent))
from comparators import construct2d as c2d  # noqa: E402
from comparators import pyhyp_driver as phd  # noqa: E402

Y1 = first_cell_height(6e6, 1.0)
R = 500.0


def bnd(name, te):
    return am.build_region([am.load_airfoil(name, n_points=1000)],
                           farfield=20.0, te=te,
                           body_spacing=5e-4).boundaries[1]


def timed(f, n=5):
    ts = []
    out = None
    for _ in range(n):
        t0 = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


if __name__ == "__main__":
    work = Path(sys.argv[1]) / "timing"
    res = {"timing": {}, "domains": {}}
    for te in ("sharp", "blunt"):
        t0 = time.perf_counter()
        b = bnd("0012", te)
        t_geo = time.perf_counter() - t0
        t_c, mbc = timed(lambda: c_mesh(b, Y1, radius=R, wake_length=R,
                                        n_layers=140))
        t_o, mbo = timed(lambda: o_mesh(b, Y1, radius=R, n_layers=140))
        t_w, _ = timed(lambda: write_openfoam(mbc.assemble(),
                                              work / f"am_{te}"), n=1)
        wall = mbo.blocks[0].grid[:-1, 0]
        a = 0.5 * np.sum(wall[:, 0] * np.roll(wall[:, 1], -1)
                         - np.roll(wall[:, 0], -1) * wall[:, 1])
        wall = wall if a > 0 else wall[::-1]
        t_p, _ = timed(lambda: phd.mesh(wall, Y1, n_layers=141,
                                        march_dist=R,
                                        workdir=work / f"ph_{te}"))
        topo = "CGRD" if te == "sharp" else "OGRD"
        kw = (dict(nsrf=250, nwke=75, lesp=7.5e-4, tesp=3e-3)
              if topo == "CGRD" else dict(nsrf=250, lesp=4e-3, tesp=6e-4))
        t_k, _ = timed(lambda: c2d.mesh("0012", te, topo=topo, jmax=121,
                                        ypls=2.0, recd=6e6, radi=R,
                                        workdir=work / f"c2_{te}", **kw))
        res["timing"][te] = {
            "geometry_s": t_geo, "AeroMesh C": t_c, "AeroMesh O": t_o,
            "AeroMesh write polyMesh": t_w,
            "pyHyp O": t_p, f"Construct2D {topo[0]}": t_k,
            "cells_C": int(mbc.assemble().n_cells),
        }
        print(te, {k: round(v, 3) for k, v in res["timing"][te].items()},
              flush=True)

    for name, te in (("0012", "sharp"), ("4412", "blunt"), ("6409", "sharp"),
                     ("2424", "blunt")):
        b = bnd(name, te)
        for d in ("c", "box", "circle"):
            t0 = time.perf_counter()
            mb = domain_mesh(b, Y1, d, size=20.0)
            dt = time.perf_counter() - t0
            m = mb.assemble()
            rep = mesh_report(mb, Y1)
            lo_out, hi_out = cell_angles(mb.blocks[-1].grid)
            res["domains"][f"{name}_{te}_{d}"] = {
                "cells": m.n_cells, "seconds": dt,
                "blocks": [b_.name for b_ in mb.blocks],
                "min_angle": rep["min_angle"], "max_angle": rep["max_angle"],
                "outer_min_angle": float(lo_out.min()),
                "bl_skew_max": rep["bl"]["skew_max"],
                "y1_within_2pct": rep["wall"]["y1_within_2pct"],
                "fv_nonorth_max": float(assembled_nonorthogonality(m).max()),
            }
            print(name, te, d, res["domains"][f"{name}_{te}_{d}"]["min_angle"])
    out = Path(__file__).parent / "results" / "timing_domains.json"
    out.write_text(json.dumps(res, indent=1))
    print("written", out)

"""Mesh-only measurements for report section 14.1 (items 2 and 5).

    python validation/mesh_next.py

Writes ``results/mesh_next.json``: the standard C-mesh against the wake soft
cap (item 2), and three ways of making the far field exactly the supplied
500-chord C-shape (item 5), two of them rejected.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_standard import R_TARGET, measure  # noqa: E402
from study_next import am_c, am_snap, boundary, dims  # noqa: E402

from aeromesh.mesh import c_mesh  # noqa: E402
from aeromesh.mesh.farfield import c_domain  # noqa: E402


def row(label, level, fn):
    try:
        mb, y1 = fn()
        _, m = measure(mb, y1)
        r = dict(fv_max=m["fv_max"], fv_p99=m["fv_p99"],
                 skew_max=m["skew_max"], bl_skew=m["bl"]["skew_max"])
    except ValueError as err:
        r = {"error": str(err).split(" have")[0] + " inverted"}
    r.update(label=label, level=level)
    print(r, flush=True)
    return r


def march_then_tfi(level, inner, frac):
    ni, nj, nw, y1 = dims(level)
    n_in = int(round((nj - 1) * frac))
    cm = c_mesh(boundary("0012", "sharp"), y1, radius=inner,
                wake_length=R_TARGET, n_layers=n_in, n_wall=ni - 2 * nw,
                n_wake=nw, max_growth=1.8)
    return c_domain(cm, "c", R_TARGET), y1


def thin_band(level, rin, n_out):
    ni, nj, nw, y1 = dims(level)
    cm = c_mesh(boundary("0012", "sharp"), y1, radius=rin * R_TARGET,
                wake_length=R_TARGET, n_layers=nj - 1 - n_out,
                n_wall=ni - 2 * nw, n_wake=nw, max_growth=1.8)
    return c_domain(cm, "c", R_TARGET, n_layers=n_out), y1


rows = []
for L in (1, 2):
    rows.append(row("standard C (march, tuned)", L, lambda: am_c(L)))
    rows.append(row("wake soft cap 10", L, lambda: am_c(L, wake_soft_cap=10.0)))
    rows.append(row("wake soft cap 3", L, lambda: am_c(L, wake_soft_cap=3.0)))
rows.append(row("march to 3c + TFI to 500c", 2,
                lambda: march_then_tfi(2, 3.0, 0.5)))
rows.append(row("march to 25c + TFI to 500c", 2,
                lambda: march_then_tfi(2, 25.0, 0.5)))
rows.append(row("march to 250c + 10-row TFI band", 2,
                lambda: thin_band(2, 0.5, 10)))
rows.append(row("march + snap along grid lines", 2, lambda: am_snap(2)))
Path(__file__).parent.joinpath("results", "mesh_next.json").write_text(
    json.dumps({"rows": rows}, indent=1))

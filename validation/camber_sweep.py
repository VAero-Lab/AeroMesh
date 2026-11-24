"""The claim, measured: do the boundary-layer metrics hold as the section changes?

Three ways of getting a mesh for each section of a camber sweep:

``regenerated``  AeroMesh re-derives and re-marches the C-mesh for the section.
``deformed``     one baseline mesh is warped onto the section by inverse-
                 distance weighting **with rotations** (Luke, Collins & Blades
                 2012 -- the algorithm behind IDWarp), which is what a mesh
                 movement package does. Rotations matter: without them the
                 comparison would be unfair to deformation, because near-wall
                 lines would not turn with the surface at all.

Every mesh is measured by the same ``mesh_report`` and, independently, by
OpenFOAM's ``checkMesh``.

    python validation/camber_sweep.py <work_dir>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import aeromesh as am
from aeromesh.mesh import (Block, MultiBlock, c_mesh, first_cell_height,
                           wall_segments, write_openfoam)
from aeromesh.quality.metrics import mesh_report

sys.path.insert(0, str(Path(__file__).parent))
from foam_case import check_mesh  # noqa: E402

RE = 6e6
CAMBER = [f"{m}412" for m in range(0, 10)]
THICKNESS = ["2406", "2409", "2412", "2415", "2418", "2421", "2424"]
OTHERS = ["23012", "63012", "6409", "4415"]


def build(name, te):
    body = am.load_airfoil(name, n_points=1000)
    region = am.build_region([body], farfield=20.0, te=te, body_spacing=5e-4)
    return region.boundaries[1]


def keep(rep):
    out = {k: v for k, v in rep["wall"].items() if not k.startswith("_")}
    out.update({f"bl_{k}": v for k, v in rep["bl"].items()})
    out["min_angle_all"] = rep["min_angle"]
    out["max_angle_all"] = rep["max_angle"]
    out["cells"] = rep["cells"]
    return out


def referee(mb, work, label):
    case = work / "sweep" / label
    write_openfoam(mb.assemble(), case)
    (case / "system").mkdir(exist_ok=True)
    (case / "system" / "controlDict").write_text(
        "FoamFile{format ascii; class dictionary; object controlDict;}\n"
        "application foamRun; startFrom startTime; startTime 0; "
        "stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; "
        "writeInterval 1;\n")
    for f in ("fvSchemes", "fvSolution"):
        (case / "system" / f).write_text(
            f"FoamFile{{format ascii; class dictionary; object {f};}}\n")
    c = check_mesh(case)
    return {"foam_nonorth_max": c["nonorth_max"],
            "foam_nonorth_avg": c["nonorth_avg"],
            "foam_skewness_max": c["max_skewness"]}


def idw_deform(mb, wall_mask, new_wall, a=3.0, b=5.0, alpha=0.25):
    """Warp a C-mesh so its wall nodes land on ``new_wall``.

    Sources are the wall nodes (displacement and rotation known) and the outer
    boundary nodes (held). Each source proposes a rigid motion; a volume node
    takes the weighted mean with ``w = A [(L/d)^a + (alpha L/d)^b]``.
    """
    g = mb.blocks[0].grid
    ni = g.shape[0]
    wi = np.where(wall_mask)[0]
    old = g[wi, 0]
    disp = new_wall - old

    def angles(p):
        t = np.gradient(p, axis=0)
        return np.unwrap(np.arctan2(t[:, 1], t[:, 0]))

    rot = angles(new_wall) - angles(old)
    outer = np.vstack([g[:, -1], g[0, :], g[-1, :]])
    src = np.vstack([old, outer])
    d_src = np.vstack([disp, np.zeros_like(outer)])
    r_src = np.concatenate([rot, np.zeros(len(outer))])
    area = np.concatenate([
        np.linalg.norm(np.gradient(old, axis=0), axis=1),
        np.linalg.norm(np.gradient(g[:, -1], axis=0), axis=1),
        np.linalg.norm(np.gradient(g[0, :], axis=0), axis=1),
        np.linalg.norm(np.gradient(g[-1, :], axis=0), axis=1)])
    L = float(np.max(np.linalg.norm(old - old.mean(axis=0), axis=1)))
    cs, sn = np.cos(r_src), np.sin(r_src)

    def warp(pts):
        out = np.empty_like(pts)
        for k0 in range(0, len(pts), 4000):
            p = pts[k0:k0 + 4000]
            rel = p[:, None, :] - src[None, :, :]
            d = np.maximum(np.linalg.norm(rel, axis=2), 1e-14)
            w = area[None, :] * ((L / d) ** a + (alpha * L / d) ** b)
            rx = cs * rel[..., 0] - sn * rel[..., 1]
            ry = sn * rel[..., 0] + cs * rel[..., 1]
            sx = rx - rel[..., 0] + d_src[:, 0]
            sy = ry - rel[..., 1] + d_src[:, 1]
            ws = w.sum(axis=1)
            out[k0:k0 + 4000, 0] = p[:, 0] + (w * sx).sum(axis=1) / ws
            out[k0:k0 + 4000, 1] = p[:, 1] + (w * sy).sum(axis=1) / ws
        return out

    blocks = []
    for blk in mb.blocks:
        new = warp(blk.grid.reshape(-1, 2)).reshape(blk.grid.shape)
        blocks.append(Block(new, blk.sides, blk.name))
    # exact on the wall
    blocks[0].grid[wi, 0] = new_wall
    return MultiBlock(blocks)


def wall_mask_of(mb):
    tag = mb.blocks[0].sides["jmin"]
    e = np.array([t == "wall" for t in tag])
    node = np.zeros(len(e) + 1, bool)
    node[:-1] |= e
    node[1:] |= e
    return node


def target_wall(boundary, base_wall):
    """Nodes of the target wall at the baseline's arc-length fractions."""
    segs, _ = wall_segments(boundary)
    seg = segs[int(np.argmax([s.length for s in segs]))]
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(
        np.diff(base_wall, axis=0), axis=1))))
    return seg(s / s[-1] * seg.length)


if __name__ == "__main__":
    work = Path(sys.argv[1])
    y1 = first_cell_height(RE, 1.0)
    results = {"y1": y1, "regenerated": {}, "deformed": {}, "battery": {}}

    for te in ("sharp", "blunt"):
        for name in CAMBER:
            t0 = time.time()
            bnd = build(name, te)
            t1 = time.time()
            mb = c_mesh(bnd, y1)
            mb.assemble()
            t2 = time.time()
            rep = keep(mesh_report(mb, y1))
            rep.update(referee(mb, work, f"regen_{te}_{name}"))
            rep["seconds_geometry"] = t1 - t0
            rep["seconds_mesh"] = t2 - t1
            rep["blocks"] = [list(b.shape) for b in mb.blocks]
            results["regenerated"][f"{te}_{name}"] = rep
            print("regen", te, name, f"BL skew max {rep['bl_skew_max']:.2f}",
                  f"wall face {rep['wall_face_nonorth_max']:.2f}",
                  f"foam {rep['foam_nonorth_max']:.1f}", flush=True)

    # deformation from two baselines, sharp trailing edge
    for base in ("0012", "4412"):
        mb0 = c_mesh(build(base, "sharp"), y1)
        mask = wall_mask_of(mb0)
        base_wall = mb0.blocks[0].grid[mask, 0]
        for name in CAMBER:
            new_wall = target_wall(build(name, "sharp"), base_wall)
            mbd = idw_deform(mb0, mask, new_wall)
            try:
                mbd.assemble()
                rep = keep(mesh_report(mbd, y1))
                rep.update(referee(mbd, work, f"deform_{base}_{name}"))
                rep["valid"] = True
            except ValueError as err:
                rep = {"valid": False, "error": str(err)}
            results["deformed"][f"{base}_to_{name}"] = rep
            if rep["valid"]:
                print("deform", base, "->", name,
                      f"BL skew max {rep['bl_skew_max']:.2f}",
                      f"wall face {rep['wall_face_nonorth_max']:.2f}",
                      f"y1 {rep['y1_min_ratio']:.3f}-{rep['y1_max_ratio']:.3f}",
                      f"foam {rep['foam_nonorth_max']:.1f}", flush=True)
            else:
                print("deform", base, "->", name, "INVALID", rep["error"])

    for te in ("sharp", "blunt"):
        for name in THICKNESS + OTHERS:
            mb = c_mesh(build(name, te), y1)
            mb.assemble()
            rep = keep(mesh_report(mb, y1))
            rep.update(referee(mb, work, f"bat_{te}_{name}"))
            results["battery"][f"{te}_{name}"] = rep
            print("battery", te, name, f"BL skew max {rep['bl_skew_max']:.2f}",
                  f"min angle {rep['min_angle_all']:.1f}", flush=True)

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "camber_sweep.json").write_text(json.dumps(results, indent=1))
    print("written", out / "camber_sweep.json")

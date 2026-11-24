"""Mesh export.

``write_meshio`` goes through ``meshio`` (SU2, VTK, CGNS, ... by extension).

``write_openfoam`` writes a ``polyMesh`` directly, because meshio has no writer
for it: the planar mesh is extruded one cell in ``z`` and the two ``z`` faces
form an ``empty`` patch, which is how OpenFOAM runs a two-dimensional case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_HEADER = """FoamFile
{{
    format      ascii;
    class       {cls};
    location    "constant/polyMesh";
    object      {obj};
}}
"""


def write_meshio(mesh, path) -> None:
    """Write through meshio; boundary edges carry their patch as a tag."""
    import meshio
    pts = np.column_stack([mesh.points, np.zeros(len(mesh.points))])
    names = sorted(mesh.patches)
    lines = np.concatenate([mesh.patches[n] for n in names])
    tags = np.concatenate([np.full(len(mesh.patches[n]), k + 1)
                           for k, n in enumerate(names)])
    path = Path(path)
    if path.suffix == ".su2":
        # written directly: meshio 5.3's SU2 writer fails on its own CellBlock
        out = ["NDIME= 2", f"NELEM= {len(mesh.quads)}"]
        out += [f"9 {q[0]} {q[1]} {q[2]} {q[3]} {k}"
                for k, q in enumerate(mesh.quads)]
        out.append(f"NPOIN= {len(mesh.points)}")
        out += [f"{p[0]:.16g} {p[1]:.16g} {k}"
                for k, p in enumerate(mesh.points)]
        out.append(f"NMARK= {len(names)}")
        for name in names:
            ed = mesh.patches[name]
            out += [f"MARKER_TAG= {name}", f"MARKER_ELEMS= {len(ed)}"]
            out += [f"3 {a} {b}" for a, b in ed]
        path.write_text("\n".join(out) + "\n")
    else:
        meshio.write(path, meshio.Mesh(pts, [("quad", mesh.quads)]))


def write_openfoam(mesh, case_dir, span: float = 1.0,
                   patch_types: dict | None = None) -> dict:
    """Write ``constant/polyMesh`` for a one-cell-thick extrusion.

    Returns ``{patch: n_faces}``.
    """
    patch_types = patch_types or {}
    n = len(mesh.points)
    q = mesh.quads
    n_cells = len(q)

    # --- edges -> owner / neighbour
    e = np.stack([q, np.roll(q, -1, axis=1)], axis=-1).reshape(-1, 2)
    cell = np.repeat(np.arange(n_cells), 4)
    key = np.sort(e, axis=1)
    order = np.lexsort((cell, key[:, 1], key[:, 0]))
    key_s, cell_s, e_s = key[order], cell[order], e[order]
    same = np.all(key_s[1:] == key_s[:-1], axis=1)
    first = np.where(same)[0]                      # internal: pairs (k, k+1)
    owner_i = cell_s[first]                        # lower cell id (lexsort)
    neigh_i = cell_s[first + 1]
    edge_i = e_s[first]                            # as traversed by the owner
    # upper-triangular order
    o = np.lexsort((neigh_i, owner_i))
    owner_i, neigh_i, edge_i = owner_i[o], neigh_i[o], edge_i[o]

    def quad_face(a, b):                            # normal to the right of a->b
        return np.column_stack([a, b, b + n, a + n])

    faces = [quad_face(edge_i[:, 0], edge_i[:, 1])]
    owners = [owner_i]

    # boundary: owner cell of each directed boundary edge
    is_single = np.ones(len(key_s), bool)
    is_single[first] = False
    is_single[first + 1] = False
    lookup = {tuple(k): c for k, c in zip(map(tuple, e_s[is_single]),
                                          cell_s[is_single])}
    boundary = []
    start = len(owner_i)
    for name in sorted(mesh.patches):
        edges = mesh.patches[name]
        own = np.array([lookup[tuple(ed)] for ed in edges])
        o = np.argsort(own, kind="stable")
        faces.append(quad_face(edges[o, 0], edges[o, 1]))
        owners.append(own[o])
        boundary.append((name, patch_types.get(
            name, "wall" if name == "wall" else "patch"), len(edges), start))
        start += len(edges)
    # back (z = 0, normal -z) and front (z = span, normal +z)
    faces.append(q[:, ::-1])
    owners.append(np.arange(n_cells))
    faces.append(q + n)
    owners.append(np.arange(n_cells))
    boundary.append(("frontAndBack", "empty", 2 * n_cells, start))

    faces = np.concatenate(faces)
    owners = np.concatenate(owners)

    poly = Path(case_dir) / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    pts3 = np.vstack([np.column_stack([mesh.points, np.zeros(n)]),
                      np.column_stack([mesh.points, np.full(n, span)])])

    def dump(name, cls, body, note=None):
        head = _HEADER.format(cls=cls, obj=name)
        if note:
            head = head.replace("    object", f"    note        \"{note}\";\n"
                                               f"    object")
        (poly / name).write_text(head + body)

    dump("points", "vectorField",
         f"{len(pts3)}\n(\n" + "\n".join(
             f"({p[0]:.15g} {p[1]:.15g} {p[2]:.15g})" for p in pts3) + "\n)\n")
    dump("faces", "faceList",
         f"{len(faces)}\n(\n" + "\n".join(
             f"4({f[0]} {f[1]} {f[2]} {f[3]})" for f in faces) + "\n)\n")
    note = (f"nPoints: {len(pts3)} nCells: {n_cells} nFaces: {len(faces)} "
            f"nInternalFaces: {len(owner_i)}")
    dump("owner", "labelList",
         f"{len(owners)}\n(\n" + "\n".join(map(str, owners)) + "\n)\n", note)
    dump("neighbour", "labelList",
         f"{len(neigh_i)}\n(\n" + "\n".join(map(str, neigh_i)) + "\n)\n", note)
    body = f"{len(boundary)}\n(\n"
    for name, typ, nf, st in boundary:
        body += (f"    {name}\n    {{\n        type {typ};\n"
                 f"        nFaces {nf};\n        startFace {st};\n    }}\n")
    dump("boundary", "polyBoundaryMesh", body + ")\n")
    return {b[0]: b[2] for b in boundary}

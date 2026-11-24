"""The multiblock mesh container and its assembly into one conformal mesh.

A :class:`Block` is a structured grid ``(ni, nj, 2)`` whose four sides carry a
tag: the name of a boundary patch, or ``None`` for a side shared with another
block (or with itself, across a periodic seam or a wake cut). A side may also
carry one tag per edge, because a C-mesh's ``jmin`` side is wall over the body
and interior along the wake cut.

:meth:`MultiBlock.assemble` merges coincident nodes and then *proves* the
result is a closed quad complex, exactly, with no tolerance on area:

* every cell has positive area;
* every edge is used by one cell or by two;
* an edge used by one cell carries a patch tag, and an edge used by two
  carries none.

The third is the check that a missing block, a crack or an unglued interface
cannot pass: an interior side that pairs with nobody is single-owned and
untagged, and that raises. It is the structured-grid form of the
``closure_report`` written after the October audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

SIDES = ("imin", "imax", "jmin", "jmax")


@dataclass
class Block:
    grid: np.ndarray
    sides: dict = field(default_factory=dict)
    name: str = ""

    @property
    def shape(self):
        return self.grid.shape[:2]

    def side_edges(self, side: str) -> np.ndarray:
        """Local (i, j) index pairs of the edges on one side, shape (m, 2, 2)."""
        ni, nj = self.shape
        if side == "imin":
            a = np.column_stack([np.zeros(nj, int), np.arange(nj)])
        elif side == "imax":
            a = np.column_stack([np.full(nj, ni - 1), np.arange(nj)])
        elif side == "jmin":
            a = np.column_stack([np.arange(ni), np.zeros(ni, int)])
        elif side == "jmax":
            a = np.column_stack([np.arange(ni), np.full(ni, nj - 1)])
        else:
            raise KeyError(side)
        return np.stack([a[:-1], a[1:]], axis=1)


@dataclass
class Mesh:
    """An assembled, conformal, unstructured view of a multiblock mesh."""
    points: np.ndarray            # (N, 2)
    quads: np.ndarray             # (M, 4), counter-clockwise
    patches: dict                 # name -> (K, 2) node ids, owner cell on left
    block_nodes: list             # per block, (ni, nj) node ids
    block_cells: list             # per block, (ni-1, nj-1) cell ids
    blocks: list

    @property
    def n_cells(self) -> int:
        return len(self.quads)


class MultiBlock:
    def __init__(self, blocks):
        self.blocks = list(blocks)

    def assemble(self, tol: float = 1e-11) -> Mesh:
        pts = np.concatenate([b.grid.reshape(-1, 2) for b in self.blocks])
        # --- merge coincident nodes (union-find over KD-tree pairs)
        parent = np.arange(len(pts))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in cKDTree(pts).query_pairs(tol):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        roots = np.array([find(i) for i in range(len(pts))])
        uniq, new_id = np.unique(roots, return_inverse=True)
        points = pts[uniq]

        block_nodes, block_cells, quads = [], [], []
        off = 0
        for b in self.blocks:
            ni, nj = b.shape
            ids = new_id[off:off + ni * nj].reshape(ni, nj)
            off += ni * nj
            block_nodes.append(ids)
            q = np.stack([ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:],
                          ids[:-1, 1:]], axis=-1).reshape(-1, 4)
            block_cells.append((len(np.concatenate(quads)) if quads else 0)
                               + np.arange(len(q)).reshape(ni - 1, nj - 1))
            quads.append(q)
        quads = np.concatenate(quads)

        # --- every cell positive
        p = points[quads]
        area = 0.5 * np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, axis=1)
                            - np.roll(p[:, :, 0], -1, axis=1) * p[:, :, 1],
                            axis=1)
        if np.any(area <= 0.0):
            raise ValueError(f"{int(np.sum(area <= 0))} of {len(area)} cells "
                             f"have non-positive area")

        # --- edge ownership
        e = np.stack([quads, np.roll(quads, -1, axis=1)], axis=-1).reshape(-1, 2)
        key = np.sort(e, axis=1)
        _, inv, counts = np.unique(key, axis=0, return_inverse=True,
                                   return_counts=True)
        inv = inv.ravel()
        if counts.max() > 2:
            raise ValueError("an edge is shared by more than two cells")
        if np.any(key[:, 0] == key[:, 1]):
            raise ValueError("a cell has a collapsed edge")
        single = {tuple(k) for k in key[counts[inv] == 1]}
        directed = {tuple(np.sort(d)): tuple(d) for d in e[counts[inv] == 1]}

        # --- tags
        tagged = {}
        for b, ids in zip(self.blocks, block_nodes):
            for side in SIDES:
                tag = b.sides.get(side)
                if tag is None:
                    continue
                edges = b.side_edges(side)
                tags = [tag] * len(edges) if isinstance(tag, str) else list(tag)
                for (a0, a1), t in zip(edges, tags):
                    if t is None:
                        continue
                    k = tuple(sorted((ids[a0[0], a0[1]], ids[a1[0], a1[1]])))
                    tagged[k] = t
        loose = single - set(tagged)
        if loose:
            raise ValueError(f"{len(loose)} interior edges pair with nothing "
                             f"(a crack or a missing block)")
        buried = set(tagged) - single
        if buried:
            raise ValueError(f"{len(buried)} tagged boundary edges are shared "
                             f"by two cells")
        patches: dict = {}
        for k, t in tagged.items():
            patches.setdefault(t, []).append(directed[k])
        patches = {t: np.array(v) for t, v in patches.items()}
        return Mesh(points, quads, patches, block_nodes, block_cells,
                    self.blocks)

"""Report section 14.1: far field snapped onto the supplied curve, the blunt
base resolved, the wake's first-cell growth softly capped.

Each test names the measured defect it guards against.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.mesh import c_mesh, first_cell_height
from aeromesh.mesh.farfield import far_curve, snap_farfield
from aeromesh.quality.metrics import assembled_nonorthogonality, mesh_report

Y1 = first_cell_height(6e6, 1.0)


def boundary(te):
    body = am.load_airfoil("0012", n_points=1000)
    return am.build_region([body], farfield=20.0, te=te,
                           body_spacing=5e-4).boundaries[1]


def _dist_to_polyline(p, c):
    a, b = c[:-1], c[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab)
                / np.einsum("ij,ij->i", ab, ab), 0, 1)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))


@pytest.fixture(scope="module")
def sharp():
    return c_mesh(boundary("sharp"), Y1, radius=50.0, wake_length=50.0,
                  n_layers=90)


def test_snapped_outer_row_lies_on_the_supplied_curve(sharp):
    """A tuned march lands within 0.5 % of the far field, never on it."""
    s = snap_farfield(sharp, "c", 50.0, n_blend=10)
    g = s.blocks[0].grid
    xo = 0.5 * (g[0, 0, 0] + g[-1, 0, 0])
    cur = far_curve("c", xo, 50.0)
    d = max(_dist_to_polyline(p, cur) for p in g[:, -1])
    assert d < 1e-6 * 50.0
    s.assemble()                                       # no inverted cell


def test_snap_leaves_the_boundary_layer_untouched(sharp):
    """The correction is confined to the outer rows."""
    s = snap_farfield(sharp, "c", 50.0, n_blend=10)
    a, b = sharp.blocks[0].grid, s.blocks[0].grid
    assert np.array_equal(a[:, :-11], b[:, :-11])


def test_base_refinement_no_longer_fans_the_wake_lines():
    """Widening measured against the first wake spacing made the wake lines
    part 17x faster when the base cell was refined, and cells inverted at a
    base aspect of 8. Against the default TE spacing they stay valid down to
    1.5 first cells."""
    bnd = boundary("blunt")
    with pytest.raises(ValueError):
        c_mesh(bnd, Y1, radius=50.0, wake_length=50.0, n_layers=90,
               base_aspect=3.0, te_aspect=3.0).assemble()
    mb = c_mesh(bnd, Y1, radius=50.0, wake_length=50.0, n_layers=90,
                base_aspect=3.0, te_aspect=6.0, spread_ref="te",
                volume_smoothing=1.0)
    rep = mesh_report(mb, Y1)
    assert rep["bl"]["negative_cells"] == 0
    # the base's first cell is the wake's first spacing
    aft = mb.blocks[1].grid
    h_base = np.linalg.norm(aft[1, :] - aft[0, :], axis=1).max()
    assert h_base < 4.0 * Y1


def test_soft_cap_reduces_wake_nonorthogonality():
    """50.4 degrees on the level-1 C-mesh came from wake rows tapering at
    different rates; a soft cap on the wake's first-cell growth removes it."""
    bnd = boundary("sharp")
    kw = dict(radius=500.0, wake_length=500.0, n_layers=75, n_wall=112,
              n_wake=40, max_growth=1.8)
    y1 = 2.0 * Y1
    plain = assembled_nonorthogonality(c_mesh(bnd, y1, **kw).assemble())
    capped = assembled_nonorthogonality(
        c_mesh(bnd, y1, wake_soft_cap=3.0, **kw).assemble())
    assert capped.max() < plain.max() - 10.0

"""Marched single-body meshes: spacing, the marcher, topology and assembly.

Each test names the defect it exists to catch. Several of those defects were
found by OpenFOAM's ``checkMesh`` rather than by our own metrics, and the
checks here are the ones that would have caught them.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.mesh import (Block, MultiBlock, c_mesh, first_cell_height,
                           graded_parameters, layer_heights, march, o_mesh,
                           wall_segments)
from aeromesh.quality.metrics import (assembled_nonorthogonality,
                                      fv_nonorthogonality, mesh_report)

Y1 = first_cell_height(6e6, 1.0)
SECTIONS = ["0012", "4412", "8412", "2424", "23012"]


def boundary(name, te):
    body = am.load_airfoil(name, n_points=1000)
    return am.build_region([body], farfield=20.0, te=te,
                           body_spacing=5e-4).boundaries[1]


@pytest.fixture(scope="module")
def meshes():
    out = {}
    for name in SECTIONS:
        for te in ("sharp", "blunt"):
            mb = c_mesh(boundary(name, te), Y1)
            out[name, te] = (mb, mb.assemble())
    return out


# ---------------------------------------------------------------- spacing

def test_first_cell_height_scales_inversely_with_yplus_and_reynolds():
    assert first_cell_height(6e6, 2.0) == pytest.approx(2 * Y1)
    assert first_cell_height(6e7, 1.0) < Y1 / 7.0
    assert 3e-6 < Y1 < 6e-6


@pytest.mark.parametrize("ratio", [1.08, 1.15, 1.3])
def test_graded_nodes_never_exceed_the_ratio(ratio):
    """The audit's 36x along-wall jump came from a per-block node count. A
    size field limited in gradient cannot produce one."""
    t = np.linspace(0, 1, 3000)
    line = np.column_stack([t * 2.0, 0.05 * np.sin(6 * t)])
    s = graded_parameters(line, h_max=0.05, h_start=1e-4, h_end=3e-4,
                          ratio=ratio)
    ds = np.diff(s)
    r = ds[1:] / ds[:-1]
    assert max(r.max(), (1 / r).max()) <= ratio * 1.05
    assert ds[0] <= 1.2e-4 and ds[-1] <= 3.6e-4
    assert s[0] == 0.0 and s[-1] == pytest.approx(
        np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))


def test_layer_heights_start_exact_and_reach_the_distance():
    first = np.array([Y1, Y1, 5 * Y1])
    h = layer_heights(first, 120, growth=1.15, distance=50.0)
    assert h[0, 0] == pytest.approx(Y1, rel=1e-12)
    assert h[2, 0] == pytest.approx(5 * Y1, rel=1e-12)
    assert h[0].sum() == pytest.approx(50.0, rel=1e-6)
    g = h[0, 1:] / h[0, :-1]
    assert np.allclose(g[:30], 1.15)          # the boundary layer keeps its law
    assert g.max() <= 1.3 + 1e-9


# ---------------------------------------------------------------- marcher

def test_march_from_a_circle_gives_concentric_circles():
    """Closed form: every node moves radially by exactly its layer height."""
    n = 180
    th = -np.linspace(0, 2 * np.pi, n, endpoint=False)        # clockwise
    circle = np.column_stack([np.cos(th), np.sin(th)])
    h = np.tile(0.01 * 1.1 ** np.arange(40), (n, 1))
    g = march(circle, h)
    r = np.linalg.norm(g, axis=2)
    assert np.ptp(r, axis=0).max() < 1e-9
    step = np.diff(r[0]) / h[0]
    # thin cells take their prescribed height (without the curvature
    # correction the first step here is 0.9804 of the request) ...
    assert abs(step[0] - 1.0) < 5e-3
    # ... and thick ones are allowed to fall short: the far field has no
    # first-cell height to protect and the correction is switched off there.
    assert 0.8 < step.min() <= step.max() < 1.01


def test_march_orientation_gives_positive_cells():
    """The October run read 4872 of 4872 negative areas off a correct mesh;
    the convention is fixed here: region on the left, cells counter-clockwise."""
    n = 64
    th = -np.linspace(0, 2 * np.pi, n, endpoint=False)
    g = march(np.column_stack([2 * np.cos(th), np.sin(th)]),
              np.full((n, 10), 0.02))
    g = np.concatenate([g, g[:1]])
    MultiBlock([Block(g, {"jmin": "wall", "jmax": "farfield"})]).assemble()


# ---------------------------------------------------------------- topology

def test_element_counts_drive_the_wake_lines():
    """Sharp: one corner asking for 4 elements. Blunt: two asking for 3."""
    _, e_sharp = wall_segments(boundary("2412", "sharp"))
    _, e_blunt = wall_segments(boundary("2412", "blunt"))
    assert list(e_sharp) == [4]
    assert list(e_blunt) == [3, 3]


def test_blunt_and_sharp_share_one_code_path(meshes):
    """A sharp edge is the blunt construction with a zero-length aft face:
    no strip block, and the two wake lines coincide."""
    mb_s, _ = meshes["4412", "sharp"]
    mb_b, _ = meshes["4412", "blunt"]
    assert [b.name for b in mb_s.blocks] == ["C"]
    assert [b.name for b in mb_b.blocks] == ["C", "aft"]
    g = mb_s.blocks[0].grid
    n_w = sum(t is None for t in mb_s.blocks[0].sides["jmin"]) // 2
    assert np.array_equal(g[:n_w + 1, 0], g[::-1][:n_w + 1, 0])


def test_corner_free_body_has_no_wake_line():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    ell = am.Loop.from_points(np.column_stack([0.5 + 0.5 * np.cos(t),
                                               0.1 * np.sin(t)]))
    bnd = am.build_region([ell], farfield=20.0).boundaries[1]
    with pytest.raises(ValueError, match="corner-free"):
        c_mesh(bnd, 1e-5)
    mb = o_mesh(bnd, 1e-5, radius=20.0, n_layers=80)
    mb.assemble()


# ---------------------------------------------------------------- assembly

def test_assembly_is_a_closed_quad_complex(meshes):
    for (name, te), (mb, mesh) in meshes.items():
        n_bnd = sum(len(v) for v in mesh.patches.values())
        # Euler: V - E + F = 1 - holes... for a disc with one hole, chi = 0
        e = np.sort(np.stack([mesh.quads, np.roll(mesh.quads, -1, axis=1)],
                             axis=-1).reshape(-1, 2), axis=1)
        n_edges = len(np.unique(e, axis=0))
        assert len(mesh.points) - n_edges + mesh.n_cells == 0, (name, te)
        assert set(mesh.patches) == {"wall", "farfield", "outflow"}
        assert n_bnd == 2 * n_edges - 4 * mesh.n_cells


def test_assembly_catches_a_missing_block(meshes):
    """Deleting a wedge block passed the October validator with ok=True."""
    mb, _ = meshes["0012", "blunt"]
    with pytest.raises(ValueError, match="pair with nothing"):
        MultiBlock(mb.blocks[:1]).assemble()


def test_assembly_catches_a_crack(meshes):
    mb, _ = meshes["0012", "blunt"]
    g = mb.blocks[1].grid.copy()
    g[5, 0] += 1e-7                       # one interface node, 0.02 first cells
    with pytest.raises(ValueError, match="pair with nothing"):
        MultiBlock([mb.blocks[0], Block(g, mb.blocks[1].sides)]).assemble()


def test_assembly_catches_a_fold(meshes):
    mb, _ = meshes["0012", "sharp"]
    g = mb.blocks[0].grid.copy()
    g[100, 3] = g[100, 1]
    with pytest.raises(ValueError):
        MultiBlock([Block(g, mb.blocks[0].sides)]).assemble()


# ---------------------------------------------------------------- quality

def test_first_cell_is_honoured_at_every_wall_node(meshes):
    """Measured at the worst node, not at the minimum: the interpolated collar
    honoured the request on 5.7 per cent of the wall and passed its test."""
    for key, (mb, _) in meshes.items():
        w = mesh_report(mb, Y1)["wall"]
        assert 0.985 < w["y1_min_ratio"] <= w["y1_max_ratio"] < 1.01, key
        assert w["y1_within_2pct"] == 1.0, key


def test_first_cell_follows_the_request():
    bnd = boundary("2412", "blunt")
    for y1 in (1e-4, 1e-5, 1e-6):
        mb = c_mesh(bnd, y1)
        w = mesh_report(mb, y1)["wall"]
        assert 0.98 < w["y1_min_ratio"] <= w["y1_max_ratio"] < 1.02
        assert w["y1_within_2pct"] == 1.0
        # a finer first cell brings a finer trailing edge with it, so the
        # finite-volume non-orthogonality does not grow (45.7 degrees at
        # y1 = 1e-6 when the trailing-edge spacing was a fixed number)
        assert assembled_nonorthogonality(mb.assemble()).max() < 25.0


def test_boundary_layer_quality_does_not_depend_on_camber(meshes):
    """The claim the project stands on, as an assertion."""
    for te in ("sharp", "blunt"):
        skew = [mesh_report(meshes[n, te][0], Y1)["bl"]["skew_max"]
                for n in ("0012", "4412", "8412")]
        assert max(skew) < 6.0     # the floor is alpha / n_c - 90 at the corner
        assert max(skew) - min(skew) < 0.5, (te, skew)


def test_whole_mesh_angles_and_wall_face_orthogonality(meshes):
    for key, (mb, _) in meshes.items():
        rep = mesh_report(mb, Y1)
        assert rep["min_angle"] > 78.0, key
        assert rep["max_angle"] < 102.0, key
        assert rep["wall"]["wall_face_nonorth_max"] < 5.0, key
        assert rep["wall"]["along_wall_ratio_max"] < 1.2, key
        assert rep["wall"]["bl_growth_max"] < 1.22, key


def test_finite_volume_nonorthogonality_including_interfaces(meshes):
    """Corner angles cannot see this. checkMesh found 87 degrees in the far
    wake, 35 inside the boundary layer and 35 across the wake line, each on
    cells whose corners were within 5 degrees of square; all three came from
    cells tapering at different rates."""
    for key, (mb, mesh) in meshes.items():
        assert assembled_nonorthogonality(mesh).max() < 20.0, key
        for b in mb.blocks:
            a_i, a_j = fv_nonorthogonality(b.grid)
            assert max(a_i.max(), a_j.max()) < 20.0, key


def test_o_mesh_is_limited_by_the_corner():
    """Recorded, not hidden: the single annulus rounds the trailing edge."""
    mb = o_mesh(boundary("0012", "blunt"), Y1)
    mb.assemble()
    rep = mesh_report(mb, Y1)
    assert 40.0 < rep["min_angle"] < 60.0


# ---------------------------------------------------------------- export

def test_exports(tmp_path, meshes):
    from aeromesh.mesh import write_meshio, write_openfoam
    _, mesh = meshes["0012", "blunt"]
    write_meshio(mesh, tmp_path / "m.su2")
    text = (tmp_path / "m.su2").read_text()
    assert f"NELEM= {mesh.n_cells}" in text and "MARKER_TAG= wall" in text
    patches = write_openfoam(mesh, tmp_path / "case")
    assert patches["wall"] == len(mesh.patches["wall"])
    assert patches["frontAndBack"] == 2 * mesh.n_cells
    owner = (tmp_path / "case/constant/polyMesh/owner").read_text()
    n_faces = int(owner.split("nFaces: ")[1].split()[0])
    n_int = int(owner.split("nInternalFaces: ")[1].split('"')[0])
    # each cell has 6 faces; internal faces are shared
    assert 6 * mesh.n_cells == 2 * n_int + (n_faces - n_int)


# ---------------------------------------------------------------- domains

@pytest.mark.parametrize("domain", ["c", "box", "circle"])
@pytest.mark.parametrize("te", ["sharp", "blunt"])
def test_domain_meshes_close_and_reach_the_far_field(domain, te):
    from aeromesh.mesh import domain_mesh
    bnd = boundary("4412", te)
    if domain == "circle" and te == "sharp":
        # an O-mesh cannot meet a four-element corner (180-degree face,
        # solver crash); the request is refused rather than meshed badly
        with pytest.raises(ValueError, match="outflow"):
            domain_mesh(bnd, Y1, domain, size=20.0)
        return
    mb = domain_mesh(bnd, Y1, domain, size=20.0)
    mesh = mb.assemble()
    far = mesh.points[np.unique(mesh.patches["farfield"])]
    if domain == "circle":
        r = np.linalg.norm(far - [0.5, 0.0], axis=1)
        assert np.allclose(r, 20.0, rtol=1e-3)
    if domain == "box":
        # the four corners are nodes
        for c in [(-20.0, -20.0), (-20.0, 20.0)]:
            assert np.min(np.linalg.norm(mesh.points - c, axis=1)) < 1e-9
    rep = mesh_report(mb, Y1)
    # the boundary layer is the C-mesh's (or O-mesh's) own, untouched
    if domain == "circle":
        # an O-mesh rounds the trailing-edge corner, where no node can be
        # normal to both faces (0.14 of the request at a sharp edge); the
        # rest of the wall is exact
        assert rep["wall"]["y1_within_2pct"] > 0.97
    else:
        assert 0.98 < rep["wall"]["y1_min_ratio"] <= rep["wall"]["y1_max_ratio"] < 1.02


@pytest.mark.parametrize("domain,floor", [("c", 70.0), ("box", 40.0)])
def test_c_domain_outer_block_quality(domain, floor):
    """Matching far-field nodes by arc length gave 22-24 degrees; continuing
    the marched lines gives 45-77."""
    from aeromesh.mesh import domain_mesh
    mb = domain_mesh(boundary("4412", "sharp"), Y1, domain, size=20.0)
    assert mesh_report(mb, Y1)["min_angle"] > floor


def test_o_mesh_refuses_a_cusp_and_c_mesh_takes_it():
    """The O-mesh's 180-degree face at a sharp trailing edge crashed the
    solver; the element count (4) is what rules the O-mesh out."""
    bnd = boundary("0012", "sharp")
    with pytest.raises(ValueError, match="asks for 4 elements"):
        o_mesh(bnd, Y1)
    mb = c_mesh(bnd, Y1)
    assert assembled_nonorthogonality(mb.assemble()).max() < 20.0


def test_pinned_counts():
    """Comparison at equal grid dimensions needs the counts to be exact."""
    bnd = boundary("0012", "blunt")
    mb = c_mesh(bnd, Y1, n_wall=256, n_wake=60, n_layers=100)
    g = mb.blocks[0].grid
    assert g.shape[:2] == (256 + 2 * 60, 101)
    om = o_mesh(bnd, Y1, n_wall=300, n_layers=80)
    assert om.blocks[0].grid.shape[:2] == (301, 81)

"""Randomised geometry sweep (the S5 gate, in miniature).

The claim this project makes is one of *generality* — that a single aerofoil, a
three-element arrangement and a wing–body crossflow cut are the same kind of
object — and a fixed test matrix cannot support a claim about geometry nobody
chose. So the bodies here are drawn: NACA sections with random digits and point
counts, ellipses of random aspect ratio, and polygons of random vertex count
with random radial jitter, placed at random scale, incidence and offset inside a
box, C or circular far field of random extent.

The invariant asserted is **not** "nothing ever fails". It is that there is no
*silent* failure: every configuration either comes out sound — it tiles, no
outline crosses itself, every side is either on the boundary or shared with
exactly one other block, and the index budget balances — or it says in its own
notes which feature it could not handle. That is the project's one rule, and it
is the rule a green suite of shape-only assertions once broke, when a medial
axis carrying branches of radius zero passed as validated.

The seed is fixed so a failure is reproducible. Raise ``TRIALS`` when hunting.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.geometry.loop import Loop

TRIALS = 10
SEED = 4471


def _random_body(rng, name):
    kind = rng.choice(["naca", "ellipse", "polygon"], p=[0.5, 0.3, 0.2])
    if kind == "naca":
        digits = (f"{rng.integers(0, 7)}{rng.integers(0, 7)}"
                  f"{rng.integers(8, 21):02d}")
        return am.load_airfoil(digits, n_points=int(rng.integers(220, 420)))
    if kind == "ellipse":
        a = float(rng.uniform(0.4, 1.8))
        b = a * float(rng.uniform(0.12, 1.0))
        t = np.linspace(0, 2 * np.pi, int(rng.integers(260, 480)),
                        endpoint=False)
        return Loop.from_points(
            np.column_stack([a * np.cos(t), b * np.sin(t)]), name)
    k = int(rng.integers(3, 9))
    t = np.linspace(0, 2 * np.pi, k, endpoint=False) + float(rng.uniform(0, 1))
    r = 1.0 + 0.25 * rng.standard_normal(k) * (rng.random() < 0.4)
    g = Loop.from_points(np.column_stack([r * np.cos(t), r * np.sin(t)]), name)
    return g.resample(int(rng.integers(400, 700)))


def _random_case(rng):
    bodies = []
    for i in range(int(rng.choice([1, 1, 1, 2, 2, 3]))):
        g = _random_body(rng, f"b{i}")
        bodies.append(g.transform(scale=float(rng.uniform(0.5, 1.2)),
                                  angle=float(rng.uniform(-0.35, 0.35)),
                                  dx=1.5 * i + float(rng.uniform(-0.2, 0.2)),
                                  dy=float(rng.uniform(-0.4, 0.4)),
                                  name=f"b{i}"))
    shape = rng.choice(["circle", "box", "c"])
    radius = float(rng.uniform(5.0, 12.0))
    outer = {"circle": lambda: am.circle_farfield(bodies, radius),
             "box": lambda: am.box_farfield(bodies, radius * 0.5,
                                            radius * 0.8, radius * 0.5),
             "c": lambda: am.c_farfield(bodies, radius, radius * 1.7)}[shape]()
    return bodies, outer, f"{len(bodies)} bodies / {shape} / R={radius:.1f}"


def _valid_cases(rng, n):
    """Draw ``n`` configurations that are actually well posed.

    A random placement can put two bodies through each other, and
    ``build_region`` rejects that -- correctly, since overlapping bodies are not
    a meshing problem. Redraw instead, or the sweep spends its failures on
    invalid inputs and hides the real ones.
    """
    out = []
    while len(out) < n:
        bodies, outer, label = _random_case(rng)
        try:
            am.build_region(bodies, outer=outer)
        except ValueError:
            continue
        out.append((bodies, outer, label))
    return out


CASES = _valid_cases(np.random.default_rng(SEED), TRIALS)


@pytest.mark.parametrize("bodies,outer,label", CASES,
                         ids=[c[2] for c in CASES])
class TestRandomGeometry:

    @staticmethod
    def _stages(bodies, outer):
        region = am.build_region(bodies, outer=outer)
        axis = am.exterior_axis(region)
        field = am.solve_singularities(axis, region)
        blocked = am.decompose(axis, region, field=field)
        layered = am.insert_layer(blocked, delta_target=0.02, rho_max=0.05)
        return region, axis, blocked, layered, am.to_quads(layered)

    def test_the_retract_certificate_holds(self, bodies, outer, label):
        """b1 = h, whatever the geometry. This one is never allowed to fail --
        it is a property of the medial axis, not of any choice made later."""
        region = am.build_region(bodies, outer=outer)
        axis = am.exterior_axis(region)
        assert axis.betti_1 == region.n_bodies, label

    def test_no_stage_fails_silently(self, bodies, outer, label):
        """Sound, or it says why. Never neither."""
        _, _, blocked, layered, quads = self._stages(bodies, outer)
        for system, stage in ((blocked, "blocking"), (layered, "layer"),
                              (quads, "quads")):
            assert system.is_sound() or system.notes, f"{label}: {stage}"

    def test_a_sound_blocking_stays_sound_through_the_layer(
            self, bodies, outer, label):
        """Inserting the near-wall layer and quadrilateralising must not break
        a blocking that was sound: both are splits of it, and a split cannot
        introduce a defect the original did not have."""
        _, _, blocked, layered, quads = self._stages(bodies, outer)
        if not blocked.is_sound():
            pytest.skip(f"{label}: blocking already unsound, reported")
        assert layered.is_sound(), f"{label}: {layered.notes[-2:]}"
        assert quads.is_sound(), f"{label}: {quads.notes[-2:]}"

    def test_the_layer_never_changes_the_index_budget(
            self, bodies, outer, label):
        _, _, blocked, layered, _ = self._stages(bodies, outer)
        if not blocked.is_sound():
            pytest.skip(f"{label}: blocking already unsound, reported")
        assert layered.index_certificate()["ok"], label

    def test_no_hanging_or_stranded_sides(self, bodies, outer, label):
        _, _, blocked, layered, quads = self._stages(bodies, outer)
        for system, stage in ((blocked, "blocking"), (layered, "layer"),
                              (quads, "quads")):
            if not system.is_sound():
                continue
            assert system.hanging_nodes() == [], f"{label}: {stage}"
            assert system.interface_report()["stranded"] == [], \
                f"{label}: {stage}"

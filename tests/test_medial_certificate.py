"""The retract certificate on randomised geometry.

Salvaged from ``test_sweep.py`` when the medial-axis blocking pipeline was
retired in November 2026. Everything else in that file exercised
``solve_singularities``, ``decompose``, ``insert_layer`` and ``to_quads``,
which went with ``blocking/`` and ``topology/``. This one test did not: it
asserts a property of the *medial axis*, which is kept, and it does so on
random bodies in random far fields rather than on a fixed list.

The identity is that the medial axis deformation-retracts onto its region, so
the cycle count of the exterior axis equals the number of bodies. It is a
theorem, not a tuning parameter, and it is the cheapest end-to-end check the
medial engine has. It is the first thing that should fail if the engine rots
while the multi-element work is waiting.
"""

from __future__ import annotations

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
def test_the_retract_certificate_holds_on_random_geometry(bodies, outer,
                                                          label):
    """b1 = h, whatever the geometry."""
    region = am.build_region(bodies, outer=outer)
    axis = am.exterior_axis(region)
    assert axis.betti_1 == region.n_bodies, label

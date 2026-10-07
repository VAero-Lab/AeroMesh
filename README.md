# AeroMesh

Automatic structured multiblock mesh generation for 2-D aerodynamic
configurations, driven by the medial axis. Companion to
[AeroShape](https://github.com/victor-alulema/aeroshape); Pillar III of the
MAPS research programme.

**Status: under active reconstruction.** S0 (geometry foundation) is complete;
the medial engine is next. See [PROJECT_TRACKER.md](PROJECT_TRACKER.md).

---

## What it does

The input is a **set of closed loops** — one outer boundary and any number of
bodies:

```python
import aeromesh as am
import numpy as np

main = am.load_airfoil("2412")
flap = main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02, dy=-0.10,
                      name="flap")

region = am.build_region([main, flap], farfield=15.0)           # circular
```

The outer boundary is a curve you supply, built with the conventional
constructors:

```python
am.circle_farfield(bodies, radius=15)                  # O-type, 0 corners
am.c_farfield(bodies, radius=15, wake_length=25)       # C-type, 2 corners
am.box_farfield(bodies, upstream=15, downstream=25, lateral=15)   # H-type, 4
am.offset_farfield(bodies, distance=15)                # level set, 0 corners
```

Choosing one is a modelling decision, like choosing how far out to truncate —
not a topology selection. It is passed as a curve and stored as a curve, and no
later stage can tell which constructor made it. What it *does* influence is how
many corners the fluid boundary has, and a corner generates a medial flare, and
a flare generates a block.

There is no domain type to choose and no topology to select. A single airfoil,
a three-element high-lift arrangement and a wing–body crossflow cut are the same
kind of object, and every stage downstream sees them identically.

## The design rule

> Nothing is prescribed. The number of blocks, their connectivity, the
> singularity positions and types, and the topology class itself are **derived**
> from the geometry. The bodies, where the domain is truncated, and the target
> resolution are **supplied**.

C, O and H are labels you read off the result — never inputs.

## Method

1. **Medial axis**, of the fluid region *and* of each body. The exterior axis
   supplies everything between and around the bodies; the body's own interior
   axis supplies its shape, and unlike the exterior one it does not degenerate
   when the far field sits fifty chords away.
2. **Singularities** solved by flux balance on the medial angle θ_m
   (Fogg, Armstrong & Robinson 2015), not looked up in a rule table.
3. **Certificates** before construction: the medial graph's cycle count must
   equal the body count, and the singularity budget must balance against the
   Euler characteristic.
4. **Decomposition** as a ranked search over splits that are curves of the
   medial coordinate system (s, ρ), so interfaces meet walls orthogonally by
   construction.
5. **Mesh** by transfinite interpolation and elliptic smoothing, with wall
   clustering derived from y⁺, Reynolds number and altitude.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, and `aeroshape` for geometry intake.

## Try it

```bash
python examples/10_region_any_bodies.py
```

Builds seven configurations through one call and writes
`output/10_region_any_bodies.png`.

## Tests

```bash
python -m pytest tests -q
```

The suite is invariant-based by policy: total turning equals 2π, resampling
preserves a polygon's perimeter, a rigid motion preserves the corner set, an
opened cusp conserves turning. Shape-only assertions are not enough — a green
suite of them is how a medial axis carrying branches of radius zero once passed
as validated.

## Documents

| File | Contents |
|---|---|
| [PROJECT_TRACKER.md](PROJECT_TRACKER.md) | Stage status, gates, decision log, invariant registry |
| [Long_term_plan.md](Long_term_plan.md) | Architecture and build order |
| [MAPS_Proposal_v2.md](MAPS_Proposal_v2.md) | The scientific argument and claims |
| [MAPS_Implementation_Plan.md](MAPS_Implementation_Plan.md) | Stage-by-stage implementation detail |

## References

1. Rigby, D.L. (2004). *TopMaker: a technique for automatic multi-block topology
   generation using the medial axis.* NASA/CR-2004-213044.
2. Fogg, H.J., Armstrong, C.G., Robinson, T.T. (2015). Enhanced medial-axis-based
   block-structured meshing in 2-D. *Computer-Aided Design*.
3. Bunin, G. (2008). A continuum theory for unstructured mesh generation in two
   dimensions. *Computer Aided Geometric Design*, 25, 14–40.
4. Thompson, J.F., Thames, F.C., Mastin, C.W. (1974). Automatic numerical
   generation of body-fitted curvilinear coordinate systems. *JCP*, 15(3).

## Licence

MIT

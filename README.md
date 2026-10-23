# AeroMesh

Automatic structured multiblock mesh generation for 2-D aerodynamic
configurations, driven by the medial axis. Companion to
[AeroShape](https://github.com/victor-alulema/aeroshape); Pillar III of the
MAPS research programme.

**Status: under active construction.** The geometry foundation, the medial
engine, the singularity solver, the block decomposition, the near-wall layer and
quadrilateralisation all work. Over seventeen configurations — aerofoils, two-
and three-element arrangements, a fuselage with a wing, three bodies, discs and
regular polygon bodies, in box, C and circular far fields — every stage is
*sound*: the blocks tile, no outline crosses itself, every side is either on the
boundary or shared with exactly one other block, and the index budget balances.
Nine of the seventeen reach an all-quadrilateral blocking. Mesh construction
proper — transfinite interpolation, elliptic smoothing, export — is next. See
[PROJECT_TRACKER.md](PROJECT_TRACKER.md).

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
3. **Certificates**, two of them. Before construction: the medial graph's cycle
   count must equal the body count. After: the delivered blocking must balance
   the index budget `Σ(valence − 4) = Σ(2 − q) − 4χ`. That identity holds for
   every valid quadrilateral mesh, which is exactly why checking it on the
   *product* is a bookkeeping test and not a forecast — and it resolves cracks
   three orders finer than an area test, which is how a 5×10⁻⁷ hole between two
   vertex regions was found.
4. **Decomposition** as a ranked search over splits that are curves of the
   medial coordinate system (s, ρ), so interfaces meet walls orthogonally by
   construction. Cuts go one per gap between singularities, so each block holds
   at most one; blocks are built constructively from the medial graph and carry
   conformal connectivity.
5. **Blocking by zone**, because one decomposition cannot serve two objectives.
   The medial decomposition of steps 2–4 is a *CAD* method — well-shaped quads
   on a part — and CFD wants something else at the wall: extreme anisotropy,
   orthogonal cells and a fixed Δy₁. Asked to do both it did neither, producing
   blockings that passed every certificate and could not be meshed. So the
   near-wall region is **built** as a collar rather than decomposed: an annulus
   around the body, because a band around a closed curve cannot be anything
   else; rays from each corner's normal cone, as many as its element count asks
   for; bands between rays of different corners, wedges between rays of the
   same. Thickness is `min(δ_target, ρ_max·r_m)` — the flow sets it, the medial
   radius caps it to the room available. Nothing in it names a configuration,
   and the same code runs on sharp and blunt trailing edges. Zones 2 (mid-field,
   from the medial features *between* bodies) and 3 (far field) are next.
6. **Quadrilateralisation** by midpoint subdivision — the one prescribed
   construction, and named as such. Applied to every block, or none, because a
   partial subdivision is not conformal; and with the spokes built by a harmonic
   map onto a disc, since straight spokes to a centre fold at any choice of
   centre and Tutte's theorem is the construction that cannot.
7. **Mesh** by transfinite interpolation and elliptic smoothing, with wall
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
opened cusp conserves turning, the medial graph's cycle count equals the body
count, and the blocks tile the region with overlap, gap and spill each measured
separately.

Shape-only assertions are not enough — a green suite of them is how a medial
axis carrying branches of radius zero once passed as validated. Nor is an area
sum: it cannot see an overlap cancelled by an equal gap, and it is completely
blind to a zero-area spike. Three real defects in the block builder hid behind
one before the stricter checks were added.

Nor is a check that cannot fail. `paired + boundary == n_sides` held whatever
happened, because a side that pairs with nobody was filed as a boundary side;
ten unpaired sides passed as conformal. Every check added since names, in its
own docstring, the defect it was written to catch and the measurement that found
it.

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
5. Tutte, W.T. (1963). How to draw a graph. *Proc. London Math. Soc.*, 13.
   — the injectivity guarantee the quadrilateralisation rests on.
6. Shewchuk, J.R. (1996). Triangle: engineering a 2D quality mesh generator and
   Delaunay triangulator. *Applied Computational Geometry*.

## Licence

MIT

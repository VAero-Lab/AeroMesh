# AeroMesh — Project Tracker

**Owner:** Victor Alulema · **Thesis pillar:** III (MAPS) · **Started tracking:** 2026-09-08

The single source of truth for what is built, what is next, and what has been
decided. Planning rationale lives in `Long_term_plan.md`; the scientific
argument lives in `MAPS_Proposal_v2.md`.

---

## The one rule

> **Nothing is prescribed.** No hardcoded topology, no domain-type switch, no
> rule table, no per-airfoil or per-configuration branch. The number of blocks,
> their connectivity, the singularity positions and types, and the topology
> class itself are **derived**. The geometry of the bodies, where the domain is
> truncated, and the target resolution are **supplied**.

The only prescribed construction permitted anywhere in the codebase is the
midpoint-subdivision template for filling a logically convex *m*-gon (S3) —
a proven primitive for quad-meshing an *m*-gon, containing no geometry-specific
case. It must be named as such where it is used.

---

## Status at a glance

| Stage | Scope | Status | Est. | Gate |
|---|---|---|---|---|
| **S0** | Loop / corners / cusp policy / Region / far field | ✅ **done** 2026-09-08 | 1 wk | met |
| **S1** | Medial engine — r_m, θ_m, normals, typing, finite contact | ✅ **done** 2026-09-08 | 2 wk | met |
| **S2** | Singularity solver + certificates | ⚠️ **mostly done** | 1.5 wk | 4 of 6 |
| **S3** | Decomposition | 🔨 **blocks build and tile** | 2.5 wk | partial |
| **S4** | Mesh construction (TFI, TTM, export) | ⬜ not started | 1.5 wk | — |
| **S5** | Adversarial geometry gate | ⬜ not started | 1 wk | — |
| **S6** | Quality metrics + optimisation | ⬜ not started | 1.5 wk | — |
| **S7** | Bunin φ-field by FEM — *optional* | ⬜ not started | 1 wk | — |

**Ordering is not negotiable: S0 → S1 → S2 → S3.** The hardcoded blocking in the
previous iteration appeared because S3 was attempted while S1 was incomplete and
S2 did not exist.

---

## S0 — Foundation and data model ✅

Delivered 2026-09-08. New modules; nothing was deleted.

- [x] `geometry/loop.py` — `Loop`: closed, CCW, stored open. Arc-length
      interpolation and resampling, corner-preserving so a polygon keeps its
      perimeter exactly. Area centroid, containment, rigid transforms.
- [x] `geometry/corners.py` — scale-invariant C0 detection by extrapolating
      total turning to zero window width. Returns the corner angle as a
      by-product. `fluid_interior_angle` handles the hole/outer convention.
- [x] `geometry/cusp.py` — `open_cusp` cuts perpendicular to the cusp axis and
      inserts a face of exact length; `find_cusps` gates it.
- [x] `geometry/region.py` — `Region` = one outer loop + N holes. Validation,
      `hole_points()` from the largest inscribed circle, `sample()` for the
      medial engine, `euler_characteristic`.
- [x] `geometry/airfoil.py` — AeroShape intake returning a `Loop`. No TE policy
      here; cusp handling is general and lives in `prepare_boundary`.
- [x] `domain/farfield.py` — the conventional truncation curves as explicit
      constructors: `circle_farfield` (0 corners, O-type, the default),
      `c_farfield` (2 corners, C-type), `box_farfield` (4 corners, H-type or
      tunnel walls), plus `offset_farfield` (a distance level set, for tight or
      widely spread configurations) and `rectangle`. Every one returns a plain
      `Loop`; the shape is never recorded and cannot be branched on downstream.
- [x] `aeromesh/__init__.py` — `build_region(bodies, farfield=…)` entry point.
- [x] Tests: `test_loop.py`, `test_corners.py`, `test_cusp.py`, `test_region.py`,
      `test_farfield.py` — 160 new tests, all invariant-based. Suite: **194 passed**.
- [x] Examples: `examples/10_region_any_bodies.py`,
      `examples/11_farfield_shapes.py`.

**Gate — met.** Seven configurations build through one call with no branch on
body type or count:

| Configuration | far field | bodies | χ | points | corners |
|---|---|---|---|---|---|
| single airfoil | circle | 1 | 0 | 2126 | outer 0, main 2 |
| main + flap | C-shape | 2 | −1 | 3284 | outer 2, main 2, flap 2 |
| slat + main + flap | circle | 3 | −2 | 4116 | outer 0, main 2, flap 2, slat 2 |
| airfoil + fuselage | circle | 2 | −1 | 3225 | outer 0, main 2, fuselage 0 |
| fuselage alone | circle | 1 | 0 | 2210 | 0 |
| circle body | C-shape | 1 | 0 | 2382 | outer 2 |
| airfoil in a box | box | 1 | 0 | 2414 | outer 4, main 2 |

Corner sets contain exactly the real corners: none on a circle or a bluff body,
two on a C-shape (the downstream pair — its arc joins the straight sides
tangentially), four on a box, and two per blunted trailing edge (n_c = 3 at
each, from a fluid interior angle of 261.8°).

---

## S1 — Medial engine ✅

Delivered 2026-09-08. New modules `medial/axis.py` and `medial/fields.py`;
`medial/cdt.py` and `medial/graph.py` untouched and unused (D7).

- [x] Geometric separation filter, on **smooth boundary elements** rather than
      loops. A corner divides a loop into distinct elements, and two generators
      on different elements are always a genuine pair however close they sit.
      `Region.segment_ids()` derives the segmentation from detected corners.
- [x] CDT dropped from this path.
- [x] Full field set on every medial point: `r_m`, `theta_m`, `n1`, `n2`, touch
      points, arc length `s`. Touch points are the **exact feet on the polyline**,
      not the nearest samples, which makes θ_m second order.
- [x] Pruned on θ_m, not arc length; branches anchored at a known corner are
      never pruned.
- [x] Vertex typing N / C / D, corners taken from S0.
- [x] Flares extended to exactly r_m = 0 at the corner, with θ_m set
      analytically to π − α from the corner angle.
- [x] Finite-contact handling for a body whose axis collapses to a point.
- [x] `interior_axis(loop)` and `exterior_axis(region)`.
- [x] 53 tests in `test_medial_axis.py`; legacy tests preserved in
      `test_medial_legacy.py`. Example: `examples/12_medial_axis.py`.

**Gate — met.**

| Check | Result |
|---|---|
| b₁ = h, 1/2/3 bodies × circle/C/box far fields | exact in all 15 |
| Annulus r_m vs analytic | **0.0021%** (gate < 1%) |
| Annulus θ_m vs analytic 180° | 179.977°–180.000° |
| Ellipse interior axis on y = 0 | 1.3 × 10⁻¹⁵ |
| Ellipse half-extent (a²−b²)/a | 0.046% |
| Ellipse r_m at centre = b | 0.0001% |
| Exact circle | `FINITE_CONTACT` vertex, θ_max = 180° → k = −2 |
| Zero-radius branches | only at CORNER vertices, nowhere else |
| Scale invariance of θ_m | identical to 10⁻⁴ deg over 5000× scale range |
| Metamorphic (whole-problem rotate/scale/translate) | graph isomorphic |

Derived structure, no rule anywhere:

| Far field | Vertices | Edges |
|---|---|---|
| circle | none — Rigby's doughnut | 1 LOOP |
| C-shape | 2 NORMAL, 2 CORNER | 2 PRIMARY, 2 FLARE |
| box | 4 NORMAL, 4 CORNER | 4 PRIMARY, 4 FLARE |
| 3 bodies + box | 8 NORMAL, 4 CORNER | 10 PRIMARY, 4 FLARE |

θ_m collapses to **9.2°** in a main–flap slot and **6.5°** in a three-element
configuration, so the flux balance will demand singularities there; over a
featureless exterior loop at 15 chords it stays above 170° and Fogg's index is
zero everywhere. Extraction costs ~0.35 s for 2400 boundary points.

## S2 — Singularity solver and certificates ⚠️ partial

New module `topology/singularities.py`. Machinery and certificate built; the
solver does not yet close the budget on every configuration, and it says so.

- [x] `n_opt = round((π − θ_m)/(π/2))` and the residual flux
      `Φ = (π − θ_m) − n·π/2`, bounded by π/4 and maximal at the critical angles.
- [x] **Class 1** — θ_m crossings of π/4 and 3π/4, restricted to touches on
      smooth edges (a crossing whose touches have converged on a corner is
      already carried by that corner's n_c).
- [x] **Class 3** — concave-corner reference switches, where the cross direction
      θ_m is measured from jumps by one sector.
- [x] **Class 2 — flux balance at medial vertices.** `k_V = Σ_j (2 − n_j) − 4`
      over the incident medial edges, with `n_j` read one local radius out.
      Exact on six of seven known polygons; an L-shape is off by one, where one
      vertex reads an incident flow index of 0 where the balance needs 1.
- [x] **Finite contact at a branch end** (Fogg Fig. 10) — a semicircular end
      has contact over π and gives k = −2. Previously only a fully collapsed
      axis was detected, so a stadium reported nothing.
- [x] **Crossings adjacent to a vertex are excluded** — a branch point or a
      finite-contact end changes the flow index through its own balance, and
      counting the crossing as well double-counts it. This removed every
      spurious crossing on all seven test polygons.
- [x] `k = −⌊contact extent/(π/2)⌋` at finite-contact vertices. The quantity is
      the *angular extent of contact*, not the largest angle between two touches:
      a semicircular end spans π → k = −2, a full circle spans 2π → k = −4, the
      four superimposed negatives of the 0-sided template. (Required removing a
      clamp to π left in S1.)
- [x] n_c at corners from the fluid interior angle.
- [x] Local merging within [r_m/4, r_m], cancelling ± pairs, total conserved.
- [x] Retract certificate (b₁ = h) — delivered in S1.
- [x] **Index budget.** Derived from discrete Gauss–Bonnet for quad meshes,
      `Σ_int (4−v) + Σ_bdry (3−v) = 4χ`, which with `k = v−4` and `n_c = v_b−1`
      gives

      **Σ_interior k = Σ_corners (2 − n_c) − 4·χ**

      Verified against six configurations whose correct blocking is known —
      square, disk, triangle, L-shape, annulus — and it **predicts** Fogg's
      stated `k = +2` at the centre of a regular hexagon, a case not used to
      derive it.

**Gate — 4 of 6 configurations certify, and 6 of 7 known polygons are exact.**

Closed-form polygons, where the correct blocking is known:

| shape | budget | solved | |
|---|---|---|---|
| triangle | −1 | −1 | ✅ |
| square | 0 | 0 | ✅ |
| pentagon | +1 | +1 | ✅ |
| hexagon | +2 | +2 | ✅ Fogg's value, from an actual hexagon |
| rectangle 2:1 | 0 | 0 | ✅ |
| stadium | −4 | −4 | ✅ on its two semicircular ends |
| L-shape | 0 | +1 | ❌ off by one |

Target configurations:

| Configuration | h | b₁ | χ | Σk | required | |
|---|---|---|---|---|---|---|
| bluff body / circle | 1 | 1 | 0 | 0 | 0 | ✅ |
| airfoil / circle | 1 | 1 | 0 | −2 | −2 | ✅ |
| airfoil / box | 1 | 1 | 0 | +2 | +2 | ✅ |
| slat+main+flap / circle | 3 | 3 | −2 | +2 | +2 | ✅ |
| airfoil / C-shape | 1 | 1 | 0 | −2 | 0 | −2 |
| main+flap / circle | 2 | 2 | −1 | +1 | 0 | +1 |

The retract certificate (b₁ = h) holds everywhere. Residuals on the open cases
are ±1–2, down from −3…−11 before class 2.

**The vertex rule is `k_V = Σ_j (2 − n_j) − 4`** over the incident medial edges.
Each cut across an incident edge carries `n_j` quarter-turns of the cross field
and enters the balance exactly as a boundary corner's `n_c` does, so the same
accounting that gives the global budget gives this for the disc around one
vertex. Domain corners are *not* counted here — they are already in the global
corner term. It reduces to `m − 4` when every incident edge has `n_j = 1`, which
is why that simpler form worked on the regular polygons and failed on a
rectangle, whose central medial edge has `n = 0`.

**The crossing sign is settled: `k = -|dn|`.** One negative singularity per
critical angle crossed, independent of the direction of travel. The thin-sliver
derivation suggests a signed `k = -dn`, but `dn` reverses with the direction of
travel along an edge. Two closed-form checks rule that out: an ellipse's axis
yields `dn = [−1, −1, +1, +1]` at every aspect ratio from 3:1 to 1.01:1, summing
to zero under either signed rule and to −4 under `−|dn|`, which is what the
budget requires for a disk-like region; and an elongated ellipse (resolved by
crossings) must agree with a circle (resolved by finite contact, a separate
route) — both give −4, and only `−|dn|` is continuous across that handover.

**Known open defects**, all recorded as tests rather than hidden:

1. **An L-shape is off by one**, and two target configurations are off by ±1–2.
   One incident flow index is misread; the evaluation point for `n_j` is the
   likely cause.
2. **A narrow transition band at the circular limit.** At a ≈ 1.01:1 the ends
   have enough contact to be promoted to finite-contact vertices, which guards
   their crossings out, but not enough extent for `−⌊extent/(π/2)⌋` to reach −2
   apiece; the total comes to −2 where −4 is required. Every aspect ratio from
   3:1 down to 1.05:1 is exact by crossings and the circle itself is exact by
   finite contact, so only the handover is wrong. The two detection fixes below
   narrowed this band from ≈1.05–1.2 to ≈1.01 alone.

Two detection bugs were found and fixed along the way. The finite-contact
tolerance was 2% of r_m, loose enough to call an airfoil nose a circular arc;
it is now 0.2%, since true finite contact means zero deviation. And the
full-contact shortcut compared the largest angular gap against 2π/n, which grows
as fewer points are in contact and so called a small arc a full circle — a NACA
leading edge was reporting 360° of contact. It now compares against the median
gap.

Note on the gate wording: "a lone airfoil returns zero singularities" was
written before the budget existed and is **wrong**. An airfoil with a blunt
trailing edge has two corners at n_c = 3, so the budget requires Σk = −2. A pure
O-mesh avoids them only by taking n_c = 2 at the trailing edge, which makes each
cell there span ~180°. Choosing the quality-optimal n_c is what forces the
interior singularities — the method working, not failing. A lone *bluff body*
does correctly return zero.

---

## S3 — Decomposition 🔨 started

Replaces `blocking/`. The hardest stage, and the one that must contain no
per-case code at all.

**Built so far.**

- [x] `blocking/splits.py` — constant-s splits (medial radius pairs,
      `T1 → p(s) → T2`) with Fogg's ranking: `n = 0` preferred over `n = 1`
      because it leaves no concave corner, and within a class the angle nearer
      the ideal (π for `n = 0`, π/2 for `n = 1`) wins; `n = 2`, a thin slot, is
      ranked worst. Candidate generation per edge, best first, with a margin
      that keeps splits off the vertices.
- [x] **Orthogonality verified against the boundary polyline**, not against the
      construction: worst |cos| between a split and the wall tangent is under
      0.08 across every edge of a box configuration. Touches landing *on* a
      corner are excluded — the wall has no tangent there and a radius ending at
      a corner is a flare, a different object.
- [x] Prerequisites in the medial layer: touch **indices** are now carried on
      every axis edge (`i1`, `i2`) so a block can walk the boundary arc between
      consecutive splits.
- [x] **Stable touch ordering.** Touches were sorted by distance — but a medial
      point is equidistant from its touches *by definition*, so that sort was a
      coin flip that reordered `t1`/`t2` from one sample to the next and would
      have scrambled every block boundary. Now ordered by (loop, sample index).

- [x] `blocking/blocks.py` — **block construction, constructively from the medial
      graph**. A span between two cuts on a medial edge is a four-sided block
      (two radius pairs, two boundary arcs); a medial vertex is an m-sided
      region bounded by the innermost cut on each incident edge; a flare ending
      at a corner is a three-sided wedge; a medial loop with no vertices is one
      ring wrapping on itself. The face structure is the medial graph's, so no
      planar arrangement is computed.
- [x] Boundary arcs recovered by stitching the short path between consecutive
      touch indices, which handles direction and wrap without having to work
      either out.
- [x] **Tiling validation** — the decisive check. Blocks' areas must sum to the
      fluid area; a gap or an overlap shows up at once.

**Review of the builder, 2026-09-24.** The first version passed an area-sum
check, which turned out to be far too weak. Three defects were found and fixed.

1. **Every join between a split and an arc was left open by half a boundary
   sample spacing.** A split ends at the *exact* perpendicular foot on the
   boundary -- that exactness is what made θ_m second order in S1 -- while an arc
   was built from boundary *samples*. The two differ by up to half a spacing, so
   each block outline had a hole at every one of its four corners. Fixed by
   splicing the exact feet in as the arcs' endpoints. Joins now close to 10⁻¹⁶.
2. **A name-shadowing bug put a boundary *index* into an outline as a
   *coordinate*.** Inside `boundary_arc`, a local `start, n = int(offsets[loop])
   …` overwrote the `start` parameter, so `arc[0] = start` assigned the integer
   loop offset to both components — producing outline vertices at (1399, 1399)
   and (0, 0). This is the one that matters most, because **it is invisible to
   every area-based check**: a spike out to such a point is a zero-area sliver.
   It also slipped past a shapely overlap test because `buffer(0)` silently
   repaired it.
3. **Collapsed sides were kept.** Two cuts can terminate at exactly the same
   boundary point — typically a sharp corner on a body, where the medial radii
   from either side both land on the corner. The arc between them is then empty
   and the region is one side shorter; keeping the empty side left a repeated
   vertex that made the outline non-simple for no geometric reason.

`validate` now checks joins, stray vertices and degeneracy, and
`validate_strict` adds the shapely tests — self-intersection, and overlap, gap
and spill measured *separately* rather than inferred from one area total.

**Results after the fixes.** Every configuration is simple, closed, and tiles
with overlap, gap and spill all below 10⁻¹⁵ of the region area:

| configuration | blocks | composition | area error |
|---|---|---|---|
| 1 body, smooth far field | 1 | ring ×1 | 0 |
| 1 body, C-shape | 8 | span 4, vertex 2, corner 2 | 7.9e−12 |
| 1 body, box | 16 | span 8, vertex 4, corner 4 | 6.1e−11 |
| 2 bodies, box | 21 | span 11, vertex 6, corner 4 | 3.9e−10 |
| 3 bodies, box | 26 | span 14, vertex 8, corner 4 | 6.4e−10 |
| 1 body, box, 4 splits/edge | 40 | span 32, vertex 4, corner 4 | 9.5e−11 |

(Before the fixes these read 10⁻⁵ — five orders of magnitude worse, and the
spikes were not visible at all.)

**Still to build.**

- [ ] **Fogg's neighbour database and the ranked split search.** This is the
      gap that matters now. Splits are currently placed at a fixed fraction
      along each edge, so the decomposition is *valid* but not *good*: vertex
      regions sprawl and spans are thin slabs. The search is what places cuts so
      that singularities end up inside logically convex regions, and it is what
      turns a tiling into a blocking worth meshing.
- [ ] Constant-ρ splits (level sets of ρ = d_wall/r_m), the ring interfaces.
- [ ] Conformal connectivity between blocks, and the edge-division integer
      programme.
- [ ] Midpoint-subdivision templates for logically convex m-gons — the one
      permitted primitive, to be named as such where it is used.
- [ ] Region builders for `FINITE_CONTACT` vertices (reported in `notes`, not
      silently skipped).

**Finding to carry into the block builder.** A collapsed medial edge can span a
change in *which loops* its two touches lie on — the ring edge's inner touch
sweeps a whole body, and an edge can cover both a flare stretch and a ring
stretch. Checked and it is not a missing vertex: degree and touch count agree
exactly across every configuration (degree 2 ↔ 2 touches, degree 3 ↔ 3 touches).
So the block builder has to handle a block whose two arcs are not on the same
pair of loops, or place splits densely enough that none spans the transition —
and validate that no block does.

**Gate — structure met, quality not yet.** A lone body gives a ring; a box far
field gives a ring of spans plus one wedge per corner; two- and three-element
configurations give slot blocks; all from one code path, and the
anti-prescription grep over `src/aeromesh/blocking/` returns nothing. What is
not yet met is that the blocks be *good* — that waits on the ranked split
search.

---

## S4 — Mesh construction

- [ ] Real TFI per block (currently returns zeros).
- [ ] Real TTM elliptic smoothing with wall spacing as a BC (currently a no-op).
- [ ] tanh clustering from Δy₁, r, N_wall.
- [ ] Export via `meshio`. No custom writers.

**Gate.** SU2 opens the mesh; minimum cell Jacobian positive everywhere.

---

## S5 — Adversarial geometry gate

The direct answer to the false-confidence problem. What makes "no prescription"
checkable rather than asserted.

- [ ] Metamorphic: rotate, scale, translate, mirror → the block graph must come
      back isomorphic. *(Partial: already in place for corner detection.)*
- [ ] Randomised: random CST sections at random positions in random convex outer
      loops → a certified blocking every time.
- [ ] Per-module invariants: r_m against an analytic annulus, θ_m against an
      analytic wedge, b₁ = h, positive block areas, conformal shared edges,
      every boundary arc covered exactly once.

**Gate.** A thousand randomised configurations give a thousand certified block
graphs; every metamorphic pair is isomorphic.

---

## S6 — Quality and optimisation

- [ ] Exact metrics: wall orthogonality, min Jacobian, BL aspect ratio, metric
      jump across interfaces, achieved y⁺. Cross-checked against `vtkMeshQuality`.
- [ ] Δy₁ from y⁺, Re and altitude.
- [ ] L-BFGS over the continuous parameters (ρ*, Δy₁, r, N_wall). CMA-ES only if
      the landscape proves multimodal.

**Gate.** F(x*) beats a sensible hand-set default, measurably.

---

## S7 — Bunin φ-field (optional)

- [ ] FEM solve of ∇²φ = Σ kᵢ(π/2)δ(r − pᵢ) with clustering and far-field BCs.
- [ ] Let singularities float inside their polygons to equalise element size.

**Gate.** Repositioned singularities measurably reduce peak distortion.

---

## Decision log

| # | Date | Decision | Rationale |
|---|---|---|---|
| D1 | 2026-09-08 | **Blunt trailing edge is the default**; closed and cusped supported via explicit corner seeding. | A cusp is the degenerate case for a medial axis — the branch it implies is tangent to both surfaces and ill-conditioned, and the wake branch cannot be recovered from a Voronoi. **Reverses** the closed-TE-only decision in the earlier plan. |
| D2 | 2026-09-08 | **`DomainType` leaves the pipeline.** The data model is a set of closed loops. | C/O/H were an *input* that switched the blocking — prescription at the top of the pipeline. They are now output labels read off the block graph. Also what makes multi-body free. |
| D3 | 2026-09-08 | **TopMaker's rule table is not implemented.** Singularities are solved by flux balance; the decomposition is a ranked search. | Fogg's paper exists because the table fails — it assumes all corners are flat except those on flares, which distorts blocks on a multi-element aerofoil. |
| D4 | 2026-09-08 | ~~Far field defaults to a distance level set of the body set's convex hull.~~ **Superseded by D8.** | Reasoning at the time: it needs no shape name, so no letter could leak back into the pipeline. The hull kept it C1 — offsetting a union of separated bodies leaves a concave crease, a real corner that would drive real topology from an arbitrary modelling choice. That part still holds; making it the *default* did not. |
| D9 | 2026-09-08 | **The wake cut is not a medial-axis feature.** It is a decomposition split from a reflex corner (S3), seeded by the corner data S0 provides. | The medial axis terminates at *convex* corners of the fluid domain — the far-field box corners, which do give flares. A convex corner of a **body** is reflex for the fluid, and at a reflex vertex the corner is the unique nearest point for a whole fan of directions, so no medial branch emanates. Confirmed numerically: what looked like a wake branch had `n_touch = 1, θ_m = 0`, an artefact, now filtered. A circular far field around an airfoil correctly gives a bare ring. **Corrects a claim in the review.** |
| D8 | 2026-09-08 | **Far field defaults to a circle. The conventional constructors — `circle_farfield`, `c_farfield`, `box_farfield` — are provided alongside `offset_farfield`.** | D4 overcorrected. By Table C in the review, the truncation curve is a *supplied* input: a circle is geometry, not prescription. Three measured reasons. (1) The blob's only technical claim, uniform truncation distance, is worth **3.1% of R at 15 chords and 0.9% at 50** — a circle is already uniform at any realistic far field. (2) The blob and a circle produce **identical** topology: 0 corners, 0 flares, 0 junctions. It buys nothing. (3) A corner-free default **removes the C and H topologies entirely**, because their flares come from the far-field corners — measured: circle 0 flares, C-shape 2, box 4. Non-standard shapes also make comparison against pyHyp, construct2d and published results harder for no gain. `offset_farfield` is kept for tight or widely spread configurations, where the hull genuinely wraps better. |
| D5 | 2026-09-08 | **Neural operator (old Phase D) cut from the critical path; Bunin φ kept as an optional FEM stage (S7).** | The operator existed to make ~1000 CMA-ES quality evaluations cheap. Topology is now solved and certified rather than searched, leaving ~6 smooth continuous parameters — tens of L-BFGS evaluations, where exact 100 ms evaluations cost seconds. Revisit only for the 3-D extension (per-section cost × 15–20 stations) or for ∂quality/∂shape in shape optimisation. |
| D6 | 2026-09-08 | **Validation targets deferred; invariants not.** | Comparing against pyHyp / construct2d / SU2 drag can wait for meshes to exist. The invariant gates cannot — they are what prevents the next false-confidence build. S5 sits before the optimisation work, not at the end. |
| D7 | 2026-09-08 | ~~Superseded modules kept in place because the repository is not under version control.~~ **Resolved: the repository is under git** (`main`, everything committed), so the superseded modules can be removed whenever convenient. They still import cleanly and nothing on the current path uses them. |

---

## Invariant registry

Every gate that must hold at every stage. A green suite that checks none of
these is how a skeleton with zero-radius branches shipped as "strictly validated".

| Invariant | Statement | Where | Status |
|---|---|---|---|
| Total turning | Σ vertex turns = 2π on any simple closed loop | `test_corners.py` | ✅ |
| Corner discrimination | Ellipses have none; a square has four at 90°; a NACA has one, at the TE | `test_corners.py` | ✅ |
| Metamorphic (corners) | Rotation, scaling, translation and mirroring preserve the corner set | `test_corners.py` | ✅ |
| Turning conservation | Opening a cusp splits its turn; it creates none | `test_cusp.py` | ✅ |
| Face exactness | An opened cusp's face length is exact and its two corners symmetric | `test_cusp.py` | ✅ |
| Multi-cusp | A body with two tips gets two faces; neither is reopened | `test_region.py` | ✅ |
| Polygon perimeter | Resampling preserves a polygon's perimeter exactly | `test_loop.py` | ✅ |
| Interior hole point | The largest inscribed circle centre lies inside its body | `test_region.py` | ✅ |
| χ = 1 − h | Region reports the right Euler characteristic | `test_region.py` | ✅ |
| Truncation corners | circle 0, C-shape 2, box 4 — all right angles; a generated level set carries none | `test_farfield.py` | ✅ |
| Shape does not leak | `Region` records no shape, kind or family; an unnamed hand-built curve is accepted | `test_farfield.py` | ✅ |
| **b₁ = h** | The medial graph's cycle count equals the body count | `test_medial_axis.py` | ✅ |
| Budget identity | Σk = Σ(2−n_c) − 4χ on six known blockings + Fogg's hexagon | `test_singularities.py` | ✅ |
| Flux residual | Bounded by π/4, zero at π/2 and π, maximal at the critical angles | `test_singularities.py` | ✅ |
| Merging conserves Σk | ± pairs cancel; same-sign combine; total preserved | `test_singularities.py` | ✅ |
| Budget closes | Σk of the solved field equals the required total | S2 | ⬜ **all vertex-free cases; class 2 open** |
| Crossing sign | Ellipse gives Σk = −4 at every aspect ratio, and tends to the circle's −4 | `test_singularities.py` | ✅ |
| r_m accuracy | Within 1% of analytic for an annulus and an ellipse | `test_medial_axis.py` | ✅ |
| θ_m accuracy | 180° on an annulus; exactly π − α on a flare | `test_medial_axis.py` | ✅ |
| Finite contact | An exact circle yields a vertex, not an empty graph | `test_medial_axis.py` | ✅ |
| Metamorphic (skeleton) | Whole-problem rigid motion or scaling gives an isomorphic graph | `test_medial_axis.py` | ✅ |
| Index budget | Σkᵢ balances against χ | S2 | ⬜ |
| Block validity | Positive areas, conformal shared edges, every boundary arc covered once | S3 | ⬜ |
| Metamorphic (blocks) | A rigid motion gives an isomorphic block graph | S5 | ⬜ |
| Randomised | 1000 random configurations → 1000 certified block graphs | S5 | ⬜ |

---

## Removed

Deleted 2026-09-24, once the repository was under git so the removal is
recoverable. None of it was on the current code path, and nothing that remains
imports any of it.

| Path | Superseded by | Why |
|---|---|---|
| `geometry/boundary.py` | `geometry/loop.py`, `geometry/airfoil.py` | `Boundary` assumed one airfoil; split upper/lower at min-x, a parameterisation artefact |
| `domain/outer.py` | `domain/farfield.py` | `DomainType` and the C/O/H builders are prescription (D2) |
| `medial/cdt.py`, `medial/graph.py` | `medial/axis.py`, `medial/fields.py` | Label-based medial test; no θ_m, n̂ or touch points; CDT computed then discarded |
| `topology/classify.py`, `topology/design.py` | `topology/singularities.py` | Magic-number classification; `np.full(20, 40)` interfaces |
| `blocking/blocks.py` | `blocking/splits.py` + the S3 builder | Hardcoded C template behind a dispatch that could not branch |
| `mesh/algebraic.py`, `mesh/smoothing.py` | S4 | Stubs returning zeros and their own input, documenting behaviour they did not have |
| `_viz.py` | per-example plotting | Only the removed examples used it |
| `tests/test_geometry.py`, `test_domain.py`, `test_medial_legacy.py`, `conftest.py` | the S0–S3 suites | Exercised the removed modules; no surviving test used the fixtures |
| `examples/01`–`03` | `examples/10`–`13` | Exercised the superseded path |
| 11 stale `output/*.png` | `output/10`–`13` | Produced by the removed examples |

The planning documents were also removed from `.gitignore` and are now tracked:
`Long_term_plan.md`, `MAPS_Implementation_Plan.md`, `MAPS_Proposal_v2.md` and the
Obsidian note. Only `PAPER/references/` stays ignored — 14 MB of third-party
PDFs, not ours to redistribute.

## Changelog

### 2026-09-24 — block builder reviewed and corrected
- Audited the builder with shapely rather than an area sum, and found three
  defects the area sum could not see: open joins at every split/arc junction, a
  name-shadowing bug writing boundary *indices* into outlines as *coordinates*
  (a zero-area spike, invisible to any area check), and collapsed sides kept
  instead of dropped.
- All three fixed. Area error improved from ~1e-5 to ~1e-11; overlap, gap and
  spill now below 1e-15 of the region area on every configuration.
- `validate` strengthened (joins, stray vertices, degeneracy) and
  `validate_strict` added (shapely simplicity, overlap, gap, spill). 37 new
  tests. Suite at **355 passed**.

### 2026-09-24 — block builder
- `blocking/blocks.py`: constructive decomposition off the medial graph. Span,
  vertex, corner-wedge and ring blocks; boundary arcs by short-path stitching.
- Every configuration tiles, area error ≤ 8.5e−5, including bluff bodies, cusped
  trailing edges and refined split densities.
- Fixed a real bug on the way: at a medial vertex with three or more touches the
  collapse took the first two as the edge's pair, so the touch jumped to another
  loop for exactly one sample at each edge end. Now the pair that continues the
  neighbouring sample is chosen; loop changes along every edge went to zero.
- 43 tests in `test_blocks.py`; `examples/14_blocks.py`. Suite at **318 passed**.

### 2026-09-24 — superseded code removed, documents tracked
- Deleted 29 files: the superseded modules, their tests, the four old examples
  and the stale output they produced. Package `__init__` files rewritten; every
  package still imports and the suite is unchanged.
- `.gitignore` reduced to build artefacts, the reference PDFs and the Obsidian
  internals, so the planning documents and the proposal are now under version
  control.

### 2026-09-08 — S3 started
- `blocking/splits.py`: constant-s splits with Fogg's ranking; orthogonality to
  the wall verified against the boundary polyline (worst |cos| < 0.08).
- Medial layer now carries touch indices, and touches are ordered by
  (loop, index) rather than by distance — the old sort was a coin flip, since a
  medial point is equidistant from its touches by definition.
- 12 tests in `test_splits.py`.

### 2026-09-08 — S2 partial; crossing sign resolved
- Singularity solver built: flux residual, class 1 (θ_m crossings), class 3
  (concave-corner switches), finite contact, merging, and the index budget.
- **Budget identity** derived from discrete Gauss–Bonnet and verified on six
  known blockings; it predicts Fogg's k = +2 for a regular hexagon.
- **Crossing sign settled as `k = -|dn|`** by the ellipse budget and the
  ellipse→circle limit; both signed conventions give 0 for an ellipse at every
  aspect ratio and are ruled out.
- Fixed a clamp left in S1 that capped finite-contact θ at π, which made a
  circle give k = −2 instead of −4.
- Fixed the concave-switch reference edge, which was assumed rather than chosen.
- Remaining gap localised to class 2 alone. Suite at **276 passed**.

### 2026-09-08 — class 2 implemented; two detection bugs fixed
- Vertex rule `k = m − 4`, exact on four regular polygons including Fogg's
  hexagon +2; rectangle and L-shape remain open.
- Finite contact at a branch end (Fogg Fig. 10) — a stadium's two
  semicircular ends now give k = −2 each and it balances exactly.
- Crossings within a medial radius of a vertex excluded; this removed every
  spurious crossing on all seven test polygons.
- Fixed the finite-contact tolerance (2% → 0.2% of r_m) and the full-contact
  shortcut, which compared the largest angular gap against 2π/n and so read
  a NACA leading edge as 360° of contact.
- **Vertex rule replaced by `k_V = Σ(2 − n_j) − 4`**, derived from the same
  accounting as the global budget. Six of seven known polygons exact and
  4 of 6 target configurations certify; residuals on the rest fell from
  −3…−11 to ±1–2. Added `examples/13_singularities.py`.
  Suite at **297 passed**.

### 2026-09-08 — S1 complete
- Medial engine rebuilt: both skeletons, full field set, Rigby vertex typing,
  finite contact, and the retract certificate.
- **D9**: the wake cut is not a medial-axis feature. The medial axis terminates
  at *convex* corners of the fluid domain, but a convex corner of a **body** is
  reflex for the fluid, where the corner is the unique nearest point for a fan
  of directions — so no branch emanates. A circular far field around an airfoil
  correctly gives a bare ring. The wake cut is a **decomposition split from a
  reflex corner** (Fogg §4.1), which is S3 work, seeded by the corner data S0
  already provides. This corrects a claim in the review.
- Two accuracy fixes found by the metamorphic tests: touch points are the exact
  feet on the polyline rather than the nearest samples, and a flare tip takes
  θ_m = π − α from the corner angle instead of from the worst-conditioned point
  in the extraction. Together these took θ on a 90° flare from a
  scale-dependent 71–76° to exactly 90.0000°.
- Suite at **247 passed**.

### 2026-09-08 — far-field shapes corrected
- D4 superseded by **D8**: the nameless hull offset was an overcorrection. The
  conventional constructors are back as first-class, the default is a circle,
  and `offset_farfield` is retained for tight or widely spread configurations.
- Measured: the blob and a circle give identical topology; the blob's
  uniform-distance advantage is 3.1% of R at 15 chords; a corner-free default
  made C and H topologies unreachable.
- Added `examples/11_farfield_shapes.py`. Suite at **194 passed**.

### 2026-09-08 — S0 complete
- Review of the previous iteration recorded; root cause identified as a starved
  topology stage (θ_m never computed) feeding a hardcoded blocking.
- Decisions D1–D7 taken.
- New foundation: `Loop`, corner detection, cusp opening, `Region`, far field.
- 132 invariant tests added; suite at **166 passed**.
- Fixed: a body with more than one cusp lost the corners of every face but the
  last, and a face narrower than the corner detector's smallest window read as a
  cusp again on the next pass, reopening it until the safety limit tripped.
  Faces are now tracked by position across cuts and excluded from re-detection.
- Verified: one code path over seven configurations including two- and
  three-element arrangements, a bluff body, and a circle.

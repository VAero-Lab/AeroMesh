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
| **S2** | Singularity solver + certificates | ⚠️ **realised certificate exact; predictor open** | 1.5 wk | 5 of 6 |
| **S3** | Decomposition | ⚠️ **sound on 17/17; all-quad on 9/17 — but soundness proved insufficient, see SZ** | 2.5 wk | met, and not enough |
| **SZ** | Zoned blocking — collar / mid-field / far field | 🔨 **zone 1 done and measured; zones 2–3 not started** | 2 wk | — |
| **S4** | Mesh construction (TFI, TTM, export) | 🔨 **quadrilateralisation done; TFI done inside the collar** | 1.5 wk | — |
| **S5** | Adversarial geometry gate | ⬜ not started | 1 wk | — |
| **S6** | Quality metrics + optimisation | ⬜ not started | 1.5 wk | — |
| **S7** | Bunin φ-field by FEM — *optional* | ⬜ not started | 1 wk | — |

**Ordering is not negotiable: S0 → S1 → S2 → S3.** The hardcoded blocking in the
previous iteration appeared because S3 was attempted while S1 was incomplete and
S2 did not exist.

**SZ comes after S3 and does not replace it.** S3 is still what builds zones 2
and 3; what SZ establishes is that the near-wall region is not S3's to build.
The lesson is narrower than "S3 was wrong": a certificate answers the question
it is asked, and *is this blocking coherent* is not *is this blocking meshable*.

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

- [x] `blocking/search.py` — **Fogg's ranked split search** (sec. 4.2). On a
      medial edge carrying singularities at s₁ < … < s_k, one cut goes in each
      gap — (0, s₁), (s₁, s₂), …, (s_k, L) — giving k+1 cuts that separate the
      k singularities into k spans and bound the vertex regions at either end.
      An edge with no singularity still takes one cut, or the two vertex regions
      at its ends would run into each other. Within a gap the best-ranked
      candidate is taken: n = 0 before n = 1, and within a class the medial angle
      nearer the ideal. Singularities belonging to a *vertex* place no cut — the
      vertex region already contains them.

**The objective it meets.** Each block holds **at most one singularity**, which
is what "logically convex subregions around the singularities" means once the
subregions are spans and vertex regions of the medial graph. Verified across
eight configurations. The search also spends no cut where none is needed, so it
is never the more fragmented option:

| configuration | even spacing | ranked search | singularities |
|---|---|---|---|
| 1 body, smooth far field | 1 | 1 | 1 |
| 1 body, C-shape | 8 | **4** | 1 |
| 1 body, box | 16 | **8** | 2 |
| 2 bodies, box | 21 | **11** | 5 |
| 3 bodies, box | 26 | **15** | 6 |
| airfoil + fuselage, box | 21 | **11** | 5 |

**Two fixes the search forced.**

4. **The vertex-region builder assumed its region was local.** It chose each
   boundary arc by walking both ways from a cut end and keeping the shorter run
   to the next one — which is only the right arc while the region is small
   compared to the boundary. The ranked search makes vertex regions span half
   the domain, and the assumption broke, producing overlaps. Rewritten to trace
   the arcs from the edges' touch arrays, exactly as a span block does, and to
   chain the incident edges by the vertex touch they share. No locality
   assumption remains.
5. **Joins at a vertex's shared touch could land on adjacent samples.** Two arcs
   meeting there are the same point represented twice, half a spacing apart.
   Joins below a couple of sample spacings are now snapped to their midpoint.

**A — block connectivity. Done.** Every side is now either on the region
boundary or shared with exactly one other block, and the two blocks traverse a
shared side in opposite directions. `connectivity()` pairs sides by endpoints
plus a midpoint, which works because adjacent blocks are built from the *same*
`radius_split` call and so carry byte-identical curves; `interface_report()`
states whether the decomposition is conformal. Verified on eight configurations.

Two things this turned up. Blocks came out with **mixed winding** — some
clockwise — because they are assembled by walking the medial graph, which fixes
no handedness; `orient_ccw` now normalises them, which matters both for
interfaces and for transfinite interpolation. And `decompose` was **mutating its
input**, stashing cut indices on the medial graph, so a second call saw the
first call's cuts; cut positions are now local, keyed by a direction-independent
edge key (`graph.edges(node, …)` yields the endpoints in the opposite order for
one of the two ends, which a raw tuple key silently gets wrong).

**B — flares are no longer cut. Done.** Measured and now recorded as fact: along
a flare θ_m is *exactly* π − α at every sample, and r_m is *exactly* linear in
arc length (slope −1/√2, residual 10⁻¹⁴). So every candidate ranks identically
and the wedge a cut leaves is a scaled copy of itself wherever it goes — **no
local criterion can exist**. Fogg marks a convex corner with n_c ≠ 0 as
*resolved*, so no split is needed between a singularity and that corner: the
wedge merges into the neighbouring vertex region and the corner becomes one of
its polygon vertices. A flare is cut only when a singularity sits on it.

| configuration | even spacing | ranked, flares cut | ranked, flares uncut |
|---|---|---|---|
| 1 body, smooth far field | 1 | 1 | 1 |
| 1 body, C-shape | 8 | 4 | **2** |
| 1 body, box | 16 | 8 | **4** |
| 2 bodies, box | 21 | 11 | **7** |
| 3 bodies, box | 26 | 15 | **11** |
| airfoil + fuselage, box | 21 | 11 | **7** |

All eight configurations tile, pass the strict shapely checks, and are
conformal. Each block still holds at most one singularity.

**Still to build.**

**C — constant-ρ offsets. Built; the thickness law needs one more idea.**

`blocking/layers.py` supplies the second family of medial curves. A
**constant-s** curve is a medial radius pair — the cuts the decomposition
already uses. A **constant-ρ** curve is the offset
`C(s) = T(s) + rho·(p(s) − T(s))`, the outer edge of a near-wall block.

*How it is built, and why it matters.* Reading the offset points straight off
the medial samples gives a curve whose segment lengths span **four orders of
magnitude** — medial sampling has nothing to do with boundary sampling — and
smoothing *that curve* to tidy it shortens it, dragging the offset onto the wall
where curvature is high (measured: the wall distance range blew out from 1.5× to
66×). The construction that works instead moves each boundary sample along its
**own outward normal** by ρ times the medial radius seen from it, and smooths
the **scalar radius field** rather than the curve. A scalar can be smoothed
without moving anything sideways, and the normal is exact. `rho_offset` now
yields simple closed curves that contain their body and stay in the fluid, with
the offset distance scaling linearly in ρ.

`peel` splits a region into near-wall layers and the region that remains, so
everything downstream is unchanged machinery: the remaining region is decomposed
exactly as before, and the body count — hence χ and both certificates — is
untouched.

*The measurement that matters.* How uniform a constant-ρ layer is depends
entirely on how much `r_m` varies along the wall:

| configuration | r_m along the body | layer thickness at ρ = 0.05 |
|---|---|---|
| 1 body, box | 2.00 – 3.01 (**1.5×**) | 0.100 – 0.143 (1.4×) |
| 1 body, circle | 1.75 – 1.97 (**1.1×**) | 0.088 – 0.099 (1.1×) |
| 1 body, box R = 15 | 7.50 – 12.52 (**1.7×**) | 0.375 – 0.588 (1.6×) |
| 2 bodies, box | 0.05 – 2.38 (**47×**) | 0.003 – 0.115 (42×) |

For a lone body the medial coordinate behaves exactly as the proposal claims —
the layer is near-uniform without being told anything. In a **slot** `r_m`
collapses, so a constant fraction of it collapses too. That is the coordinate
working correctly and the wrong *thickness law* for a boundary layer.

**The fix to make: cap, do not scale.** Thickness should be
`min(delta_target, rho_max · r_m)`, with `delta_target` set by the flow
(Δy₁, growth rate, N_wall) and the medial term acting as a ceiling that keeps
the layer inside the available space. ρ\* then stops being the sole law and
becomes what it should be — the safety margin, and the quantity S6 varies.

**Also still open on C.** The offset curve, though simple, carries enough
small-scale structure to add medial vertices to the peeled region (a body in a
box goes from 4 blocks to 15), and two of four configurations show a ~0.5%
coverage error after peeling. Both point at the same thing: the offset needs to
be a *smoother* curve than the one the radius field currently gives.

**D. Midpoint-subdivision templates for m-gons.** *(needed before TFI)* Vertex
regions are m-sided with m ≥ 3; TFI needs quads. This is **the one prescribed
construction the project permits** and must be named as such where used.

**E. Finite-contact vertex regions.** *(completeness)* Reported in `notes`, not
silently skipped.

**F. Close the S2 budget residuals.** *(parallel)* An L-shape off by one and two
configurations by ±1–2; the suspicion is the single-sample evaluation of `n_j`.

Then **S4**: real TFI per block, TTM elliptic smoothing with the wall spacing as
a boundary condition, tanh clustering from Δy₁, r and N_wall, export through
`meshio`.

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

## SZ — Zoned blocking: the collar 🔨 started

**Why this stage exists.** S3 produced blockings that were *certified* and
unusable. On the 17-configuration gallery the block boundaries were not straight,
the trailing-edge treatment was awkward, and the near-wall blocks had a topology
that would be hard or impossible to mesh. The certificate was not lying — the
index budget balanced, the areas were positive, the interfaces were conformal.
It was measuring the wrong thing: **a sound block graph is not a meshable one.**

The diagnosis, agreed with the author: S3 applies one global decomposition to a
domain whose requirements are not global. Fogg's medial-axis decomposition is a
*CAD* meshing method — it optimises for well-shaped quads on a part. CFD wants
something else near the wall: a layer of extreme anisotropy, cells orthogonal to
the surface, Δy₁ fixed by y⁺, and nothing the blocking does allowed to disturb
that. Those two objectives fight, and asking one decomposition to serve both is
why the result served neither.

So the blocking is staged by zone instead:

| Zone | Scope | Requirement that dominates |
|---|---|---|
| **1 — collar** | wall to `δ(s)` | wall orthogonality, Δy₁, anisotropy; topology forced by the wall |
| **2 — mid-field** | collar to the medial features between bodies | resolving the gaps and wakes *between* bodies |
| **3 — far field** | mid-field to the truncation curve | smooth expansion, cheap cells |

This is a change of *strategy*, not of the one rule. Zone 1's topology is still
found, not prescribed: the number of rays, where they go, how many blocks result
and what shape they are all come from the detected corners, their fluid angles
and Fogg's element counts. What is *forced* is that a band around a closed curve
is an annulus — that is a consequence of the geometry being a closed loop, not a
template chosen for aerofoils. Nothing in `collar.py` names a configuration, and
the same code runs unchanged on sharp and blunt trailing edges.

### Built

- [x] `blocking/collar.py` — zone 1 end to end: offset curve, rays, blocks,
      node distributions, TFI meshes, and the validation that proves it.
- [x] **Thickness** `δ(s) = min(δ_target, ρ_max·r_m(s))` — the flow sets it, the
      medial radius caps it to the room available. Carried over from S3.
- [x] **Offset by smooth-and-reproject.** Offsetting each wall sample along its
      own normal gives a curve whose segment lengths vary by 19,000:1 near a
      trailing edge. Alternating Laplacian relaxation with re-projection to
      distance δ converges to wall distance 0.0199–0.0200 against a target of
      0.02, with a **segment-length ratio of 1.0** and maximum turn 15.5°.
- [x] **Rays from the corner's normal cone, clamped.** A reflex corner of fluid
      angle α asking for `n_c` elements emits `n_c + 1` rays. Their directions
      are the interior cross directions **clamped into `[π/2, α − π/2]`** —
      Fogg's own rule (CAD 2015 §4.1: the radii "are almost aligned and
      perpendicular to the edges forming the corner"). The cone is exactly the
      set of directions for which the corner is the nearest boundary point, so
      a ray inside it cannot cross a neighbour's territory.
- [x] **Blocks.** Between rays of *different* corners: a four-sided band —
      wall, ray, outer, ray. Between rays of the *same* corner: a three-sided
      wedge with its apex on the corner, split into three quad sub-blocks about
      an interior irregular node. Decided by measured arc length, not by corner
      index, so a one-corner body and a two-corner body both work.
- [x] **The wedge centre is the Fermat point** of its three corner nodes — the
      point minimising total distance, where the three spokes meet at 120°.
- [x] **Node distributions from a single cache.** Every ray is resampled once,
      tanh-clustered from Δy₁ by bisection on the stretch factor, and both
      blocks sharing it read the same node list. Position mismatch across a
      shared ray is exactly 0.0.
- [x] **Wall-to-outer correspondence inverted, not assumed.** The outer curve is
      *longer* than the wall it came from — 1.83× around a leading edge, 1.39×
      at mid-chord — so sampling both uniformly pairs nodes that are nowhere
      near each other and the grid lines fan. The correspondence is known
      because the outer curve was built as the offset, so it is inverted and the
      outer side is sampled where the wall nodes project. Endpoints pinned,
      monotonicity enforced.
- [x] **Curvature-adapted along-wall spacing** — equidistributes
      `1 + strength·|κ|·L/n` instead of arc length alone, so a leading edge gets
      the points it needs without the whole band paying for them. Default
      strength 2: the band's minimum angle goes 67.0 → 73.1°, coverage 99.89 →
      99.97%, and the worst case in the battery (NACA 6409) 53.0 → 58.7°, for
      3% on peak aspect ratio. The optimum is flat — strength 4 gives 74.5° in
      the band, 8 gives less.
- [x] `validate_mesh` — **coverage, overlap and outside-area**, not just
      per-cell quality. See the lesson below.

### Measured

NACA 2412, blunt trailing edge, δ = 0.02, Δy₁ = 1e-4, 25 nodes across:

| stage | min angle | max angle | cells < 45° | median |
|---|---|---|---|---|
| first working version | 17.7° | 166.0° | 23.7% | 58.1° |
| + Fermat centre, + cone clamp, + smoothing | 35.4° | 145.7° | — | — |
| + correspondence inverted, **no smoothing** | **59.0°** | **121.0°** | **0.00%** | **85.4°** |
| the same, 200 Winslow sweeps | 50.4° | 128.7° | 0.00% | 89.0° |

Eight aerofoils × both trailing-edge kinds, at the defaults (curvature 2.0):
minimum angle **57.8–59.4°**, maximum 120.6–122.2°, **no cell below 45°**,
median exactly 90.0°, all Jacobians positive, coverage 99.95–99.97%, aspect
ratio 103–104 (δ/Δy₁ = 200, so ~100 is the anisotropy asked for, not a defect).
The sections are 0012, 2412, 4415, 4412, 6409, 6412, 23012 and 63012 — the
high-camber ones are in the list deliberately, see the changelog for 21 Oct.

### Lessons recorded, because they cost real time

1. **Aggregate per-cell metrics hide placement errors.** A folded band reported
   "all Jacobians positive, minimum angle 25.6°" while covering the collar
   **204%** — it was doubled over on itself. Three separate folds passed the
   per-cell checks and were caught only by coverage/overlap/outside. The author
   caught the first one before the metrics did, from a picture. Any new mesh
   stage gets a coverage check before it gets a quality number.
2. **Smoothing is not the fix for a bad parameterisation.** Winslow smoothing
   *worsened* the extremes once the correspondence was right (59.0° → 50.4°).
   Order: blocking → irregular-node placement → parameterisation → *then*
   smoothing. Smoothing applied earlier hides the defect it should expose.
3. **Clamp is not replace.** Having found that Fogg clamps cross directions into
   the normal cone, the first implementation replaced them with the cone
   boundaries wholesale — which deleted the sharp trailing edge's middle ray and
   left its apex cell at 163°.
4. **Assembled Laplacian smoothing of the wedges is harmful** — and the first
   measurement of that was itself wrong, which is the more useful half of the
   story. `smooth_assembly` identified its held nodes by matching the fixed
   sides' *sample positions*, and the mesh nodes are resampled along each side
   and never land on them, so nothing was held, the shared boundary drifted, and
   the measurement blamed the smoother for what the loose boundary did. A test
   written to assert the boundary is held is what found it. With the boundary
   genuinely held — coverage 99.97%, Δy₁ preserved at 1e-4 — the honest result
   is still negative: the wedge's minimum angle goes 59.0° → 34.3° / 38.2° /
   41.2° at 20 / 60 / 200 sweeps, at either relaxation factor tried. That is
   plain Laplacian behaving as expected on a graded mesh: the interior relaxes
   towards uniform spacing while the boundary stays clustered, and the cells
   between them shear. `smooth_assembly` is kept, correct and unused; the right
   form of the idea is Winslow with control functions, which preserves the
   grading, and that is deferred rather than attempted.
5. **Measurement beat the reasoning, twice.** The wedge's ray is split at the
   node-index midpoint, which under wall clustering sits at 18% of the ray's
   *length* — three visibly unequal sub-blocks, which looked like a defect worth
   correcting. Splitting by length instead was measured and is worse at every
   value tried: 59.0° at the node midpoint against 57.4° / 53.6° / 52.0° / 47.2°
   at 30 / 40 / 50 / 80% of length. A related derivation, "`n_across` must be
   odd", was simply wrong — it assumed the two halves of a ray must match each
   other, when only opposite sides *within* each sub-block must.

### Open

- [ ] Zone 2 — mid-field, using the medial features *between* bodies.
- [ ] Zone 3 — far-field expansion.
- [ ] Interval assignment across zone interfaces (the integer problem).
- [ ] The wedge's irregular node is still the binding constraint (59°, against
      67°+ in the bands). A Winslow or angle-based smoother on the assembled
      wedge, which may move that node, is the remaining idea; plain Laplacian is
      ruled out by measurement (lesson 4).
- [ ] Multi-body collars: built per body, so they work, but the zone-1/zone-2
      interface between two near bodies is untested.

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
| **Gauss–Bonnet, ground truth** | Σ(val−4) = Σ(2−q) − 4χ on hand-built grids with 0–3 holes | `test_certificate.py` | ✅ |
| **Realised certificate** | The blocking balances its own index budget | `test_certificate.py` | ✅ **17/17** |
| **No hanging node** | No block corner lies inside another block's side | `test_certificate.py` | ✅ |
| **No stranded side** | A side called "boundary" lies on the region boundary | `test_certificate.py` | ✅ |
| **Side key reversal-invariant** | Two blocks sharing a side compute the same key either way round | `test_certificate.py` | ✅ |
| **Pinned endpoints** | An analytic endpoint (a cut's foot) never moves when a join is snapped | `test_certificate.py` | ✅ |
| **No buried corner** | No C0 boundary corner lies inside a block side | `test_certificate.py` | ✅ aerofoils; ⚠️ polygons |
| **No flat block corner** | After canonicalisation no block corner is within 5° of 180° | `test_certificate.py` | ✅ |
| **n_c determinism** | A regular m-gon gives the same corner term at any rotation | `test_certificate.py` | ✅ |
| **n_c ≥ 1** | No corner ever asks for zero elements | `test_certificate.py` | ✅ |
| **Thickness cap** | δ ≤ ρ_max·r_m *pointwise*, so the offset is simple | `test_certificate.py` | ✅ |
| **Resolution never breaks** | Corner resolution is rolled back if it would make the blocking unsound | `test_certificate.py` | ✅ |
| **Layer adds no index** | Inserting the near-wall layer leaves the certificate unchanged | `test_layers.py` | ✅ |
| **Subdivision is uniform** | Every block subdivided or none — a partial subdivision is not conformal | `test_templates.py` | ✅ |
| **No smooth peak as a corner** | A detected corner must be window-independent *or* fit the model exactly | `test_corner_smooth_peaks.py` | ✅ |
| **Corner detection converges** | Refining the sampling can only remove a false corner, never add one | `test_corner_smooth_peaks.py` | ✅ |
| **r_m positive on the boundary** | Every boundary sample's medial radius exceeds the local point spacing | `test_medial_radius_degeneracy.py` | ✅ |
| **Collar covers its own area** | Coverage within 0.5%, no overlap, nothing outside — *not* per-cell quality | `test_collar.py`, `test_collar_spacing.py` | ✅ |
| **Shared sides are held** | Smoothing an assembly must not move a node on a fixed side | `test_collar_spacing.py` | ✅ |
| **Δy₁ survives** | The first cell height is what was asked for, after every stage | `test_collar_spacing.py` | ✅ |
| Block validity | Positive areas, conformal shared edges, every boundary arc covered once | S3 | ✅ |
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

### 2026-10-21 — zoned blocking; two defects the narrow battery hid

**Strategy.** S3's certified-but-unmeshable blockings led to zoned blocking
(new stage **SZ**): zone 1 the near-wall collar, zone 2 the mid-field, zone 3
the far field. Zone 1 is built and measured; zones 2 and 3 are not started.
The certificate was never wrong — it was answering a different question than
"can this be meshed", and nothing in it could have caught this.

**Collar (zone 1), `blocking/collar.py`.** Offset by smooth-and-reproject,
rays from the clamped normal cone, bands and wedges decided by measured arc
length, Fermat-point wedge centres, one shared node cache per ray, the
wall-to-outer correspondence inverted rather than assumed, and
curvature-adapted along-wall spacing. Eight aerofoils × two trailing-edge
kinds: minimum angle **57.8–59.4°**, maximum 120.6–122.2°, **no cell below
45°**, median exactly 90.0°, all Jacobians positive, coverage 99.95–99.97%.

**Defect 1 — a smooth leading edge detected as a corner.** `detect_corners`
gated on the extrapolated angle alone. Its model, `T(w) = a + c·w` with "no
saturation at any w", holds in the continuum and fails on a *sampled* curve as
soon as the window exceeds the local radius of curvature: all the turning is
already inside the narrowest window, `T` flattens, and the fit reports the
saturated total as a corner angle. A 10:1 ellipse — no corner anywhere — was
detected as a 38° corner at 1200 samples; a 50:1 ellipse as 154°. On NACA 6409
this put a false 20.5° corner on the *leading* edge, which added three rays and
two wedges to the collar and folded the mesh: minimum angle **0.5°**, aspect
ratio 4779, negative Jacobians.

The fix is the one invariant that separates the cases: a real corner's angle
does not depend on the window. Refining a square 16-fold leaves 90.0°
untouched; the 10:1 ellipse decays 74.7 → 38.0 → 12.4 → 2.5 → 0.5. So the
lower and upper halves of the window band are fitted separately and must agree
— **except** that two genuine corners closer than the widest window straddle
it and disagree too, so rejection also requires a poor fit, which straddling
does not produce. Both conditions, never one. The residual was already computed
and returned and nothing had ever gated on it.

**Two things about this fix were wrong on the first attempt, and the suite
caught both.**

*The threshold was inside the real range.* Set at 0.9 from a quick reading of
shoulder samples, it discarded the crease in a per-body offset far field — a
genuine corner the far-field tests exist to document. Measured properly, at the
indices that survive suppression rather than at their shoulders:

| | stability | rel. residual |
|---|---|---|
| real — polygons, 3–12 sides, 600–2400 samples | 1.000 | 0.0000 |
| real — two corners inside one window | 1.000 | 0.0000 |
| real — far-field crease | 0.824–0.890 | 0.0215–0.0347 |
| false — cambered nose (6409, 6412) | 0.265 | 0.0500 |
| false — 10:1 ellipse | 0.402–0.608 | 0.046–0.055 |
| false — 20:1 ellipse, 2400 samples | 0.608 | 0.0464 |

The gap is 0.608 to 0.824; the threshold is 0.7.

*The gate ran before non-maximum suppression.* Rejecting a peak merely let the
next flagged sample along the same nose win instead, so the feature survived
one sample over and the rejection achieved nothing — the 10:1 ellipse was still
detected after being rejected. Judged after suppression it is now rejected at
every resolution tested.

What remains detected is thin ellipses at coarse sampling, and that is the
honest answer rather than a residual bug: a 20:1 ellipse at 600 samples has a
nose radius below its own point spacing, so the polyline really does turn
through 131° in a few segments and nothing computed from it can know otherwise.
Refinement removes it, which is the only guarantee worth making.

Note on what the earlier tests could not have caught: the aerofoil trailing-edge
corners are *constructed* and injected by `_merge_corners`, so their zero
residual is an assignment, not a measurement. The straddle case needed its own
geometry to test at all.

**Defect 2 — one degenerate medial sample zeroing the thickness field.**
`medial_radius_on_boundary` takes the *minimum* r_m over the medial samples
touching each boundary sample — correct, and maximally sensitive to one bad
sample. NACA 6409 and 6412 each produce exactly one medial sample with
`r_m = 0` and `θ_m = 0` out of ~2075, an artefact of taking the medial axis of
the *polygon* instead of the curve it approximates. That one sample set r_m = 0
on the wall near x = 0.38, the cap `ρ_max·r_m` took δ to zero with it, and the
collar's outer curve touched the wall: aspect ratio 8715, negative Jacobians.
Samples below the loop's own point spacing, or with a collapsed separation
angle, are now treated as *missing* and filled by the interpolation that was
already there for untouched samples. The genuine slot minimum on a two-element
configuration is unaffected — it sits orders of magnitude above the spacing.

**Both defects were invisible because the battery was too narrow.** Every
aerofoil tested before was 0012/2412/4415-like. Widening it to 6% camber found
two independent failures immediately, and the second only became visible once
the first was fixed. The battery is now eight sections × two trailing edges.

**Measured and rejected.** Splitting the wedge ray by length rather than node
index (worse at every value: 59.0° against 57.4/53.6/52.0/47.2°). Assembled
Laplacian smoothing of the wedges (59.0° → 34.3° with the boundary correctly
held; the first measurement of this was itself invalid because the boundary was
not held, which a test found). Winslow smoothing of the collar at all, now that
the parameterisation is right (59.0° → 50.4°).

### 2026-10-16 — second audit round: four defects, one of them mine

Three reviewers were run against the code in parallel. Two converged
independently on the same structural finding, and a fourth round found a
high-severity bug introduced earlier the same day. All four are fixed, each with
a test naming the reproduction.

| defect | how it showed | fix |
|---|---|---|
| **The near-wall layer was built *inside* the body.** `_normals_along` decides which side is fluid by stepping off the wall, and the probe was scaled to the *largest* layer thickness on the arc. A 0.3-chord flap is 0.036 thick while its layer reaches 0.194, so the probe stepped clean through the body, found fluid on **both** sides at every sample, and the tie returned the inward normal. | `tiles=True`, relative error 4.5e-4, **nothing in the notes**. At `rho_max=0.3` — inside the range S6 is meant to vary — *every* aerofoil case failed. The tests missed it because they all passed `delta_target=0.02`, which caps the probe; the failing path was the documented default of `None`. | probe scaled to the *smallest* thickness and capped by the arc's own sampling; the vote discards samples that answer ambiguously and runs over the whole arc |
| **A detected corner's index was one sample off.** At a polyline corner the extrapolated turn is a *plateau* — on a unit triangle two adjacent samples both read exactly 120.000° — so the non-maximum suppression's stable tie order kept the lower index, whose own vertex turn was 0.00°. The angle was right; only the position moved. | Everything downstream matches corners by position. `corner_cuts` found no medial foot within tolerance and gave up silently; `buried_corners` reported corners that were in fact resolved — **10 of 10** on polygonal bodies were this artefact. | snap the index onto the vertex carrying the turn, but only where the turn is genuinely concentrated on one sample, so a smooth nose or a cusped TE is left alone |
| **`canonicalise` merged a genuine 120° wall corner.** Its guard was `coincidence_tol` = 1e-9·scale — a tolerance for points that should be bit-identical — against a corner position a full sample away. And a cut's foot at a *reflex* corner can leave almost tangentially (179.6° measured), so a cut-into-wall join reads as flat. | A triangular body lost its conformality: certificate −2, one hanging node, two stranded sides 2.0 from the nearest wall. The rollback caught it — silently, so `m_gons` and `design_gap` were then reported on a representation with spurious sides. | merge only two *boundary arcs*; guard against the boundary's own spacing; say so when the rollback fires |
| **`design_index` was not the residual of an identity.** It took the block term from the corner angles and the required term from `index_certificate`, whose boundary term counts one element per corner — differencing two different models — and omitted the interior term, which is not optional because `round` is not additive: three corners of 100°, 130° and 130° at one node ask for 3 elements, not the 4 that 360° would. | It reported `−Σ(n_i − 1)`: a quality number wearing an identity's clothes. | all three terms from the design counts, interior term included. Residual is now **exactly 0 on all 7** configurations, as an identity must be, and `gap_to_realised` carries the quality number honestly |

**One speculative fix was reverted, and a second was wrongly condemned with
it** — worth recording, because the lesson is about attribution rather than
about either change.

Truncating `boundary_arc` at its first arrival at the last touch, to stop a
non-monotonic touch array overshooting, is **catastrophic on a closed edge**:
there `i1[-1] == i1[0]`, so the truncation fires at index 0 and the arc
collapses to a single point — a ring block with sides of one point each,
relative error 1.000, and an `IndexError` out of `_normals_along`. Reverted, and
it should not be reattempted in that form.

The other change — taking the **short boundary path** to a flare's corner
instead of following the touch array — was reverted at the same time, on the
evidence that it "broke a working ring". It did not. The ring was broken by the
truncation, which was active in the same tree; the two changes were measured
together and the blame went to the wrong one. Re-applied on its own, the ring is
fine and the curated matrix is unchanged, and it fixes a real double-coverage:
near a flare's corner `r_m` goes to zero and the exact feet jitter, so the touch
array runs 2993 → 2949 → 2954 and covers five samples twice. It is also a *rare*
defect — it moved one case in 35 — so it is kept on its own merits and not as
the answer to the self-touching regions, which have a different cause.

**What the randomised sweep says.** Bodies drawn at random — 99 NACA sections,
56 ellipses, 34 polygons of random vertex count and radial jitter — at random
scale, incidence and offset, in box, C and circular far fields, 101 valid draws
(an overlapping placement is redrawn, since overlapping bodies are not a meshing
problem). The invariant asserted is not "nothing fails" but that **nothing fails
silently**: every configuration is either sound or says which feature it could
not handle.

| | of 101 valid random draws |
|---|---|
| retract certificate, b₁ = h | **101 / 101** |
| blocking sound | 67 / 101 |
| layer sound | 67 / 101 — never breaks a sound blocking |
| quadrilateralisation sound | 66 / 101 |
| no buried corner | 98 / 101 |
| all-quadrilateral | 57 / 101 |

`b₁ = h` holding on every draw is the strongest number in the table: it is the
one certificate that is a property of the medial axis alone, independent of
every choice made downstream.

**The one remaining defect class, characterised.** Of the 35 failures, 29 are a
**vertex region whose outline touches itself**, which also registers a hanging
node and a ±2 certificate residual. 21 of the 35 are single-body cases, so this
is not a multi-body or slot phenomenon. The area is almost always right — most
such cases still tile to 1e-10 — because the outline runs out along a stretch of
wall and back, and the overshoot and the return cancel. Only a simplicity test
sees it, which is why `validate_strict` was made to *measure* the defect rather
than raise on it, and why `_is_simple` now gates the region and notes it.

Two degeneracies were isolated by name, and they account for 5 of the 35:

* **A self-loop medial edge incident to a real vertex.** `networkx` yields a
  self-loop *once*, but an edge that leaves a vertex and returns to it must
  contribute **two** legs to that vertex's region. With one, the chaining walks
  its arc out and back. 4 cases.
* **Two of a vertex's touches on the same boundary sample**, with both legs'
  arcs leaving it in the same direction along the wall, so they double-cover
  that stretch. Measured on one case: two PRIMARY edges both with
  `i1[0] = 442`, their arcs running 442 → 593 and 442 → 1133. 1 case.

The other 24 are **not yet explained** — the vertex named in each has neither a
self-loop nor duplicated touches by the obvious test, so the cause is a third
thing. What is *not* the cause, measured: the non-monotonic touch array near a
flare's corner. That is a real double-coverage (2993 → 2949 → 2954, five samples
covered twice) and is now fixed by taking the short boundary path to the corner,
and it moved exactly one case in 35.

This is a redesign of `_vertex_block`'s arc construction rather than a bug fix:
the arcs have to *partition* the boundary between consecutive touches by
construction, which tracing each leg's own touch array independently does not
guarantee. That function's chaining has now been reworked twice; a third pass
should change the construction, not the traversal.

**Timing, measured for the first time.** Speed is one of the three novelty
claims and had never been instrumented. End to end, on one core:

| configuration | medial | blocks | layer | quads | **total** |
|---|---|---|---|---|---|
| 1 aerofoil / C | 1.05 | 0.17 | 0.13 | 0.71 | **2.06 s** |
| 1 aerofoil / box | 0.99 | 0.24 | 0.18 | 0.39 | **1.81 s** |
| 2 elements / box | 1.73 | 0.33 | 0.42 | 0.63 | **3.11 s** |
| 3 elements / box | 2.19 | 0.48 | 0.90 | 1.10 | **4.66 s** |
| 3 bodies / circle | 1.89 | 0.41 | 0.85 | 0.99 | **4.14 s** |

That is after a 2.2x speed-up found by instrumenting rather than guessing. The
layer stage was 4.6 s on the three-element case and is now 0.90 s, because
`delta_at` was called once per arc point and each call did its own
`segment_distance` and tree query -- several hundred times per side. Vectorising
it over the arc took that stage 5x faster. A second cost was `_side_kinds`
rebuilding a KD-tree once per *side*; one tree for the system is 20% off the
total. Sub-sampling the containment vote in `_normals_along` to 256 points was
tried and is worth almost nothing (8.29 s to 8.23 s), so it is kept only because
it bounds the work, not because it was the bottleneck.

The medial stage is now the dominant cost and has not been touched.

**Not reproduced on the target matrix.** All seventeen curated configurations —
aerofoils, two- and three-element, fuselage with wing, three bodies, discs and
regular polygon bodies, in all three far-field shapes — are sound at all three
stages, and ten of seventeen reach all-quadrilateral. The sweep's failures come
from geometry nobody would mesh: random polygons with radial jitter at random
incidence. It is a stress test, and it is doing its job.

**Also hardened**: `validate_strict` repairs an invalid polygon with `buffer(0)`
before measuring, so the check that exists to measure a self-touching outline is
no longer the one thing that cannot survive one; `_vertex_block` searches every
chaining of its legs rather than taking the first, since a touch shared by more
than two legs leaves a genuine choice and `nxt[0]` was a coin flip; `to_quads`
uses transfinite interpolation for a four-sided block, because a harmonic map
onto a disc is violently distorted for an aspect ratio of 54 and a perfectly
well-shaped quad still folded through it; and a self-crossing polygon is
rejected before Triangle sees it, since that library reports an internal error
and then **segfaults**, taking the interpreter with it.

### 2026-10-16 — realised certificate; layer by block split; quadrilateralisation

**The headline: the certificate moved from the predictor to the product.**

The discrete Gauss–Bonnet identity for a quad mesh,

    Σ_interior (valence − 4) = Σ_boundary (2 − q) − 4χ,

holds for *every* valid quad mesh — it is Euler's formula rearranged, verified
here on hand-built grids with 0 to 3 holes. That makes it useless as a
prediction and valuable as a consistency test of a blocking's **own**
bookkeeping. `BlockSystem.index_certificate()` computes it from the blocks, and
it resolves defects an area test cannot: the crack it found between two vertex
regions on a fuselage-plus-wing was 5e-7 of the region area, below the 1e-6
threshold `validate_strict` uses, and it was a genuine hole.

**Defects found and fixed.** Each was found by measurement, and each has a test
naming the reproduction.

| defect | how it showed | fix |
|---|---|---|
| `clean_sides` snapped a *cut*'s analytic endpoint to a midpoint | a lens 0.24 long and 7e-4 wide between two vertex regions; cert −2 | `pinned` sides: the approximate end moves onto the analytic one, never the reverse |
| `snap_tol` took the **max** spacing over all boundaries | a far field 40 across at 720 points set a 0.11 tolerance that erased real sides on a 0.3-chord flap | `local_spacing`: the tolerance is read where the point is |
| `corner_valences` measured to boundary *samples* | every cut foot sits half a spacing off the wall, so every wall corner read as interior; cert off by −12 | measure to the boundary **polyline** |
| `_side_key` used `side[len(side)//2]` | not reversal-invariant: a 118-point side keys as 59 one way and 58 the other, so ten of the layer's sides paired with nothing | the **arc-length** midpoint |
| `interface_report` could not see an unpaired interior side | `paired + boundary == n_sides` is a tautology; an unglued side is filed as "boundary" | a `stranded` check: a boundary side must lie on the boundary |
| `wall_distance` smoothed the radius field | smoothing *raises* the cap where it is the only thing preventing a collision — 12.5× in a slot, and layers intersect from ρ_max = 0.3 | `ρ_max · min(smoothed, raw)`; the guarding test had used `r_m.max()` and was vacuous |
| `corner_elements` was non-deterministic at a tie | a regular octagon's angle is exactly 2.5 quarter-turns at every corner; rotating it swung the corner term over −6, −7, −8 | snap to the tie, then round half up, clamp to ≥ 1 |
| `corner_cuts` matched the touch **index** | the index is the sample nearest the foot; ±1 admits the run's ends, where the foot has left the corner, so the corner stayed buried anyway | match the exact **foot** position |
| `decompose` ran one resolution pass, unguarded | a pentagon body converged on two kinked wall sides with every check passing | loop to a fixed point; **roll back** a refinement that makes the blocking unsound; note what is left |
| `n_sides` was a representation count | eight of 23 block corners on one blocking were flat joins at exactly 180°; blocks reported `[8,8,8,7]` for logical `[5,6,5,5]` | `canonicalise()` merges flat non-corner joins |

**Near-wall layer, second attempt (item C).** Peeling the layer off and
re-deriving the medial axis was the wrong order of operations: the offset curve
carries enough small-scale structure to grow medial branches of its own (a body
in a box went 4 blocks → 15), and smoothing the curve to tame it is worse —
averaging a curve shortens it, which dragged the offset onto the wall and took
the wall distance from a 1.5× range to 66×. `insert_layer` now **splits the
blocking** instead. Topology is inherited, not recomputed, and the split is
exact: a medial radius meets the wall at a right angle and a constant-ρ curve is
parallel to it, so the offset crosses a cut at the point reached by travelling δ
along it from its wall foot — an arc-length step with no tolerance in it.

Two things that only the certificate caught. Trimming must be a property of the
**cut**, not of the block: the wake region of a C far field touches the wall
only at the trailing-edge *point*, so it has no wall side, kept its cuts at full
length while its neighbours trimmed theirs, and overlapped their layer by 2e-8
of the region — two orders below the tiling check. And a cut between two wall
runs needs trimming at **both** feet; one trim overwrote the other.

**Thickness law (item C, closed).** δ(s) = min(δ_target, ρ_max·r_m(s)).
Constant-ρ is the medial coordinate working correctly and the wrong thickness
law — it is near-uniform for a lone body (1.4× over a chord) and collapses 37×
in a flap slot, because a boundary layer's thickness is set by the flow, not by
how much room there happens to be. Measured: with δ_target = 0.02 the layer is
uniform to 1.0× for a lone body and varies 6.5× for a two-element case, entirely
in the slot where the cap must bind. And δ < r_m pointwise is not merely prudent
but **sufficient** for the offset to be simple, which is why the cap has to cap.

**Quadrilateralisation (item D).** Midpoint subdivision of an m-gon into m
quads — the one prescribed construction allowed, and named as such. Two things
had to be got right.

It must be **uniform**: subdivision splits every side at its midpoint, so it is
conformal only if both blocks sharing a side do it. Subdividing the m-gons and
leaving the quads alone puts a node in the middle of a side the neighbour runs
straight past.

And straight spokes to a centre fold, at **any** centre. The area centroid of a
layer strip landed exactly on one of its own corners; the inscribed-circle
centre is interior by construction but the straight lines to it still leave a
vertex region wrapping a quarter of the domain; even a four-sided span around
half an aerofoil folded — 5 of 8 blocks. The construction that cannot fold is
**Tutte's**: a discrete harmonic map of a triangulated disc onto a convex region
is injective, so triangulate the block, send its boundary to the unit circle by
arc length, solve the Laplace system, take the spokes as straight radii there,
and map them back. Folding went 10/42 → 3/54 once Triangle was told (`Y`) not to
insert Steiner points on the boundary — without that, boundary vertices get no
Dirichlet value, are solved as free, and Tutte's hypothesis fails, which is why
refining the spokes had not converged (10, 8, 4, 7).

**Where it stands.** Over 17 configurations — aerofoils, two- and three-element,
fuselage-plus-wing, three bodies, discs and regular 3/5/8-gon bodies, in box, C
and circular far fields — the blocking, the layered blocking and the
quadrilateralised blocking are all sound (tiles, strict, conformal, certificate
0, no hanging or stranded sides). 9 of 17 reach **all-quad**. 557+ tests pass.

**Open, with the diagnosis recorded.**

* **The reflex corner.** A blunt trailing edge is two reflex corners of
  α ≈ 261.8°, wanting n_c = 3 elements and so two grid lines each. The set of
  directions in which the corner is the nearest boundary point is its normal
  cone, of width exactly α − π = 81.8°, while the wanted cross directions span
  α — so the outermost are *necessarily* unreachable by a medial radius, by
  (n_c·π/2 − α)/n_c = 2.7°. Fogg's own answer is to clamp to the cone
  (sec. 4.1: the radii are "almost aligned and perpendicular to the edges
  forming the corner"), which gives sectors 90/81.8/90 — a ratio of 1.10 — and
  that is now what `corner_cuts` asks for. What still fails is downstream: the
  two *downstream* rays from the two corners converge at 5.45° and cross
  0.021 chord behind the trailing edge, so forcing both lines produces
  near-coincident feet and the blocking stops being sound. It is rolled back and
  reported. The identified route is to resolve the corner **inside the near-wall
  collar**, where the α − π restriction does not apply because a ray that stops
  at distance δ never has to reach the medial axis.
* **The design gap.** `design_index()` computes what the blocking *asks for* from
  its corner angles, `Σ(2 − round(θ/90°)) − 4` per block. It agrees with the
  singularity field's budget (+2 for an aerofoil in a box) while midpoint
  subdivision delivers +5; the gap is exactly the unresolved reflex corners, and
  is now reported as `quality()["design_gap"]`.
* **The predictor.** `solve_singularities` is off by −1 to +3 on four
  configurations while the realised certificate is exact. Diagnosed:
  `_flow_index_near` samples θ_m at `min(r_m, L/2)`, which on a far-field edge
  resolves to the edge midpoint, where the exterior axis is scale-degenerate
  (θ_m 172.9°–180.0°) and n collapses to 0. The identified route is a per-tile
  angle index, which makes the budget an identity of the tiling rather than a
  sum of independent local estimates.
* **FINITE_CONTACT is unreachable** from an exterior axis — measured across the
  matrix, exterior vertices are only NORMAL and CORNER. It arises only in a
  body's *interior* axis (an exact circle). Item E is therefore closed as moot
  rather than implemented; the guard in `decompose` stays.

### 2026-09-24 — constant-ρ offsets
- `blocking/layers.py`: ρ-offset curves and `peel`. Offsets move each boundary
  sample along its own normal by ρ·r_m, smoothing the radius field rather than
  the curve — reading points off the medial samples gives segment lengths
  spanning 1e4, and smoothing the curve itself drags it onto the wall.
- Measured: r_m varies 1.1–1.7× along a lone body, so a constant-ρ layer is
  near-uniform; it varies 47× with a slot present, so the layer thins 42×.
  Correct for the coordinate, wrong for a boundary layer — the thickness law
  needs to be `min(delta_target, rho_max·r_m)`, a cap rather than a scale.
- `peel` is clean on a lone body and a bluff body; two of four configurations
  show ~0.5% coverage error and the peeled region gains medial structure.
- 17 tests in `test_layers.py`.

### 2026-09-24 — connectivity and uncut flares
- Block connectivity: sides paired by endpoints, `interface_report()` states
  conformality. Conformal on all eight configurations.
- Blocks normalised to counter-clockwise winding; they were mixed, which breaks
  both interfaces and TFI.
- `decompose` no longer mutates the medial graph. It stashed cut indices there,
  so a second call inherited the first call's cuts — found because the example
  calls it twice. Edge keys are now direction-independent, since
  `graph.edges(node, …)` reverses the endpoint order for one end.
- Flares left uncut unless a singularity sits on them, on the measured grounds
  that no local criterion can choose a position. Block counts roughly halve
  again (16 → 4 for a body in a box).
- 47 tests in `test_connectivity.py`.

### 2026-09-24 — ranked split search
- `blocking/search.py`: cuts placed one per gap between singularities on each
  medial edge, best-ranked candidate in the gap. Each block now holds at most
  one singularity, verified over eight configurations, and block counts roughly
  halve against even spacing (16→8, 21→11, 26→15).
- Rewrote the vertex-region builder: it chose arcs by "shorter run to the next
  cut", which only holds while the region is local, and the search makes regions
  span half the domain. Now traced from the touch arrays like a span block.
- Added join snapping for arcs meeting at a vertex's shared touch, which can
  land on adjacent samples.
- Known gap: along a flare θ_m is constant, so the ranking cannot discriminate
  and the cut is effectively arbitrary — the large corner wedges in the figure.
- 37 tests in `test_search.py`.

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

# AeroMesh — index

> [!NOTE]
> This note used to be a verbatim copy of `Long_term_plan.md`. Two copies drift,
> and this one did. It is now an index into the canonical documents, which live
> in the repository root.

## Canonical documents

- [[../PROJECT_TRACKER|PROJECT_TRACKER.md]] — live status, gates, decision log,
  invariant registry. **Start here.**
- [[../Long_term_plan|Long_term_plan.md]] — architecture and build order.
- [[../MAPS_Proposal_v2|MAPS_Proposal_v2.md]] — the scientific argument and the
  claims. Revision 2, 8 Sep 2026.
- [[../MAPS_Implementation_Plan|MAPS_Implementation_Plan.md]] — stage-by-stage
  implementation detail.
- [[../README|README.md]] — what the library does and how to run it.

## The governing rule

**Nothing is prescribed.** No hardcoded topology, no domain-type switch, no rule
table, no per-airfoil or per-configuration branch.

*Derived:* number of blocks, connectivity, which boundary arc belongs to which
block, interfaces, singularity positions and types, vertex classification, and
the topology class itself.

*Supplied:* the geometry of the bodies, where the domain is truncated, and the
target resolution.

The only prescribed construction permitted anywhere is the midpoint-subdivision
template for filling a logically convex *m*-gon — a proven primitive at the level
of a quadratic formula, with no geometry-specific case in it.

## Where the work stands — 21 Oct 2026

| Stage | Scope | Status |
|---|---|---|
| S0 | Loop, corners, cusp policy, Region, far field | done |
| S1 | Medial engine | done |
| S2 | Singularity solver + certificates | realised certificate exact on 17/17; predictor residuals open |
| S3 | Decomposition | sound on 17/17 — and soundness turned out not to be enough, see SZ |
| SZ | **Zoned blocking** — collar / mid-field / far field | zone 1 (collar) built and measured; zones 2–3 not started |
| S4 | Mesh construction | quadrilateralisation done; TFI done inside the collar; TTM and export to come |
| S5 | Adversarial geometry gate | — |
| S6 | Quality + optimisation | — |
| S7 | Bunin φ-field by FEM (optional) | — |

## Decisions taken (full rationale in the tracker)

1. **Blunt trailing edge by default**; closed and cusped supported through
   explicit corner seeding. Reverses the earlier closed-TE-only decision — a
   cusp is the degenerate case for a medial axis.
2. **`DomainType` removed.** The input is a set of closed loops. C, O and H are
   labels read off the result.
3. **No TopMaker rule table.** Singularities are solved by flux balance; the
   decomposition is a ranked search. Fogg's paper is a critique of that table.
4. **Far field is a curve you supply**, with the conventional constructors —
   `circle_farfield` (0 corners, the default), `c_farfield` (2), `box_farfield`
   (4), `offset_farfield` (0, for tight or spread configurations). Choosing one
   is a modelling decision, not a topology selection: it is stored as a curve
   and no later stage can tell which constructor made it. But its *corners* do
   drive topology — a corner makes a flare, a flare makes a block — so a
   corner-free default would have put C and H topologies out of reach. *(D4,
   superseded by D8.)*
5. **Neural operator cut** from the critical path; Bunin φ kept as an optional
   FEM stage. The search it was built to accelerate no longer exists.
6. **Validation targets deferred, invariants not.** The adversarial gate sits
   before the optimisation work, not at the end.
7. **Superseded modules kept in place** until the repository is under version
   control. Run `git init` before S1 removes any of them.

## The turn to zoned blocking — 21 Oct 2026

S3 produced blockings that were **certified and unusable**: block boundaries not
straight, awkward trailing edges, near-wall blocks with a topology that would be
hard or impossible to mesh. The certificate was not wrong, it was answering a
different question. A sound block graph is not a meshable one, and no index
budget can tell the difference.

The cause is that Fogg's medial decomposition is a **CAD** method — well-shaped
quads on a part — while CFD wants extreme anisotropy at the wall, orthogonal
cells and a fixed Δy₁. One decomposition serving both objectives served neither.
So the blocking is now staged by zone: **1** the near-wall collar, **2** the
mid-field using the medial features *between* bodies, **3** the far field.

This changes the strategy, not the one rule. The collar's topology is still
found — rays, block count, block kind all come from the detected corners, their
fluid angles and Fogg's element counts, and nothing in `collar.py` names a
configuration. What is *forced* is that a band around a closed curve is an
annulus, which is a consequence of the geometry, not a template.

Zone 1 on eight aerofoils × two trailing-edge kinds: minimum angle 57.8–59.4°,
maximum 120.6–122.2°, **no cell below 45°**, median exactly 90.0°, all Jacobians
positive, coverage 99.95–99.97%, Δy₁ exactly as asked.

## Measurements worth remembering

- **Aggregate per-cell metrics hide placement errors.** A folded band reported
  "all Jacobians positive, minimum angle 25.6°" while covering the collar
  **204%** — doubled over on itself. Three separate folds passed the per-cell
  checks; coverage/overlap/outside caught all three. Victor caught the first one
  from a picture before any metric did.
- **A corner is a feature that survives refinement; high curvature is not.**
  The turning-extrapolation model `T(w) = a + c·w` claims no saturation at any
  w. True of the continuum, false of a sampled curve: once the window exceeds
  the local radius of curvature all the turning is already inside the narrowest
  window, `T` flattens, and the fit reports the saturated total as a corner
  angle. A 10:1 ellipse reads as a 38° corner at 1200 samples, a 50:1 ellipse as
  154°. The invariant that separates them is window-independence — a square
  holds 90.0° under 16-fold refinement while the 10:1 ellipse decays 74.7 → 38.0
  → 12.4 → 2.5 → 0.5.
- **Where a rejection rule sits in the pipeline is part of the rule.** The same
  test applied before non-maximum suppression achieves nothing: rejecting a
  peak lets the next flagged sample along the same feature win instead, so the
  detection survives one sample over. Applied after suppression it works.
- **Measure at the representative, not at its shoulders.** Reading the
  statistics off whichever flagged sample looked worst put the threshold inside
  the range that real corners occupy, and it discarded a genuine far-field
  crease. At the kept indices the real/false gap is 0.824 against 0.608, and
  the threshold belongs between them.
- **One bad sample out of 2075 can zero a whole field,** when the field is
  defined as a *minimum* over contributors. The medial axis of a polygon
  approximating a smooth curve grows a spurious branch at essentially every
  vertex; one such sample on NACA 6409 had `r_m = 0`, which took the thickness
  cap to zero and put the collar's outer curve on the wall. Minima need a
  resolution floor.
- **Widen the battery before trusting the result.** Both defects above were
  invisible on 0012/2412/4415-like sections and appeared immediately at 6%
  camber — and the second only became visible once the first was fixed.
- **Smoothing is not a fix for a bad parameterisation.** Winslow smoothing
  *worsened* the collar's extremes (59.0° → 50.4°) once the wall-to-outer
  correspondence was right. Order: blocking → irregular-node placement →
  parameterisation → *then* smoothing.
- **Plain Laplacian smoothing destroys a deliberately graded mesh** — 59.0° →
  34.3° on the wedges even with the boundary correctly held, because the
  interior relaxes towards uniform spacing while the boundary stays clustered.

- A **flare carries no information about where to cut it**: θ_m is exactly
  π − α at every sample and r_m is exactly linear in arc length (slope −1/√2,
  residual 10⁻¹⁴), so the wedge a cut leaves is a scaled copy of itself wherever
  it goes. The remedy is structural, not a better ranking — the corner is
  *resolved* and the wedge merges into the neighbouring region.
- **A constant-ρ layer is uniform only where r_m is.** Measured 1.1–1.7× along
  a lone body (so the layer is near-uniform) but 47× once a slot is present.
  The thickness law has to be `min(δ_target, ρ_max·r_m)` — flow sets it, the
  medial radius caps it.
- **Smoothing a curve shortens it.** Tidying the raw ρ-offset by averaging its
  points dragged it onto the wall where curvature is high. Smooth the scalar
  radius field instead and offset along the exact normal.
- **An area sum cannot verify a decomposition.** It is blind to an overlap
  cancelled by an equal gap, and completely blind to a zero-area spike. Three
  real defects hid behind it. Measure overlap, gap and spill separately.
- **The index certificate resolves what the area test cannot.** It found a crack
  of 5×10⁻⁷ of the region area between two vertex regions — below the 10⁻⁶ the
  shapely check uses — and a layer overlap of 2×10⁻⁸. Both were real holes. The
  identity holds for every valid quad mesh, which is exactly why checking it on
  the *product* is a bookkeeping test rather than a forecast.
- **`n_sides` was not the logical degree.** Two boundary arcs of a vertex region
  meet at the medial vertex's shared touch at exactly 180°, and eight of 23
  block corners on one blocking were such joins. Everything downstream reads
  `n_sides` as the degree, so each one was a spurious unit of index and a 180°
  one-quad cell on the wall.
- **A tautology can hide an unglued side.** `paired + boundary == n_sides` holds
  whatever happens once over-shared sides are taken out, because a side that
  pairs with nobody is filed as a boundary side. Ten of the layer's sides were
  unpaired and the report said conformal. The cause was `_side_key` taking
  `side[len//2]`, which is **not reversal-invariant** — 59 one way round, 58 the
  other. Cuts escaped it by luck: `radius_split` resamples to 33 points, and 33
  is odd.
- **Smoothing a cap raises it.** `ρ_max·r_m` is the only thing keeping two
  layers apart in a slot, and 12 smoothing passes diffused the slot's r_m
  minimum up by 12.5×, so the layers intersected from ρ_max = 0.3. Clip the
  smoothed field back to the raw one. And the guarding test had compared against
  `r_m.max()`, so it was vacuous.
- **An exact tie is not measure-zero.** A regular octagon's fluid angle is
  exactly 2.5 quarter-turns at *every* corner, so `round` decides each one on
  floating-point noise and rotating the octagon swung the corner term over −6,
  −7, −8. Snap to the tie, then round half up.
- **Straight spokes fold at any centre.** Midpoint subdivision needs a centre
  and lines to it; the area centroid of a layer strip landed exactly on one of
  its own corners, and even a four-sided span around half an aerofoil folded —
  5 of 8 blocks. **Tutte's theorem** is the construction that cannot: a discrete
  harmonic map onto a convex region is injective. Note it constrains the
  *target*, so a re-entrant block is handled, not rejected.
- **Tell Triangle `Y`.** Without it, Steiner points are inserted on the
  boundary, get no Dirichlet value, are solved as free, and the boundary map
  stops being a homeomorphism — so Tutte's hypothesis fails silently. That is
  why refining the spokes had not converged: folds went 10, 8, 4, 7 out of 42.
  Handing it a self-crossing polygon segfaults the interpreter, so validity has
  to be checked before the call, not after.
- **A probe can be longer than the body is thick.** Deciding which side of a
  wall is fluid by stepping off it only works if the step is smaller than the
  body. A 0.3-chord flap is 0.036 thick while its near-wall layer reaches 0.194
  on the outboard side, so a probe scaled to the largest thickness on the arc
  went straight through and found fluid on *both* sides at every sample. The
  tie built the whole layer inside the flap, and the area check passed at a
  relative error of 4.5e-4 with nothing in the notes. Scale the probe to the
  *smallest* thickness, cap it by the local sampling, and discard the samples
  that answer ambiguously.
- **An extrapolated corner angle is a plateau.** At a polyline corner the
  window straddles the vertex either way, so two adjacent samples read exactly
  the same angle — 120.000° on a unit triangle. A stable sort then hands the tie
  to the lower index, whose own vertex turn is 0.00°, and the corner's *angle*
  is right while its *position* is one sample out. Everything downstream matches
  corners by position, so ten of ten corners reported as buried on polygonal
  bodies were that one sample.
- **A tolerance for bit-identical points is not a tolerance for nearly-identical
  ones.** `coincidence_tol` is 1e-9 of the scale, which is right for two
  constructions of the same point and useless as a feature guard: at that
  setting a genuine 120° wall corner was merged away. And a cut's foot at a
  reflex corner can leave almost tangentially — 179.6° measured — so "this join
  is flat" does not imply "this join is not a corner".
- **`round` is not additive, so an index model needs its interior term.** Three
  corners of 100°, 130° and 130° meeting at a node ask for 1 + 1 + 1 = 3
  elements, not the 4 that 360° would. An identity built from per-corner
  element counts that drops the interior term is not an identity.
- **Transfinite interpolation beats a harmonic map at high aspect ratio.**
  Sending a 54:1 strip's boundary to a circle by arc length gives each long side
  half the circumference, and the map is distorted enough that a quad with
  interior angles 92/88/82/102° still folded. For four sides the Coons patch is
  both better conditioned and the map the mesher will use anyway.
- **A C library can take the process with it.** Handing Triangle a
  self-crossing polygon prints "topological inconsistency" and then segfaults.
  Validity has to be checked *before* the call; checking it after is no check at
  all.
- **The reflex corner is provably unreachable by a medial radius.** The set of
  directions in which a reflex corner is the nearest boundary point is its
  normal cone, of width exactly α − π. The cross directions span α, so whenever
  n_c rounds up the outermost lie outside the cone by (n_c·π/2 − α)/n_c — 2.7°
  at a 261.8° blunt trailing edge. Fogg clamps to the cone, giving sectors
  90/81.8/90.

- Exterior medial axis of a far-field domain is **scale-degenerate**: θ_m over
  the whole loop measures 172.9°–180.0° at ten chords, so Fogg's index is 0 at
  all 950 sampled points. Interface positions sᵢ placed there have no gradient.
- The **interior** axis is scale-free and carries the shape: θ_m = 68° at the LE
  dangle (n = 1, a nose singularity) against 179° at mid-chord (n = 0),
  identically for 0012, 2412 and 8412.
- **b₁ = h** — the medial graph's cycle count equals the body count — held
  exactly across six configurations. A free correctness test.
- Multi-element slots drive θ_m down to **10.9°**, so the flux balance places
  singularities in the gap with nothing told about a slot.
- The interior skeleton **degenerates** as a body becomes a disc: relative extent
  0.97 (NACA 2412) → 0.84 (3:1 ellipse) → 0.49 (1.55:1) → 0.07 (1.05:1) → empty
  for an exact circle. Finite contact must be built deliberately.

## Related notes

- [[TTM]] — Thompson–Thames–Mastin elliptic smoothing (S4).

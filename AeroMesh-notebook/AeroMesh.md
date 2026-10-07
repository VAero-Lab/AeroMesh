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

## Where the work stands — 8 Sep 2026

| Stage | Scope | Status |
|---|---|---|
| S0 | Loop, corners, cusp policy, Region, far field | done |
| S1 | Medial engine | next |
| S2 | Singularity solver + certificates | — |
| S3 | Decomposition | — |
| S4 | Mesh construction | — |
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

## Measurements worth remembering

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

# MAPS — Medial-Axis Parameterized Structured Meshing

### Automatic, fast, geometry-general structured mesh generation for RANS

**Victor Alulema** — Escuela Politécnica Nacional / KU Leuven  
**Deliverable:** `AeroMesh` (Python library, companion to `AeroShape`)  
**Target venue:** _Aerospace Science and Technology_

> **Revision 2 — 8 September 2026.** Revised after implementation of the
> geometry foundation and a measurement campaign on the medial axis. Three
> substantive changes: a two-sided medial analysis is added (Section 4d), the
> neural operator is withdrawn from the method and demoted to future work
> (Section 5), and the input model becomes a set of closed loops with no domain
> type (Section 6). Every number quoted as _measured_ below was obtained from
> the implementation, not estimated.

---

## Nomenclature

### Geometry and medial axis

| Symbol                       | Meaning                                                                                               |
| ---------------------------- | ----------------------------------------------------------------------------------------------------- |
| $\Gamma$                     | airfoil boundary curve (the wall)                                                                     |
| $\Gamma_{\text{out}}$        | outer boundary of the computational domain                                                            |
| $\Omega$                     | fluid domain, between $\Gamma$ and $\Gamma_{\text{out}}$                                              |
| $\mathcal{M}$                | medial axis of $\Omega$ — locus of centres of maximal inscribed circles                               |
| $s$                          | arc-length coordinate along the medial axis                                                           |
| $\hat{p}(s)$                 | the medial point at arc-length $s$                                                                    |
| $r_m(s)$                     | **medial radius** — radius of the maximal inscribed circle at $\hat p(s)$                             |
| $\theta_m(s)$                | **medial angle** — angle subtended at $\hat p(s)$ by the two medial radii                             |
| $\hat n_1(s), \hat n_2(s)$   | unit vectors from $\hat p(s)$ toward each touching boundary point                                     |
| $\mathbf{T}_1, \mathbf{T}_2$ | **touch points** — where the inscribed circle contacts the boundary                                   |
| $L$                          | total arc length of the medial edge                                                                   |
| $\rho$                       | **medial coordinate**, $\rho = d_{\text{wall}}/r_m(s) \in [0,1]$                                      |
| $\rho^\star$                 | ring-interface offset — the continuous design variable replacing $s_i$ on a featureless exterior loop |
| $h$                          | number of bodies (inner loops)                                                                        |
| $\chi$                       | Euler characteristic of the fluid region, $\chi = 1 - h$                                              |
| $b_1$                        | first Betti number of the medial graph; must equal $h$                                                |
| $n$                          | Fogg's optimum mesh-flow index, $n = \mathrm{round}\big((\pi-\theta_m)/(\pi/2)\big)$                  |
| $n_c$                        | number of elements at a boundary corner                                                               |
| $\eta$                       | normalized spanwise coordinate (3-D only), $\eta \in [0,1]$                                           |

### Mesh and quality

| Symbol                                                              | Meaning                                                                                           |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| $\phi$                                                              | Bunin scalar field controlling the mesh, $\phi = -\ln h$                                          |
| $h$                                                                 | local cell size (edge length of an infinitesimal square element)                                  |
| $\mathbf{p}_i$                                                      | position of the $i$-th **mesh singularity** — a node where a number of quads other than four meet |
| $k_i$                                                               | **type** of singularity $i$; integer, $k_i \geq -4$; the number of quads at the node minus four   |
| $\delta(\cdot)$                                                     | Dirac delta function                                                                              |
| $G_\varepsilon$                                                     | Gaussian of width $\varepsilon$, used as a smooth numerical stand-in for $\delta$                 |
| $K$                                                                 | Gaussian curvature of the surface ($K = 0$ for a planar airfoil domain)                           |
| $\rho$                                                              | normalized wall distance, $\rho = d_{\text{wall}} / r_m(s)$                                       |
| $\Delta y_1$                                                        | height of the first cell off the wall                                                             |
| $r$                                                                 | geometric stretching ratio in the wall-normal direction                                           |
| $N_{\text{wall}}$                                                   | number of grid points in the wall-normal direction                                                |
| $n_{\text{int}}$                                                    | number of block interfaces                                                                        |
| $s_i$                                                               | arc-length position of block interface $i$ on the medial axis                                     |
| $Q_{\text{orth}}, Q_{\text{Jac}}, Q_{\text{smooth}}, Q_{\text{AR}}$ | orthogonality, minimum Jacobian, smoothness, aspect-ratio metrics                                 |

### Flow and optimization

| Symbol              | Meaning                                                       |
| ------------------- | ------------------------------------------------------------- |
| $\mathrm{Re}$       | Reynolds number                                               |
| $U_\infty$          | freestream velocity                                           |
| $\nu(z)$            | kinematic viscosity at altitude $z$                           |
| $u_\tau$            | friction velocity, $u_\tau = \sqrt{\tau_w/\rho_{\text{air}}}$ |
| $C_f$               | skin-friction coefficient                                     |
| $y^+$               | dimensionless wall distance, $y^+ = \Delta y_1 u_\tau / \nu$  |
| $\mathbf{x}$        | **design vector** — the quantities being optimized            |
| $F(\mathbf{x})$     | composite mesh-quality objective                              |
| $w_1 \dots w_4$     | objective weights                                             |
| $\xi_i$             | unconstrained variable used to enforce interface ordering     |
| $B_k(\eta), c_{ik}$ | B-spline basis functions and control points (3-D interfaces)  |

### Acronyms

|                    |                                                                                                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RANS**           | Reynolds-Averaged Navier–Stokes                                                                                                                                   |
| **CFD**            | Computational Fluid Dynamics                                                                                                                                      |
| **CDT**            | Constrained Delaunay Triangulation                                                                                                                                |
| **TFI**            | **Transfinite Interpolation** — algebraic method that fills a block's interior by blending its four boundary curves; produces a valid but unsmoothed initial mesh |
| **TTM**            | **Thompson–Thames–Mastin** — the classical elliptic mesh smoother; solves a Poisson system to make grid lines smooth and near-orthogonal                          |
| **CMA-ES**         | Covariance Matrix Adaptation Evolution Strategy (gradient-free optimizer; held in reserve, see Section 5)                                                         |
| **L-BFGS**         | Limited-memory Broyden–Fletcher–Goldfarb–Shanno (quasi-Newton gradient optimizer)                                                                                 |
| **CST**            | Class–Shape Transformation (airfoil shape parameterization)                                                                                                       |
| **NURBS**          | Non-Uniform Rational B-Spline                                                                                                                                     |
| **MAT**            | Medial Axis Transform                                                                                                                                             |
| **PINN**           | Physics-Informed Neural Network                                                                                                                                   |
| **FNO / DeepONet** | Fourier Neural Operator / Deep Operator Network — architectures that learn maps between function spaces (future work only, see Section 5)                         |
| **VLM**            | Vortex Lattice Method                                                                                                                                             |
| **DDPM**           | Denoising Diffusion Probabilistic Model                                                                                                                           |
| **LE / TE**        | Leading Edge / Trailing Edge                                                                                                                                      |
| **CGNS**           | CFD General Notation System (standard mesh file format)                                                                                                           |
| **AR**             | Aspect Ratio                                                                                                                                                      |

---

## 1. Problem

### 1.1 What has to be decided to build a mesh

RANS analysis of an airfoil needs a structured multiblock mesh. Producing one requires four decisions:

| Decision                                                                   | Nature                  | Currently made by                    |
| -------------------------------------------------------------------------- | ----------------------- | ------------------------------------ |
| **Block topology** — how many blocks, how they connect                     | discrete, combinatorial | engineer, from experience            |
| **Interface placement** — where block boundaries land on the geometry      | continuous              | engineer, by eye, iteratively        |
| **Wall clustering** — first cell height $\Delta y_1$, stretching ratio $r$ | continuous              | engineer, from a $y^+$ rule of thumb |
| **Smoothness** — metric continuity across block interfaces                 | emergent                | trial and error                      |

These are not independent. Moving an interface changes the block shapes, which changes the achievable orthogonality, which changes how much clustering the block can absorb before cells skew. An engineer resolves this coupling by iterating — build, inspect, adjust, rebuild — typically for hours per configuration.

### 1.2 Why it matters that these decisions are good

Mesh quality is not cosmetic. Each of the four decisions maps to a specific failure mode in the RANS solution:

- **Poor wall orthogonality** introduces cross-derivative terms in the discretized viscous fluxes. The resulting error contaminates skin friction and cannot be distinguished from a physical effect. This is worst exactly where it matters most — at the leading edge and in the boundary layer.
- **Wrong $\Delta y_1$** means the wall $y^+$ misses its target. For a low-Re turbulence model requiring $y^+ \lesssim 1$, being off by a factor of three changes the predicted drag by percent-level amounts, which is the same order as the design differences being studied.
- **Low cell Jacobian** (highly skewed or near-degenerate cells) degrades solver convergence and, in the worst case, produces negative volumes and outright divergence.
- **Metric jumps across block interfaces** produce spurious pressure oscillations that pollute the pressure distribution and hence lift and pressure drag.

So the mesh is not a preprocessing detail. It is a _source of error of the same magnitude as the physical effects being resolved_, and it is the one source of error currently controlled by human judgement rather than by an algorithm.

### 1.3 Why it is a bottleneck now

For a single airfoil, hours of manual meshing is tolerable. The problem becomes structural when many meshes are needed:

- Building a training database for a surrogate model — hundreds of geometries.
- Aerodynamic shape optimization — a new mesh at every design iteration.
- Design-space exploration or sensitivity studies — sweeps over thickness, camber, Reynolds number.

In all three cases the mesh generation step is the only step that cannot run unattended. This is the concrete blocker. NASA's _CFD Vision 2030_ study names mesh generation as one of the principal obstacles to routine, automated high-fidelity simulation, and it remains so.

---

## 2. Motivation

The motivation is specific, not generic.

**Immediate.** Pillar II of this thesis is a multi-fidelity aerodynamic surrogate: a low-fidelity VLM/panel baseline corrected by sparse RANS data. "Sparse" means few _relative to the design space_, not few in absolute terms — it means hundreds of RANS solutions spread across a geometry family. Every one of those requires a mesh of consistent, verified quality. If a human must generate each mesh, the surrogate cannot be trained at the required scale, and worse, mesh-to-mesh quality variation becomes a hidden noise source in the training data. **Automating mesh generation with quality guarantees is a precondition for Pillar II, not a parallel activity.**

**Structural.** The existing `AeroShape` library already produces NURBS geometry for 2-D airfoils and 3-D aircraft surfaces. It ends at the geometry boundary. The natural next component — the one that turns geometry into something a solver can consume — does not exist. `AeroMesh` fills that gap and completes the geometry-to-solution pipeline.

**Scientific.** The medial axis has been used for mesh topology since the 1990s, but always _topologically_: to decide how blocks connect. Its geometric content — the continuous fields $r_m(s)$ and $\theta_m(s)$ — has been discarded. Those fields encode exactly the information a human engineer uses when placing an interface by eye: how thick the domain is here, how the two nearby walls are oriented relative to each other. Recovering that information and using it to _parameterize_ the mesh, rather than merely to classify it, is the scientific opening.

---

## 3. Competitors

| Method                                                                                            | What it automates                                  | What it does not                                                                                                                   |
| ------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **pyHyp** (Secco 2021) — hyperbolic marching                                                      | O-mesh generation; fast, robust, production-proven | O-topology only; no clustering optimization; degrades on concave regions such as re-entrant trailing edges                         |
| **Parametric multiblock** (Qi 2024, _Appl. Sci._) — TFI on parameterized control vertices         | very fast (50M cells / 10 s), high Jacobian        | topology hand-built once _per airfoil family_; no quality optimization; does not adapt to a genuinely different geometry class     |
| **DRL-MeshGen** (Qi 2025, _Eng. Comput._) — reinforcement learning + Ricci-flow conformal mapping | topology and singularity placement                 | not RANS-specific (validated on CAD surfaces); no $y^+$ awareness; expensive training; no geometric theory underneath              |
| **PINN-MG** (Wang 2025) — PINN on the Navier–Lamé elasticity equation                             | point distribution, without labeled data           | **retrains for every new boundary curve** (their stated limitation); elasticity is a proxy, not the physics of a mesh; no topology |
| **TopMaker** (Rigby 2004) — medial axis rules                                                     | topology                                           | uses only medial _topology_ (vertex types, connectivity); discards $r_m$, $\theta_m$; open loop — no quality feedback              |
| **Fogg et al.** (2014) — medial angle + Bunin flux balance                                        | topology and singularity _types_                   | stops at singularity identification; no interface geometry, no clustering, no connection to mesh quality                           |

**The gap.** No method treats _where an interface goes_ as a continuous, optimizable quantity derived from the geometry itself. And no method couples the mesh to the flow conditions through $y^+$ — the clustering is always a user input, never a consequence of the physics.

**A second gap, specific to external aerodynamics.** Every medial-axis decomposition method — Nackman & Srinivasan, Tam & Armstrong, TopMaker, Fogg — operates on a region whose boundaries sit at comparable scale. Fogg's own multi-element aerofoil case has its outer boundary about one chord away. External aerodynamics puts it at ten to a hundred chords, and there the medial axis of $\Omega = \text{box} \setminus \text{airfoil}$ sits at half the far-field distance and the body enters it only as a point-like hole. _Measured:_ for a NACA 0012 with the boundary at ten chords, $\theta_m$ over the entire medial loop lies in $172.9^\circ$–$180.0^\circ$, so the optimum mesh-flow index is zero at all 950 sampled points. The loop is featureless. No surveyed method addresses this, because none was aimed at external flow. Section 4(d) is the response.

---

## 4. Core idea

The medial axis is not just a skeleton for choosing topology. It is a **coordinate system for the entire design problem**.

Three consequences follow, and they are the whole proposal.

### (a) Interfaces become scalars

A block interface is fully specified by a single number: its arc-length position $s_i$ on the medial axis. Its geometry then follows analytically. The two touch points are

$$\mathbf{T}_{1,2}(s_i) \;=\; \hat{p}(s_i) \;+\; r_m(s_i)\,\hat{n}_{1,2}(s_i)$$

and the interface curve runs from $\mathbf{T}_1$, through the medial point $\hat p(s_i)$, to $\mathbf{T}_2$, following the medial radii. Because a medial radius meets the boundary perpendicularly by definition, **the interface meets the wall orthogonally by construction** — orthogonality is not something the optimizer has to discover, it is built into the parameterization.

A 2-D geometric design problem (place a curve in the plane) collapses to one scalar per interface.

_Correction, from measurement._ This holds wherever $\theta_m$ varies. On the exterior axis of a far-field domain it does not — see Section 3 — so an interface position $s_i$ placed on that loop has no gradient to optimise against. The variable that does carry one is the **ring offset** $\rho^\star$: interfaces are then level sets ($\rho = \rho^\star$) and gradient lines ($s = s_i$) of the medial coordinate system of Section 4(b). Constant-$s$ lines are medial radii and meet the wall perpendicularly by definition, so orthogonality by construction is unaffected. The scalar-per-interface property is retained where it is real — on the _interior_ axis of each body, and in the gaps of a multi-body configuration, where $\theta_m$ varies strongly.

_A second correction, also from measurement._ The constant-$\rho$ family is the
outer edge of the near-wall block, and how uniform that block is depends on how
much $r_m$ varies along the wall. Measured: $1.1$–$1.7\times$ along a lone body,
so a constant-$\rho$ layer is near-uniform without being told anything — the
property claimed here. But with a second body present $r_m$ collapses in the
slot, varying $47\times$, and a constant fraction of it collapses with it. The
coordinate is behaving correctly; a constant fraction is simply the wrong
_thickness law_ for a boundary layer. The law to use is

$$\delta(s) \;=\; \min\!\big(\delta_{\text{target}},\; \rho_{\max}\, r_m(s)\big)$$

with $\delta_{\text{target}}$ set by the flow through $\Delta y_1$, the growth
rate and $N_{\text{wall}}$, and the medial term acting as a ceiling that keeps
the layer inside the space available. $\rho^\star$ is then the margin rather
than the whole law — and it remains the continuous variable the optimisation
stage varies, which was the point of introducing it.

### (b) Every airfoil becomes the same domain

Define **medial coordinates** $(s, \rho)$ with $\rho = d_{\text{wall}}/r_m(s)$. Each medial edge maps the region around it to a canonical rectangle $[0,L] \times [0,1]$.

In these coordinates a NACA 0012 and a supercritical section are _the same domain_. They differ only in the one-dimensional profiles $r_m(s)$ and $\theta_m(s)$.

This is the property that will let a learned model generalize instead of memorize (Section 5).

_Caveat, stated precisely:_ the map is canonical **per medial edge**. At medial vertices — where branches meet — several patches join. This is not a problem: those patches are exactly the blocks. The decomposition into canonical rectangles _is_ the blocking.

### (c) Mesh quality has a governing equation

Bunin's continuum theory gives the governing PDE of an orthogonal quadrilateral mesh. For a planar domain ($K = 0$):

$$\boxed{\;\nabla^2 \phi \;=\; \sum_{i=1}^{N} k_i \,\frac{\pi}{2}\, \delta(\mathbf{r} - \mathbf{p}_i), \qquad \phi = -\ln h\;}$$

with boundary conditions

$$\frac{\partial \phi}{\partial n}\bigg|_{\Gamma} = f_{\text{wall}}(s;\, \Delta y_1, r), \qquad \phi\big|_{\Gamma_{\text{out}}} = 0$$

The wall condition $f_{\text{wall}}$ is the clustering law — a hyperbolic-tangent distribution whose first spacing is $\Delta y_1$ and whose growth rate is $r$:

$$f_{\text{wall}}(s;\Delta y_1, r) \;=\; -\ln \Delta y_1 \;-\; \tfrac{1}{2}\ln r \quad\text{(evaluated at the wall)}$$

The singularity types $k_i$ and positions $\mathbf p_i$ are obtained from Fogg's flux-balance analysis on the medial angle, $k = -\lfloor \max\theta_m / (\pi/2) \rfloor$.

**All mesh quality metrics are derivatives of $\phi$.** Solve for $\phi$ and you know the quality of the mesh without ever building it.

### (d) Two skeletons, not one

The exterior medial axis of $\Omega$ is scale-dominated by the far field (Section 3) and carries almost no information about the body. The body's _own_ interior medial axis does, and it is scale-free: it does not care where the domain is truncated.

So compute both, and give each the job it is competent for.

| Skeleton                          | Supplies                                                                                                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Exterior**, of the fluid region | everything between and around the bodies: the ring against each wall, flares to any corners of the truncation curve, branches through the gaps, the wake structure |
| **Interior**, of each body        | that body's own shape: the leading-edge dangle and its singularity, chordwise flow over the mid-chord, the trailing-edge terminus                                  |

The two are coupled at the wall: the singularity budget established inside is carried out through the boundary layer and must be accounted for by the outer decomposition.

_Measured_, on the interior axis, with 800 boundary samples per section:

| Section   | $\theta_m$ at LE dangle | $n$ at LE | $\theta_m$ mid-chord | $n$ mid-chord |
| --------- | ----------------------- | --------- | -------------------- | ------------- |
| NACA 0012 | $68.1^\circ$            | **1**     | $179.0^\circ$        | 0             |
| NACA 2412 | $67.4^\circ$            | **1**     | $179.7^\circ$        | 0             |
| NACA 8412 | $72.1^\circ$            | **1**     | $179.7^\circ$        | 0             |

with $n = \mathrm{round}\big((\pi - \theta_m)/(\pi/2)\big)$ (Fogg's optimum mesh-flow index). $n = 1$ at the nose means the cross-field makes one quarter-turn between the upper and lower surfaces there — a mesh singularity at the leading edge. $n = 0$ over the mid-chord means grid lines run along the chord. That contrast is what distinguishes a C-topology from an O-topology, it emerges from $\theta_m$ with no rule written by hand, and it is invisible to the exterior axis.

_Stated limit._ The interior skeleton degenerates as a body becomes a disc — measured extent relative to body size runs 0.97 for a NACA 2412, 0.84 for a 3:1 ellipse, 0.49 at 1.55:1, 0.07 at 1.05:1, and an exact circle has a single-point medial axis that a discrete Voronoi returns as nothing. It degenerates _to the correct answer_ (no preferred axis, hence no nose singularity, hence a ring), but the finite-contact case must be built deliberately, using Bunin's $k = -\lfloor \max\theta_m/(\pi/2) \rfloor$ at the collapsed vertex.

### (e) The topology can be certified before it is built

Two constraints hold exactly, are cheap, and are checked before a single cell exists.

**Retract.** The medial axis deformation-retracts onto $\Omega$, so the first Betti number of the medial graph must equal the number of bodies, $b_1 = h$. _Measured exact_ — to the integer — for one, two and three bodies, for an airfoil beside a fuselage section, for a lone bluff body and for a circle.

**Index budget.** The singularity indices are constrained by the Euler characteristic $\chi = 1 - h$, in the same way a cross field on a surface must satisfy Gauss–Bonnet. Checking $\sum_i k_i$ against that budget turns _this blocking looks reasonable_ into _this blocking is admissible_. It scales with body count for free: adding a flap changes $\chi$ and the budget demands more singularities with no new code. The annulus case is a first non-trivial confirmation — $\chi = 0$, budget zero, and $n = 0$ measured at every point. Pinning the exact constant for each boundary convention is real work, not a formality.

No medial-axis method certifies the global budget; Fogg identifies singularities and verifies by eye and by example.

**A correction from measurement, and it sharpens the claim.** The budget as written above is checked against the _predicted_ singularity field, and that was the wrong object to check. The identity

$$\sum_{\text{interior}} (\mathrm{val} - 4) \;=\; \sum_{\text{boundary}} (2 - q) \;-\; 4\chi$$

is Euler's formula rearranged, so it holds for **every** valid quadrilateral mesh — which makes it useless as a prediction and valuable as a consistency test of a blocking's own bookkeeping. Checked against the _product_ it is exact on all seventeen configurations tested, including cases where the predicted field is off by $-1$ to $+3$; and it resolves defects an area test cannot. The crack it found between two vertex regions on a fuselage-with-wing was $5\times10^{-7}$ of the region area — below the $10^{-6}$ threshold a polygon-union check uses — and it was a genuine hole, caused by a snapping rule that moved a cut's analytic endpoint to a midpoint so that the two blocks sharing the cut disagreed at one end.

So the certificate is not a forecast that happens to be checkable; it is a _property of the delivered blocking_, and the predicted field becomes a heuristic for where to cut. That is a stronger claim than the one above and a cheaper one to defend.

The companion quantity is the **design index**: what the blocking asks for from its own corner angles, $\sum_i (2 - \mathrm{round}(\theta_i/(\pi/2))) - 4$ per block. It agrees with the predicted budget where midpoint subdivision does not, so their difference is a measurement of how much index is still trapped in corners the decomposition has not resolved — presently the reflex corners of a blunt trailing edge, and nothing else.

### (f) A certificate is not a guarantee of meshability — hence zoning

The certificates in (e) are exact and they are not sufficient, and finding that
out is a result in its own right. The decomposition they certify produced
blockings whose index budget balanced, whose areas were positive, whose
interfaces were conformal — and whose near-wall blocks had a topology that could
not be meshed usefully: boundaries that were not straight, awkward trailing
edges, aspect ratios of 400 inside a block with a 262 degree interior angle.

The certificate was answering the question it was asked. **A sound block graph
is not a meshable one**, and no bookkeeping identity can tell the difference,
because the identity is satisfied by any valid quadrilateral mesh including bad
ones. What the certificates exclude is incoherence, not unsuitability.

The cause is a mismatch of objective. Fogg's medial decomposition is a method
for **CAD** meshing: it optimises for well-shaped quads on a mechanical part,
with no $\Delta y_1$ in it, no wall-normality requirement and no anisotropy.
RANS wants the opposite near the wall — cells of aspect ratio $10^2$–$10^4$,
orthogonal to the surface, with the first height fixed by $y^+$ — and wants the
CAD-like objective only away from it. A single decomposition asked to serve both
serves neither.

So MAPS blocks **by zone**: zone 1 a near-wall collar, zone 2 the mid-field
built from the medial features *between* bodies, zone 3 the far-field expansion.
Only zone 2 is a medial decomposition in Fogg's sense; zone 1 is constructed,
and its construction makes no choices — a band around a closed curve is an
annulus, the rays come from each corner's normal cone and its element count, and
the block count and kinds follow. Zone 1 measures, on eight aerofoil sections
and both trailing-edge conventions: minimum angle 57.8–59.4 degrees, no cell
below 45, median exactly 90, all Jacobians positive, coverage within 0.05 per
cent, $\Delta y_1$ exactly as requested.

This is a change of strategy and not of the governing rule. Nothing about zone 1
is prescribed per geometry; what is forced is forced by the topology of a closed
curve.

---

## 5. The surrogate decision — withdrawn

The first version of this proposal made a geometry-conditioned, physics-informed
neural operator the centrepiece: a DeepONet acting on medial coordinates
$(s,\rho)$ and the 1-D profiles $r_m(\cdot)$, $\theta_m(\cdot)$, trained once
offline against the Bunin PDE residual, supplying $\phi$ in ~1 ms so that a
CMA-ES search over interface positions could afford a thousand evaluations.

**That component is withdrawn from the method.** The reasoning is worth stating
plainly, because the argument for it was sound given what the method was then.

The operator existed to make a large search cheap. With Section 4(d) and 4(e) in
place, the search is gone: **the topology is solved and certified, not searched.**
Singularity positions and types follow from a flux balance on $\theta_m$, and
their admissibility is checked against $\chi$. What remains to optimise is a
handful of smooth, continuous parameters — the ring offset $\rho^\star$, the
first spacing $\Delta y_1$, the growth rate $r$, the wall count $N_{\text{wall}}$
— roughly six dimensions with a unimodal landscape. That is L-BFGS in tens of
evaluations, where an exact evaluation at ~100 ms costs seconds in total. A
1 ms surrogate is a fast answer to a problem the method no longer has.

Against that it is the highest-effort and highest-risk component in the
programme, it cannot be trained until the physics-free path exists to generate
its data, and it is the only part that can never be verified except against the
path it replaces.

**What is kept is Bunin's field itself**, solved directly by finite elements.
That is a small, well-posed problem, and it earns its place for a different
reason than speed: Fogg leaves singularity _placement within a subregion_ open —
"by not fixing the singularity positions they are free to be repositioned inside
their polygon subregions to suit the target element sizes" — and the $\phi$-field
is what decides where they should sit.

**When the operator becomes worth reviving.** Not for 2-D speed. Two honest
reasons: the 3-D extension of Section 7, where per-section cost multiplies by
fifteen to twenty spanwise stations; and coupling into aerodynamic shape
optimisation, which needs gradients of mesh quality with respect to _shape_, not
with respect to the design vector. Both are real. Neither is on the path to a
working generator.

### One shipping mode

The library ships with no learned component. Mesh quality is measured exactly,
from meshes that exist. There is no fallback path to maintain and no possibility
of a learned component silently producing a bad mesh.

## 6. Method, in phases

```
   a set of closed loops  (bodies + a truncation curve)
              │
   ┌──────────▼──────────┐
   │  PHASE 1 — MA-TOP   │   two medial axes → r_m, θ_m, n̂ everywhere
   └──────────┬──────────┘
              │
   ┌──────────▼──────────┐
   │  PHASE 2 — MA-SING  │   flux balance → singularities {(p_i, k_i)}
   │                     │   certificates: b₁ = h,  Σk_i vs χ = 1 − h
   └──────────┬──────────┘
              │   a CERTIFIED block topology — solved, not searched
   ┌──────────▼──────────┐
   │  PHASE 3 — BUILD    │   ranked split search → blocks → TFI → TTM
   └──────────┬──────────┘
              │
   ┌──────────▼──────────┐
   │  PHASE 4 — MA-OPT   │   L-BFGS over ρ*, Δy₁, r, N_wall
   └──────────┬──────────┘
              ▼
      CGNS / SU2 / Plot3D mesh
```

**The input model.** One outer loop and any number of inner loops. There is no
domain type to choose: C, O and H are labels read off the resulting block graph,
never inputs. A single airfoil, a three-element high-lift arrangement and a
wing–body crossflow cut are the same kind of object.

**The truncation curve is supplied, in the conventional shapes.** Where the
domain ends is a modelling decision, and the standard curves — a circle, a
C-shape with an upstream semicircle and an outflow plane, a rectangle for
tunnel walls — are offered as such. What is removed is the _coupling_: in the
earlier formulation, `domain="C"` chose a curve and selected a blocking in the
same breath. Now the curve is passed as a curve and stored as a curve, and no
later stage can tell which constructor produced it. What the choice influences
is how many corners the fluid boundary has, and a corner generates a medial
flare, and a flare generates a block — _measured_ on a NACA 2412 at five
chords: circle 0 flares, C-shape 2, box 4. So the shape matters, but it
supplies geometry to the flux balance rather than a topology to obey.

**Nothing is prescribed.** _Derived:_ the number of blocks, their connectivity,
which boundary arc belongs to which block, the interfaces, the singularities,
and the topology class. _Supplied:_ the geometry of the bodies, where the domain
is truncated, and the target resolution — which follows from Re, $y^+$ and
altitude. The one prescribed construction permitted anywhere is the
midpoint-subdivision template for filling a logically convex $m$-gon, a proven
primitive containing no geometry-specific case.

TopMaker's rule table — a primary edge gives two block faces, a flare one, a
dangle five — is **not** used. Fogg's paper is a critique of exactly that table:
it assumes all corners are flat except those connected to flares, which distorts
blocks on a multi-element aerofoil.

---

### Phase 1 — MA-TOP: geometry → design vector

**Step 1.1** — Compute the medial axis $\mathcal{M}$ by constrained Delaunay triangulation of $\Gamma \cup \Gamma_{\text{out}}$. The circumcentres of triangles interior to the domain approximate $\mathcal{M}$. Extract $r_m(s)$, $\theta_m(s)$, $\hat n_{1,2}(s)$. Prune spurious short branches using a threshold on $r_m$.

**Step 1.2** — Read the structure of the medial graph. The table below is a
_description of outcomes_, not a classifier the pipeline runs: nothing selects a
branch on it. The topology class is a label, and it is derived.

| Medial graph structure                           | Reads as    | Blocks it yields                    |
| ------------------------------------------------ | ----------- | ----------------------------------- |
| closed loop, no medial vertices                  | O           | one block wrapping on itself        |
| loop + branch from a reflex corner               | C           | ring + wake blocks                  |
| loop + flares to corners of the truncation curve | H           | ring + one block per corner         |
| loop per body + gap branches between them        | _(unnamed)_ | as many as the flux balance demands |

_Measured:_ the exterior axis gives a bare loop for a lone airfoil or a lone
bluff body, a loop with four flares inside a rectangular truncation curve, and
one cycle per body for multi-element configurations — $b_1 = h$ exactly in all
six cases tested.

**Step 1.3** — Locate mesh singularities $\{(\mathbf p_i, k_i)\}$ by Fogg's flux balance on $\theta_m(s)$, evaluated only at the three positions where an imbalance can occur: crossings of the critical angles $\pi/4$ and $3\pi/4$, medial vertices, and concave-corner switch points. At finite contact, $k = -\lfloor \max\theta_m/(\pi/2) \rfloor$; at boundary corners, $n_c$ follows from the fluid interior angle. Nearby singularities merge within a **local** tolerance of $[r_m/4,\, r_m]$, so bodies at different scales in one domain each get their own. Then **certify** (Section 4e) before anything is built.

**Step 1.4** — Bound $\Delta y_1$ from the flow, not from a rule of thumb:

$$\Delta y_1 \;=\; \frac{y^+ \, \nu(z)}{u_\tau}, \qquad u_\tau = U_\infty \sqrt{C_f/2}, \qquad C_f \approx 0.026\,\mathrm{Re}^{-1/7}$$

At Quito's altitude ($z = 2850$ m) $\nu$ is roughly 20 % higher than at sea level, which shifts $\Delta y_1$ accordingly. The mesh is therefore altitude-aware — a property no surveyed method has.

**Output:** the design vector

$$\mathbf{x} \;=\; \{\,\rho^\star,\; \Delta y_1,\; r,\; N_{\text{wall}},\; \dots\,\}, \qquad \dim(\mathbf x) \approx 4\text{–}8$$

_Why this way._ The topology is no longer part of the search — it is solved and certified in Phase 2 — so the design vector holds only continuous, smooth quantities. Interface arc-lengths $s_i$ appear only where $\theta_m$ actually varies (a body's interior axis, a multi-element gap); on a featureless exterior loop the ring offset $\rho^\star$ replaces them. The dimension is set by the topology, not by the geometric complexity of the section: a cambered supercritical section has the same number of design variables as a NACA 0012.

---

### Phase 2 — MA-SING: geometry → a certified topology

The singularity field is the **solution** of the flux balance of Step 1.3, not
an entry in a rule table. It fixes the block corners, and with them the number
of blocks and their connectivity.

Two certificates then run, before any construction (Section 4e): $b_1 = h$ on the
medial graph, and $\sum_i k_i$ against $\chi = 1 - h$. A configuration that
fails either is rejected with a diagnostic rather than meshed.

_Why this way._ Mesh topology is the one decision in the pipeline that is
discrete, and therefore the one that cannot be repaired by smoothing afterwards.
Making it a solved-and-checked step rather than a searched one removes the
combinatorial part of the optimisation entirely — which is what makes the
learned surrogate of Section 5 unnecessary.

### Phase 3 — BUILD: certified topology → mesh

1. Decompose by a **ranked split search**: on each medial edge, one cut per gap
   between consecutive singularities, so that every block holds at most one.
   Candidates are ranked with $n = 0$ preferred over $n = 1$ and medial angles
   nearer $\pi$ and $\pi/2$ favoured within each.

   A _flare_ — a medial edge running out to a convex corner of the fluid — is
   left uncut. Along one, $\theta_m$ is exactly $\pi - \alpha$ at every point
   and $r_m$ is exactly linear in arc length, so every candidate ranks
   identically and the wedge a cut would leave is a scaled copy of itself
   wherever it goes: **no local criterion exists**. Fogg's treatment applies —
   a convex corner with $n_c \neq 0$ is _resolved_, so no split is needed, and
   the wedge merges into the neighbouring subregion with the corner as one of
   its vertices.

2. Build the blocks **constructively from the medial graph**: a span between two
   cuts is four-sided, a medial vertex is an $m$-sided region, a medial loop with
   no vertices is a single ring wrapping on itself. The face structure is the
   skeleton's, so no planar arrangement is computed. Blocks carry connectivity:
   every side is either on the domain boundary or shared with exactly one other
   block, traversed in opposite directions by the two.

3. Fill logically convex $m$-sided subregions by midpoint subdivision.

4. **TFI** per block — an algebraic blend of the four edge curves. Fast, always
   valid, not smooth.

5. Smooth with the **TTM** elliptic solver, imposing $\Delta y_1$ and $r$ as
   boundary conditions.

_On verifying a decomposition._ An area sum is a necessary condition and a
** weak one**: it cannot distinguish an overlap cancelled by an equal gap, and a
zero-area spike changes no area at all. Overlap, gap and spill are measured
separately, alongside self-intersection, join closure and conformality. Three
real defects in the implementation were invisible to the area sum and caught
only by those checks.

### Phase 4 — MA-OPT: polish the continuous parameters

### Phase 4 — Build and verify

**L-BFGS**, roughly 30 steps, over $\rho^\star$, $\Delta y_1$, $r$ and
$N_{\text{wall}}$ — a smooth, unimodal landscape. Each evaluation builds the
mesh and measures it exactly: orthogonality at wall nodes, minimum cell
Jacobian, boundary-layer aspect ratio, metric jump across interfaces, achieved
$y^+$. Cross-checked against `vtkMeshQuality`.

$$F(\mathbf{x}) \;=\; w_1\big(1 - Q_{\text{orth}}\big) \;+\; w_2\, Q_{\text{smooth}} \;+\; w_3\, Q_{\text{AR}} \;+\; w_4\big(1 - Q_{\text{Jac}}\big)$$

with a sensitivity study on $w_1 \dots w_4$ reported in the validation.

_Why this way._ An exact evaluation costs ~100 ms; thirty of them cost three
seconds. There is nothing here to accelerate, and nothing that can silently go
wrong — the quantity optimised is measured on a mesh that exists. CMA-ES is held
in reserve for the case where the landscape proves multimodal, which the
certified topology makes unlikely.

**Target end-to-end cost:** a few seconds per configuration on a laptop.

---

## 7. Extension to 3-D wings

The extension is natural, and — importantly — **it does not require a 3-D medial surface**. Extracting a medial surface for a swept, tapered wing with a winglet is notoriously fragile, and this fragility is the main reason 3-D medial methods never became practical. MAPS sidesteps it entirely.

### The construction

A C- or H-topology wing mesh is a stack of 2-D topologies swept along the span. For any wing without a **topological event** — a junction, a nacelle, a discontinuous planform break — the topology is _constant_ in $\eta$. Therefore:

**Step 1.** Take spanwise stations $\eta_1 \dots \eta_m$ (typically $m = 10$–$20$). At each, cut the wing with a plane normal to the quarter-chord line, giving a 2-D section $\Gamma(\eta_j)$.

**Step 2.** Run **Phase 1** on each section independently. Because the sections vary smoothly, so do $r_m(s;\eta)$ and $\theta_m(s;\eta)$. Each section is a 2-D problem the method already solves.

**Step 3.** Promote interface positions from scalars to **spanwise functions**:

$$s_i \;\longrightarrow\; s_i(\eta) \;=\; \sum_{k=1}^{K} c_{ik}\, B_k(\eta), \qquad K \approx 3\text{–}5$$

represented with a B-spline — the same NURBS basis `AeroShape` already uses for the wing surface. The clustering parameters are promoted the same way, since $\Delta y_1$ must vary along the span as the local Reynolds number varies with chord.

**Step 4.** The design vector grows to

$$\dim(\mathbf{x}_{3\text{D}}) \;=\; K \cdot n_{\text{int}} \;+\; 3K \;\approx\; 20\text{–}40$$

Still small. CMA-ES handles this dimension comfortably.

**Step 5.** The objective becomes a spanwise integral plus a spanwise-smoothness penalty:

$$F_{3\text{D}} \;=\; \int_0^1 F\big(\mathbf{x}(\eta)\big)\, d\eta \;+\; w_\eta \int_0^1 \left\| \frac{\partial^2 s_i}{\partial \eta^2} \right\|^2 d\eta$$

The second term is the only genuinely new ingredient. It guarantees the block interfaces form **smooth ruled surfaces** rather than a stack of independently-optimized, misaligned 2-D cuts.

### Why this works

| Property                           | Reason                                                                                       |
| ---------------------------------- | -------------------------------------------------------------------------------------------- |
| No 3-D medial surface needed       | topology is constant along the span; only 2-D medial axes are ever computed                  |
| Sweep and taper handled            | sections are cut normal to the quarter-chord line, so the section is always a proper airfoil |
| The method is reused **unchanged** | it acts per section, in $(s, \rho)$; 3-D never enters the 2-D pipeline                       |
| Cost scales linearly               | $m$ sections × one forward pass each                                                         |
| Spanwise coherence guaranteed      | the B-spline representation of $s_i(\eta)$ is $C^2$ by construction                          |

### Stated limit

Wings with a **topological event** along the span — a winglet junction, a nacelle, a planform break — change the medial graph at that station. Handling these requires splitting the span into topologically-uniform segments. That segmentation is currently manual, and it is the natural next problem.

---

## 8. Connection to the PhD

The three pillars share one geometric primitive — the medial axis — and one data pipeline.

```
  PILLAR I                              PILLAR III
  MAT + DDPM                            MAPS
  geometry generation                   mesh generation
  (MAT skeleton → 3-D wing)             (MAT → block interfaces)
         │                                     │
         └──────────────┬──────────────────────┘
                        │   the same rm , θm fields
                        ▼
                 AeroShape (NURBS)
                        │
                 AeroMesh (MAPS)
                        │
                 RANS solver (SU2)
                        │
                 CL , CD , Cp database
                        │
                        ▼
                    PILLAR II
        multi-fidelity aerodynamic surrogate
          (VLM + panel + sparse CFD → CL/CD)
```

**Pillar II depends on this work.** The multi-fidelity surrogate is trained on sparse RANS corrections to a low-fidelity VLM/panel baseline. Each of those RANS runs needs a mesh of consistent, verified quality. Without automation, the surrogate cannot be trained; with inconsistent meshes, mesh-quality variation becomes a hidden noise source in the training data. MAPS is the enabling infrastructure for Pillar II.

**Pillar I shares the primitive.** The MAT skeleton and envelope that condition the diffusion model in Pillar I are the same $r_m$, $\theta_m$ fields that parameterize interfaces here. That is the thesis's structural argument: the medial axis is not a tool applied three times — it is the object that makes the three pillars one piece of work.

### AeroMesh

```python
from aeroshape import Airfoil, Wing
from aeromesh  import MeshGenerator

import aeromesh as am

main = am.load_airfoil("sd7037.dat")          # -> a Loop
flap = main.transform(scale=0.3, angle=-0.5, dx=1.0, dy=-0.1, name="flap")

mg = am.MeshGenerator(
    region       = am.build_region([main, flap], farfield=15.0, downstream=25.0),
    reynolds     = 2.1e6,
    altitude     = 2850,       # m - sets nu, hence Delta_y1
    yplus_target = 1.0,
)

mesh = mg.generate()           # Phases 1-4
mesh.topology_report()         # blocks, singularities, certificates
mesh.quality_report()          # orthogonality, Jacobian, y+ distribution
mesh.export("sd7037.cgns")
```

3-D:

```python
wing = Wing.from_sections([...])              # AeroShape
mg   = am.MeshGenerator.from_wing(wing, span_stations=15, farfield=15.0, ...)
mesh = mg.generate()                          # spanwise B-spline interfaces
```

`AeroMesh` consumes `AeroShape` geometry and emits solver-ready meshes. It re-implements no geometry handling; the dependency is one-directional and confined to a single module.

Note what is _not_ in the call: no domain type, no topology, no block count. The
region is a set of closed loops, and everything discrete about the mesh is
derived from it.

---

## 9. Software to build on

Nothing here needs to be written from scratch. The stack:

| Component              | Library                                            | Role                                                                                                                  |
| ---------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Geometry**           | `AeroShape` (own)                                  | NURBS airfoil and wing definition — the input                                                                         |
|                        | `geomdl` (NURBS-Python)                            | B-spline evaluation, if `AeroShape` needs extending for $s_i(\eta)$                                                   |
|                        | `shapely`                                          | polygon boolean ops, boundary offsetting, robustness checks                                                           |
| **Medial axis**        | `triangle` (Shewchuk's _Triangle_, Python wrapper) | constrained Delaunay triangulation — the workhorse for $\mathcal{M}$                                                  |
|                        | `scipy.spatial` (`Delaunay`, `Voronoi`)            | circumcentres and the Voronoi dual; sufficient for a first implementation                                             |
|                        | `networkx`                                         | medial graph — branch pruning, topology classification, vertex typing                                                 |
| **Mesh construction**  | _(implement directly)_                             | TFI is ~50 lines; TTM elliptic smoother ~200 lines. Both are classical and better implemented than imported           |
|                        | `meshio`                                           | export to CGNS, Plot3D, VTK, SU2 — do not write file writers                                                          |
| **Mesh quality**       | `vtk` (`vtkMeshQuality`)                           | exact Jacobian, skew, aspect ratio, warpage — the _ground truth_ for the Phase 4 verification                         |
| **Bunin $\phi$-field** | `scikit-fem` or `FEniCS`                           | direct FEM solve of the Poisson system, for repositioning singularities inside their subregions (optional, Section 5) |
| **Optimization**       | `scipy.optimize` (`L-BFGS-B`)                      | Phase 4, the continuous parameters                                                                                    |
|                        | `pycma`                                            | held in reserve, only if the landscape proves multimodal                                                              |
| **Airfoil data**       | UIUC Airfoil Coordinates Database                  | the 50-airfoil ensemble for V4                                                                                        |
|                        | `aerosandbox`                                      | CST parameterization for generating random training airfoils; also has XFOIL bindings                                 |
| **CFD (validation)**   | `SU2`                                              | RANS solver; native structured mesh support; open source                                                              |
|                        | `ADflow` (MDO Lab)                                 | alternative; pairs naturally with the `pyHyp` baseline comparison                                                     |
| **Baselines**          | `pyHyp` (MDO Lab)                                  | the O-grid comparison, V1                                                                                             |
|                        | `construct2d`                                      | classic open-source airfoil C/O grid generator — a second, independent baseline                                       |

**Note on what is absent.** There is no machine-learning stack. Section 5 explains why the neural operator was withdrawn: the search it was built to accelerate no longer exists. If it is revived for the 3-D extension, `DeepXDE`'s physics-informed DeepONet remains the right starting point — a branch net on the profiles $r_m(\cdot)$, $\theta_m(\cdot)$ and a trunk net on $(s,\rho)$, trained against the PDE residual with no labelled data.

---

## 10. What is claimed

1. **Two-sided medial blocking.** Coupling each body's interior skeleton with the fluid region's exterior skeleton, because at CFD far-field distances the exterior one is scale-degenerate. Every prior medial-axis method works on a region with boundaries at comparable scale; none addresses external flow. The degeneracy is measured, and so is the interior skeleton's response (Section 4d).
2. **Body-agnostic by construction.** The input is a set of closed loops. Airfoil, multi-element wing, fuselage section, wing–body crossflow cut and any mixture run through one code path with no branch on body type or count. Verified across six configurations; the medial graph's cycle count matches the body count exactly in every one, and singularities appear inside multi-element slots unprompted because $\theta_m$ collapses to $11^\circ$ there.
3. **Interfaces are curves of a geometry-intrinsic coordinate system.** Constant-$s$ and constant-$\rho$ curves of the medial coordinates, giving wall orthogonality by construction. Rigby and Fogg use the medial axis to decide connectivity and discard $r_m$, $\theta_m$ as geometry; interface placement is never a continuous optimisable quantity.
4. **Mesh design is coupled to the flow.** $\Delta y_1$ bounded by $y^+$, Reynolds number and altitude, not chosen by rule of thumb. No automated method surveyed does this.
5. **Topology with certificates.** Two exact pre-construction checks — the medial graph's cycle count against the body count, and the singularity budget against the Euler characteristic. Admissibility as a proof obligation rather than an inspection. No medial method states a topological invariant its skeleton must satisfy.
6. **3-D follows without a medial surface.** Spanwise B-spline interfaces over per-section 2-D medial axes, with a spanwise-smoothness penalty.

## 11. Validation

| Case | Configuration                                                 | Truncation              | Baseline                                         |
| ---- | ------------------------------------------------------------- | ----------------------- | ------------------------------------------------ |
| V1   | NACA 0012                                                     | smooth offset           | pyHyp, TopMaker                                  |
| V2   | SD7037 (low-Re UAV)                                           | offset + wake extension | Fogg 2015; manual expert mesh                    |
| V3   | NACA 4415                                                     | prescribed box          | TopMaker; Qi 2024 parametric multiblock          |
| V4   | 50 airfoils, UIUC database                                    | smooth offset           | Qi 2024 — ensemble statistics on $F(\mathbf{x})$ |
| V5   | **Two-element: main + deployed flap**                         | smooth offset           | Fogg 2015 multi-element aerofoil                 |
| V6   | **Airfoil beside a fuselage cross-section**                   | smooth offset           | manual ICEM mesh                                 |
| V7   | **1000 randomised CST sections in random convex outer loops** | random                  | _none_ — this is the anti-prescription gate      |
| V8   | Tapered swept wing, AR = 8 (3-D)                              | offset                  | manual ICEM mesh                                 |

V5 and V6 are the cases a prescribed topology cannot reach at all. V7 is not a
comparison but a proof obligation: any hardcoded topology fails on the first
unfamiliar shape, so a thousand random configurations returning a thousand
_certified_ block graphs is what makes claim 2 checkable rather than asserted.
Metamorphic tests accompany it — a rotation, scaling, translation or mirroring
of any input must return an isomorphic block graph.

**Reported for each:** wall orthogonality angle, minimum cell Jacobian, maximum boundary-layer aspect ratio, metric jump across interfaces, achieved $y^+$ distribution, wall-clock time. For V2 and V5, additionally: RANS drag prediction compared against experiment, to show that better mesh quality translates into better aerodynamics — not just better numbers on a mesh-quality report.

---

## References

1. Rigby, D.L. (2004). _TopMaker: a technique for automatic multi-block topology generation using the medial axis._ NASA/CR-2004-213044.
2. Fogg, H.J., Armstrong, C.G., Robinson, T.T. (2014). New techniques for enhanced medial axis based decompositions in 2-D. _Procedia Engineering_, 23rd International Meshing Roundtable.
3. Bunin, G. (2008). A continuum theory for unstructured mesh generation in two dimensions. _Computer Aided Geometric Design_, 25, 14–40.
4. Qi, L. et al. (2025). DRL-MeshGen: automated block-structured mesh generation framework via deep reinforcement learning and optimal conformal mapping. _Engineering with Computers_, 41, 4293–4315.
5. Qi et al. (2024). Automatic structured mesh generation method for airfoil configuration based on parametric multi-block topology. _Applied Sciences_, 16(2), 1116.
6. Wang, M. et al. (2025). PINN-MG: a physics-informed neural network for mesh generation. _arXiv:2503.00814_.
7. Khairullin, B., Rykovanov, S., Zagidullin, R. (2025). Neural networks for structured grid generation. _Scientific Reports_, 15, 12526.
8. Secco, N.R., Kenway, G.K.W., He, P., Mader, C.A., Martins, J.R.R.A. (2021). Efficient mesh generation and deformation for aerodynamic shape optimization. _AIAA Journal_, 59(4).
9. Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G.E. (2021). Learning nonlinear operators via DeepONet. _Nature Machine Intelligence_, 3, 218–229.
10. Li, Z., Kovachki, N., Azizzadenesheli, K. et al. (2021). Fourier neural operator for parametric partial differential equations. _ICLR_.
11. Tancik, M. et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. _NeurIPS_.
12. Hansen, N. (2016). The CMA evolution strategy: a tutorial. _arXiv:1604.00772_.
13. Thompson, J.F., Thames, F.C., Mastin, C.W. (1974). Automatic numerical generation of body-fitted curvilinear coordinate systems. _Journal of Computational Physics_, 15(3), 299–319.
14. Shewchuk, J.R. (1996). Triangle: engineering a 2D quality mesh generator and Delaunay triangulator. _Applied Computational Geometry_, Springer.
15. Slotnick, J. et al. (2014). _CFD Vision 2030 Study: a path to revolutionary computational aerosciences._ NASA/CR-2014-218178.

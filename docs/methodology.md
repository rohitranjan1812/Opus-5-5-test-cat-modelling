# CatForge methodology

This document specifies every model component with its equations, the numerical methods used to
evaluate them, and the invariants the test-suite enforces. Parameter values are an *illustrative
calibration* to the order of magnitude of public US statistics — the framework is the product; the
numbers are placeholders to be re-fitted to proprietary data.

---

## 1. Hazard

### 1.1 Tropical cyclone — landfall-gate stochastic tracks

The US Gulf and Atlantic coastline is a polyline of "gates" (`catforge/data/coast.py`) oriented so that
land is always on the left; the landward normal of a segment is `bearing − 90°`. Each coastal region
*r* carries an annual hurricane landfall rate Λ<sub>r</sub>, a climatological heading and an intensity
scale.

**Landfall position (importance sampled).** The target density along arc length *s* is
f(s) ∝ Λ<sub>r(s)</sub>/L<sub>r</sub>. Samples are drawn (stratified) from g(s) ∝ f(s)<sup>α</sup>,
α = 0.5, which flattens coverage so that low-rate, high-value coasts (New York, New England) receive
enough events. Each event carries the weight w<sub>pos</sub> = f/g.

**Landfall intensity (importance sampled).** For hurricanes,
V<sub>max</sub> = 33 m/s + X, X ~ Weibull(k = 1.6, c = 15.5·scale<sub>r</sub>), truncated at 88 m/s.
The quantile u is drawn from a tail-tilted proposal u = 1 − (1 − v)<sup>γ</sup>, v ~ U(0,1)
(stratified), γ = 2.2, whose density is g(u) = γ<sup>−1</sup>(1 − u)<sup>1/γ − 1</sup>; the
importance weight is w<sub>int</sub> = γ(1 − u)<sup>1 − 1/γ</sup>, with E<sub>g</sub>[w] = 1.
Tropical storms use V<sub>max</sub> ~ U(18, 33) m/s at 0.85× the hurricane rate.

Each event's rate is λ<sub>i</sub> = Λ · w<sub>pos,i</sub> · w<sub>int,i</sub> / N, which makes
Σλ an unbiased estimator of the regional climatology while putting ≈5× more events in the tail.

**Storm parameters.**

- Translation speed: ln V<sub>t</sub> ~ N(ln(4.5 + 0.35·max(φ − 25, 0)), 0.35²), clipped to [1.5, 22] m/s.
- Heading: normal + δ, δ ~ N(wrap(clim − normal), 25²), truncated to |δ| ≤ 75° (the track must cross from sea to land).
- Recurvature: ω ~ N(ω̄(φ), 0.35²) °/h, where ω̄ is 0 (φ < 27), 0.25 (27–33) or 0.45 (φ > 33).
- Surface Holland parameter (Holland 2008, ∂p/∂t = 0): B<sub>s</sub> = −4.4·10⁻⁵Δp² + 0.01Δp − 0.014|φ| + 0.15 V<sub>t</sub><sup>x</sup> + 1, x = 0.6(1 − Δp/215).
- Pressure deficit from the fixed point of V<sub>max</sub> = √(B<sub>s</sub>Δp/(ρe)) + 0.55 V<sub>t</sub>, capped at 140 hPa (B then absorbs the remainder).
- Radius of maximum winds (Vickery & Wadhera 2008 form): ln R<sub>max</sub> = 3.015 − 6.291·10⁻⁵Δp² + 0.0337φ + 0.40ε.

**Track and decay.** Tracks are integrated hourly along great circles from −36 h to +60 h around
landfall. Over land, Kaplan & DeMaria (1995) filling applies: V(t + 1 h) = V<sub>b</sub> + (V(t) − V<sub>b</sub>)e<sup>−α</sup>,
with α = 0.095 h⁻¹ and V<sub>b</sub> = 13.75 m/s. There is no re-intensification over water, and the
land mask is a point-in-polygon test.

**Wind field.** At distance r from the centre, the Holland profile is

V(r) = √( B Δp/ρ · (R<sub>m</sub>/r)<sup>B</sup> e<sup>−(R<sub>m</sub>/r)<sup>B</sup></sup> + (rf/2)² ) − rf/2

It is rotated inward by the Sobey, Harper & Stark (1977) inflow profile — 10° inside R<sub>max</sub>,
rising linearly to 25° at 1.2 R<sub>max</sub> and constant beyond — and the translation vector is added
scaled by 0.55·V(r)/V<sub>max</sub> (the right-of-track asymmetry). The site gust is
K<sub>terrain</sub>·|**V**|, where K is 1.18 (coastal), 1.10 (open), 1.00 (suburban) or 0.94 (urban).
The footprint is the lifetime maximum: evaluated hourly, then refined at 7.5-minute sub-steps within
±1 h of the peak.

**Kernel.** The footprint kernel runs in numba, parallel over events. Sites are bucketed on a 0.5°
spatial grid, and the influence radius per track point is min(700, 300 + 3R<sub>m</sub>) km. The
output is a CSR sparse event×site table with gust ≥ 20 m/s.

### 1.2 Earthquake

**Sources** (`catforge/data/seismic_sources.py`): named faults (San Andreas, Hayward, Cascadia, New
Madrid, …) and rectangular background zones. The magnitude–frequency distribution is truncated
Gutenberg–Richter:

N(M ≥ m) = ν · (10<sup>−b(m−m<sub>min</sub>)</sup> − 10<sup>−b(m<sub>max</sub>−m<sub>min</sub>)</sup>) / (1 − 10<sup>−b(m<sub>max</sub>−m<sub>min</sub>)</sup>)

binned at ΔM = 0.1 on faults and 0.2 in areas.

**Ruptures.** Rupture length follows Wells & Coppersmith (1994), log L = −3.22 + 0.69M, or Strasser
et al. (2010) for the subduction interface, log L = −2.477 + 0.585M. On a fault, ruptures slide
along the trace at several positions (the rate is split evenly). In an area source, epicentres are
drawn by Latin hypercube, the strike is uniform, and the position count scales with area/(πr<sub>dmg</sub>²).

**Finite, dipping ruptures.** Every rupture is a planar fault below its trace (right-hand rule: it
dips to the right of the trace direction) with dip δ, top depth z<sub>tor</sub> and down-dip width
W = min(10<sup>−1.01+0.32M</sup>, (z<sub>bot</sub> − z<sub>tor</sub>)/sin δ) (interface:
10<sup>−0.882+0.351M</sup>). Faults carry their own dip, depth range and mechanism (strike-slip,
reverse, subduction interface); background zones use vertical strike-slip planes. R<sub>JB</sub> is the
distance to the rupture's **surface projection** — zero for every site above the plane — which
produces the hanging-wall amplification of dipping thrusts (Northridge-type analog: 0.31 g on the
hanging wall vs 0.19 g at the same distance on the footwall).

**Ground motion.** PGA uses the Boore & Atkinson (2008) functional form:

ln Y = F<sub>M</sub>(M) + [c₁ + c₂(M − 4.5)] ln R + a·c₃(R − 1) + b<sub>lin</sub> ln(V<sub>s30</sub>/760) + b<sub>nl</sub> ln(max(PGA<sub>rock</sub>, 0.1)/0.1)

Here R = √(R<sub>JB</sub>² + h²), and the anelastic factor *a* is 0.3 in the central/eastern US,
0.6 on the subduction interface and 1 elsewhere. Aleatory variability is τ = 0.26 (inter-event)
and φ = 0.50 (intra-event).

### 1.3 Hazard curves & maps

For a site, λ(x) = Σ<sub>e</sub> λ<sub>e</sub> [1 − Φ((ln x − ln m<sub>e</sub>)/σ)] with
σ = √(σ<sub>b</sub>² + σ<sub>w</sub>²). The annual exceedance probability is 1 − e<sup>−λ(x)</sup>.
Return-period maps invert λ(x) = −ln(1 − 1/T) by log-log interpolation.

### 1.4 Storm surge in the stochastic engine — multi-fidelity

Surge is part of the hurricane peril. It is on by default (`AnalysisConfig.tc_surge`) and flows
through the kernel with the wind. The full 2-D model of §9.1 takes 3–12 s per event. A 4,800-event
catalog, of which about 3,900 events bring ≥ 30 m/s gusts to a surge-reachable site, would take hours.
So the engine runs a cheap model on every event and learns, statistically, how it differs from the
full model.

**Low fidelity, for every event (`hazard/surge.py`).** This is the same solver as §9.1, with three
differences:

- a coarser, cheaper domain: the landfall box (±2.0° lat × ±2.5° lon), united with the padded
  bounding box of the track points within 250 km of the coast, at 4′ (stride 2 on the 2′ DEM);
- Δt = 90 s, with forcing refreshed every 12 steps, over the full model's window (−18 h to +12 h);
- the whole time loop is fused into one `nogil` numba call, so events run concurrently on threads.

The domain follows the track along the coast for a reason. Version 1 used the landfall box alone,
and the fidelity-allocation diagnostics (§1.5) exposed it. A storm that passes Tampa Bay offshore at
hurricane strength, then makes landfall in the Panhandle 400 km away, floods Tampa, but Tampa was
outside the box. The surrogate saw 3–8 % of that surge, with zero uncertainty, because a site
with no low-fidelity water gets no surge draws at all. Storms running up the coast from Florida to
the Carolinas had the same problem.

That is about 0.13 s per event threaded and 0.5 s serial, **roughly 10–40× cheaper than the full
model**. Open-coast peaks agree with the full model within 0–10 % (Hugo 5.93 vs 5.92 m, Ian 6.29 vs
6.49 m, Michael 4.57 vs 4.86 m).

The event product is portfolio-independent: the wet near-shore cells (z > −20 m, η > 0.1 m) with
their peak water surface. It is cached on disk per catalog signature
(`~/.cache/catforge/surge/lf_<sig>.npz`), so only the first run of a catalog pays.

**Site level x.** Friction-limited penetration from the event's cells:

x<sub>s</sub> = max<sub>c: d ≤ R</sub> [η<sub>c</sub> − α·max(d(c, s) − r₀, 0)]

It is evaluated only for surge-reachable locations: within 30 km of the coast and below 20 m on the
2′ DEM.

**Correction to the full model — a two-part (hurdle) model (`hazard/surge_calib.py`).** The design
set has three parts, and both models are run on every storm:

- 84 catalog storms stratified by 7 coastal regions × 4 intensity classes;
- an intense-storm stratum, one storm per 1.5° coastal reach × {Cat 1–2, Cat 3, Cat 4+} (131
  storms), as in JPM-OS surge studies, so that every bay sees big water;
- the 9 historical analogs.
 Every coastal 2′ node inside the box is recorded: land nodes,
plus shoreline water nodes, which is where waterfront buildings geocode on a coarse DEM. That gives
448,584 node-events from 224 storms, and the full model's value is the peak water surface in the
node's 3×3 stencil — the rule the engine applies to buildings. Two processes decide what a node sees:

- **connectivity.** P(wet) = σ(θ·[1, x, z, shore, x·z, x − z, (x − z)₊, Δ, Δ·shore]), a logistic
  model fitted by IRLS.
- **level given wet.** y = s(x) + γ·[shore, z, x·shore, x·z, Δ, Δ·shore, min(Δ, 3)·x] +
  u<sub>e</sub> + δ<sub>n</sub> + ε,
  where s is a linear spline with knots at 1, 2, 3, 4 and 6 m, u<sub>e</sub> ~ N(0, σ<sub>e</sub>²) is
  an event term, δ<sub>n</sub> ~ N(0, τ²) is the node's **site response** (the harbour/bay term), and
  ε ~ N(0, σ²). The crossed random effects are fitted by EM with BLUP shrinkage.

Here z is the node's 2′ ground, clipped to 0–10 m, and shore = 1 when the stencil touches the sea.
Both are computed from the DEM by the same function (`site_covariates`) in calibration and in the
engine. Δ = x(α = 0) − x is the friction loss the site rule applied. When a site draws on distant
water across a bay, that water travels over water and amplifies (funnelling) instead of attenuating.
Δ lets the model learn this, and it cuts held-out error in water over 1 m deep by 8 %. A per-node
random *slope* on x (a multiplicative site amplification, fitted as a 2×2 random effect) was tried
instead. With two or three big storms per bay it overfits, and held-out error rises; physical
covariates and design coverage do the job instead. The shrunk δ̂<sub>n</sub> and its posterior variance are stored for 8,664 nodes
(`data/surge_site_response.npz`). Nodes the design set never reached get the prior (0, τ²).

Penetration parameters are chosen by 5-fold, event-grouped cross-validation of flood depth over the
engine's default ground (the 2′ DEM floored at 1 m). The CV surface is flat for α between 0.2 and
0.3 m/km. Among fits within 1 % of the best, the largest R is taken, because misses (flooded nodes
that no 4′ cell reaches) are one-sided. The result is **α = 0.3 m/km, R = 20 km, r₀ = 3.5 km,
σ = 0.37 m, τ = 0.27 m.**

**The error model has three components.**

- **An event-wide error that grows with the surge,** u<sub>e0</sub> + u<sub>e1</sub>·(x − 2), with
  (u<sub>e0</sub>, u<sub>e1</sub>) ~ N(0, Ψ). The event-mean residual's sd rises from 0.18 m at
  x ≈ 1 to 0.32 m at x ≥ 3, so a constant event term misstates big storms. The EM fits Ψ jointly
  with the node terms, giving sd 0.13 m at x = 2 and 0.23 m at x = 5 (correlation 0.83). A per-event
  slope is well identified, because each event carries hundreds of nodes.
- **A spatially correlated residual within the event.** Its correlation is about 0.8 at 3 km, 0.44
  at 9 km, 0.19 at 17 km and about 0 beyond 40 km, because whole bays are off together. The fit is
  σ²·[0.54·exp(−d²/4·3.9²) + 0.32·exp(−d²/4·9.5²)] with d in km, which leaves a 14 % nugget.
  Treating node residuals as independent would average them away over the hundreds of buildings an
  event floods, and the event's loss uncertainty would be several times too small.
- **The node's site response δ<sub>n</sub>** with its posterior variance.

**Why a hurdle model and not a Tobit.** Both look principled. At x ≥ 2 m, the held-out residuals of a
censored single-Gaussian model have skew −0.8 to −2.1 and excess kurtosis 3–7. The error is a
mixture: connected nodes sit above the fit, and sheltered nodes (behind barriers, in basins the 4′
grid merges) sit far below it. A pooled Gaussian spends its lower tail on connected nodes, which cuts
their expected depth. The sheltered nodes cannot give the depth back, because it is floored at 0 on
their higher ground. So the Tobit is biased low exactly where losses are made. The table gives
held-out E[depth] / full-model depth:

| low-fidelity level x | 1–2 m | 2–3 m | 3–4 m | 4–6 m | ≥ 6 m | ground 3–5 m (x ≥ 2) |
|---|---|---|---|---|---|---|
| **hurdle (engine)** | **0.93** | **0.98** | **0.97** | **0.98** | **1.01** | **0.93** |
| Tobit + site term | 1.02 | 0.77 | 0.69 | 0.65 | 0.63 | 0.38 |
| OLS on wet-in-both pairs | 1.14 | 0.81 | 0.70 | 0.65 | 0.64 | 0.33 |

Overall, the hurdle model has a held-out depth RMSE of 0.35 m (misses included), 0.45 m where the
water is over 1 m deep, an aggregate depth bias of −1.7 %, a hit rate of 0.89 and a false-alarm
ratio of 0.10. The Tobit and the OLS reach 0.84–0.86 m in deep water. The OLS on wet pairs is a truncated
sample, so it is biased too: +16 % in aggregate.
Recalibrate with `python scripts/calibrate_surge.py collect && … fit` (about 10 minutes, then about
10 seconds).

**In the kernel.** Each hurricane pair carries the calibrated wet level
WSE = s(x) + γ·covariates + δ̂<sub>n</sub> and the connectivity probability π. The draws are keyed
by the building's **2′ node** n, not by the building. In the full model, buildings in one node share
one stencil water level, so a concentrated book must not diversify that risk away:

- wet if u(k, WET + n) < π;
- ζ = WSE + U<sub>k</sub>(x) + C<sub>k</sub>(n) + σ<sub>s,j</sub>Φ⁻¹(u(k, SURGE + n)), with:
  - U<sub>k</sub>(x) = u₀ + u₁(x − 2), the event error, drawn through the Cholesky factor of Ψ;
  - C<sub>k</sub>(n) = Σ<sub>m</sub> w<sub>nm</sub>Φ⁻¹(u(k, FIELD + m)), the correlated field, a
    two-lattice **process convolution**. Lattice knots of spacing h<sub>i</sub> get counter-RNG
    normals, and Gaussian weights normalised to Σw² = 1 give every node exactly the variance
    σ²s<sub>i</sub> and correlation exp(−d²/4h<sub>i</sub>²), reproduced to within 0.013. Knot draws
    are memoised per occurrence, so the field stays bit-reproducible and independent of the thread
    count.
  - σ<sub>s,j</sub>² = σ²·nugget + Var δ̂<sub>n</sub>;
- depth = ζ − g<sub>j</sub>, where g<sub>j</sub> is measured ground (§9.4) or else the 2′ DEM
  floored at 1 m;
- D<sub>s</sub> = m<sub>j</sub>·f(depth − ff<sub>j</sub>), using USACE-style depth–damage curves,
  first-floor heights by era and occupancy (overridable with `first_floor_height_m`), and
  storey/material/mobile-home modifiers.

Wind and surge combine per building as D ← 1 − (1 − D<sub>w</sub>)(1 − D<sub>s</sub>), before any
financial terms, so deductibles and limits see the combined loss. The surge share is attributed
exactly: the kernel re-evaluates the coverage loss without surge from the *same* wind draw. The
surge-attributed ground-up loss per occurrence is returned alongside, and the analysis summary gives
the surge AAL share and the hurricane AEP with and without surge (`summary.surge`, the Results page).

**Validation on the tail against the full model (`scripts/validate_surge_tail.py`,
`docs/validation/surge_tail.json`).** The test case is the demo book (5,000 locations), hurricane
only, 10,000 years. Every event that matters takes the full 2′ model's water levels through the
engine's own path, and exactly its occurrences are re-simulated with the occurrence keys unchanged,
so wind uses common random numbers. "Every event that matters" is 606 events: the tail of the 200
worst years, every event of the 120 worst years, and every event fidelity allocation (§1.5) picked
at any budget.

| Hurricane GU, surge included | engine (surrogate only) | full-model reference | engine vs reference | wind only |
|---|---|---|---|---|
| AEP 1-in-100 | $488.0m | $497.8m | −2.0 % | $381.9m |
| AEP 1-in-250 | $605.2m | $606.5m | −0.2 % | $497.3m |
| AEP 1-in-500 | $683.7m | $692.3m | −1.3 % | $584.8m |
| TVaR 1-in-250 | $728.6m | $737.3m | **−1.2 %** | $618.9m |

Summed over the tail events, the surrogate's surge loss is 0.95× the full model's. The earlier
release was at 0.80–0.84× before the low-fidelity domain followed the track and the error model got
its severity-scaled event term and correlated field.

**Is the surrogate's uncertainty honest?** For 166 tail events, the full model's expected surge loss
was compared with the surrogate's distribution under identical wind draws (64 samples):
z = (full − mean)/epistemic sd.

| | z mean (ideal 0) | z sd (ideal 1) | share with \|z\| > 2 (ideal ≈ 5 %) |
|---|---|---|---|
| v1 domain, global event term, independent node residuals | +3.48 | 14.0 | 23 % |
| **v2 domain, event slope, correlated field (engine)** | **+0.44** | **1.81** | **9.6 %** |

The uncertainty is now informative but still about 1.8× under-dispersed. The stopping rule of §1.5
absorbs this in an effective correlation ρ̄.

Compute: the full model takes about 5 s per event, the low-fidelity model 0.5 s serial and 0.13 s
threaded. A fresh 10,000-year hurricane analysis with surge takes 21 s once the low-fidelity product
is cached; building that cache for a catalog takes about 10 minutes, once. On this book surge is 43 %
of hurricane ground-up AAL (1,604 of 5,000 locations reached; the book has no measured ground). It
adds +71 % at 1-in-10, +28 % at 1-in-100 and +22 % at 1-in-250.

### 1.5 Fidelity allocation — full-model surge where the tail needs it (`engine/fidelity.py`)

With `surge_fidelity = {"budget": K, "tol": τ, "rp": T}`, the engine spends up to K full-model runs
on the hurricanes that reduce the surge error of the reported 1-in-T AEP TVaR the most. After the
first ELT and YLT passes:

1. **Tail sensitivity a<sub>e</sub>.** This is the *realised* Euler gradient of this simulation's
   TVaR: the number of e's occurrences in the k = n/T worst simulated years, divided by k, with the
   year weight tapering linearly to 0 at rank 2k. The taper hedges against years crossing the tail
   boundary once losses change.
2. **Resolvable uncertainty V<sub>e</sub>.** The variance of e's surge-attributed loss over its ELT
   samples. The track is fixed per catalog event, so every surge draw is epistemic with respect to
   the full model: the event error, the correlated field, the nugget and connectivity. One full-model
   run removes them all.
3. **Order and stopping.** Events are upgraded in order of a<sub>e</sub>·sd<sub>e</sub>. The
   stopping rule uses U² = (1 − ρ̄)·Σa²V + ρ̄·(Σa·sd)², which interpolates between independent
   (ρ̄ = 0) and fully coherent (ρ̄ = 1) event errors. The ordering by a·sd is optimal at both ends.
   The engine stops at the first K with U ≤ τ·TVaR, or at the budget.
4. **Upgrade and exact re-simulation.** The chosen events take the full model's water levels (cached
   per event, portfolio-independent), with residual and connectivity draws switched off. Exactly
   their ELT rows and YLT occurrences are re-simulated, and every other draw is unchanged (common
   random numbers). Location AAL is patched with the exact delta (new − old, same keys), so it still
   sums to the ELT AAL to 10⁻⁹.

**Why the realised gradient, not the expected one.** The first version used the expected tail
participation, λ<sub>e</sub>·T·E<sub>s</sub>[P(A ≥ VaR − L<sub>e,s</sub>)]. An oracle study on the
demo book decomposed the TVaR error exactly by event. The error is extremely concentrated, with 6
events holding 59 % of Σc<sub>e</sub>². An event that never lands in a simulated tail year carries
none of the error in the reported number, however likely it is to do so in expectation. The table
gives the exact TVaR₂₅₀ error after upgrading K events, against the full-model reference:

| K | 3 | 6 | 12 | 24 | 48 | 96 |
|---|---|---|---|---|---|---|
| expected participation | −1.23 % | −1.29 % | −1.24 % | −1.29 % | −1.26 % | −1.07 % |
| random events from tail years | −1.17 % | −1.17 % | −1.16 % | −1.12 % | −1.13 % | −1.01 % |
| **realised participation, taper 2k (engine)** | −1.44 % | −1.26 % | −1.11 % | **−0.44 %** | **−0.47 %** | **−0.08 %** |
| oracle (knows each event's error) | −0.52 % | −0.38 % | −0.29 % | +0.04 % | −0.03 % | 0.00 % |

The engine run through `run_analysis` reproduces its row exactly. Event errors are signed, so a
small K can overshoot; −1.44 % at K = 3 is an example.

**Choosing ρ̄.** The baseline's realised TVaR₂₅₀ error of −1.17 % sits between U(ρ̄ = 0) = 0.53 %
and U(ρ̄ = 1) = 5.46 %, at ρ̂ = 0.037. The default ρ̄ = 0.04 is an *effective* parameter: it absorbs
both the cross-event correlation of errors and the 1.8× under-dispersion above. With it, the
stopping rule's estimate tracks the realised error:

| tolerance τ | events picked | U after (estimate) | realised TVaR₂₅₀ error after |
|---|---|---|---|
| 1 % | 9 | 0.99 % | −1.19 % |
| 0.5 % | 50 | 0.49 % | −0.42 % |
| 0.25 % | 96 | 0.25 % | −0.08 % |

The cost is one full-model run per pick, about 5 s the first time; the fields are cached and shared
by every portfolio. The Results page shows the uncertainty before and after, the realised TVaR
change, and the events upgraded. ρ̄ was estimated on one book. A second, differently concentrated
book is the obvious next check.

---

## 2. Vulnerability

**Mean damage.**

- Wind uses the Emanuel (2011) form μ(V) = v³/(1 + v³), with v = (V − 27)₊/(V<sub>½</sub> − 27).
  V<sub>½</sub> depends on construction and is modified multiplicatively by occupancy, year-built
  band, storeys, roof shape and shutters.
- Earthquake uses HAZUS-style lognormal fragilities (slight, moderate, extensive, complete) on PGA,
  with damage ratios (2, 10, 50, 100)% and a year-built code factor on the medians.

**Secondary uncertainty — a zero-one-inflated Beta on damage bins.**

- P(D = 0) = π₀ = 0.95 (1 − μ/0.15)₊²
- P(D = 1) = π₁ = ((μ − 0.3)/0.7)₊²
- D | 0 < D < 1 ~ Beta(ms, (1−m)s), with m = (μ − π₁)/(1 − π₀ − π₁) and s = 1/k − 1

So E[D] = μ exactly. The distribution is discretised onto 51 interior bins plus two point masses;
inside a bin the density is uniform.

**Hazard uncertainty folded in analytically.** With ln I = ln m + ε and ε ~ N(0, σ<sub>w</sub>²),
the effective bin probabilities are P<sub>eff</sub>(b | m) = ∫ P(b | m e<sup>ε</sup>) φ(ε) dε. This
is computed once per vulnerability class as a Gaussian-kernel matrix product on a 160-node
log-intensity grid. The zero-one-inflated Beta probabilities are pre-tabulated on a dense μ-grid,
and tables are memoised. The engine therefore needs **one uniform per (occurrence, site)** to sample
damage, plus the inter-event residual η.

**Coverages.** Contents damage is min(1, a·d<sup>b</sup>) and business interruption is
min(1, (d/d₀)<sup>p</sup>), both mapped from building damage.

---

## 3. Financial terms

For each occurrence, site and account:

```
GU_j = Σ_c TIV_jc g_c(D_j)
X_j  = min((GU_j − ded_j)₊, lim_j)                                  (ded ≤ 1 ⇒ fraction of TIV)
G_a  = share_a · min((min((Σ_{j∈a} X_j − accded_a)₊, acclim_a) − attach_a)₊, layer_a)
```

Pairs are pre-sorted by (event, account), so account aggregation is a streaming sum.
Account gross is allocated back to its sites pro rata to X<sub>j</sub>, which keeps allocation
additive. Per-risk XL applies per account per occurrence.

---

## 4. Loss engine

### 4.1 Stateless counter-based randomness

Each occurrence *k* gets a stream h = mix(seed ⊕ mix(key<sub>k</sub>)), where *mix* is the SplitMix64
finaliser, and draw *i* is u = (mix(h + (i + 1)·φ<sub>64</sub>) ≫ 11 + ½)·2⁻⁵³. Draws are indexed
as follows:

- η uses index 0.
- The event factor Z uses index 1.
- Cell factors use index 2⁴⁰ + cell id.
- Site draws use index 16 + site id.
- Surge uses index 2 for the event water-level error, 2³⁸ + node for the site water-level error, and
  2³⁹ + node for connectivity. The node is the building's 2′ cell (§1.4).

Consequences:

- Results are bit-identical regardless of thread count.
- Any occurrence can be re-simulated alone. This is used for exact tail re-simulation, and the
  tests assert zero drift.
- What-ifs get **common random numbers**: changing one account leaves every other site's draws
  unchanged.

### 4.2 Dependence — spatially explicit hazard residuals (default)

The log-intensity at site *j* in occurrence *k* is

ln I<sub>kj</sub> = ln Ī<sub>ej</sub> + η<sub>k</sub> + σ<sub>w</sub> W<sub>k</sub>(s<sub>j</sub>),  η<sub>k</sub> ~ N(0, τ²)

Here W<sub>k</sub> is a unit-variance **Matérn Gaussian random field** with smoothness ν and practical
range ℓ (correlation 0.05 at ℓ): TC ν = 1.5, ℓ = 60 km (smooth wind residuals); EQ ν = 0.5, ℓ = 30 km
(the exponential kernel of Jayaram & Baker 2009). Vulnerability tables are built with σ<sub>w</sub> = 0,
because the within-event hazard variability is now simulated rather than convolved.

**Sampling at scale — Vecchia / nearest-neighbour GP.** For a portfolio of n sites the joint density
factorises exactly as Π p(w<sub>i</sub> | w<sub>1..i−1</sub>). Vecchia (1988) conditions each site on
its m nearest *previously ordered* neighbours only:

w<sub>i</sub> = Σ<sub>t∈N(i)</sub> b<sub>it</sub> w<sub>t</sub> + d<sub>i</sub> z<sub>i</sub>,  b<sub>i</sub> = C<sub>NN</sub>⁻¹ c<sub>N,i</sub>,  d<sub>i</sub>² = 1 − c<sub>N,i</sub>ᵀ b<sub>i</sub>

- **Ordering:** approximate maximin (coarse-to-fine hierarchical grid), which is what makes small m
  accurate (Guinness 2018).
- **Neighbours:** ordered k-NN via a KD-tree on 3-D unit-sphere coordinates, with a brute-force
  fallback for the first sites.
- **Coefficients:** batched m×m Cholesky solves with a 10⁻⁶ nugget; |b| < 10⁻³ pruned.
- **Per-event closures:** an event touches only some sites, but a site's conditional needs its
  ancestors. The closure of each event's site set under the parent relation is precomputed once
  (numba), so an occurrence samples exactly the sites it needs — on average 1.5× the touched set.
- **Randomness:** z<sub>i</sub> comes from the counter RNG at index 2³⁶ + site, so fields are
  reproducible per occurrence and tail re-simulation stays bit-identical.

Accuracy (tests): with m = 30 the implied correlation matrix is within 0.013 (exponential) and 0.026
(ν = 1.5) of the exact Matérn on 500 clustered sites, with KL divergence 0.010. On the 5,000-site demo
book (20,000 years), switching from the copula to the field preserves AAL (23.12 vs 22.99 $m) and
moves the tail as expected from the correlation structure (1-in-250 AEP 300 vs 306 $m).

**Damage residuals** keep a light two-level copula on the damage uniform,
x<sub>kj</sub> = √ρ<sub>e</sub> Z<sub>k</sub> + √ρ<sub>c</sub> Z<sub>k,cell(j)</sub> + √(1 − ρ<sub>e</sub> − ρ<sub>c</sub>) ε<sub>kj</sub>,
with ρ<sub>e</sub> = 0.04 and ρ<sub>c</sub> = 0.10 (0.25° cells).

`dependence="copula"` restores the legacy model: σ<sub>w</sub> convolved into the vulnerability
tables, and the copula coupling hazard and damage together. Grids use circulant embedding
(`circulant_field`) — exact and O(N log N) for regular lattices.

### 4.3 Frequency — mixed Poisson with climate regimes

N | Θ ~ Poisson(ΘΛ), Θ = M<sub>R</sub>·G

- R ~ Categorical over ENSO states (La Niña, Neutral, El Niño), with multipliers normalised so that E[M] = 1.
- G ~ Gamma(r, 1/r).

Because thinning preserves the mixing law, only events that touch the portfolio are simulated. Θ
and R are stored per year, which makes regime-conditional EP an exact sub-setting of years and
turns likelihood-ratio reweighting into a closed form (§6).

### 4.4 Passes

1. **ELT pass** — every relevant event × S samples, each weighted λ/S. It produces the event mean,
   SD, P(0) and exposed limit, plus location AAL. Location AAL is exactly additive, so AAL by any
   dimension is a group-by.
2. **YLT pass** — occurrences in (year, day) order with seasonality, plus a per-segment loss matrix.
3. **Tail re-simulation** — only the occurrences in the top ⌊n/T⌋ AEP years, at full site detail.
   This yields the Euler co-TVaR per site. The test-suite asserts that Σ co-TVaR = TVaR exactly.

---

## 5. Risk metrics

- **EP.** OEP uses the annual maximum occurrence; AEP uses the annual aggregate. VaR<sub>T</sub> is
  the Hazen-position quantile at 1 − 1/T. TVaR is the mean of the top ⌊n/T⌋ years.
- **Confidence intervals.**
  - Quantiles: distribution-free order statistics. The rank of the q-quantile is Binomial(n, q),
    so [x<sub>(l)</sub>, x<sub>(u)</sub>] has ≥ 95% coverage.
  - TVaR: vectorised bootstrap.
  - AAL: σ/√n.
- **Analytic cross-check (independent of the YLT).** Each event is L<sub>i</sub> = 0 with
  probability p₀, otherwise cap<sub>i</sub>·Beta matched to the ELT's conditional moments. The
  rate-weighted mixture severity per peril is deposited on a grid, using exact conditional bin means
  (via I<sub>x</sub>(a + 1, b)) so that the mean is preserved exactly. Then:
  - OEP(x) = 1 − Π<sub>p</sub> P<sub>p</sub>(F<sub>p</sub>(x))
  - AEP = IFFT[Π<sub>p</sub> P<sub>p</sub>(FFT(f<sub>p</sub> e<sup>−θk</sup>))] · e<sup>θk</sup>, with θ = 20/N (exponential tilting against aliasing).

  The pgf of the mixed Poisson is P(z) = Σ<sub>R</sub> p<sub>R</sub>(1 − M<sub>R</sub>Λ(z − 1)/r)<sup>−r</sup>.
- **Allocation.**
  - Euler co-TVaR by any exposure dimension, per account and per location.
  - Stand-alone versus contribution by segment, giving the diversification benefit.
  - Marginal impact by CRN re-run with the account removed. This differs from the Euler share by
    design: the difference is the account's share of diversification.

## 6. What-ifs

| What-if | Method | Exactness |
|---|---|---|
| Frequency change, TC intensity-distribution shift (climate) | Year likelihood ratio w<sub>y</sub> = Π<sub>k∈y</sub>(λ'<sub>e</sub>/λ<sub>e</sub>)·Π<sub>p</sub>e<sup>−Θ<sub>p,y</sub>(Λ'<sub>p</sub>−Λ<sub>p</sub>)</sup>; the intensity shift is λ'/λ = f'(V)/f(V) for a Weibull-scale change; ESS reported | Exact (importance sampling) |
| ENSO conditioning | Subset of years by simulated regime | Exact |
| Vulnerability, correlation, σ<sub>b</sub> | Kernel re-run on the same YLT and keys | CRN |
| Mitigation (shutters, roofs, code upgrades, URM retrofit) | Modified attributes → rebuilt tables → CRN re-run | CRN |
| Reinsurance structures | Re-applied to stored occurrence losses (no re-simulation) | Exact |

## 7. Reinsurance

Contracts are applied in inuring *stages*; contracts in the same stage share one subject loss.

For a cat XL, the occurrence layer loss is x<sub>k</sub> = min((S<sub>k</sub> − A)₊, L). Its
running annual sum is C<sub>k</sub>, and the recovery is

R<sub>k</sub> = min((C<sub>k</sub> − AAD)₊, AAL) − min((C<sub>k−1</sub> − AAD)₊, AAL)

with AAL = L(1 + n) by default. A stop-loss is the case A = 0, L = ∞. A quota share recovers
c·S<sub>k</sub>, optionally capped per event.

Pricing is either EL + kσ or EL + CoC·(TVaR<sub>α</sub> − EL). Reinstatement premium is pro rata
as to amount, which gives the closed form

P = (EL + load) / (1 + r·E[min(R, nL)]/(L·placed))

The optimiser grid-searches single layers attaching and exhausting at gross OEP return periods and
reports the Pareto frontier of expected net cost against net 1-in-200 AEP.

## 8. Verified invariants (test-suite)

- **AAL agreement.** YLT, ELT and location-sum AAL agree. ELT and location-sum are exactly equal;
  YLT is within MC error.
- **Euler allocation is exact.** Σ co-TVaR equals TVaR, and tail re-simulation is bit-identical.
- **Financial terms.** Per-account terms hold, and gross ≤ ground-up on every occurrence.
- **Common random numbers.** Rerunning with identical parameters is identical; damage × 1.1 is
  pathwise monotone.
- **Reweighting.** Frequency × 1.2 gives AAL × 1.2 within MC error, with ESS reported.
- **Reinsurance.** Reinstatement exhaustion, AAD and inuring order are checked on hand-worked
  examples.
- **Analytic compound Poisson.** The FFT matches closed forms (P(N ≥ 2) etc.).
- **Mixed Poisson.** The pgf and the variance-to-mean ratio match simulation.
- **Hazard physics.** The Holland peak sits at R<sub>max</sub>, right-of-track winds are stronger,
  winds decay inland, the Sobey inflow profile is continuous, hanging-wall sites exceed footwall
  sites, and GR bin rates sum to ν.
- **Random fields.** The Vecchia implied covariance matches Matérn (max |Δρ| < 0.03, KL < 0.02);
  Monte-Carlo correlations match; circulant fields hit the target lag correlation.
- **Surge solver.** A uniform wind stress on a closed flat basin reaches the analytic set-up
  Δη = τL/(ρgh) within 6 %, and volume is conserved to 10⁻⁹ m. The Hugo analog peaks at 4–8 m
  near Bulls Bay. Buildings do not start flooded, coastal buildings flood, inland ones stay dry.
- **Fidelity allocation.** Allocation orders events by tail weight × sd and stops at the tolerance
  or the budget. Realised tail participation equals the Euler count with the taper. An end-to-end
  run with K = 2 upgrades exactly those events: every other occurrence is bit-identical to the run
  without allocation, and location AAL still sums to the ELT AAL to 10⁻⁹. The two-lattice process
  convolution gives each node exactly σ²Σs<sub>i</sub> and the calibrated correlation to within 0.03.
- **Stochastic surge.** The low-fidelity peak tracks the full model (Hugo 4.5–7.5 m near Bulls
  Bay). Friction-limited site extraction matches its closed form. The crossed mixed model recovers
  known parameters from censored synthetic data (slope ±0.03, τ ±0.05, δ̂ correlation > 0.95) while
  the wet-pair OLS is visibly biased. On synthetic data where connectivity and level separate, the
  hurdle model's deep-water depth is within 6 % and beats the Tobit. In the kernel, surge adds loss
  only through water above a building's ground: the same water on 12 m ground and an inland building
  are unchanged to 10⁻⁹. The wind/surge split is exact (GU − surge = the wind-only run), and re-runs
  are bit-identical.
- **Rupture kinematics.** Σ μ·A·slip = M<sub>0</sub> exactly; S arrives after P; the vertical S travel
  time matches depth/β. The stochastic seismogram's geometric-mean PGA is within a factor 3 of the
  GMPE median.
- **Client parity.** The browser's wind field matches the server at probe points (typically
  |Δ| ≈ 0.002 m/s — rounding only).

---

## 9. Event development — one realization resolved in time (`/api/develop`)

The engine prices an event through its lifetime-maximum footprint. The development view resolves one
physically consistent realization **in time** and on real terrain, so a user can watch the hazard
arrive and each building's damage accrue. A realization has three parts:

- η drawn from N(0, τ²);
- the residual field W on a regular grid (circulant embedding, same ν and ℓ as the engine);
- damage uniforms from the same two-level copula.

### 9.1 Hurricane

**Wind.** The browser evaluates the *identical* analytic field — the TypeScript port of `wind_uv` and
`track_state` in `frontend/src/dev/physics.ts`. Each draped pixel's 3-s gust is

g = K<sub>terrain</sub>(land/sea) · |**V**| · exp(η + σ<sub>w</sub> W(s))

The server sends 18 probe gusts (open terrain, no residual) so the client can prove parity. Other
diagnostics come from the same field:

- NHC-style **wind radii** (34/50/64 kt, per quadrant), found by an eyewall peak scan followed by
  bisection on the monotone outer profile;
- a running-max footprint;
- **tracer particles** advected through the instantaneous field.

**Storm surge — 2-D depth-averaged shallow-water equations** on NOAA ETOPO1 (2′), in flux form:

∂η/∂t + ∇·**q** = 0
∂**q**/∂t + gH∇η = **τ**<sub>s</sub>/ρ<sub>w</sub> − g n²|**q**|**q**/H<sup>7/3</sup> − f**k**×**q**

The numerical scheme:

- Arakawa C-grid; forward–backward time stepping (Δt = 30 s).
- Face depths use the higher-bed (upwind-safe) rule; a positivity limiter plus wetting/drying (H<sub>dry</sub> = 5 cm) let water flood land up to +15 m.
- Manning friction is semi-implicit: n = 0.025 over water, 0.045 over land.
- Wind stress uses Garratt's C<sub>d</sub> (capped at 2.5·10⁻³) on 10-min wind (×0.93), reduced ×0.6 over land.
- Inverse-barometer forcing, with the open boundary clamped to the inverse barometer.
- Deep water is capped at 200 m for the CFL limit.

Validation — peak coastal water level (m), model vs observed:

| Analog | Model | Observed | Note |
|---|---|---|---|
| Hugo 1989 | 5.9 | 6.0 | Bulls Bay |
| Michael 2018 | 4.9 | 4.7 | Mexico Beach |
| Sandy 2012 | 2.7 | ≈2.9 | Raritan Bay |
| Harvey 2017 | 3.9 | 3.0 | |
| Ian 2022 | 6.5 | 4.5 | |
| Andrew 1992 | 3.7 | 5 | Biscayne Bay (sub-grid funnelling) |
| 1938 New England | 2.6 | ≈4 | Narragansett Bay (sub-grid funnelling) |
| Katrina 2005 | 4.2 | 8.5 | the analog lacks the Cat-5 history over the Gulf |

The median ratio is ≈0.96.

**Building inundation (sub-grid).** A building's water depth is the highest water surface among the
*wet* cells of its 3×3 stencil, minus its own ground elevation. The ground elevation comes from the
full-resolution DEM, floored at 1 m because a building stands on land even where its coarse cell
averages to sea. Surge damage uses USACE-style depth–damage curves (first-floor heights by era and
occupancy; stories/material/mobile-home modifiers). It combines with wind damage as
1 − (1 − d<sub>wind</sub>)(1 − d<sub>surge</sub>), each evolving with its running maximum.

**Rain.** R-CLIPER (Tuleya et al. 2007): the rain-rate profile scales with intensity and the radius
of maximum rain, accumulated every 10 min and framed every 3 h.

### 9.2 Earthquake

**Kinematic finite fault.**

- The rupture plane is discretised into ≤ 900 sub-faults.
- Slip is a von Kármán random field (Hurst 0.75; correlation lengths from Mai & Beroza 2002 scaling) with a lognormal marginal, tapered at the edges and rescaled so that Σ μ A D = M<sub>0</sub> exactly.
- Rupture nucleates at the hypocentre and spreads at V<sub>r</sub> = 0.8 β: t<sub>r</sub> = |ξ − ξ<sub>h</sub>|/V<sub>r</sub>.
- The moment-rate function Σ sub-fault triangles; rise time 2.03·10⁻⁹ M<sub>0</sub>[dyne·cm]<sup>1/3</sup> (Somerville et al. 1999).

**Waves.** First arrivals over the ground grid are t<sub>P,S</sub>(x) = min over sub-faults of
[t<sub>r</sub> + R/α or R/β], with α = 6.2 and β = 3.5 km/s. The strong-shaking window runs from S
arrival to t<sub>S</sub> + source duration + distance-dependent path duration. The displayed shaking is
PGA<sub>realized</sub> × a Saragoni–Hart-type envelope, and the wavefield mesh is its phase-locked,
vertically exaggerated rendering (explicitly *not* a full-waveform solution).

**Seismograms on demand** (`/api/develop/seismogram`) use an EXSIM-style stochastic finite-fault
simulation (Motazedian & Atkinson 2005):

- each sub-fault is a Brune ω² source with a dynamic corner frequency, triggered at its rupture time;
- Boore (2003) windowed Gaussian noise, geometric spreading, and Q(f) with κ₀;
- WNA uses Q = 180 f<sup>0.45</sup>, κ = 0.035, Δσ = 50 bar; CEUS uses Q = 680 f<sup>0.36</sup>, κ = 0.006, Δσ = 140 bar;
- a Vs30 site term.

The output is acceleration, velocity, Fourier spectrum and 5 %-damped PSA (frequency-domain SDOF),
compared against the GMPE median ±1σ. The ratio to the GMPE median is 0.6–1.3 over M 6.5–7.5 and
10–60 km.

### 9.3 Building-level reference

**Footprint matching.** Exposed locations in view are matched to building footprints from OpenStreetMap
vector tiles (OpenMapTiles schema via OpenFreeMap; `render_height` gives the building height):

- A point inside a footprint (ray-casting test, candidates from a ~110 m grid index) matches that
  footprint. Where footprints overlap, the smallest wins.
- Otherwise the nearest footprint edge within 35 m is taken.
- Anything else is flagged as unmatched.

The in-view match rate, the share of locations inside a footprint, and the median snap distance are
reported as a geocoding-quality signal.

**Flood water at a building.** The plane is drawn at z<sub>ground</sub> + d(t), where d(t) is the same
sub-grid inundation depth (§9.1) that drives the building's surge damage. Referencing the plane to the
building's own ground makes it independent of the vertical datum: the photoreal mesh uses ellipsoidal
heights, the DEM uses MSL, and the difference (the geoid undulation, about −25 to −35 m over the US)
cancels.

- **On Google Photorealistic 3D Tiles**, z<sub>ground</sub> is sampled from the mesh: 36 screen points
  on a golden-angle spiral around the building are unprojected onto the mesh, and the 12th percentile
  of their heights is used, so streets and yards win over roofs and facades. The sample is repeated
  as finer tiles stream in.
- **On the OSM rendering**, the terrain elevation is used.

**Photorealistic tiles.** Google's tileset is loaded with deck.gl's `Tile3DLayer` using
`operation: 'terrain+draw'`, so hazard fields and damage-tinted footprints can be draped onto it with
the `TerrainExtension`. By default the tiles come through the server's `/v1/3dtiles/*` pass-through,
so the key stays server-side:

- the path is restricted to `root.json` and `datasets/…`;
- a client-supplied `key` parameter is dropped;
- responses are relayed without caching.

A failed tileset (for example an invalid key) reports Google's error and falls back to the OSM view.

**Synthetic exposure on land.** The demo portfolio generator redraws any location whose ETOPO1
bilinear elevation is below −5 m, falling back to its hub if needed. This happens on a separate random
stream, so every other location is unchanged. It removed 676 of 5,000 demo locations that had landed
in open water. The 2′ DEM cannot resolve the coastal fringe, so the footprint match is the finer
check.

### 9.4 Exposure enrichment and the coarse-DEM bias

`POST /api/portfolios/{id}/enrich` (SDK `enrich()`) returns a **new** portfolio. By default it covers
coastal locations (within 30 km of the coast and below 20 m); `scope="all"` covers every location.
Each location in scope gets two things.

**Building-scale ground elevation (`ground_elev_m`).**

- It is the bilinear value from Terrain Tiles (AWS Open Data, Tilezen "terrarium" encoding; USGS
  3DEP/NED in the US) at z14, about 8–9 m per pixel at Gulf latitudes.
- The PNGs are decoded in-house: numba scanline unfiltering covering all five PNG filter types.
- A cross-check against USGS's 1 m lidar point service at a Cape Coral building: 2.42 m from the tiles
  vs 2.47 m from lidar. The same point service is exposed at `GET /api/geodata/elevation?lidar=true`.
- `ground_elev_m` replaces the surge model's building ground, which was the 2′ ETOPO1 cell value
  floored at 1 m (§9.1).
- A known `first_floor_height_m`, for example from an elevation certificate, likewise replaces the
  era default in the depth–damage function.

**Mapped footprint.**

- OpenStreetMap buildings come from the OpenMapTiles vector tiles, decoded by a minimal in-house
  protobuf reader. Producers merge same-attribute buildings into one MultiPolygon, so candidate
  rings are filtered by their bounding boxes around each location.
- The match rule is the same as the 3-D view's: inside a footprint, else the nearest edge within 35 m.
- Footprint area, and the mapped height when it is informative, give `building_height_m`,
  `floor_area_m2` and a re-derived storey count. OpenMapTiles reports 5 m when OSM has neither a
  height nor a level count, so only taller buildings change storeys.

**QA report.** It gives the match rate, inside vs snapped, the median snap distance, the distribution
of coarse-minus-measured ground, and the count of possible offshore geocodes (on water with no
building within 35 m). Tiles are fetched concurrently into a disk cache, and `CATFORGE_OFFLINE=1`
replays a run from the cache.

**Why it matters (demo book, 1,640 coastal locations).** The ground the surge model assumed without
measurement sits a median **1.9 m above the measured ground**, and **58 %** of locations are more than
1 m lower than assumed; 118 are below mean sea level. The 2′ DEM smooths coastal cities upward, as the
point checks in the table below show.

| Point | USGS 3DEP (tiles) | ETOPO1 2′ | Assumed by the surge model |
|---|---|---|---|
| Downtown Miami | 0.52 m | 6.7 m | 6.7 m |
| New Orleans | 1.66 m | 4.45 m | 4.45 m |
| Cape Coral | 2.42 m | 2.30 m | 2.30 m |

**Loss impact.** Ground-up loss from the time-resolved development (§9.1), same realization (seed 1),
original vs enriched demo book. Buildings flagged as possible offshore geocodes are reported
separately, because measured ground puts them at the waterline: they are a data problem, not a
physics result.

| Event | On-land buildings | GU loss, coarse ground | GU loss, measured ground | Change | Flooded > 0.3 m |
|---|---|---|---|---|---|
| Ian 2022 (SW Florida) | 990 | $115.1m | $140.7m | **+22 %** | 18 → 45 |
| Katrina 2005 (New Orleans / Gulf Coast) | 262 | $12.9m | $16.6m | **+29 %** | 35 → 36 |
| Michael 2018 (Panhandle) | 53 | $10.56m | $10.60m | +0.4 % | 0 → 2 |
| Harvey 2017 (Texas coast) | 100 | $3.01m | $3.10m | +3 % | 1 → 1 |

The flagged offshore geocodes move far more: Katrina's 22 go from $16.2m to $41.5m, and Michael's
and Harvey's 5 each more than double. Two readings follow.

- **Where the surge meets populated low ground** (SW Florida, New Orleans), coarse-cell ground
  understates event surge loss by roughly a fifth to a third.
- **Geocode quality** (points on water) can move an event's loss more than the physics does, so
  enrichment should run before any surge analysis.

The wind-only loss is unchanged in every case: wind damage does not depend on ground elevation.

### 9.5 Rendering (why the picture is quantitatively right)

- **Terrain** comes from the same ETOPO1 grid, served as Terrarium tiles by `/api/terrain/{z}/{x}/{y}.png`. It drives both MapLibre terrain and a hypsometric/bathymetric tint.
- **Fields are draped as canvases whose pixel rows are uniform in Web-Mercator y.** MapLibre warps image sources linearly in Mercator space, so every pixel is evaluated at its true latitude; there is no drift across tall domains.
- **Draped tiles are refreshed deliberately.** With terrain on, MapLibre caches draped tiles by a source fingerprint that ignores canvas content. A feature-state write bumps that fingerprint, so only the tiles under a changed drape re-render.
- **3-D objects** are placed with the same vertical exaggeration as the terrain: building columns at ground elevation × exaggeration, fault sub-faults at depth, and the hypocentre.
- **Live-scene access.** `window.catforge3d` exposes the live scene (map, deck overlay, per-stage timings) for automation.

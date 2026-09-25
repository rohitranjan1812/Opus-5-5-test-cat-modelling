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

### 9.3 Rendering (why the picture is quantitatively right)

- **Terrain** comes from the same ETOPO1 grid, served as Terrarium tiles by `/api/terrain/{z}/{x}/{y}.png`. It drives both MapLibre terrain and a hypsometric/bathymetric tint.
- **Fields are draped as canvases whose pixel rows are uniform in Web-Mercator y.** MapLibre warps image sources linearly in Mercator space, so every pixel is evaluated at its true latitude; there is no drift across tall domains.
- **Draped tiles are refreshed deliberately.** With terrain on, MapLibre caches draped tiles by a source fingerprint that ignores canvas content. A feature-state write bumps that fingerprint, so only the tiles under a changed drape re-render.
- **3-D objects** are placed with the same vertical exaggeration as the terrain: building columns at ground elevation × exaggeration, fault sub-faults at depth, and the hypocentre.
- **Live-scene access.** `window.catforge3d` exposes the live scene (map, deck overlay, per-stage timings) for automation.

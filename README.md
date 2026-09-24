# CatForge — multi-peril catastrophe modelling platform

CatForge is a catastrophe model for property portfolios. It covers hurricane (US Gulf and Atlantic)
and earthquake (US West Coast, Pacific Northwest, Central and South-East US). It is built as three
layers:

1. **A numerically serious engine**: numba kernels, a counter-based RNG, importance-sampled
   catalogs, and an analytic FFT cross-check.
2. **An API-first service**: FastAPI with OpenAPI docs, a Python SDK and a CLI.
3. **An analyst UI**: React, ECharts and Leaflet, with dark and light themes.

The engine produces a full distribution of losses, not just point estimates, and turns it into
insights with numbers attached:

- where the tail comes from
- what reinsurance buys
- what climate regimes, model assumptions and mitigation change
- how much sampling noise remains

> **Calibration disclaimer.** The hazard and vulnerability parameters are an *illustrative*
> calibration to public order-of-magnitude statistics (HURDAT-era landfall rates, NGA-style ground
> motions, HAZUS-style fragilities). The architecture and mathematics are production-grade; the
> numbers are placeholders to be re-fitted before any pricing or regulatory use.

![Overview dashboard](docs/img/overview.png)

## What's inside

| Layer | Highlights |
|---|---|
| **Hazard** | **Hurricane:** landfall-gate stochastic tracks, importance-sampled in both position and intensity; Holland (2008) surface wind field with translational asymmetry; Kaplan–DeMaria inland decay; terrain gust factors. **Earthquake:** fault and area sources with truncated Gutenberg–Richter recurrence; finite ruptures (Wells–Coppersmith / Strasser); BA08-form GMPE with non-linear site terms. **Both:** site hazard curves and return-period maps. |
| **Vulnerability** | Emanuel wind curves and HAZUS-style EQ fragilities. Modifiers for construction, occupancy, year built, storeys, roof shape and shutters. Secondary uncertainty is a zero-one-inflated Beta on damage bins, with within-event hazard uncertainty convolved in analytically. |
| **Financial** | Site deductibles (% of TIV or flat) and limits. Account deductibles, limits, layers and shares. Per-risk XL. Reinsurance programmes with inuring stages: cat XL with reinstatements and AAD/AAL, quota share, stop-loss. Technical pricing by EL + kσ or cost of capital. |
| **Engine** | Numba-parallel kernel driven by a stateless SplitMix64 counter-based RNG, so any draw can be regenerated exactly. Two-level Gaussian copula (event and 0.25° cell factors) plus inter-event hazard residuals. Mixed-Poisson frequency (gamma × ENSO regimes). Three passes: ELT (event × samples), YLT (occurrences), and an exact tail re-simulation for Euler allocation. |
| **Analytics** | OEP/AEP/TVaR with distribution-free order-statistic CIs. An analytic ELT → PGF → exponentially-tilted FFT EP as an independent check. Euler co-TVaR by any dimension. Stand-alone vs diversified segments. Marginal account impact. Automated, quantified insights. |
| **What-ifs** | Exact year likelihood-ratio reweighting for frequency and intensity (climate) changes, with ESS reported. Exact ENSO conditioning. Common-random-number re-runs for vulnerability, correlation and mitigation. Tornado sensitivity. Instant reinsurance evaluation and an efficient-frontier optimiser. Historical analog and custom scenarios with a full loss distribution. |
| **Interfaces** | REST API (`/docs`), Python SDK (`catforge.client`), in-process library, CLI, and a web UI. |

The full mathematics is in **[docs/methodology.md](docs/methodology.md)**.

![Risk lab: climate re-weighting, ENSO conditioning, CRN tornado, mitigation and marginal impact](docs/img/risklab.png)

## Quick start

```bash
# backend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# frontend (optional — the API works without it)
cd frontend && npm ci && npm run build && cd ..

# run: builds a 5,000-location demo book and a 20,000-year analysis in the background (~20 s cold)
catforge serve --port 8000          # UI at http://localhost:8000, OpenAPI at /docs
```

Or with Docker: `docker build -t catforge . && docker run -p 8000:8000 catforge`.

UI development with hot reload: run `catforge serve` and `cd frontend && npm run dev`, then open
http://localhost:5173. The Vite dev server proxies `/api` to port 8000.

### As a library

```python
from catforge import CatModel, AnalysisConfig, generate_portfolio

model = CatModel.default()                          # TC (4,800 events) + EQ (~3,400 events)
pf = generate_portfolio(5000)                       # or Portfolio.from_csv(open("book.csv").read())
res = model.run(pf, AnalysisConfig(n_years=20000, reinsurance={"contracts": [
    {"name": "150m xs 100m", "type": "cat_xl", "attachment": 1e8, "limit": 1.5e8, "reinstatements": 1}]}))

res.summary["rp_table"]      # AEP/OEP/TVaR with confidence intervals, net of reinsurance
res.elt.head()               # event loss table
res.insights                 # ranked, quantified findings
```

### Via the REST API / SDK

```python
from catforge.client import CatForgeClient

cf = CatForgeClient("http://localhost:8000")
pf = cf.create_synthetic_portfolio(n_locations=20000, states=["FL", "TX", "LA"])
an = cf.run_analysis(pf["id"], n_years=50000)
cf.climate(an["id"], tc_frequency=1.1, tc_intensity=1.05)   # exact re-weighting, no re-simulation
cf.optimise_reinsurance(an["id"])                            # efficient frontier
cf.sensitivity(an["id"])                                     # CRN tornado
cf.scenario(pf["id"], analog="cascadia_m90")                 # deterministic event, 1,000 realisations
```

### CLI

```bash
catforge portfolio --n 10000 --states FL TX --out book.csv
catforge run --portfolio book.csv --years 50000 --reinsurance program.json --out results/
catforge hazard-curve --peril EQ --lat 34.05 --lon -118.24
catforge scenario --analog andrew_1992 --portfolio book.csv
```

## Exposure format

CSV files are read with an OED-inspired schema. Common OED aliases are accepted, for example
`Latitude`, `BuildingTIV`, `ConstructionCode` and `AccNumber`.

- **Required:** `lat`, `lon`, `tiv_building`.
- **Optional:**
  - identifiers: `loc_id`, `acc_id`, `state`, `lob`
  - building: `construction`, `occupancy`, `year_built`, `stories`
  - other coverages: `tiv_contents`, `tiv_bi`
  - site terms: `ded_tc`, `ded_eq` (a value ≤ 1 means a fraction of TIV), `loc_limit`
  - site conditions: `terrain`, `vs30`, `roof_shape`, `shutters`
  - account terms: `acc_ded`, `acc_limit`, `acc_attach`, `acc_layer_limit`, `acc_share`

Validation maps aliases, coerces types, applies defaults and reports problems as warnings.

## API map

| Endpoint | Purpose |
|---|---|
| `GET /api/catalogs`, `/catalogs/{peril}/events`, `/events/{id}`, `/events/{id}/footprint`, `/catalogs/{peril}/stats` | Stochastic catalogs, geometry, footprints, climatology |
| `GET /api/hazard/curve`, `/hazard/map` | Site hazard curves, return-period maps |
| `GET /api/vulnerability/curve` | Damage distribution for any building class |
| `POST /api/portfolios/synthetic`, `/portfolios/upload`; `GET /api/portfolios/{id}[/locations]` | Exposure management |
| `POST /api/analyses` (`?wait=true` for synchronous) | Run an analysis as a background job |
| `GET /api/analyses/{id}`, `/ep`, `/analytic`, `/elt`, `/ylt`, `/locations` | Results (JSON or CSV) |
| `GET /api/analyses/{id}/allocation?dimension=…`, `/segments`, `/insights`, `/enso` | Allocation, diversification, insights, climate regimes |
| `POST /api/analyses/{id}/reinsurance`, `/reinsurance/optimize`, `/reinsurance/apply` | Programme evaluation, frontier, persist |
| `POST /api/analyses/{id}/climate`, `/sensitivity`, `/mitigation`, `/marginal` | What-ifs |
| `POST /api/scenarios/run`; `GET /api/scenarios/analogs` | Deterministic events and historical analogs |
| `GET /api/jobs/{id}` | Job progress and results |

## Validation & performance

These numbers come from the 5,000-location demo book ($16.25bn TIV), with 20,000 years, run on
4 vCPUs.

**Convergence checks**

- **AAL:** the three independent estimators agree. YLT gives $23.0m, the ELT gives $23.0m, and the
  analytic FFT gives $23.0m.
- **AEP curve:** simulated and analytic (FFT) agree within about 1% from the 1-in-50 to the 1-in-1000.
- **Euler allocation:** co-TVaR contributions sum exactly to TVaR.
- **Tail re-simulation:** re-running the tail years is bit-identical to the original run.

**Timings**

| Stage | Time |
|---|---|
| Cold start (numba compilation, footprints, ELT, YLT, allocation) | ~20 s |
| Warm re-run | ~6 s |
| Hurricane-only run, 10,000 years | ~4 s |
| Reinsurance evaluation | ~15 ms |
| Sensitivity tornado (13 CRN re-runs and re-weightings) | ~7 s |

**Tests:** `pytest` runs 32 tests covering the hazard physics, the maths identities, the financial
terms, reinsurance path logic, the API and the SDK.

## Project layout

```
catforge/
  hazard/         tropical_cyclone.py · earthquake.py · footprint.py · base.py (catalogs, mixed-Poisson)
  vulnerability/  damage.py (curves, zero-one-inflated Beta bins, convolution, tables)
  exposure/       portfolio.py (schema/validation/IO) · synthetic.py
  financial/      terms.py · reinsurance.py (programmes, pricing, optimiser)
  engine/         kernel.py (numba loss kernel) · ylt.py · model.py (orchestration)
  analytics/      ep.py · analytic.py (FFT) · allocation.py · sensitivity.py · insights.py
  api/            app.py · schemas.py · store.py (jobs, persistence)
  scenario.py · client.py · cli.py · rng.py · geo.py · data/
frontend/         React + TypeScript + ECharts + Leaflet
docs/             methodology.md
tests/
```

## Where to push next — open problems worth solving together

1. **Spatially explicit dependence.** Replace the two-level copula with a Gaussian random field on
   intra-event residuals, using Matérn covariance and range ~ 10–40 km (Jayaram & Baker). The
   challenge is sampling at 10⁵–10⁶ sites per occurrence. Options are an SPDE/GMRF representation
   (sparse precision, Cholesky on a triangulated mesh) or a low-rank Nyström or random-Fourier-feature
   approximation, with an error bound on the tail metrics.
2. **Event-set compression.** Choose a weighted subset of events that preserves the portfolio EP to
   ±x% at chosen return periods. This can be framed as a quadrature or optimal-transport problem on
   the loss distribution, or as loss-based importance sampling with a variance-optimal proposal.
   The payoff is 10–50× faster pricing loops.
3. **Tail-optimal simulation.** Target variance reduction at the 1-in-250 rather than at the AAL:
   cross-entropy importance sampling over (event, η, Z), or multilevel splitting on annual loss.
   Can the same CI width be reached with 10× fewer years?
4. **Clustering beyond regimes.** Model within-season TC clustering with a Hawkes or Cox process
   driven by a latent SST/steering state, then fit it by maximum likelihood to HURDAT. How do
   reinstatement and aggregate-cover prices move relative to the mixed-Poisson baseline?
5. **Capital allocation beyond TVaR.** Use spectral or distortion risk measures (Wang transform,
   proportional hazards) for allocation. Kernel-smoothed co-VaR gradients with CRN would give
   capital-consistent risk-adjusted pricing per account.
6. **Structure optimisation as a real optimiser.** Replace the grid with gradient-based optimisation
   of a multi-layer tower. Loss is piecewise linear in (A, L), so subgradients over the simulated
   occurrences are available, giving a convex-in-practice programme for minimum cost at a target
   net VaR/TVaR.
7. **Calibration pipeline.** Fit the landfall climatology to HURDAT2, fit vulnerability to claims
   (a Bayesian hierarchical model over construction × code era), and run posterior-predictive checks
   against historical industry losses. Epistemic uncertainty then propagates as an outer loop,
   giving the "EP of EPs".

## License

MIT

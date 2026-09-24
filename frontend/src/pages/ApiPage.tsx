import { Card } from '../components/ui'
import { useApp } from '../state'

const PY = `from catforge.client import CatForgeClient

cf = CatForgeClient("http://localhost:8000")
pf = cf.create_synthetic_portfolio(n_locations=20000, states=["FL", "TX", "LA"])
an = cf.run_analysis(pf["id"], n_years=50000, reinsurance={"contracts": [
    {"name": "150m xs 100m", "type": "cat_xl", "attachment": 1e8, "limit": 1.5e8, "reinstatements": 1}]})

print(an["aal"], [r for r in an["rp_table"] if r["rp"] in (100, 250)])
frontier = cf.optimise_reinsurance(an["id"], reinstatements=1)
tornado  = cf.sensitivity(an["id"])                       # common-random-number re-runs
climate  = cf.climate(an["id"], tc_frequency=1.1, tc_intensity=1.05)  # exact year re-weighting
andrew   = cf.scenario(pf["id"], analog="andrew_1992")`

const LIB = `from catforge import CatModel, AnalysisConfig, generate_portfolio

model = CatModel.default()                    # TC + EQ stochastic catalogs
pf = generate_portfolio(5000)                 # or Portfolio.from_csv(open("book.csv").read())
res = model.run(pf, AnalysisConfig(n_years=20000))
res.summary["rp_table"]; res.elt.head(); res.insights`

const CURL = `# create a portfolio from CSV (OED aliases accepted)
curl -F file=@book.csv -F name=book http://localhost:8000/api/portfolios/upload

# run an analysis synchronously
curl -X POST 'http://localhost:8000/api/analyses?wait=true' -H 'Content-Type: application/json' \\
  -d '{"portfolio_id": "pf_…", "config": {"n_years": 20000, "perils": ["TC","EQ"]}}'

# results
curl http://localhost:8000/api/analyses/an_…/ep?basis=gross
curl 'http://localhost:8000/api/analyses/an_…/allocation?dimension=state'
curl http://localhost:8000/api/analyses/an_…/elt?format=csv > elt.csv`

const CLI = `catforge serve --port 8000
catforge portfolio --n 10000 --states FL TX --out book.csv
catforge run --portfolio book.csv --years 50000 --reinsurance program.json --out results/
catforge hazard-curve --peril TC --lat 25.77 --lon -80.19
catforge scenario --analog cascadia_m90 --portfolio book.csv`

const ENDPOINTS: [string, string][] = [
  ['GET /api/catalogs · /catalogs/{peril}/events · /events/{id}/footprint', 'Stochastic catalogs, event geometry and footprints'],
  ['GET /api/hazard/curve · /hazard/map', 'Site hazard curves and return-period hazard maps'],
  ['GET /api/vulnerability/curve', 'Damage distribution for any building class'],
  ['POST /api/portfolios/synthetic · /portfolios/upload', 'Create exposure'],
  ['POST /api/analyses', 'Run an analysis (background job; ?wait=true for sync)'],
  ['GET /api/analyses/{id} · /ep · /elt · /ylt · /locations', 'Summary, EP curves, event & year loss tables, location results'],
  ['GET /api/analyses/{id}/allocation · /segments · /insights', 'Euler allocation, diversification, automated insights'],
  ['POST /api/analyses/{id}/reinsurance · /reinsurance/optimize · /reinsurance/apply', 'Programme evaluation, efficient frontier, persist'],
  ['POST /api/analyses/{id}/climate · /sensitivity · /mitigation · /marginal', 'What-ifs (re-weighting, CRN re-runs)'],
  ['GET /api/analyses/{id}/enso', 'Climate-regime conditional risk'],
  ['POST /api/scenarios/run', 'Historical analogs / custom deterministic events'],
  ['GET /api/jobs/{id}', 'Background job status and results'],
]

export default function ApiPage() {
  const { meta } = useApp()
  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>API &amp; SDK</h1>
          <p>Everything in the UI is an API call. Interactive OpenAPI documentation lives at <a href="/docs" target="_blank" rel="noreferrer">/docs</a>
            {' '}(schema at <a href="/openapi.json" target="_blank" rel="noreferrer">/openapi.json</a>) — generate clients in any language from it.</p>
        </div>
        <span className="pill">v{meta?.version}</span>
      </div>
      <div className="grid g2">
        <Card title="Python SDK (REST)"><pre>{PY}</pre></Card>
        <Card title="Python library (in-process)"><pre>{LIB}</pre></Card>
        <Card title="curl"><pre>{CURL}</pre></Card>
        <Card title="CLI"><pre>{CLI}</pre></Card>
      </div>
      <Card title="Endpoint map">
        <div className="table-wrap" style={{ maxHeight: 'none' }}>
          <table className="data">
            <thead><tr><th>Endpoint</th><th>Purpose</th></tr></thead>
            <tbody>{ENDPOINTS.map(([e, d]) => <tr key={e}><td><code>{e}</code></td><td>{d}</td></tr>)}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

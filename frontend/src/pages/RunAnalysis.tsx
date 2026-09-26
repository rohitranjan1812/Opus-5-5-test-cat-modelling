import { useState } from 'react'
import { api } from '../api'
import type { AnalysisBrief, Job, Program } from '../api'
import ProgramEditor, { defaultProgram } from '../components/ProgramEditor'
import { Card, DataTable, JobStatus, NumberField } from '../components/ui'
import { money, num, when } from '../format'
import { useApp, useFetch } from '../state'

export default function RunAnalysis() {
  const { portfolios, portfolioId, setPortfolioId, refresh, setAnalysisId, go } = useApp()
  const [name, setName] = useState('Analysis')
  const [perils, setPerils] = useState<string[]>(['TC', 'EQ'])
  const [years, setYears] = useState(20000)
  const [samples, setSamples] = useState(24)
  const [seed, setSeed] = useState(20240601)
  const [allocRp, setAllocRp] = useState(250)
  const [groupBy, setGroupBy] = useState('state')
  const [dmgTC, setDmgTC] = useState(1)
  const [dmgEQ, setDmgEQ] = useState(1)
  const [rho, setRho] = useState(1)
  const [sigB, setSigB] = useState(1)
  const [rateTC, setRateTC] = useState(1)
  const [rateEQ, setRateEQ] = useState(1)
  const [tcInt, setTcInt] = useState(1)
  const [enso, setEnso] = useState(true)
  const [surge, setSurge] = useState(true)
  const [hfBudget, setHfBudget] = useState(24)
  const [hfTol, setHfTol] = useState(0.5)
  const [dependence, setDependence] = useState<'grf' | 'copula'>('grf')
  const [rangeTC, setRangeTC] = useState(60)
  const [rangeEQ, setRangeEQ] = useState(30)
  const [vecchiaM, setVecchiaM] = useState(30)
  const [dispersion, setDispersion] = useState(20)
  const [usePR, setUsePR] = useState(false)
  const [pr, setPr] = useState({ retention: 5e6, limit: 2e7 })
  const [useRe, setUseRe] = useState(true)
  const [program, setProgram] = useState<Program>(defaultProgram)
  const [job, setJob] = useState<Job | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const jobs = useFetch(() => api.get<Job[]>('/jobs'), [job?.status])
  const { analyses } = useApp()

  async function submit() {
    if (!portfolioId) return
    setErr(null)
    const config = {
      name, perils, n_years: years, elt_samples: samples, seed, allocation_rp: allocRp, group_by: groupBy,
      damage_scale: { TC: dmgTC, EQ: dmgEQ }, rho_scale: rho, sigma_between_scale: sigB,
      rate_multiplier: { TC: rateTC, EQ: rateEQ }, tc_intensity_scale: tcInt,
      frequency: { TC: { dispersion_r: dispersion > 0 ? dispersion : null, regimes: enso ? [
        { name: 'La Niña', prob: 0.25, multiplier: 1.3 }, { name: 'Neutral', prob: 0.5, multiplier: 1.0 },
        { name: 'El Niño', prob: 0.25, multiplier: 0.65 }] : [] } },
      dependence, vecchia_m: vecchiaM,
      grf: dependence === 'grf' ? { TC: { range_km: rangeTC }, EQ: { range_km: rangeEQ } } : {},
      tc_surge: surge,
      surge_fidelity: { budget: surge ? hfBudget : 0, tol: hfTol / 100, rp: allocRp },
      per_risk: usePR ? pr : null,
      reinsurance: useRe && program.contracts.length ? program : null,
    }
    try {
      const j = await api.post<Job>('/analyses', { portfolio_id: portfolioId, config })
      setJob(j)
      const done = await api.waitJob(j, setJob)
      await refresh()
      if (done.result_id) {
        setAnalysisId(done.result_id)
        go('overview')
      }
    } catch (e) {
      setErr(String(e))
    }
  }

  const running = job && (job.status === 'running' || job.status === 'queued')
  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>Run analysis</h1>
          <p>Configure the simulation, model assumptions and financial structure. The engine computes footprints (cached per
            portfolio), an event loss table (event × samples), and a year-loss table with mixed-Poisson frequency, then
            applies reinsurance and re-simulates tail years exactly for Euler allocation.</p>
        </div>
      </div>
      <div className="grid g2">
        <Card title="Simulation">
          <div className="form-grid">
            <label className="field"><span>Portfolio</span>
              <select value={portfolioId ?? ''} onChange={(e) => setPortfolioId(e.target.value)}>
                {portfolios.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select></label>
            <label className="field"><span>Name</span><input type="text" value={name} onChange={(e) => setName(e.target.value)} /></label>
            <NumberField label="Simulated years" value={years} step={5000} min={200} onChange={setYears} />
            <NumberField label="ELT samples / event" value={samples} min={2} max={512} onChange={setSamples} />
            <NumberField label="Seed" value={seed} onChange={setSeed} />
            <NumberField label="Allocation return period" value={allocRp} min={2} onChange={setAllocRp} />
            <label className="field"><span>Segment by</span>
              <select value={groupBy} onChange={(e) => setGroupBy(e.target.value)}>
                {['state', 'construction', 'occupancy', 'lob', 'terrain'].map((g) => <option key={g}>{g}</option>)}
              </select></label>
          </div>
          <div className="row mt">
            {['TC', 'EQ'].map((p) => (
              <label key={p} className="check"><input type="checkbox" checked={perils.includes(p)}
                onChange={(e) => setPerils(e.target.checked ? [...perils, p] : perils.filter((x) => x !== p))} />
                {p === 'TC' ? 'Hurricane' : 'Earthquake'}</label>
            ))}
          </div>
        </Card>
        <Card title="Model assumptions" desc="Multipliers on the calibrated model — 1.0 = base view">
          <div className="form-grid">
            <NumberField label="TC damage ×" value={dmgTC} step={0.05} onChange={setDmgTC} />
            <NumberField label="EQ damage ×" value={dmgEQ} step={0.05} onChange={setDmgEQ} />
            <NumberField label="TC frequency ×" value={rateTC} step={0.05} onChange={setRateTC} />
            <NumberField label="EQ frequency ×" value={rateEQ} step={0.05} onChange={setRateEQ} />
            <NumberField label="TC intensity scale" value={tcInt} step={0.01} min={0.8} max={1.2} onChange={setTcInt} />
            <NumberField label="Correlation ρ ×" value={rho} step={0.1} min={0} max={3} onChange={setRho} />
            <NumberField label="Inter-event σ ×" value={sigB} step={0.1} min={0} max={3} onChange={setSigB} />
            <NumberField label="TC gamma dispersion r (0 = off)" value={dispersion} min={0} onChange={setDispersion} />
          </div>
          <div className="row mt">
            <label className="check"><input type="checkbox" checked={enso} onChange={(e) => setEnso(e.target.checked)} />ENSO regimes for hurricane frequency</label>
          </div>
          <div className="row mt" style={{ gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <label className="check"><input type="checkbox" checked={surge} onChange={(e) => setSurge(e.target.checked)} />Storm surge (multi-fidelity 2-D)</label>
            <NumberField label="Full-fidelity tail events (budget)" value={hfBudget} min={0} max={500} onChange={setHfBudget} />
            <NumberField label="Tail surge-error tolerance (% of TVaR)" value={hfTol} step={0.25} min={0} max={50} onChange={setHfTol} />
          </div>
          <p className="muted small">Hurricanes whose surge uncertainty moves the 1-in-{allocRp} TVaR most are re-run with the full 2′
            model (~6 s each the first time, then cached), until the remaining error is within tolerance or the budget is spent.</p>
          <div className="row mt" style={{ gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <label className="field"><span>Spatial dependence</span>
              <select value={dependence} onChange={(e) => setDependence(e.target.value as 'grf' | 'copula')}>
                <option value="grf">Matérn random field (Vecchia NNGP)</option>
                <option value="copula">Two-level Gaussian copula (legacy)</option>
              </select></label>
            {dependence === 'grf' && <>
              <NumberField label="TC range" suffix="km" value={rangeTC} step={10} min={5} max={500} onChange={setRangeTC} />
              <NumberField label="EQ range" suffix="km" value={rangeEQ} step={5} min={5} max={300} onChange={setRangeEQ} />
              <NumberField label="Vecchia m" value={vecchiaM} step={2} min={4} max={64} onChange={setVecchiaM} />
            </>}
          </div>
          <div className="small muted mt">{dependence === 'grf'
            ? 'Intra-event hazard residuals are a Matérn Gaussian random field (TC ν = 1.5, EQ ν = 0.5), sampled exactly per event with a nearest-neighbour Gaussian process; “Correlation ρ ×” scales the ranges.'
            : 'Legacy mode: hazard and damage residuals coupled through an event factor and 0.25° cell factors.'}</div>
        </Card>
      </div>
      <Card title="Reinsurance programme" tools={<label className="check"><input type="checkbox" checked={useRe} onChange={(e) => setUseRe(e.target.checked)} />include</label>}
        desc="Programmes can also be evaluated and optimised instantly after the run on the Reinsurance page">
        {useRe && <ProgramEditor program={program} onChange={setProgram} />}
        <div className="row mt">
          <label className="check"><input type="checkbox" checked={usePR} onChange={(e) => setUsePR(e.target.checked)} />Per-risk XL (per account, inures to cat)</label>
          {usePR && <>
            <NumberField label="Retention ($m)" value={pr.retention / 1e6} onChange={(v) => setPr({ ...pr, retention: v * 1e6 })} />
            <NumberField label="Limit ($m)" value={pr.limit / 1e6} onChange={(v) => setPr({ ...pr, limit: v * 1e6 })} />
          </>}
        </div>
      </Card>
      <Card>
        <div className="row">
          <button className="btn primary" disabled={!portfolioId || !perils.length || !!running} onClick={submit}>▶ Run analysis</button>
          <div style={{ flex: 1, minWidth: 240 }}><JobStatus job={job} /></div>
        </div>
        {err && <div className="error small mt">{err}</div>}
      </Card>
      <div className="grid g2">
        <Card title="Analyses">
          <DataTable rows={analyses} maxHeight={300} onRowClick={(a: AnalysisBrief) => { setAnalysisId(a.id); go('overview') }} columns={[
            { key: 'name', label: 'Name' }, { key: 'portfolio_name', label: 'Portfolio' },
            { key: 'n_years', label: 'Years', align: 'r', render: (a) => num(a.n_years) },
            { key: 'aal', label: 'AAL', align: 'r', value: (a) => a.aal.gross, render: (a) => money(a.aal.gross) },
            { key: 'aep_250', label: '1-in-250', align: 'r', render: (a) => money(a.aep_250) },
            { key: 'created', label: 'Created', render: (a) => when(a.created) },
          ]} />
        </Card>
        <Card title="Recent jobs">
          <DataTable rows={jobs.data ?? []} maxHeight={300} columns={[
            { key: 'kind', label: 'Kind' }, { key: 'status', label: 'Status' },
            { key: 'progress', label: 'Progress', align: 'r', render: (j) => `${Math.round(j.progress * 100)}%` },
            { key: 'elapsed_s', label: 'Time', align: 'r', render: (j) => `${j.elapsed_s.toFixed(1)}s` },
            { key: 'message', label: 'Message', render: (j) => j.error ?? j.message },
          ]} />
        </Card>
      </div>
    </div>
  )
}

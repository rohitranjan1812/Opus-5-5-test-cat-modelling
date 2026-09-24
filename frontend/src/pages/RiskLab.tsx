import { useEffect, useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { api } from '../api'
import type { AnalysisSummary, Job } from '../api'
import Chart from '../components/Chart'
import { Card, DataTable, Empty, JobStatus, Tile } from '../components/ui'
import { money, moneyAxis, num, pct } from '../format'
import { useApp, useFetch } from '../state'
import { TOKENS, barItem, catAxis, lineSeries, logAxis, valueAxis } from '../theme'

interface M4 { aal: number; aep_100: number | null; aep_250: number | null; tvar_250: number | null }
interface Climate { base: M4; scenario: M4; aep: { rp: number; loss: number }[]; effective_sample_size: number; n_years: number }
interface Enso { regimes: { regime: string; prob: number; multiplier: number; n_years: number; aal: number; tc_aal: number; aep_100: number | null; aep_250: number | null; p_loss_gt_aep10: number }[]; unconditional_aal: number }
interface Tornado { base: M4; rows: { parameter: string; low_label: string; high_label: string; low: M4; high: M4 }[] }
interface Mitig { label: string; n_locations: number; tiv_affected: number; base: M4; mitigated: M4; aal_saving: number; aal_saving_affected_pct: number; aal_saving_per_location: number; top_locations: { loc_id: string; saving: number }[]; message?: string }
interface Marginal { acc_ids: string[]; n_locations: number; tiv: number; base: M4; without: M4; marginal: M4; aal_standalone_in_portfolio: number; euler_tvar_contribution: number }

function delta(a: number | null, b: number | null) {
  if (a === null || b === null || !a) return ''
  const d = b / a - 1
  return `${d >= 0 ? '+' : ''}${(100 * d).toFixed(1)}% vs base`
}

export default function RiskLab() {
  const { analysisId, mode, meta } = useApp()
  const t = TOKENS[mode]
  const sum = useFetch(analysisId ? () => api.get<AnalysisSummary>(`/analyses/${analysisId}`) : null, [analysisId])
  const [cl, setCl] = useState({ tc_frequency: 1.1, tc_intensity: 1.04, eq_frequency: 1.0 })
  const [climate, setClimate] = useState<Climate | null>(null)
  const enso = useFetch(analysisId ? () => api.get<Enso>(`/analyses/${analysisId}/enso`) : null, [analysisId])
  const [tJob, setTJob] = useState<Job | null>(null)
  const [tornado, setTornado] = useState<Tornado | null>(null)
  const [preset, setPreset] = useState('shutters_fl_pre2002')
  const [mJob, setMJob] = useState<Job | null>(null)
  const [mit, setMit] = useState<Mitig | null>(null)
  const accs = useFetch(analysisId ? () => api.get<{ rows: { key: string; cotvar: number; aal: number; tiv: number }[] }>(`/analyses/${analysisId}/allocation`, { dimension: 'acc_id', top: 25 }) : null, [analysisId])
  const [acc, setAcc] = useState<string | null>(null)
  const [gJob, setGJob] = useState<Job | null>(null)
  const [marg, setMarg] = useState<Marginal | null>(null)

  useEffect(() => { setTornado(null); setMit(null); setMarg(null) }, [analysisId])
  useEffect(() => {
    if (!analysisId) return
    const h = setTimeout(() => { api.post<Climate>(`/analyses/${analysisId}/climate`, cl).then(setClimate).catch(() => undefined) }, 200)
    return () => clearTimeout(h)
  }, [cl, analysisId])

  const hasTC = ((sum.data?.config.perils as string[]) ?? []).includes('TC')

  const climOpt = useMemo((): EChartsOption | null => {
    if (!climate || !sum.data) return null
    const base = sum.data.rp_table.map((r) => [r.rp, r.gross_aep])
    const sc = climate.aep.map((r) => [r.rp, r.loss])
    return {
      legend: { data: ['Base view', 'Climate-conditioned'] },
      tooltip: { trigger: 'axis', valueFormatter: (v) => money(v as number) },
      grid: { left: 12, right: 20, top: 36, bottom: 30, containLabel: true },
      xAxis: logAxis(mode, { name: 'Return period (years)', min: 2, max: 1000 }), yAxis: valueAxis(mode),
      series: [lineSeries('Base view', t.series[0], base, { showSymbol: true }), lineSeries('Climate-conditioned', t.series[1], sc, { showSymbol: true })],
    }
  }, [climate, sum.data, mode, t])

  const ensoOpt = useMemo((): EChartsOption | null => {
    const r = enso.data?.regimes
    if (!r?.length) return null
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => money(v as number) },
      grid: { left: 8, right: 20, top: 16, bottom: 8, containLabel: true },
      xAxis: catAxis(mode, r.map((x) => `${x.regime} (${pct(x.prob, 0)})`)),
      yAxis: valueAxis(mode),
      series: [{ name: 'Hurricane AAL', type: 'bar', data: r.map((x) => x.tc_aal), barMaxWidth: 24, itemStyle: barItem(t.series[0]),
        label: { show: true, position: 'top', color: t.text2, formatter: (p: { value: unknown }) => moneyAxis(p.value as number) } }],
    }
  }, [enso.data, mode, t])

  const tornadoOpt = useMemo((): EChartsOption | null => {
    if (!tornado) return null
    const base = tornado.base.aep_250 ?? 0
    const rows = [...tornado.rows].reverse()
    return {
      legend: { data: ['Low setting', 'High setting'] },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => `${(v as number) >= 0 ? '+' : ''}${money(v as number)}` },
      grid: { left: 16, right: 24, top: 36, bottom: 8, containLabel: true },
      xAxis: { type: 'value', axisLabel: { color: t.text3, formatter: (v: number) => `${v >= 0 ? '+' : ''}${moneyAxis(v)}` }, splitLine: { lineStyle: { color: t.grid } } },
      yAxis: catAxis(mode, rows.map((r) => r.parameter.replace(/ \(.*\)$/, '')), { axisLabel: { color: t.text2 } }),
      series: [
        { name: 'Low setting', type: 'bar', stack: 's', data: rows.map((r) => (r.low.aep_250 ?? base) - base), barMaxWidth: 16, itemStyle: { color: t.series[0], borderRadius: 4 } },
        { name: 'High setting', type: 'bar', stack: 's2', barGap: '-100%', data: rows.map((r) => (r.high.aep_250 ?? base) - base), barMaxWidth: 16, itemStyle: { color: t.series[1], borderRadius: 4 } },
      ],
    }
  }, [tornado, mode, t])

  async function runTornado() {
    if (!analysisId) return
    const j = await api.post<Job>(`/analyses/${analysisId}/sensitivity`)
    setTJob(j)
    const done = await api.waitJob<Tornado>(j, setTJob)
    setTornado(done.result)
  }
  async function runMitigation() {
    if (!analysisId) return
    const j = await api.post<Job>(`/analyses/${analysisId}/mitigation`, { preset })
    setMJob(j)
    const done = await api.waitJob<Mitig>(j, setMJob)
    setMit(done.result)
  }
  async function runMarginal(a: string) {
    if (!analysisId) return
    setAcc(a)
    setMarg(null)
    const j = await api.post<Job>(`/analyses/${analysisId}/marginal`, { acc_ids: [a] })
    setGJob(j)
    const done = await api.waitJob<Marginal>(j, setGJob)
    setMarg(done.result)
  }

  if (!analysisId) return <Card><Empty>No analysis selected.</Empty></Card>
  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>Risk lab</h1>
          <p>What-ifs that stay statistically exact: rate changes (climate, frequency) are evaluated by <b>importance
            re-weighting of simulated years</b> with exact likelihood ratios (no re-simulation), while vulnerability,
            correlation and exposure changes re-run the kernel with <b>common random numbers</b>, so deltas are not
            swamped by Monte Carlo noise.</p>
        </div>
      </div>

      <div className="grid g2">
        <Card title="Climate-conditioned view" desc={climate ? `Year re-weighting · effective sample size ${num(climate.effective_sample_size)} of ${num(climate.n_years)} years` : ''}>
          <div className="grid g3">
            {hasTC && <label className="field"><span>Hurricane frequency × {cl.tc_frequency.toFixed(2)}</span>
              <input type="range" min={0.7} max={1.5} step={0.01} value={cl.tc_frequency} onChange={(e) => setCl({ ...cl, tc_frequency: +e.target.value })} /></label>}
            {hasTC && <label className="field"><span>Landfall intensity scale × {cl.tc_intensity.toFixed(2)}</span>
              <input type="range" min={0.9} max={1.1} step={0.005} value={cl.tc_intensity} onChange={(e) => setCl({ ...cl, tc_intensity: +e.target.value })} /></label>}
            <label className="field"><span>Earthquake frequency × {cl.eq_frequency.toFixed(2)}</span>
              <input type="range" min={0.7} max={1.5} step={0.01} value={cl.eq_frequency} onChange={(e) => setCl({ ...cl, eq_frequency: +e.target.value })} /></label>
          </div>
          {climate && (
            <div className="grid g3 mt">
              <Tile label="AAL" value={money(climate.scenario.aal)} foot={delta(climate.base.aal, climate.scenario.aal)} />
              <Tile label="1-in-100 AEP" value={money(climate.scenario.aep_100)} foot={delta(climate.base.aep_100, climate.scenario.aep_100)} />
              <Tile label="1-in-250 AEP" value={money(climate.scenario.aep_250)} foot={delta(climate.base.aep_250, climate.scenario.aep_250)} />
            </div>
          )}
          <div className="mt">{climOpt ? <Chart option={climOpt} height={260} ariaLabel="Climate EP" /> : <Empty>…</Empty>}</div>
        </Card>
        <Card title="ENSO-conditional hurricane risk" desc="Exact conditioning on the simulated climate regime of each year (mixed-Poisson frequency)">
          {ensoOpt ? <Chart option={ensoOpt} height={260} ariaLabel="ENSO" /> : <Empty>{hasTC ? 'Loading…' : 'Hurricane peril not in this analysis.'}</Empty>}
          {enso.data?.regimes.length ? <div className="mt"><DataTable rows={enso.data.regimes} columns={[
            { key: 'regime', label: 'Regime' }, { key: 'multiplier', label: 'Rate ×', align: 'r', render: (r) => r.multiplier.toFixed(2) },
            { key: 'n_years', label: 'Years', align: 'r', render: (r) => num(r.n_years) },
            { key: 'aal', label: 'Total AAL', align: 'r', render: (r) => money(r.aal) },
            { key: 'aep_100', label: '1-in-100', align: 'r', render: (r) => money(r.aep_100) },
            { key: 'p_loss_gt_aep10', label: 'P(> 1-in-10 loss)', align: 'r', render: (r) => pct(r.p_loss_gt_aep10) },
          ]} /></div> : null}
        </Card>
      </div>

      <Card title="Sensitivity tornado — change in 1-in-250 AEP" desc="One-at-a-time ±20% perturbations of key assumptions"
        tools={<button className="btn" onClick={runTornado} disabled={tJob?.status === 'running'}>Run sensitivity</button>}>
        {tJob && tJob.status !== 'done' && <JobStatus job={tJob} />}
        {tornadoOpt ? <Chart option={tornadoOpt} height={340} ariaLabel="Tornado" /> : !tJob && <Empty>Run to quantify which assumptions move the tail most.</Empty>}
        {tornado && <div className="small muted">Base 1-in-250 AEP {money(tornado.base.aep_250)} · AAL {money(tornado.base.aal)}</div>}
      </Card>

      <div className="grid g2">
        <Card title="Mitigation what-if" desc="Modify building attributes → rebuild vulnerability → re-run with common random numbers"
          tools={<>
            <select value={preset} onChange={(e) => setPreset(e.target.value)} aria-label="Mitigation preset" style={{ maxWidth: 360 }}>
              {Object.entries(meta?.mitigation_presets ?? {}).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            <button className="btn" onClick={runMitigation}>Evaluate</button>
          </>}>
          {mJob && mJob.status !== 'done' && <JobStatus job={mJob} />}
          {mit ? (mit.n_locations === 0 ? <Empty>{mit.message}</Empty> : <>
            <div className="grid g3">
              <Tile label="Locations upgraded" value={num(mit.n_locations)} foot={`TIV ${money(mit.tiv_affected)}`} />
              <Tile label="AAL saving" value={money(mit.aal_saving)} foot={`${pct(mit.aal_saving_affected_pct)} of their AAL · ${money(mit.aal_saving_per_location)} / site / yr`} />
              <Tile label="1-in-250 AEP" value={money(mit.mitigated.aep_250)} foot={delta(mit.base.aep_250, mit.mitigated.aep_250)} />
            </div>
            <div className="mt"><DataTable rows={mit.top_locations} maxHeight={220} columns={[
              { key: 'loc_id', label: 'Location' }, { key: 'saving', label: 'AAL saving / yr', align: 'r', render: (r) => money(r.saving) }]} /></div>
          </>) : !mJob && <Empty>Choose a mitigation programme and evaluate its benefit.</Empty>}
        </Card>
        <Card title="Marginal account impact" desc="Remove one account with common random numbers — diversification-aware capital cost vs Euler contribution">
          <div className="grid g2">
            <div>{accs.data ? <DataTable rows={accs.data.rows} maxHeight={330} onRowClick={(r) => runMarginal(r.key)} selected={(r) => r.key === acc} columns={[
              { key: 'key', label: 'Account' }, { key: 'tiv', label: 'TIV', align: 'r', render: (r) => money(r.tiv) },
              { key: 'cotvar', label: 'co-TVaR', align: 'r', render: (r) => money(r.cotvar) }]} /> : <Empty>Loading…</Empty>}</div>
            <div className="col">
              {gJob && gJob.status !== 'done' && <JobStatus job={gJob} />}
              {marg ? <>
                <Tile label={`Marginal 1-in-250 AEP of ${marg.acc_ids[0]}`} value={money(marg.marginal.aep_250)} foot={`TVaR 250: ${money(marg.marginal.tvar_250)}`} />
                <Tile label="Euler co-TVaR contribution" value={money(marg.euler_tvar_contribution)} />
                <Tile label="Marginal AAL" value={money(marg.marginal.aal)} foot={`${num(marg.n_locations)} locations · TIV ${money(marg.tiv)}`} />
              </> : !gJob && <Empty>Click an account.</Empty>}
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}

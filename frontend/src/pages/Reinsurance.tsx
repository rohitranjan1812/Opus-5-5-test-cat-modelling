import { useEffect, useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { api } from '../api'
import type { AnalysisSummary, Curve, EpRow, Program, ReinsMetrics } from '../api'
import Chart from '../components/Chart'
import { epOption } from '../components/charts'
import ProgramEditor, { DEFAULT_LAYER, defaultProgram } from '../components/ProgramEditor'
import { Card, DataTable, Empty, NumberField, Tile } from '../components/ui'
import { money, moneyAxis, pct } from '../format'
import { useApp, useFetch } from '../state'
import { SLOT, TOKENS } from '../theme'

interface EvalResult { metrics: ReinsMetrics; net_aep: EpRow[]; net_aep_curve: Curve; gross_aep_curve: Curve; gross_aep: EpRow[] }
interface FrontierPoint { attach_rp: number; exhaust_rp: number; attachment: number; limit: number; premium: number; expected_loss: number; net_cost: number; net_var_99_5: number; capital_relief: number; rate_on_line: number; efficient: boolean }

export default function Reinsurance() {
  const { analysisId, mode, refresh } = useApp()
  const t = TOKENS[mode]
  const sum = useFetch(analysisId ? () => api.get<AnalysisSummary>(`/analyses/${analysisId}`) : null, [analysisId])
  const [program, setProgram] = useState<Program>(defaultProgram)
  const [res, setRes] = useState<EvalResult | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [opt, setOpt] = useState({ reinstatements: 1, pricing_load: 0.3 })
  const [frontier, setFrontier] = useState<{ points: FrontierPoint[]; gross_var_99_5: number } | null>(null)
  const [saved, setSaved] = useState<string | null>(null)

  useEffect(() => {
    const p = sum.data?.reinsurance?.program
    if (p) setProgram({ ...p, contracts: p.contracts.map((c) => ({ ...c })) })
  }, [sum.data])

  useEffect(() => {
    if (!analysisId) return
    const h = setTimeout(() => {
      api.post<EvalResult>(`/analyses/${analysisId}/reinsurance`, { program }).then((r) => { setRes(r); setErr(null) }).catch((e) => setErr(String(e)))
    }, 250)
    return () => clearTimeout(h)
  }, [program, analysisId])

  const epOpt = useMemo(() => res ? epOption(mode, [
    { name: 'Gross AEP', curve: res.gross_aep_curve, slot: SLOT.gross },
    { name: 'Net AEP', curve: res.net_aep_curve, slot: SLOT.net },
  ], { minRp: 2, maxRp: 1000 }) : null, [res, mode])

  const frontierOpt = useMemo((): EChartsOption | null => {
    if (!frontier) return null
    const eff = frontier.points.filter((p) => p.efficient).sort((a, b) => a.net_cost - b.net_cost)
    const dom = frontier.points.filter((p) => !p.efficient)
    const fmt = (p: FrontierPoint) => `${money(p.limit)} xs ${money(p.attachment)} (1-in-${p.attach_rp} → 1-in-${p.exhaust_rp})<br/>net cost ${money(p.net_cost)} · net 1-in-200 ${money(p.net_var_99_5)}<br/>ROL ${pct(p.rate_on_line)} · relief ${money(p.capital_relief)}`
    return {
      legend: { data: ['Efficient frontier', 'Dominated structures'] },
      tooltip: { trigger: 'item', formatter: (q: unknown) => fmt((q as { data: { p: FrontierPoint } }).data.p) },
      grid: { left: 12, right: 24, top: 36, bottom: 30, containLabel: true },
      xAxis: { type: 'value', name: 'Expected net cost of reinsurance (premium − expected recovery)', nameLocation: 'middle', nameGap: 26,
        nameTextStyle: { color: t.text3 }, axisLabel: { color: t.text3, formatter: (v: number) => moneyAxis(v) }, splitLine: { lineStyle: { color: t.grid } } },
      yAxis: { type: 'value', axisLabel: { color: t.text3, formatter: (v: number) => moneyAxis(v) },
        splitLine: { lineStyle: { color: t.grid } }, scale: true },
      series: [
        { name: 'Dominated structures', type: 'scatter', symbolSize: 8, data: dom.map((p) => ({ value: [p.net_cost, p.net_var_99_5], p })),
          itemStyle: { color: t.text3, opacity: 0.6, borderColor: t.surface, borderWidth: 2 } },
        { name: 'Efficient frontier', type: 'line', data: eff.map((p) => ({ value: [p.net_cost, p.net_var_99_5], p })), symbolSize: 9, showSymbol: true,
          lineStyle: { width: 2, color: t.series[0] }, itemStyle: { color: t.series[0], borderColor: t.surface, borderWidth: 2 } },
      ],
    }
  }, [frontier, t])

  async function optimise() {
    if (!analysisId) return
    setFrontier(await api.post(`/analyses/${analysisId}/reinsurance/optimize`, opt))
  }

  async function apply() {
    if (!analysisId) return
    await api.post(`/analyses/${analysisId}/reinsurance/apply`, { program })
    setSaved('Programme saved to the analysis — net results and insights updated.')
    refresh()
    sum.reload()
  }

  if (!analysisId) return <Card><Empty>No analysis selected.</Empty></Card>
  const m = res?.metrics
  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>Reinsurance</h1>
          <p>Evaluate any programme instantly against the simulated occurrence losses — inuring stages, reinstatements with
            pro-rata premium, annual aggregate deductibles/limits, quota shares and stop-loss covers, technical pricing
            (EL + k·σ or cost of capital) and the capital view of gross vs net.</p>
        </div>
        <div className="row"><button className="btn primary" onClick={apply}>Apply to analysis</button></div>
      </div>
      {saved && <div className="small" style={{ color: 'var(--good-text)' }}>{saved}</div>}
      <Card title="Programme"><ProgramEditor program={program} onChange={setProgram} /></Card>
      {err && <div className="error small">{err}</div>}
      {m && (
        <div className="grid g6">
          <Tile label="Upfront + reinstatement premium" value={money(m.total_premium)} />
          <Tile label="Expected recoveries" value={money(m.total_expected_recovery)} />
          <Tile label="Expected net cost" value={money(m.net_cost)} foot="premium − expected recovery" />
          <Tile label="Capital relief (1-in-200 AEP)" value={money(m.capital_relief_var995)} foot={`${money(m.gross.var_99_5)} → ${money(m.net.var_99_5)}`} />
          <Tile label="Relief per $ of cost" value={m.relief_per_cost ? `${m.relief_per_cost.toFixed(1)}×` : '—'} />
          <Tile label="Net AAL" value={money(m.net.aal)} foot={`gross ${money(m.gross.aal)}`} />
        </div>
      )}
      <div className="col" style={{ gap: 16 }}>
        <Card title="Contract metrics">
          {m ? <DataTable rows={m.contracts} columns={[
            { key: 'name', label: 'Contract' },
            { key: 'expected_loss', label: 'Expected loss', align: 'r', render: (r) => money(r.expected_loss) },
            { key: 'sd', label: 'SD', align: 'r', render: (r) => money(r.sd) },
            { key: 'prob_attach', label: 'P(attach)', align: 'r', render: (r) => pct(r.prob_attach, 2) },
            { key: 'prob_exhaust', label: 'P(exhaust)', align: 'r', render: (r) => pct(r.prob_exhaust, 2) },
            { key: 'premium', label: 'Premium', align: 'r', render: (r) => money(r.premium) },
            { key: 'rate_on_line', label: 'ROL', align: 'r', value: (r) => r.rate_on_line ?? 0, render: (r) => pct(r.rate_on_line, 2) },
            { key: 'loss_on_line', label: 'LOL', align: 'r', value: (r) => r.loss_on_line ?? 0, render: (r) => pct(r.loss_on_line, 2) },
            { key: 'expected_reinstatement_premium', label: 'Exp. RIP', align: 'r', render: (r) => money(r.expected_reinstatement_premium) },
          ]} /> : <Empty>Evaluating…</Empty>}
        </Card>
        <Card title="Gross vs net aggregate EP">
          {epOpt ? <Chart option={epOpt} height={340} ariaLabel="Gross vs net EP" /> : <Empty>Evaluating…</Empty>}
        </Card>
      </div>
      <Card title="Cat XL optimiser — efficient frontier" desc="Net 1-in-200 AEP (y) vs expected net cost (x) for single layers attaching/exhausting at gross OEP return periods; click a point to load it"
        tools={<>
          <NumberField label="Reinstatements" value={opt.reinstatements} min={0} max={10} onChange={(v) => setOpt({ ...opt, reinstatements: v })} />
          <NumberField label="Pricing load (k·σ)" value={opt.pricing_load} step={0.05} min={0} onChange={(v) => setOpt({ ...opt, pricing_load: v })} />
          <button className="btn" onClick={optimise} style={{ alignSelf: 'flex-end' }}>Run optimiser</button>
        </>}>
        {frontierOpt ? <Chart option={frontierOpt} height={380} ariaLabel="Efficient frontier" onClick={(p) => {
          const pt = (p.data as { p?: FrontierPoint })?.p
          if (pt) setProgram({ ...program, contracts: [{ ...DEFAULT_LAYER, name: `Cat XL ${money(pt.limit)} xs ${money(pt.attachment)}`, stage: 1, attachment: pt.attachment, limit: pt.limit, reinstatements: opt.reinstatements }] })
        }} /> : <Empty>Run the optimiser to map cost vs retained tail risk.</Empty>}
      </Card>
    </div>
  )
}

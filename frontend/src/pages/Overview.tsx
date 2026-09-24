import { useMemo } from 'react'
import type { EChartsOption } from 'echarts'
import { api } from '../api'
import type { AnalysisSummary, EpResponse } from '../api'
import Chart from '../components/Chart'
import { epOption } from '../components/charts'
import { Card, Empty, InsightList, JobStatus, Tile } from '../components/ui'
import { money, num, pct } from '../format'
import { useApp, useFetch } from '../state'
import { SLOT, TOKENS, barItem, catAxis } from '../theme'

interface AllocRow { key: string; aal_share: number; cotvar_share: number; tiv_share: number | null }

export default function Overview() {
  const { analysisId, mode, bootJob, go } = useApp()
  const sum = useFetch(analysisId ? () => api.get<AnalysisSummary>(`/analyses/${analysisId}`) : null, [analysisId])
  const ep = useFetch(analysisId ? () => api.get<EpResponse>(`/analyses/${analysisId}/ep`, { basis: 'gross' }) : null, [analysisId])
  const byState = useFetch(analysisId ? () => api.get<{ rows: AllocRow[] }>(`/analyses/${analysisId}/allocation`, { dimension: 'state', top: 8 }) : null, [analysisId])
  const byPeril = useFetch(analysisId ? () => api.get<{ rows: AllocRow[] }>(`/analyses/${analysisId}/allocation`, { dimension: 'peril' }) : null, [analysisId])
  const s = sum.data
  const t = TOKENS[mode]

  const epOpt = useMemo(() => {
    if (!ep.data || !s) return null
    return epOption(mode, [
      { name: 'AEP (gross)', curve: ep.data.aep_curve, slot: SLOT.aep, band: ep.data.aep },
      { name: 'OEP (gross)', curve: ep.data.oep_curve, slot: SLOT.oep },
    ], { maxRp: 1000, points: [{ name: 'Analytic AEP (FFT check)', slot: SLOT.analytic, data: s.analytic.map((r) => [r.rp, r.aep] as [number, number]) }] })
  }, [ep.data, s, mode])

  const stateOpt = useMemo((): EChartsOption | null => {
    const rows = byState.data?.rows
    if (!rows) return null
    const keys = rows.map((r) => r.key).reverse()
    const get = (f: keyof AllocRow) => rows.map((r) => +(((r[f] as number) ?? 0) * 100).toFixed(2)).reverse()
    return {
      legend: { data: ['TIV share', 'AAL share', 'Tail (co-TVaR) share'] },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => `${v}%` },
      grid: { left: 8, right: 24, top: 36, bottom: 8, containLabel: true },
      xAxis: { type: 'value', axisLabel: { color: t.text3, formatter: '{value}%' }, splitLine: { lineStyle: { color: t.grid } } },
      yAxis: catAxis(mode, keys),
      series: [
        { name: 'TIV share', type: 'bar', data: get('tiv_share'), barMaxWidth: 10, barGap: '20%', itemStyle: barItem(t.series[2], true) },
        { name: 'AAL share', type: 'bar', data: get('aal_share'), barMaxWidth: 10, itemStyle: barItem(t.series[0], true) },
        { name: 'Tail (co-TVaR) share', type: 'bar', data: get('cotvar_share'), barMaxWidth: 10, itemStyle: barItem(t.series[1], true) },
      ],
    }
  }, [byState.data, mode, t])

  const perilOpt = useMemo((): EChartsOption | null => {
    const rows = byPeril.data?.rows
    if (!rows) return null
    const keys = rows.map((r) => (r.key === 'TC' ? 'Hurricane' : 'Earthquake'))
    return {
      legend: { data: ['AAL share', 'Tail (co-TVaR) share'] },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => `${v}%` },
      grid: { left: 8, right: 24, top: 36, bottom: 8, containLabel: true },
      xAxis: catAxis(mode, keys),
      yAxis: { type: 'value', max: 100, axisLabel: { color: t.text3, formatter: '{value}%' }, splitLine: { lineStyle: { color: t.grid } } },
      series: [
        { name: 'AAL share', type: 'bar', data: rows.map((r) => +(r.aal_share * 100).toFixed(1)), barMaxWidth: 24, itemStyle: barItem(t.series[0]),
          label: { show: true, position: 'top', color: t.text2, formatter: '{c}%' } },
        { name: 'Tail (co-TVaR) share', type: 'bar', data: rows.map((r) => +(r.cotvar_share * 100).toFixed(1)), barMaxWidth: 24,
          itemStyle: barItem(t.series[1]), label: { show: true, position: 'top', color: t.text2, formatter: '{c}%' } },
      ],
    }
  }, [byPeril.data, mode, t])

  if (!analysisId) {
    return (
      <Card>
        <Empty>
          {bootJob ? (<div className="col" style={{ maxWidth: 420, margin: '0 auto' }}>
            <div>Building the demo portfolio and running the first analysis…</div>
            <JobStatus job={bootJob} />
          </div>) : (<div className="col" style={{ alignItems: 'center' }}>
            <div>No analyses yet.</div>
            <button className="btn primary" onClick={() => go('run')}>Run an analysis</button>
          </div>)}
        </Empty>
      </Card>
    )
  }
  if (!s) return <Card><Empty>{sum.error ?? 'Loading analysis…'}</Empty></Card>
  const rp = Object.fromEntries(s.rp_table.map((r) => [r.rp, r]))
  const r250 = rp[250]
  const r100 = rp[100]
  const net = s.aal.net !== undefined && s.reinsurance

  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>{s.name}</h1>
          <p>{s.portfolio_name} · {num(s.n_locations)} locations · {num(s.n_years)} simulated years · {num(s.n_occurrences)} loss
            occurrences · computed in {s.timings.total_s?.toFixed(1)}s</p>
        </div>
        <div className="row">
          <button className="btn" onClick={() => go('results')}>Detailed results</button>
          <button className="btn" onClick={() => go('reinsurance')}>Reinsurance</button>
        </div>
      </div>

      <div className="grid g4">
        <Tile hero label="1-in-250 aggregate loss (gross)" value={money(r250?.gross_aep)}
          foot={r250?.gross_aep_lo !== undefined ? `95% CI ${money(r250.gross_aep_lo)} – ${money(r250.gross_aep_hi)}` : undefined} />
        <div className="grid g2 span3">
          <Tile label="Average annual loss (gross)" value={money(s.aal.gross)} foot={`${s.loss_cost_per_mille.toFixed(2)}‰ of TIV · ground-up ${money(s.aal.gu)}`} />
          <Tile label="Total insured value" value={money(s.tiv)} foot={`${num(s.n_locations)} locations`} />
          <Tile label="1-in-100 aggregate loss" value={money(r100?.gross_aep)} foot={`occurrence ${money(r100?.gross_oep)}`} />
          <Tile label={net ? '1-in-250 net of reinsurance' : 'TVaR 1-in-250 (expected shortfall)'}
            value={net ? money(r250?.net_aep) : money(r250?.gross_aep_tvar)}
            foot={net ? `net AAL ${money(s.aal.net)}` : `TVaR / VaR ${(r250?.gross_aep_tvar / r250?.gross_aep).toFixed(2)}`} />
        </div>
      </div>

      <div className="grid g3">
        <Card className="span2" title="Exceedance probability" desc="Aggregate (AEP) and occurrence (OEP) loss by return period · shaded band: 95% order-statistic CI · dots: independent analytic FFT check">
          {epOpt ? <Chart option={epOpt} height={340} ariaLabel="Exceedance probability curves" /> : <Empty>Loading…</Empty>}
        </Card>
        <Card title="Peril contribution" desc={`Share of expected loss vs share of 1-in-${s.config.allocation_rp as number} tail`}>
          {byPeril.data && byPeril.data.rows.length < 2
            ? <Tile label="Single-peril analysis" value={byPeril.data.rows[0]?.key === 'EQ' ? 'Earthquake' : 'Hurricane'} foot="100% of expected loss and of the tail" />
            : perilOpt ? <Chart option={perilOpt} height={340} ariaLabel="Peril contribution" /> : <Empty>Loading…</Empty>}
        </Card>
      </div>

      <div className="grid g3">
        <Card className="span2" title="Insights" desc="Automatically derived, quantified findings — every number is traceable to the simulation">
          <InsightList items={s.insights} limit={8} />
        </Card>
        <div className="col">
          <Card title="Where the tail comes from" desc="Top states by tail contribution: exposure vs expected loss vs tail share">
            {stateOpt ? <Chart option={stateOpt} height={320} ariaLabel="State concentration" /> : <Empty>Loading…</Empty>}
          </Card>
          <Card title="Model diagnostics">
            <dl className="kv">
              <dt>AAL: YLT / ELT / analytic</dt><dd>{money(s.aal.gross)} / {money(s.elt_aal)} / {money(s.analytic_aal)}</dd>
              <dt>Euler allocation check</dt><dd>{money(s.tail_check?.tvar_alloc_sum)} = {money(s.tail_check?.tvar_direct)}</dd>
              <dt>Tail re-simulation drift</dt><dd>{s.tail_check?.max_abs_occ_diff === 0 ? 'bit-identical' : money(s.tail_check?.max_abs_occ_diff)}</dd>
              <dt>Annual loss CoV</dt><dd>{(s.sd / s.aal.gross).toFixed(2)}</dd>
              <dt>Events touching portfolio</dt><dd>{num(s.n_events_relevant)} · {num(s.n_pairs)} event-site pairs</dd>
              <dt>P(annual loss &gt; 0)</dt><dd>{pct(ep.data?.stats.p_nonzero)}</dd>
            </dl>
          </Card>
        </div>
      </div>
    </div>
  )
}

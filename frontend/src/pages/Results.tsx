import { useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { api } from '../api'
import type { AnalysisSummary, EpResponse, SurgeFidelity, SurgeSummary } from '../api'
import Chart from '../components/Chart'
import { epOption } from '../components/charts'
import MapView, { MapLegend } from '../components/MapView'
import type { Layer } from '../components/MapView'
import { Card, DataTable, Download, Empty, Seg, Tabs, Tile } from '../components/ui'
import { money, moneyAxis, num, pct, rpLabel } from '../format'
import { useApp, useFetch } from '../state'
import { BLUE, SLOT, TOKENS, barItem, catAxis, rampCss, seqColor } from '../theme'

type Tab = 'ep' | 'map' | 'alloc' | 'elt' | 'ylt'
interface LocRes { loc_id: string[]; acc_id: string[]; lat: number[]; lon: number[]; state: string[]; construction: string[]; tiv: number[]; aal: number[]; cotvar: number[]; loss_cost_pm: number[] }
interface AllocRow { key: string; n: number; tiv: number; aal: number; aal_share: number; cotvar: number; cotvar_share: number; tiv_share: number; loss_cost_pm: number; tail_leverage: number }
interface Segments { rows: (Record<string, number> & { key: string })[]; portfolio: Record<string, { var: number; tvar: number; sum_standalone_tvar: number; diversification_benefit: number }> }
interface EltRow { event_uid: number; name: string; peril: string; rate: number; n_locs: number; mean: number; sd: number; mean_gu: number; cap: number; p0: number; aal_contrib: number }
interface Ylt { n_years: number; top_years: Record<string, number | string>[]; p_zero: number; histogram: { edges: number[]; counts: number[]; zero: number } }

export default function Results() {
  const { analysisId, mode } = useApp()
  const [tab, setTab] = useState<Tab>('ep')
  const sum = useFetch(analysisId ? () => api.get<AnalysisSummary>(`/analyses/${analysisId}`) : null, [analysisId])
  if (!analysisId) return <Card><Empty>No analysis selected.</Empty></Card>
  const s = sum.data
  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>Results</h1>
          <p>{s ? `${s.name} · ${s.portfolio_name} · ${num(s.n_years)} years` : 'Loading…'}</p>
        </div>
        <div className="row">
          <Download href={`/api/analyses/${analysisId}/elt?format=csv`}>ELT CSV</Download>
          <Download href={`/api/analyses/${analysisId}/ylt?format=csv`}>YLT CSV</Download>
          <Download href={`/api/analyses/${analysisId}/ylt?format=occurrences`}>Occurrences CSV</Download>
          <Download href={`/api/analyses/${analysisId}/locations?format=csv`}>Locations CSV</Download>
        </div>
      </div>
      <Tabs value={tab} onChange={setTab} options={[
        { value: 'ep', label: 'Exceedance curves' }, { value: 'map', label: 'Loss map' },
        { value: 'alloc', label: 'Allocation & diversification' }, { value: 'elt', label: 'Event loss table' },
        { value: 'ylt', label: 'Year loss table' }]} />
      {tab === 'ep' && s && <EpTab id={analysisId} s={s} mode={mode} />}
      {tab === 'map' && <MapTab id={analysisId} mode={mode} />}
      {tab === 'alloc' && <AllocTab id={analysisId} mode={mode} />}
      {tab === 'elt' && <EltTab id={analysisId} />}
      {tab === 'ylt' && <YltTab id={analysisId} mode={mode} />}
    </div>
  )
}

function EpTab({ id, s, mode }: { id: string; s: AnalysisSummary; mode: 'dark' | 'light' }) {
  const hasNet = s.aal.net !== undefined
  const [basis, setBasis] = useState<'gross' | 'gu' | 'net'>('gross')
  const [peril, setPeril] = useState('all')
  const ep = useFetch(() => api.get<EpResponse>(`/analyses/${id}/ep`, { basis, peril }), [id, basis, peril])
  const opt = useMemo(() => ep.data ? epOption(mode, [
    { name: 'AEP', curve: ep.data.aep_curve, slot: SLOT.aep, band: ep.data.aep },
    { name: 'OEP', curve: ep.data.oep_curve, slot: SLOT.oep },
  ], { minRp: 1.5, maxRp: s.n_years }) : null, [ep.data, mode, s.n_years])
  const oep = Object.fromEntries((ep.data?.oep ?? []).map((r) => [r.rp, r]))
  const an = Object.fromEntries(s.analytic.map((r) => [r.rp, r]))
  const perils = (s.config.perils as string[]) ?? []
  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="row">
        <Seg value={basis} onChange={setBasis} ariaLabel="Basis" options={[
          { value: 'gu', label: 'Ground-up' }, { value: 'gross', label: 'Gross' }, ...(hasNet ? [{ value: 'net' as const, label: 'Net' }] : [])]} />
        <Seg value={peril} onChange={setPeril} ariaLabel="Peril" options={[{ value: 'all', label: 'All perils' },
          ...perils.map((p) => ({ value: p, label: p === 'TC' ? 'Hurricane' : 'Earthquake' }))]} />
      </div>
      {ep.data && (
        <div className="grid g4">
          <Tile label="Average annual loss" value={money(ep.data.stats.mean)} foot={`±${money(1.96 * ep.data.stats.se_mean)} (95% MC error)`} />
          <Tile label="Standard deviation" value={money(ep.data.stats.sd)} foot={`CoV ${ep.data.stats.cov?.toFixed(2) ?? '—'}`} />
          <Tile label="P(annual loss > 0)" value={pct(ep.data.stats.p_nonzero)} />
          <Tile label="Largest simulated year" value={money(ep.data.stats.max)} />
        </div>
      )}
      <div className="grid g3">
        <Card className="span2" title="Exceedance probability curves" desc="AEP with 95% distribution-free order-statistic confidence band">
          {opt ? <Chart option={opt} height={400} ariaLabel="EP curves" /> : <Empty>Loading…</Empty>}
        </Card>
        <Card title="Simulation vs analytic" desc="ELT → Beta severities → mixed-Poisson PGF → tilted FFT (gross, all perils)">
          <DataTable rows={s.rp_table.filter((r) => an[r.rp])} maxHeight={400} columns={[
            { key: 'rp', label: 'Return period', render: (r) => rpLabel(r.rp) },
            { key: 'sim', label: 'Simulated AEP', align: 'r', value: (r) => r.gross_aep, render: (r) => money(r.gross_aep) },
            { key: 'an', label: 'Analytic AEP', align: 'r', value: (r) => an[r.rp].aep, render: (r) => money(an[r.rp].aep) },
            { key: 'd', label: 'Diff', align: 'r', value: (r) => an[r.rp].aep / r.gross_aep - 1, render: (r) => pct(an[r.rp].aep / r.gross_aep - 1) },
          ]} />
        </Card>
      </div>
      <Card title="Return-period table" desc={`${basis} · ${peril === 'all' ? 'all perils' : peril}`}>
        {ep.data ? <DataTable rows={ep.data.aep} columns={[
          { key: 'rp', label: 'Return period', render: (r) => rpLabel(r.rp) },
          { key: 'prob', label: 'Annual prob.', align: 'r', render: (r) => pct(r.prob, 2) },
          { key: 'loss', label: 'AEP loss', align: 'r', render: (r) => money(r.loss) },
          { key: 'ci', label: '95% CI', align: 'r', value: (r) => r.loss_lo ?? 0, render: (r) => `${money(r.loss_lo)} – ${money(r.loss_hi)}` },
          { key: 'tvar', label: 'AEP TVaR', align: 'r', render: (r) => money(r.tvar) },
          { key: 'oep', label: 'OEP loss', align: 'r', value: (r) => oep[r.rp]?.loss ?? 0, render: (r) => money(oep[r.rp]?.loss) },
          { key: 'otvar', label: 'OEP TVaR', align: 'r', value: (r) => oep[r.rp]?.tvar ?? 0, render: (r) => money(oep[r.rp]?.tvar) },
        ]} /> : <Empty>Loading…</Empty>}
      </Card>
      {s.surge && <SurgeCard sg={s.surge} />}
    </div>
  )
}

function SurgeCard({ sg }: { sg: SurgeSummary }) {
  const m = sg.model
  const f = (k: string, d = 2) => (typeof m[k] === 'number' ? (m[k] as number).toFixed(d) : '—')
  return (
    <Card title="Storm surge (hurricane)"
      desc={`Multi-fidelity 2-D surge: 4′ runs for every event, corrected to the 2′ model by a calibrated two-part model (connectivity × wet level with a per-node site response). Held-out depth RMSE ${f('cv_depth_rmse_m')} m, bias ${typeof m.cv_depth_bias === 'number' ? pct(m.cv_depth_bias as number) : '—'}; σ event ${f('sigma_event_m')} m, σ site ${f('sigma_site_m')} m, τ ${f('tau_site_response_m')} m`}>
      <div className="grid g4" style={{ marginBottom: 12 }}>
        <Tile label="Surge share of hurricane AAL" value={pct(sg.share_of_tc_aal_gu)} foot={`${money(sg.aal_gu)} of ${money(sg.tc_aal_gu)} ground-up`} />
        <Tile label="Locations reached by water" value={num(sg.n_locations_reached)} foot={`${num(sg.n_pairs_with_water)} event–location pairs`} />
        <Tile label="With calibrated site response" value={num(sg.n_locations_with_site_response)} foot="Harbour/bay term from the 2′ model" />
        <Tile label="Measured ground" value={num(sg.n_locations_measured_ground)} foot={sg.mean_p_wet == null ? 'Enrich exposure for building-scale ground' : `Mean P(connected) ${pct(sg.mean_p_wet)}`} />
      </div>
      <DataTable rows={sg.aep} columns={[
        { key: 'rp', label: 'Return period', render: (r) => rpLabel(r.rp) },
        { key: 'tc', label: 'Hurricane AEP (GU)', align: 'r', render: (r) => money(r.tc_gu) },
        { key: 'w', label: 'Wind only', align: 'r', render: (r) => money(r.tc_gu_wind_only) },
        { key: 's', label: 'Surge-attributed AEP', align: 'r', render: (r) => money(r.surge_gu) },
        { key: 'u', label: 'Uplift from surge', align: 'r', value: (r) => r.uplift ?? 0, render: (r) => (r.uplift == null ? '—' : pct(r.uplift)) },
      ]} />
      {sg.fidelity && <FidelityPanel fd={sg.fidelity} />}
    </Card>
  )
}

function FidelityPanel({ fd }: { fd: SurgeFidelity }) {
  const d = fd.tvar_after / fd.tvar_before - 1
  return (
    <div className="col mt" style={{ gap: 12 }}>
      <h3 style={{ margin: 0 }}>Fidelity allocation — 1-in-{fd.rp} TVaR</h3>
      <div className="grid g4">
        <Tile label="Full-fidelity events" value={`${fd.upgraded.length} / ${fd.budget}`}
          foot={`${num(fd.n_candidates)} candidates · ${fd.tol_met ? 'tolerance met' : 'budget spent'}`} />
        <Tile label="Surge-error sd of TVaR" value={`${pct(fd.u_before_rel, 2)} → ${pct(fd.u_after_rel, 2)}`}
          foot={`${money(fd.u_before)} → ${money(fd.u_after)} (ρ̄ ${fd.rho})`} />
        <Tile label="TVaR, surrogate → with full model" value={`${money(fd.tvar_before)} → ${money(fd.tvar_after)}`}
          foot={`Realised change ${pct(d, 1)}`} />
        <Tile label="VaR" value={`${money(fd.var_before)} → ${money(fd.var_after)}`}
          foot={fd.hf_seconds != null ? `${fd.hf_seconds.toFixed(0)} s of full-model runs` : ''} />
      </div>
      {fd.upgraded.length > 0 && <DataTable rows={fd.upgraded} maxHeight={260} columns={[
        { key: 'name', label: 'Event' },
        { key: 'share', label: 'Share of tail surge error', align: 'r', render: (r) => pct(r.share, 1) },
        { key: 'a', label: 'Tail sensitivity a', align: 'r', render: (r) => r.tail_sensitivity.toFixed(3) },
        { key: 'sd', label: 'Surge sd (surrogate)', align: 'r', render: (r) => money(r.sd_surge_gu) },
        { key: 'b', label: 'Mean surge: surrogate', align: 'r', render: (r) => money(r.mean_surge_gu_before) },
        { key: 'f', label: 'Full model', align: 'r', render: (r) => money(r.mean_surge_gu_after) },
      ]} />}
    </div>
  )
}

function MapTab({ id, mode }: { id: string; mode: 'dark' | 'light' }) {
  const [metric, setMetric] = useState<'aal' | 'cotvar' | 'loss_cost_pm'>('aal')
  const d = useFetch(() => api.get<LocRes>(`/analyses/${id}/locations`), [id])
  const L = d.data
  const { layers, lo, hi } = useMemo(() => {
    if (!L) return { layers: [] as Layer[], lo: '', hi: '' }
    const v = L[metric]
    const pos = v.filter((x) => x > 0)
    const lv = v.map((x) => Math.log10(Math.max(x, 1e-9)))
    const vmin = pos.length ? Math.log10(Math.min(...pos)) : 0
    const vmax = pos.length ? Math.log10(Math.max(...pos)) : 1
    const order = v.map((_, i) => i).sort((a, b) => v[a] - v[b]) // draw largest last
    const idx = order.filter((i) => v[i] > 0)
    const tq = (i: number) => (lv[i] - vmin) / Math.max(vmax - vmin, 1e-9)
    const layer: Layer = {
      kind: 'points', lat: idx.map((i) => L.lat[i]), lon: idx.map((i) => L.lon[i]), opacity: 0.85,
      color: (k) => seqColor(tq(idx[k]), mode), radius: (k) => 2 + 5 * tq(idx[k]),
      tooltip: (k) => { const i = idx[k]; return `<b>${L.loc_id[i]}</b> · ${L.acc_id[i]} · ${L.state[i]}<br/>${L.construction[i]} · TIV ${money(L.tiv[i])}<br/>AAL ${money(L.aal[i])} · loss cost ${L.loss_cost_pm[i].toFixed(2)}‰<br/>co-TVaR ${money(L.cotvar[i])}` },
    }
    const f = metric === 'loss_cost_pm' ? (x: number) => `${x.toFixed(2)}‰` : money
    return { layers: [layer], lo: pos.length ? f(Math.min(...pos)) : '', hi: pos.length ? f(Math.max(...pos)) : '' }
  }, [L, metric, mode])
  const top = useMemo(() => {
    if (!L) return []
    return L.loc_id.map((id2, i) => ({ loc_id: id2, acc: L.acc_id[i], state: L.state[i], con: L.construction[i], tiv: L.tiv[i], aal: L.aal[i], cotvar: L.cotvar[i], lc: L.loss_cost_pm[i] }))
      .sort((a, b) => b.cotvar - a.cotvar).slice(0, 100)
  }, [L])
  const label = { aal: 'Location AAL', cotvar: 'Tail contribution (co-TVaR)', loss_cost_pm: 'Loss cost (AAL / TIV)' }[metric]
  return (
    <div className="grid g3">
      <Card className="span2" title={label} desc="Log colour scale; locations with zero modelled loss hidden"
        tools={<Seg value={metric} onChange={setMetric} ariaLabel="Metric" options={[
          { value: 'aal', label: 'AAL' }, { value: 'cotvar', label: 'Tail contribution' }, { value: 'loss_cost_pm', label: 'Loss cost' }]} />}>
        {L ? <MapView layers={layers} height={540} bounds={[[24, -125], [49, -67]]} fitKey={id}>
          <MapLegend title={label} ramp={rampCss(mode === 'dark' ? [...BLUE].reverse().slice(0, 11) : BLUE.slice(2))} lo={lo} hi={hi} />
        </MapView> : <Empty>Loading…</Empty>}
      </Card>
      <Card title="Top tail contributors" desc="Locations ranked by Euler co-TVaR">
        <DataTable rows={top} maxHeight={540} columns={[
          { key: 'loc_id', label: 'Location' }, { key: 'state', label: 'St' },
          { key: 'cotvar', label: 'co-TVaR', align: 'r', render: (r) => money(r.cotvar) },
          { key: 'aal', label: 'AAL', align: 'r', render: (r) => money(r.aal) },
          { key: 'lc', label: 'LC ‰', align: 'r', render: (r) => r.lc.toFixed(2) },
        ]} />
      </Card>
    </div>
  )
}

function AllocTab({ id, mode }: { id: string; mode: 'dark' | 'light' }) {
  const t = TOKENS[mode]
  const [dim, setDim] = useState('state')
  const al = useFetch(() => api.get<{ rows: AllocRow[]; rp: number }>(`/analyses/${id}/allocation`, { dimension: dim, top: 15 }), [id, dim])
  const seg = useFetch(() => api.get<Segments>(`/analyses/${id}/segments`), [id])
  const opt = useMemo((): EChartsOption | null => {
    const rows = al.data?.rows
    if (!rows) return null
    const keys = rows.map((r) => r.key).reverse()
    const g = (f: 'tiv_share' | 'aal_share' | 'cotvar_share') => rows.map((r) => +((r[f] ?? 0) * 100).toFixed(2)).reverse()
    return {
      legend: { data: ['TIV share', 'AAL share', 'Tail (co-TVaR) share'] },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => `${v}%` },
      grid: { left: 8, right: 24, top: 36, bottom: 8, containLabel: true },
      xAxis: { type: 'value', axisLabel: { color: t.text3, formatter: '{value}%' }, splitLine: { lineStyle: { color: t.grid } } },
      yAxis: catAxis(mode, keys),
      series: [
        { name: 'TIV share', type: 'bar', data: g('tiv_share'), barMaxWidth: 8, itemStyle: barItem(t.series[2], true) },
        { name: 'AAL share', type: 'bar', data: g('aal_share'), barMaxWidth: 8, itemStyle: barItem(t.series[0], true) },
        { name: 'Tail (co-TVaR) share', type: 'bar', data: g('cotvar_share'), barMaxWidth: 8, itemStyle: barItem(t.series[1], true) },
      ],
    }
  }, [al.data, mode, t])
  const segOpt = useMemo((): EChartsOption | null => {
    const rows = seg.data?.rows.filter((r) => r.standalone_tvar_250 > 0).slice(0, 12)
    if (!rows?.length) return null
    const keys = rows.map((r) => r.key).reverse()
    return {
      legend: { data: ['Stand-alone TVaR 250', 'Contribution in portfolio'] },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => money(v as number) },
      grid: { left: 8, right: 24, top: 36, bottom: 8, containLabel: true },
      xAxis: { type: 'value', axisLabel: { color: t.text3, formatter: (v: number) => moneyAxis(v) }, splitLine: { lineStyle: { color: t.grid } } },
      yAxis: catAxis(mode, keys),
      series: [
        { name: 'Stand-alone TVaR 250', type: 'bar', data: rows.map((r) => r.standalone_tvar_250).reverse(), barMaxWidth: 10, itemStyle: barItem(t.series[0], true) },
        { name: 'Contribution in portfolio', type: 'bar', data: rows.map((r) => r.cotvar_250).reverse(), barMaxWidth: 10, itemStyle: barItem(t.series[1], true) },
      ],
    }
  }, [seg.data, mode, t])
  const div = seg.data?.portfolio['250']
  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="grid g2">
        <Card title="Risk allocation" desc={`Euler allocation of AEP TVaR (1-in-${al.data?.rp ?? 250}) — contributions sum exactly to the portfolio TVaR`}
          tools={<select value={dim} onChange={(e) => setDim(e.target.value)} aria-label="Dimension">
            {['state', 'construction', 'occupancy', 'lob', 'year_band', 'terrain', 'stories_band', 'acc_id', 'peril'].map((d) => <option key={d} value={d}>{d}</option>)}
          </select>}>
          {opt ? <Chart option={opt} height={420} ariaLabel="Allocation" /> : <Empty>Loading…</Empty>}
        </Card>
        <Card title="Diversification" desc="Stand-alone segment tail risk vs its contribution to the portfolio tail">
          {div && <div className="grid g3" style={{ marginBottom: 12 }}>
            <Tile label="Σ stand-alone TVaR" value={money(div.sum_standalone_tvar)} />
            <Tile label="Portfolio TVaR" value={money(div.tvar)} />
            <Tile label="Diversification benefit" value={pct(div.diversification_benefit)} />
          </div>}
          {segOpt ? <Chart option={segOpt} height={330} ariaLabel="Diversification" /> : <Empty>Loading…</Empty>}
        </Card>
      </div>
      <Card title="Allocation table">
        {al.data ? <DataTable rows={al.data.rows} columns={[
          { key: 'key', label: dim }, { key: 'n', label: 'Locations', align: 'r', render: (r) => (r.n === null ? '—' : num(r.n)) },
          { key: 'tiv', label: 'TIV', align: 'r', render: (r) => money(r.tiv) },
          { key: 'aal', label: 'AAL', align: 'r', render: (r) => money(r.aal) },
          { key: 'aal_share', label: 'AAL share', align: 'r', render: (r) => pct(r.aal_share) },
          { key: 'cotvar', label: 'co-TVaR', align: 'r', render: (r) => money(r.cotvar) },
          { key: 'cotvar_share', label: 'Tail share', align: 'r', render: (r) => pct(r.cotvar_share) },
          { key: 'loss_cost_pm', label: 'Loss cost ‰', align: 'r', render: (r) => (r.loss_cost_pm === null ? '—' : r.loss_cost_pm?.toFixed(2)) },
          { key: 'tail_leverage', label: 'Tail leverage', align: 'r', render: (r) => (r.tail_leverage ? `${r.tail_leverage.toFixed(2)}×` : '—') },
        ]} /> : <Empty>Loading…</Empty>}
      </Card>
    </div>
  )
}

function EltTab({ id }: { id: string }) {
  const [peril, setPeril] = useState('')
  const d = useFetch(() => api.get<{ total: number; rows: EltRow[] }>(`/analyses/${id}/elt`, { limit: 500, peril: peril || undefined }), [id, peril])
  return (
    <Card title="Event loss table" desc={d.data ? `${num(d.data.total)} events with non-zero loss potential · ranked by AAL contribution (rate × mean loss)` : ''}
      tools={<Seg value={peril} onChange={setPeril} options={[{ value: '', label: 'All' }, { value: 'TC', label: 'TC' }, { value: 'EQ', label: 'EQ' }]} />}>
      {d.data ? <DataTable rows={d.data.rows} maxHeight={600} columns={[
        { key: 'name', label: 'Event' }, { key: 'peril', label: 'Peril' },
        { key: 'rate', label: 'Rate /yr', align: 'r', render: (r) => r.rate.toExponential(2) },
        { key: 'n_locs', label: 'Sites hit', align: 'r', render: (r) => num(r.n_locs) },
        { key: 'mean_gu', label: 'Mean GU', align: 'r', render: (r) => money(r.mean_gu) },
        { key: 'mean', label: 'Mean gross', align: 'r', render: (r) => money(r.mean) },
        { key: 'sd', label: 'SD gross', align: 'r', render: (r) => money(r.sd) },
        { key: 'p0', label: 'P(0)', align: 'r', render: (r) => pct(r.p0, 0) },
        { key: 'cap', label: 'Exposed limit', align: 'r', render: (r) => money(r.cap) },
        { key: 'aal_contrib', label: 'AAL contrib.', align: 'r', render: (r) => money(r.aal_contrib) },
      ]} /> : <Empty>Loading…</Empty>}
    </Card>
  )
}

function YltTab({ id, mode }: { id: string; mode: 'dark' | 'light' }) {
  const t = TOKENS[mode]
  const d = useFetch(() => api.get<Ylt>(`/analyses/${id}/ylt`, { limit: 100 }), [id])
  const opt = useMemo((): EChartsOption | null => {
    const h = d.data?.histogram
    if (!h?.edges.length) return null
    const cats = h.counts.map((_, i) => moneyAxis(Math.sqrt(h.edges[i] * h.edges[i + 1])))
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: (p: unknown) => {
        const q = (p as { dataIndex: number; value: number }[])[0]
        return `${money(h.edges[q.dataIndex])} – ${money(h.edges[q.dataIndex + 1])}<br/><b>${num(q.value)}</b> years`
      } },
      grid: { left: 12, right: 20, top: 16, bottom: 30, containLabel: true },
      xAxis: catAxis(mode, cats, { name: 'Annual gross loss (log bins)', nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.text3 } }),
      yAxis: { type: 'value', axisLabel: { color: t.text3 }, splitLine: { lineStyle: { color: t.grid } } },
      series: [{ type: 'bar', data: h.counts, barMaxWidth: 24, barCategoryGap: '2px', itemStyle: barItem(t.series[0]) }],
    }
  }, [d.data, mode, t])
  const Y = d.data
  return (
    <div className="grid g2">
      <Card title="Distribution of annual loss" desc={Y ? `${num(Y.n_years)} simulated years · ${pct(Y.p_zero)} loss-free years excluded from the log histogram` : ''}>
        {opt ? <Chart option={opt} height={380} ariaLabel="Annual loss histogram" /> : <Empty>Loading…</Empty>}
      </Card>
      <Card title="Worst simulated years">
        {Y ? <DataTable rows={Y.top_years} maxHeight={380} columns={[
          { key: 'year', label: 'Year' },
          { key: 'gross', label: 'Gross', align: 'r', render: (r) => money(r.gross as number) },
          { key: 'net', label: 'Net', align: 'r', render: (r) => money(r.net as number) },
          { key: 'max_occurrence', label: 'Largest event', align: 'r', render: (r) => money(r.max_occurrence as number) },
          { key: 'n_occurrences', label: 'Events', align: 'r' },
          ...(Y.top_years[0]?.tc_regime !== undefined ? [{ key: 'tc_regime', label: 'ENSO' }] : []),
        ]} /> : <Empty>Loading…</Empty>}
      </Card>
    </div>
  )
}

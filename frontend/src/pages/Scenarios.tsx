import { useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { api } from '../api'
import type { Grid, Job } from '../api'
import Chart from '../components/Chart'
import MapView, { MapLegend } from '../components/MapView'
import type { Layer } from '../components/MapView'
import { Card, DataTable, Empty, JobStatus, NumberField, Seg, Tile } from '../components/ui'
import { money, moneyAxis, num } from '../format'
import { useApp } from '../state'
import { HEAT, TOKENS, barItem, catAxis, rampColor, rampCss, seqColor } from '../theme'

interface ScenarioResult {
  peril: 'TC' | 'EQ'; n_samples: number; n_locations_affected: number; tiv_affected: number
  gross: Record<string, number>; gu: Record<string, number>; histogram: { edges: number[]; counts: number[] }
  by_state: { state: string; n: number; gross: number; gu: number; tiv: number }[]
  locations: { lat: number[]; lon: number[]; mean_gross: number[]; damage_ratio: number[] }
  geometry: { lat: number[]; lon: number[] }; footprint: Grid; event: Record<string, unknown>
}

const TC_DEFAULT = { landfall_lat: 27.95, landfall_lon: -82.8, heading: 60, vmax: 62, rmax_km: 30, vt: 6 }
const EQ_DEFAULT = { lat: 34.05, lon: -118.25, mag: 7.0, strike: 110, depth_h: 6 }

export default function Scenarios() {
  const { meta, portfolioId, portfolios, setPortfolioId, mode } = useApp()
  const t = TOKENS[mode]
  const [analog, setAnalog] = useState<string | null>('andrew_1992')
  const [custom, setCustom] = useState<'TC' | 'EQ'>('TC')
  const [tc, setTc] = useState(TC_DEFAULT)
  const [eq, setEq] = useState(EQ_DEFAULT)
  const [job, setJob] = useState<Job | null>(null)
  const [res, setRes] = useState<ScenarioResult | null>(null)
  const [err, setErr] = useState<string | null>(null)

  async function run(useAnalog: boolean) {
    if (!portfolioId) return
    setErr(null)
    try {
      const body = useAnalog && analog ? { portfolio_id: portfolioId, analog, n_samples: 1000 }
        : { portfolio_id: portfolioId, peril: custom, params: custom === 'TC' ? tc : eq, n_samples: 1000 }
      const j = await api.post<Job>('/scenarios/run', body, { wait: false })
      setJob(j)
      const done = await api.waitJob<ScenarioResult>(j, setJob)
      setRes(done.result)
    } catch (e) { setErr(String(e)) }
  }

  const layers = useMemo((): Layer[] => {
    if (!res) return []
    const f = res.footprint
    const lo = res.peril === 'TC' ? 25 : 0.02
    const hi = res.peril === 'TC' ? 85 : 1.0
    const sc = (v: number) => res.peril === 'TC' ? (v - lo) / (hi - lo) : Math.log(v / lo) / Math.log(hi / lo)
    const L = res.locations
    const mx = Math.max(...L.mean_gross, 1)
    const idx = L.mean_gross.map((_, i) => i).sort((a, b) => L.mean_gross[a] - L.mean_gross[b])
    return [
      { kind: 'grid', lat0: f.lat0 - f.res_deg / 2, lon0: f.lon0 - f.res_deg / 2, res: f.res_deg, ny: f.ny, nx: f.nx,
        value: (iy, ix) => { const v = f.values[iy][ix]; return v >= lo ? v : null }, color: (v) => rampColor(HEAT, sc(v)), opacity: 0.6 },
      { kind: 'lines', items: [{ lat: res.geometry.lat, lon: res.geometry.lon, color: t.text1, weight: 2.5, opacity: 0.9 }] },
      { kind: 'points', lat: idx.map((i) => L.lat[i]), lon: idx.map((i) => L.lon[i]), opacity: 0.9,
        color: (k) => seqColor(Math.sqrt(L.mean_gross[idx[k]] / mx), mode), radius: (k) => 2 + 5 * Math.sqrt(L.mean_gross[idx[k]] / mx),
        tooltip: (k) => `Mean gross loss ${money(L.mean_gross[idx[k]])}<br/>Mean damage ratio ${(100 * L.damage_ratio[idx[k]]).toFixed(1)}%` },
    ]
  }, [res, mode, t])

  const histOpt = useMemo((): EChartsOption | null => {
    const h = res?.histogram
    if (!h || h.counts.length < 2) return null
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: (p: unknown) => {
        const q = (p as { dataIndex: number; value: number }[])[0]
        return `${money(h.edges[q.dataIndex])} – ${money(h.edges[q.dataIndex + 1])}<br/><b>${q.value}</b> of ${res!.n_samples} realisations`
      } },
      grid: { left: 12, right: 20, top: 16, bottom: 30, containLabel: true },
      xAxis: catAxis(mode, h.counts.map((_, i) => moneyAxis(0.5 * (h.edges[i] + h.edges[i + 1]))), { name: 'Gross event loss', nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.text3 } }),
      yAxis: { type: 'value', axisLabel: { color: t.text3 }, splitLine: { lineStyle: { color: t.grid } } },
      series: [{ type: 'bar', data: h.counts, barCategoryGap: '2px', barMaxWidth: 24, itemStyle: barItem(t.series[0]) }],
    }
  }, [res, mode, t])

  const analogs = Object.entries(meta?.analogs ?? {})
  const bounds = res ? [[Math.min(...res.geometry.lat) - 2, Math.min(...res.geometry.lon) - 2.5], [Math.max(...res.geometry.lat) + 2, Math.max(...res.geometry.lon) + 2.5]] as [[number, number], [number, number]] : null
  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>Deterministic scenarios</h1>
          <p>Run historical analogs or custom events through the full engine — hazard uncertainty, vulnerability uncertainty,
            spatial correlation and financial terms — for a distribution of event loss (1,000 realisations), not a point estimate.</p>
        </div>
        <label className="row small sub">Portfolio
          <select value={portfolioId ?? ''} onChange={(e) => setPortfolioId(e.target.value)}>
            {portfolios.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select></label>
      </div>
      <div className="grid g3">
        <Card title="Historical analogs" desc="Approximate public parameters at today's exposure">
          <div className="col" style={{ gap: 8, maxHeight: 420, overflowY: 'auto' }}>
            {analogs.map(([k, a]) => (
              <button key={k} className={`analog ${analog === k ? 'on' : ''}`} onClick={() => setAnalog(k)}>
                <span className="pill">{a.peril === 'TC' ? 'Hurricane' : 'Earthquake'}</span> <span style={{ marginLeft: 6 }}>{a.label}</span>
              </button>
            ))}
          </div>
          <button className="btn primary mt" onClick={() => run(true)} disabled={!analog || job?.status === 'running'}>▶ Run analog</button>
        </Card>
        <Card title="Custom event" tools={<Seg value={custom} onChange={setCustom} options={[{ value: 'TC', label: 'Hurricane' }, { value: 'EQ', label: 'Earthquake' }]} />}>
          {custom === 'TC' ? (
            <div className="form-grid">
              <NumberField label="Landfall lat" value={tc.landfall_lat} step={0.05} onChange={(v) => setTc({ ...tc, landfall_lat: v })} />
              <NumberField label="Landfall lon" value={tc.landfall_lon} step={0.05} onChange={(v) => setTc({ ...tc, landfall_lon: v })} />
              <NumberField label="Heading (°)" value={tc.heading} step={5} onChange={(v) => setTc({ ...tc, heading: v })} />
              <NumberField label="Vmax (m/s, 1-min)" value={tc.vmax} step={1} onChange={(v) => setTc({ ...tc, vmax: v })} />
              <NumberField label="Rmax (km)" value={tc.rmax_km} step={5} onChange={(v) => setTc({ ...tc, rmax_km: v })} />
              <NumberField label="Forward speed (m/s)" value={tc.vt} step={0.5} onChange={(v) => setTc({ ...tc, vt: v })} />
            </div>
          ) : (
            <div className="form-grid">
              <NumberField label="Epicentre lat" value={eq.lat} step={0.05} onChange={(v) => setEq({ ...eq, lat: v })} />
              <NumberField label="Epicentre lon" value={eq.lon} step={0.05} onChange={(v) => setEq({ ...eq, lon: v })} />
              <NumberField label="Magnitude" value={eq.mag} step={0.1} onChange={(v) => setEq({ ...eq, mag: v })} />
              <NumberField label="Strike (°)" value={eq.strike} step={5} onChange={(v) => setEq({ ...eq, strike: v })} />
              <NumberField label="Depth term h (km)" value={eq.depth_h} step={1} onChange={(v) => setEq({ ...eq, depth_h: v })} />
            </div>
          )}
          <button className="btn mt" onClick={() => run(false)} disabled={job?.status === 'running'}>▶ Run custom event</button>
          <div className="mt"><JobStatus job={job && job.status !== 'done' ? job : null} /></div>
          {err && <div className="error small">{err}</div>}
        </Card>
        <Card title="Result">
          {res ? (
            <div className="grid g2">
              <Tile label="Mean gross loss" value={money(res.gross.mean)} foot={`ground-up ${money(res.gu.mean)}`} />
              <Tile label="Median" value={money(res.gross.p50)} foot={`5–95%: ${money(res.gross.p5)} – ${money(res.gross.p95)}`} />
              <Tile label="1-in-100 realisation" value={money(res.gross.p99)} />
              <Tile label="Locations affected" value={num(res.n_locations_affected)} foot={`TIV ${money(res.tiv_affected)}`} />
            </div>
          ) : <Empty>Run a scenario.</Empty>}
        </Card>
      </div>
      {res && (
        <div className="grid g3">
          <Card className="span2" title={String(res.event.name ?? 'Scenario footprint')} desc="Median intensity footprint (heat) · white: track / rupture · dots: mean gross loss by location">
            <MapView layers={layers} height={480} bounds={bounds} fitKey={JSON.stringify(res.event)}>
              <MapLegend title={res.peril === 'TC' ? '3-s gust (m/s)' : 'PGA (g)'} ramp={rampCss(HEAT)} lo={res.peril === 'TC' ? '25' : '0.02'} hi={res.peril === 'TC' ? '85' : '1.0'} />
            </MapView>
          </Card>
          <div className="col">
            <Card title="Event loss distribution">{histOpt ? <Chart option={histOpt} height={220} /> : <Empty>—</Empty>}</Card>
            <Card title="Loss by state">
              <DataTable rows={res.by_state} maxHeight={200} columns={[
                { key: 'state', label: 'State' }, { key: 'n', label: 'Sites', align: 'r', render: (r) => num(r.n) },
                { key: 'gross', label: 'Mean gross', align: 'r', render: (r) => money(r.gross) },
                { key: 'gu', label: 'Mean GU', align: 'r', render: (r) => money(r.gu) }]} />
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}

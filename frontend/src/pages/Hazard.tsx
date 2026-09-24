import { Fragment, useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { api } from '../api'
import type { Grid } from '../api'
import Chart from '../components/Chart'
import MapView, { MapLegend } from '../components/MapView'
import type { GridLayer, Layer } from '../components/MapView'
import { Card, DataTable, Empty, Seg } from '../components/ui'
import { num } from '../format'
import { useApp, useFetch } from '../state'
import { HEAT, TOKENS, barItem, catAxis, lineSeries, logAxis, rampColor, rampCss } from '../theme'

type Peril = 'TC' | 'EQ'
interface HazMap { lat: number[]; lon: number[]; value: number[]; res_deg: number; unit: string }
interface GeomItem { event_id: number; lat: number[]; lon: number[]; category?: number; vmax?: number; mag?: number; source?: string }
interface EventRow { event_id: number; name: string; rate: number; vmax?: number; category?: number; region?: string; mag?: number; source?: string; rmax_km?: number; dp_hpa?: number; landfall_lat?: number; landfall_lon?: number }
interface Curve { intensity: number[]; annual_exceedance_prob: number[]; return_period_intensity: Record<string, number | null>; unit: string }
interface VulnCurve { intensity: number[]; mean: number[]; mean_no_hazard_unc: number[]; p5: number[]; p25: number[]; p50: number[]; p75: number[]; p95: number[] }

const RANGE: Record<Peril, { lo: number; hi: number; log: boolean; fmt: (v: number) => string; unit: string }> = {
  TC: { lo: 25, hi: 85, log: false, fmt: (v) => `${v.toFixed(0)} m/s`, unit: '3-s gust (m/s)' },
  EQ: { lo: 0.02, hi: 1.0, log: true, fmt: (v) => `${v.toFixed(2)} g`, unit: 'PGA (g)' },
}

function scaleT(p: Peril, v: number) {
  const r = RANGE[p]
  return r.log ? Math.log(v / r.lo) / Math.log(r.hi / r.lo) : (v - r.lo) / (r.hi - r.lo)
}

function pointsToGrid(m: HazMap): GridLayer | null {
  if (!m.lat.length) return null
  const res = m.res_deg
  const lat0 = Math.min(...m.lat)
  const lon0 = Math.min(...m.lon)
  const ny = Math.round((Math.max(...m.lat) - lat0) / res) + 1
  const nx = Math.round((Math.max(...m.lon) - lon0) / res) + 1
  const arr = new Float32Array(ny * nx).fill(NaN)
  m.lat.forEach((la, i) => { arr[Math.round((la - lat0) / res) * nx + Math.round((m.lon[i] - lon0) / res)] = m.value[i] })
  return { kind: 'grid', lat0: lat0 - res / 2, lon0: lon0 - res / 2, res, ny, nx, value: (iy, ix) => arr[iy * nx + ix], color: () => [0, 0, 0], opacity: 0.7 }
}

export default function Hazard() {
  const { mode, meta } = useApp()
  const t = TOKENS[mode]
  const [peril, setPeril] = useState<Peril>('TC')
  const [rp, setRp] = useState(100)
  const [site, setSite] = useState<[number, number] | null>([25.77, -80.19])
  const [eventId, setEventId] = useState<number | null>(null)
  const [showGeom, setShowGeom] = useState(true)
  const [vuln, setVuln] = useState({ construction: 'WOOD', occupancy: 'RES_SF', year_built: 1990, stories: 1, roof_shape: 'gable', shutters: 0 })

  const hmap = useFetch(() => api.get<HazMap>('/hazard/map', { peril, rp }), [peril, rp])
  const geom = useFetch(() => api.get<{ items: GeomItem[] }>(`/catalogs/${peril}/geometry`, { limit: peril === 'TC' ? 250 : 400 }), [peril])
  const events = useFetch(() => api.get<{ total: number; rows: EventRow[] }>(`/catalogs/${peril}/events`, { limit: 200, sort: peril === 'TC' ? 'vmax' : 'mag' }), [peril])
  const fp = useFetch(eventId ? () => api.get<Grid>(`/catalogs/${peril}/events/${eventId}/footprint`, { res_deg: peril === 'TC' ? 0.05 : 0.04 }) : null, [eventId, peril])
  const evDetail = useFetch(eventId ? () => api.get<{ geometry: { lat: number[]; lon: number[] } }>(`/catalogs/${peril}/events/${eventId}`) : null, [eventId, peril])
  const curve = useFetch(site ? () => api.get<Curve>('/hazard/curve', { peril, lat: site[0], lon: site[1] }) : null, [peril, site?.[0], site?.[1]])
  const stats = useFetch(() => api.get<Record<string, unknown>>(`/catalogs/${peril}/stats`), [peril])
  const vc = useFetch(() => api.get<VulnCurve>('/vulnerability/curve', { peril, ...vuln }), [peril, vuln])

  const layers = useMemo((): Layer[] => {
    const out: Layer[] = []
    const heat = (v: number) => rampColor(HEAT, scaleT(peril, v))
    if (eventId && fp.data) {
      const g = fp.data
      out.push({ kind: 'grid', lat0: g.lat0 - g.res_deg / 2, lon0: g.lon0 - g.res_deg / 2, res: g.res_deg, ny: g.ny, nx: g.nx,
        value: (iy, ix) => { const v = g.values[iy][ix]; return v >= RANGE[peril].lo ? v : null }, color: heat, opacity: 0.72 })
      if (evDetail.data) out.push({ kind: 'lines', items: [{ lat: evDetail.data.geometry.lat, lon: evDetail.data.geometry.lon, color: t.text1, weight: 2.5, opacity: 0.9 }] })
    } else if (hmap.data) {
      const g = pointsToGrid(hmap.data)
      if (g) out.push({ ...g, value: (iy, ix) => { const v = g.value(iy, ix); return v !== null && v >= RANGE[peril].lo ? v : null }, color: heat })
      if (showGeom && geom.data) {
        out.push({ kind: 'lines', items: geom.data.items.map((it) => ({
          lat: it.lat, lon: it.lon, weight: peril === 'TC' ? 1 : 2.5, opacity: peril === 'TC' ? 0.45 : 0.85,
          color: peril === 'TC' ? `rgb(${rampColor(HEAT, (it.category ?? 0) / 5).join(',')})` : `rgb(${rampColor(HEAT, ((it.mag ?? 6.5) - 6.5) / 2.5).join(',')})`,
          tooltip: peril === 'TC' ? `Event ${it.event_id} · Cat ${it.category} · ${it.vmax?.toFixed(0)} m/s` : `M${it.mag?.toFixed(1)} ${it.source}`,
          onClick: () => setEventId(it.event_id),
        })) })
      }
    }
    if (site) out.push({ kind: 'points', lat: [site[0]], lon: [site[1]], color: () => t.text1, radius: () => 6, opacity: 1 })
    return out
  }, [eventId, fp.data, evDetail.data, hmap.data, geom.data, showGeom, site, peril, t])

  const curveOpt = useMemo((): EChartsOption | null => {
    const c = curve.data
    if (!c) return null
    const data = c.intensity.map((x, i) => [x, c.annual_exceedance_prob[i]]).filter((d) => d[1] > 1e-5)
    return {
      tooltip: { trigger: 'axis', valueFormatter: (v) => (typeof v === 'number' ? `${(v * 100).toFixed(3)}%` : String(v)) },
      grid: { left: 12, right: 20, top: 16, bottom: 30, containLabel: true },
      xAxis: { type: 'value', name: RANGE[peril].unit, nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.text3 },
        axisLabel: { color: t.text3 }, splitLine: { lineStyle: { color: t.grid } }, min: 'dataMin' },
      yAxis: logAxis(mode, {
        axisLabel: { color: t.text3, formatter: (v: number) => (v >= 0.01 ? `${v * 100}%` : `${(v * 100).toPrecision(1)}%`) } }),
      series: [lineSeries('Hazard curve', t.series[0], data)],
    }
  }, [curve.data, peril, mode, t])

  const statsOpt = useMemo((): EChartsOption | null => {
    const s = stats.data as Record<string, unknown> | null
    if (!s || s.peril !== peril) return null
    if (peril === 'TC') {
      const rows = (s.by_region as Record<string, number | string>[]).map((r) => ({ region: r.region as string,
        rate: Object.entries(r).filter(([k]) => k.startsWith('cat')).reduce((a, [, v]) => a + (v as number), 0),
        major: ['cat3', 'cat4', 'cat5'].reduce((a, k) => a + ((r[k] as number) || 0), 0) })).sort((a, b) => a.rate - b.rate)
      return {
        legend: { data: ['All landfalls', 'Major (Cat 3+)'] },
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => `${(v as number).toFixed(3)} / yr` },
        grid: { left: 8, right: 20, top: 36, bottom: 8, containLabel: true },
        xAxis: { type: 'value', axisLabel: { color: t.text3 }, splitLine: { lineStyle: { color: t.grid } } },
        yAxis: catAxis(mode, rows.map((r) => r.region)),
        series: [
          { name: 'All landfalls', type: 'bar', data: rows.map((r) => r.rate), barMaxWidth: 8, itemStyle: barItem(t.series[0], true) },
          { name: 'Major (Cat 3+)', type: 'bar', data: rows.map((r) => r.major), barMaxWidth: 8, itemStyle: barItem(t.series[1], true) },
        ],
      }
    }
    const mfd = s.mfd as { m: number[]; rate_ge: number[] }
    return {
      tooltip: { trigger: 'axis', valueFormatter: (v) => `${(v as number).toPrecision(3)} / yr` },
      grid: { left: 12, right: 20, top: 16, bottom: 30, containLabel: true },
      xAxis: { type: 'value', min: 5, max: 9.2, name: 'Magnitude', nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.text3 }, axisLabel: { color: t.text3 }, splitLine: { lineStyle: { color: t.grid } } },
      yAxis: logAxis(mode),
      series: [lineSeries('Magnitude-frequency', t.series[0], mfd.m.map((m, i) => [m, mfd.rate_ge[i]]).filter((d) => d[1] > 0))],
    }
  }, [stats.data, peril, mode, t])

  const vulnOpt = useMemo((): EChartsOption | null => {
    const v = vc.data
    if (!v) return null
    const c = t.series[0]
    const band = (lo: number[], hi: number[], name: string, op: number) => [
      { name, type: 'line', data: v.intensity.map((x, i) => [x, lo[i]]), stack: name, lineStyle: { opacity: 0 }, symbol: 'none', silent: true, tooltip: { show: false } },
      { name, type: 'line', data: v.intensity.map((x, i) => [x, hi[i] - lo[i]]), stack: name, lineStyle: { opacity: 0 }, symbol: 'none', silent: true, areaStyle: { color: c, opacity: op }, tooltip: { show: false } },
    ]
    return {
      legend: { data: ['Mean damage (with hazard uncertainty)', 'Mean damage (median intensity)'] },
      tooltip: { trigger: 'axis', valueFormatter: (x) => `${((x as number) * 100).toFixed(1)}%` },
      grid: { left: 12, right: 20, top: 40, bottom: 30, containLabel: true },
      xAxis: { type: peril === 'EQ' ? 'log' : 'value', name: RANGE[peril].unit, nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.text3 },
        axisLabel: { color: t.text3 }, splitLine: { lineStyle: { color: t.grid } }, min: peril === 'EQ' ? 0.01 : 20, max: peril === 'EQ' ? 3 : 110 },
      yAxis: { type: 'value', max: 1, axisLabel: { color: t.text3, formatter: (x: number) => `${Math.round(x * 100)}%` }, splitLine: { lineStyle: { color: t.grid } } },
      series: [
        ...band(v.p5, v.p95, '5–95% band', 0.08), ...band(v.p25, v.p75, '25–75% band', 0.14),
        lineSeries('Mean damage (with hazard uncertainty)', c, v.intensity.map((x, i) => [x, v.mean[i]])),
        lineSeries('Mean damage (median intensity)', t.series[1], v.intensity.map((x, i) => [x, v.mean_no_hazard_unc[i]])),
      ] as EChartsOption['series'],
    }
  }, [vc.data, peril, t])

  const selected = events.data?.rows.find((r) => r.event_id === eventId)

  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>Hazard &amp; vulnerability</h1>
          <p>Stochastic catalogs: {peril === 'TC'
            ? 'importance-sampled landfall-gate hurricane tracks with a Holland (2008) surface wind field and Kaplan–DeMaria inland decay.'
            : 'fault and area sources with truncated Gutenberg–Richter recurrence, finite ruptures and a BA08-form ground-motion model with site amplification.'}
            {' '}Click the map for a site hazard curve; click a track/rupture or a table row for its footprint.</p>
        </div>
        <div className="row">
          <Seg value={peril} onChange={(p) => { setPeril(p); setEventId(null); setSite(p === 'TC' ? [25.77, -80.19] : [34.05, -118.24]) }}
            options={[{ value: 'TC', label: 'Hurricane' }, { value: 'EQ', label: 'Earthquake' }]} ariaLabel="Peril" />
        </div>
      </div>

      <div className="grid g3">
        <Card className="span2" title={eventId ? `Event footprint — ${selected?.name ?? eventId}` : `${rp}-year return-period hazard`}
          desc={eventId ? 'Median intensity; white line: track / rupture' : `Intensity with ${(100 / rp).toFixed(1)}% annual exceedance probability (open terrain / Vs30 = 400 m/s)`}
          tools={<>
            {!eventId && <Seg value={String(rp)} onChange={(v) => setRp(Number(v))} ariaLabel="Return period"
              options={['25', '50', '100', '250', '500'].map((x) => ({ value: x, label: `${x}y` }))} />}
            {!eventId && <label className="check small"><input type="checkbox" checked={showGeom} onChange={(e) => setShowGeom(e.target.checked)} />{peril === 'TC' ? 'tracks' : 'ruptures'}</label>}
            {eventId && <button className="btn sm" onClick={() => setEventId(null)}>Back to hazard map</button>}
          </>}>
          <MapView layers={layers} height={640} onMapClick={(la, lo) => setSite([la, lo])}
            bounds={peril === 'TC' ? [[24, -98], [45, -67]] : [[31.5, -125], [49.5, -88]]} fitKey={peril}>
            <MapLegend title={RANGE[peril].unit} ramp={rampCss(HEAT)} lo={RANGE[peril].fmt(RANGE[peril].lo)} hi={RANGE[peril].fmt(RANGE[peril].hi)} />
          </MapView>
          {(hmap.loading || fp.loading) && <div className="small muted" style={{ marginTop: 6 }}>Computing hazard…</div>}
        </Card>
        <div className="col">
          <Card title="Site hazard curve" desc={site ? `Annual exceedance probability at ${site[0].toFixed(3)}, ${site[1].toFixed(3)} — click the map to move` : 'Click the map'}>
            {curveOpt ? <Chart option={curveOpt} height={230} ariaLabel="Site hazard curve" /> : <Empty>{curve.loading ? 'Computing…' : '—'}</Empty>}
            {curve.data && (
              <dl className="kv" style={{ marginTop: 8 }}>
                {Object.entries(curve.data.return_period_intensity).map(([k, v]) => (
                  <Fragment key={k}><dt>{k}-year</dt><dd>{v === null ? '< threshold' : RANGE[peril].fmt(v)}</dd></Fragment>
                ))}
              </dl>
            )}
          </Card>
          <Card title={peril === 'TC' ? 'Landfall climatology' : 'Magnitude–frequency'} desc={peril === 'TC' ? 'Annual landfall rate by coastal region' : 'Annual rate of M ≥ m, all sources'}>
            {statsOpt ? <Chart option={statsOpt} height={peril === 'TC' ? 360 : 240} /> : <Empty>Loading…</Empty>}
          </Card>
        </div>
      </div>

      <div className="grid g2">
        <Card title="Event catalog" desc={events.data ? `${num(events.data.total)} stochastic events · showing the 200 most intense` : ''}>
          {events.data ? (
            <DataTable rows={events.data.rows} maxHeight={380} onRowClick={(r) => setEventId(r.event_id)} selected={(r) => r.event_id === eventId}
              columns={peril === 'TC' ? [
                { key: 'name', label: 'Event' }, { key: 'region', label: 'Landfall' },
                { key: 'category', label: 'Cat', align: 'r' },
                { key: 'vmax', label: 'Vmax m/s', align: 'r', render: (r) => r.vmax?.toFixed(1) },
                { key: 'dp_hpa', label: 'Δp hPa', align: 'r', render: (r) => r.dp_hpa?.toFixed(0) },
                { key: 'rmax_km', label: 'Rmax km', align: 'r', render: (r) => r.rmax_km?.toFixed(0) },
                { key: 'rate', label: 'Rate /yr', align: 'r', render: (r) => r.rate.toExponential(2) },
              ] : [
                { key: 'name', label: 'Event' }, { key: 'mag', label: 'M', align: 'r', render: (r) => r.mag?.toFixed(1) },
                { key: 'source', label: 'Source' },
                { key: 'rate', label: 'Rate /yr', align: 'r', render: (r) => r.rate.toExponential(2) },
              ]} />
          ) : <Empty>Loading…</Empty>}
        </Card>
        <Card title="Vulnerability explorer" desc="Discretised damage distribution: mean, quartile and 5–95% bands (zero-one-inflated Beta, convolved with within-event hazard uncertainty)"
          tools={<>
            <select value={vuln.construction} onChange={(e) => setVuln({ ...vuln, construction: e.target.value })} aria-label="Construction">
              {Object.entries(meta?.construction_classes ?? {}).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            <select value={vuln.year_built} onChange={(e) => setVuln({ ...vuln, year_built: Number(e.target.value) })} aria-label="Year built">
              {[1930, 1960, 1985, 1998, 2010].map((y) => <option key={y} value={y}>built {y}</option>)}
            </select>
            {peril === 'TC' && <select value={vuln.roof_shape} onChange={(e) => setVuln({ ...vuln, roof_shape: e.target.value })} aria-label="Roof">
              {['gable', 'hip', 'flat'].map((r) => <option key={r} value={r}>{r} roof</option>)}
            </select>}
            {peril === 'TC' && <label className="check small"><input type="checkbox" checked={!!vuln.shutters} onChange={(e) => setVuln({ ...vuln, shutters: e.target.checked ? 1 : 0 })} />shutters</label>}
          </>}>
          {vulnOpt ? <Chart option={vulnOpt} height={360} ariaLabel="Vulnerability curve" /> : <Empty>Loading…</Empty>}
        </Card>
      </div>
    </div>
  )
}

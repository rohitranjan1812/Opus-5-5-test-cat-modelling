import { useMemo, useState } from 'react'
import type { Job } from '../api'
import type { EChartsOption } from 'echarts'
import { api } from '../api'
import type { Breakdown, PortfolioSummary } from '../api'
import Chart from '../components/Chart'
import MapView, { MapLegend } from '../components/MapView'
import type { Layer } from '../components/MapView'
import { Card, Download, Empty, JobStatus, NumberField, Seg, Tile } from '../components/ui'
import { money, moneyAxis, num } from '../format'
import { useApp, useFetch } from '../state'
import { BLUE, TOKENS, barItem, catAxis, rampCss, seqColor } from '../theme'

const YEAR_BANDS = ['<1950', '1950-74', '1975-94', '1995-01', '2002+']

interface LocCols { loc_id: string[]; acc_id: string[]; lat: number[]; lon: number[]; state: string[]; construction: string[]; occupancy: string[]; year_built: number[]; tiv: number[] }

function breakdownOption(mode: 'dark' | 'light', rows: Breakdown[], top = 12): EChartsOption {
  const t = TOKENS[mode]
  const r = rows.slice(0, top).reverse()
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => money(v as number) },
    grid: { left: 8, right: 56, top: 8, bottom: 8, containLabel: true },
    xAxis: { type: 'value', axisLabel: { color: t.text3, formatter: (v: number) => moneyAxis(v) }, splitLine: { lineStyle: { color: t.grid } } },
    yAxis: catAxis(mode, r.map((x) => x.key)),
    series: [{ type: 'bar', data: r.map((x) => x.tiv), barMaxWidth: 16, itemStyle: barItem(t.series[0], true),
      label: { show: true, position: 'right', color: t.text2, formatter: (p: { value: unknown }) => moneyAxis(p.value as number) } }],
  }
}

interface EnrichReport {
  scope: string; n_locations: number; n_scope: number; elapsed_s: number
  elevation?: { n: number; pixel_m: number; tiles: number; tiles_missing: number; source: string; share_lower_by_1m: number | null
    below_1m: number; below_msl: number; coarse_minus_measured_m: { n: number; mean?: number; median?: number; p10?: number; p90?: number } }
  footprints?: { matched: number; inside: number; snapped: number; none: number; match_rate: number | null; median_snap_m: number | null
    informative_height: number; stories_changed: number; stories_up: number; stories_down: number; tiles: number; source: string }
  flags?: { possible_offshore: number; examples: string[] }
  fetch?: { requests: number; cache_hits: number; bytes: number; errors: number }
}
interface Quality { report: EnrichReport; enriched_from: string; locations: { etopo_m?: (number | null)[]; ground_elev_m?: (number | null)[] } }

/** Building-scale ground elevation + mapped footprints for the selected portfolio (server job), and its QA report. */
function Enrichment({ portfolioId, enriched, onDone }: { portfolioId: string; enriched: boolean; onDone: (id: string) => void }) {
  const { mode } = useApp()
  const t = TOKENS[mode]
  const [scope, setScope] = useState<'coastal' | 'all'>('coastal')
  const [job, setJob] = useState<Job | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const q = useFetch(enriched ? () => api.get<Quality>(`/portfolios/${portfolioId}/quality`, { limit: 100000 }) : null, [portfolioId, enriched])
  const rep = q.data?.report

  async function run() {
    setErr(null)
    try {
      const j = await api.post<Job>(`/portfolios/${portfolioId}/enrich`, { scope })
      setJob(j)
      const done = await api.waitJob<{ portfolio: { id: string } }>(j, setJob, 600)
      onDone(done.result.portfolio.id)
    } catch (e) { setErr(String(e)) }
  }

  const hist = useMemo((): EChartsOption | null => {
    const L = q.data?.locations
    if (!L?.etopo_m || !L.ground_elev_m) return null
    // what the surge model assumed without measurement: the 2′ cell value floored at 1 m
    const d = L.ground_elev_m.map((g, i) => (g == null || L.etopo_m![i] == null ? null : Math.max(L.etopo_m![i] as number, 1) - g))
      .filter((x): x is number => x != null)
    if (!d.length) return null
    // uniform 1 m bins; the end bins are open (≤ −5, ≥ 9)
    const lo = -5
    const hi = 9
    const counts = new Array(hi - lo + 1).fill(0)
    for (const x of d) counts[Math.min(hi - lo, Math.max(0, Math.floor(x) - lo))]++
    const labels = counts.map((_, k) => (k === 0 ? `≤${lo + 1}` : k === counts.length - 1 ? `≥${hi}` : `${lo + k}…${lo + k + 1}`))
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: (p: unknown) => {
        const x = (p as { dataIndex: number; value: number }[])[0]
        return `${labels[x.dataIndex]} m<br/><b>${x.value}</b> locations`
      } },
      grid: { left: 8, right: 12, top: 12, bottom: 34, containLabel: true },
      xAxis: catAxis(mode, labels, { name: 'Coarse-DEM ground − measured ground (m)', nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.text3 } }),
      yAxis: { type: 'value', axisLabel: { color: t.text3 }, splitLine: { lineStyle: { color: t.grid } } },
      series: [{ type: 'bar', data: counts, barCategoryGap: '8%', itemStyle: barItem(t.series[0]) }],
    }
  }, [q.data, mode, t])

  const pct = (x: number | null | undefined) => (x == null ? '—' : `${(100 * x).toFixed(0)}%`)
  const running = job && (job.status === 'running' || job.status === 'queued')
  return (
    <Card title="Data quality & enrichment" desc={enriched
      ? 'Building-scale ground elevation (USGS 3DEP terrain tiles, ≈10 m) and mapped OpenStreetMap footprints were attached to this portfolio.'
      : 'Attach building-scale ground elevation (USGS 3DEP, ≈10 m) and mapped building footprints (OpenStreetMap: area, height, storeys) — surge depth then uses each building’s real ground, and the match doubles as a geocoding check.'}
      tools={!enriched && <div className="row" style={{ gap: 8 }}>
        <Seg value={scope} onChange={setScope} options={[{ value: 'coastal', label: 'Coastal' }, { value: 'all', label: 'All locations' }]} ariaLabel="Enrichment scope" />
        <button className="btn primary sm" onClick={run} disabled={!!running}>Enrich</button></div>}>
      {running && <JobStatus job={job} />}
      {err && <div className="error small">{err}</div>}
      {enriched && rep && (
        <div className="grid g3">
          <div className="col" style={{ gap: 10 }}>
            <div className="grid g2">
              <Tile label="Footprint match" value={pct(rep.footprints?.match_rate)}
                foot={rep.footprints ? `${num(rep.footprints.inside)} inside · ${num(rep.footprints.snapped)} snapped (median ${rep.footprints.median_snap_m?.toFixed(0) ?? '—'} m) · ${num(rep.footprints.none)} none` : ''} />
              <Tile label="Ground lower than 2′ DEM by >1 m" value={pct(rep.elevation?.share_lower_by_1m)}
                foot={rep.elevation ? `median ${rep.elevation.coarse_minus_measured_m.median?.toFixed(2) ?? '—'} m · ${num(rep.elevation.below_msl)} below MSL` : ''} />
              <Tile label="Storeys updated" value={num(rep.footprints?.stories_changed ?? 0)} foot={rep.footprints ? `${num(rep.footprints.informative_height)} footprints with a mapped height` : ''} />
              <Tile label="Possible offshore geocodes" value={num(rep.flags?.possible_offshore ?? 0)} foot="on water, no building within 35 m" />
            </div>
            <div className="small muted">{num(rep.n_scope)} of {num(rep.n_locations)} locations in scope ({rep.scope}) · {rep.elevation ? `${num(rep.elevation.tiles)} terrain tiles at ${rep.elevation.pixel_m.toFixed(1)} m/px` : ''}
              {rep.fetch ? ` · ${num(rep.fetch.requests)} requests, ${num(rep.fetch.cache_hits)} cache hits, ${(rep.fetch.bytes / 1e6).toFixed(0)} MB` : ''} · {rep.elapsed_s.toFixed(0)} s.
              {' '}Sources: USGS 3DEP via Terrain Tiles (AWS Open Data); © OpenStreetMap contributors, OpenMapTiles, OpenFreeMap.</div>
          </div>
          <div className="span2">{hist ? <Chart option={hist} height={250} ariaLabel="Ground elevation bias histogram" /> : <Empty>—</Empty>}</div>
        </div>
      )}
    </Card>
  )
}

export default function Exposure() {
  const { portfolios, portfolioId, setPortfolioId, refresh, mode } = useApp()
  const sum = useFetch(portfolioId ? () => api.get<PortfolioSummary>(`/portfolios/${portfolioId}`) : null, [portfolioId])
  const locs = useFetch(portfolioId ? () => api.get<LocCols>(`/portfolios/${portfolioId}/locations`) : null, [portfolioId])
  const [gen, setGen] = useState({ n_locations: 5000, seed: 11, states: '', commercial_share: 0.18, name: '' })
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const s = sum.data

  const layers = useMemo((): Layer[] => {
    const L = locs.data
    if (!L) return []
    const lt = L.tiv.map((v) => Math.log10(Math.max(v, 1)))
    const lo = Math.min(...lt)
    const hi = Math.max(...lt)
    return [{
      kind: 'points', lat: L.lat, lon: L.lon, opacity: 0.75,
      color: (i) => seqColor((lt[i] - lo) / Math.max(hi - lo, 1e-9), mode),
      radius: (i) => 2 + 3 * (lt[i] - lo) / Math.max(hi - lo, 1e-9),
      tooltip: (i) => `<b>${L.loc_id[i]}</b> · ${L.acc_id[i]}<br/>${L.construction[i]} · ${L.occupancy[i]} · ${L.year_built[i]}<br/>TIV ${money(L.tiv[i])}`,
    }]
  }, [locs.data, mode])

  const tivRange = useMemo(() => {
    const t = locs.data?.tiv
    if (!t?.length) return ['', '']
    return [money(Math.min(...t)), money(Math.max(...t))]
  }, [locs.data])

  async function createSynthetic() {
    setBusy('Generating…')
    setErr(null)
    try {
      const states = gen.states.split(/[\s,]+/).map((x) => x.trim().toUpperCase()).filter(Boolean)
      const p = await api.post<PortfolioSummary>('/portfolios/synthetic', {
        n_locations: gen.n_locations, seed: gen.seed, commercial_share: gen.commercial_share,
        states: states.length ? states : null, name: gen.name || null,
      })
      await refresh()
      setPortfolioId(p.id)
    } catch (e) { setErr(String(e)) } finally { setBusy(null) }
  }

  async function upload() {
    if (!file) return
    setBusy('Uploading…')
    setErr(null)
    try {
      const p = await api.upload(file, file.name.replace(/\.csv$/i, ''))
      await refresh()
      setPortfolioId(p.id)
    } catch (e) { setErr(String(e)) } finally { setBusy(null) }
  }

  async function remove() {
    if (!portfolioId || !confirm('Delete this portfolio?')) return
    await api.del(`/portfolios/${portfolioId}`)
    setPortfolioId(null)
    refresh()
  }

  const bounds = s ? [[s.bbox[1], s.bbox[0]], [s.bbox[3], s.bbox[2]]] as [[number, number], [number, number]] : null

  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>Exposure</h1>
          <p>Portfolios in an OED-inspired schema: site coordinates, coverage TIVs, construction/occupancy/age/height, terrain,
            Vs30, mitigation features, site and account financial terms. Upload a CSV or generate a synthetic national book.</p>
        </div>
        <div className="row">
          <select value={portfolioId ?? ''} onChange={(e) => setPortfolioId(e.target.value || null)} aria-label="Select portfolio">
            {!portfolios.length && <option value="">No portfolios</option>}
            {portfolios.map((p) => <option key={p.id} value={p.id}>{p.name} ({num(p.n_locations)})</option>)}
          </select>
          {portfolioId && <Download href={`/api/portfolios/${portfolioId}/locations?format=csv`}>Download CSV</Download>}
          {portfolioId && <button className="btn sm danger" onClick={remove}>Delete</button>}
        </div>
      </div>

      {s && (
        <div className="grid g4">
          <Tile label="Locations" value={num(s.n_locations)} foot={`${num(s.n_accounts)} accounts`} />
          <Tile label="Total insured value" value={money(s.tiv_total)} foot={`avg ${money(s.tiv_total / s.n_locations)} per site`} />
          <Tile label="Building / contents / BI" value={money(s.tiv_by_coverage.building)}
            foot={`contents ${money(s.tiv_by_coverage.contents)} · BI ${money(s.tiv_by_coverage.bi)}`} />
          <Tile label="States" value={num(s.by_state.length)} foot={`largest: ${s.by_state[0]?.key} ${money(s.by_state[0]?.tiv)}`} />
        </div>
      )}

      <div className="grid g3">
        <Card className="span2" title="Location map" desc="Colour and size encode total insured value (log scale)">
          {locs.data ? (
            <MapView layers={layers} height={480} bounds={bounds} fitKey={portfolioId ?? ''}>
              <MapLegend title="TIV per location" ramp={rampCss(mode === 'dark' ? [...BLUE].reverse().slice(0, 11) : BLUE.slice(2))} lo={tivRange[0]} hi={tivRange[1]} />
            </MapView>
          ) : <Empty>{portfolioId ? 'Loading locations…' : 'Select or create a portfolio'}</Empty>}
        </Card>
        <div className="col">
          <Card title="New synthetic portfolio" desc="Insured-value hubs across hurricane and earthquake zones">
            <div className="form-grid">
              <NumberField label="Locations" value={gen.n_locations} min={10} max={250000} step={1000} onChange={(v) => setGen({ ...gen, n_locations: v })} />
              <NumberField label="Seed" value={gen.seed} onChange={(v) => setGen({ ...gen, seed: v })} />
              <NumberField label="Commercial share" value={gen.commercial_share} step={0.05} min={0} max={1} onChange={(v) => setGen({ ...gen, commercial_share: v })} />
              <label className="field"><span>States (optional)</span>
                <input type="text" placeholder="FL, TX, CA" value={gen.states} onChange={(e) => setGen({ ...gen, states: e.target.value })} /></label>
            </div>
            <div className="row mt">
              <input type="text" placeholder="Name (optional)" value={gen.name} onChange={(e) => setGen({ ...gen, name: e.target.value })} style={{ flex: 1 }} />
              <button className="btn primary" disabled={!!busy} onClick={createSynthetic}>Generate</button>
            </div>
          </Card>
          <Card title="Upload CSV" desc="Required: lat, lon, tiv_building (OED aliases like Latitude, BuildingTIV accepted)">
            <div className="row">
              <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
              <button className="btn" disabled={!file || !!busy} onClick={upload}>Upload</button>
            </div>
          </Card>
          {busy && <div className="small muted">{busy}</div>}
          {err && <div className="small error">{err}</div>}
          {s?.warnings?.length ? <Card title="Validation warnings"><ul className="small sub">{s.warnings.map((w) => <li key={w}>{w}</li>)}</ul></Card> : null}
        </div>
      </div>

      {s && portfolioId && (
        <Enrichment portfolioId={portfolioId} enriched={!!(s.meta as { enrichment?: unknown }).enrichment}
          onDone={async (id) => { await refresh(); setPortfolioId(id) }} />
      )}

      {s && (
        <div className="grid g3">
          <Card title="TIV by state">{<Chart option={breakdownOption(mode, s.by_state)} height={320} />}</Card>
          <Card title="TIV by construction">{<Chart option={breakdownOption(mode, s.by_construction)} height={320} />}</Card>
          <Card title="TIV by year built">{<Chart option={breakdownOption(mode, YEAR_BANDS.map((k) => s.by_year_band.find((b) => b.key === k)).filter((b): b is Breakdown => !!b))} height={320} />}</Card>
        </div>
      )}
    </div>
  )
}

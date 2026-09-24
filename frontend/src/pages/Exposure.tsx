import { useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { api } from '../api'
import type { Breakdown, PortfolioSummary } from '../api'
import Chart from '../components/Chart'
import MapView, { MapLegend } from '../components/MapView'
import type { Layer } from '../components/MapView'
import { Card, Download, Empty, NumberField, Tile } from '../components/ui'
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

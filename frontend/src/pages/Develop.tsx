import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { api } from '../api'
import type { Job } from '../api'
import Chart from '../components/Chart'
import { Card, DataTable, Empty, JobStatus, Seg, Tile } from '../components/ui'
import { buildingLayers } from '../dev/buildings'
import { EMPTY_MATCH, MIN_MATCH_ZOOM, matchFootprints } from '../dev/footprints'
import type { MatchResult } from '../dev/footprints'
import { storeKey, storedKey } from '../dev/googleKey'
import type { GoogleMode } from '../dev/googleKey'
import Map3D, { refreshDrapes } from '../dev/Map3D'
import type { Drape, MapHandle } from '../dev/Map3D'
import { EQ_FLAGS, EqScene } from '../dev/eqScene'
import type { EqFlags } from '../dev/eqScene'
import { saffirSimpson } from '../dev/physics'
import { TC_FLAGS, TcScene } from '../dev/tcScene'
import type { TcFlags } from '../dev/tcScene'
import type { DevPayload, EqDev, Seismogram, TcDev } from '../dev/types'
import { money, moneyAxis, num } from '../format'
import { useApp, useFetch } from '../state'
import { DAMAGE, HEAT, RAIN, SLIP, TOKENS, WATER, lineSeries, logAxis, rampCss, valueAxis } from '../theme'

type Scene = TcScene | EqScene
interface Integrations { google_3d_tiles: { available: boolean; mode: string; setup: string | null } }
const DETAIL_ZOOM = 12.5 // above this, buildings are drawn as their footprints / markers instead of km-scale columns
// last development survives page switches (the payload is large and deterministic per request)
let LAST: { key: string; data: DevPayload } | null = null

const TC_SPEEDS = [{ value: '1', label: '1 h/s' }, { value: '3', label: '3 h/s' }, { value: '6', label: '6 h/s' }]
const EQ_SPEEDS = [{ value: '0.5', label: '½×' }, { value: '1', label: 'Real time' }, { value: '3', label: '3×' }]
const LAYER_LABELS: Record<string, string> = {
  wind: 'Gust field', footprint: 'Running max', particles: 'Flow tracers', surge: 'Surge & inundation', rain: 'Rainfall',
  buildings: 'Buildings', radii: 'Wind radii', labels: 'Cities', shaking: 'Shaking', fronts: 'P/S fronts',
  mesh: 'Wavefield mesh', fault: 'Fault plane',
}

const SUP: Record<string, string> = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻' }
const pow10 = (e: number) => `10${String(e).split('').map((c) => SUP[c] ?? c).join('')}`

function hoursLabel(t: number) {
  const s = t < 0 ? '−' : '+'
  const a = Math.abs(t)
  return `T${s}${Math.floor(a)}h${String(Math.round((a % 1) * 60)).padStart(2, '0')}`
}

function Legend({ title, ramp, lo, hi }: { title: string; ramp: string[]; lo: string; hi: string }) {
  return (
    <div className="leg">
      <div className="leg-title">{title}</div>
      <div className="leg-ramp" style={{ background: rampCss(ramp) }} />
      <div className="leg-ends"><span>{lo}</span><span>{hi}</span></div>
    </div>
  )
}

export default function Develop() {
  const { meta, portfolioId, portfolios, setPortfolioId, mode } = useApp()
  const tok = TOKENS[mode]
  const analogs = useMemo(() => Object.entries(meta?.analogs ?? {}), [meta])
  const [source, setSource] = useState<'analog' | 'catalog'>('analog')
  const [analog, setAnalog] = useState('ian_2022')
  const [peril, setPeril] = useState<'TC' | 'EQ'>('TC')
  const [eventId, setEventId] = useState('')
  const [seed, setSeed] = useState(1)
  const [surge, setSurge] = useState(true)
  const [job, setJob] = useState<Job | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [data, setData] = useState<DevPayload | null>(LAST?.data ?? null)
  const [exag, setExag] = useState(2)
  const [depthScale, setDepthScale] = useState(1.5)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState('3')
  const [tUi, setTUi] = useState(0)
  const [tcFlags, setTcFlags] = useState<TcFlags>(TC_FLAGS)
  const [eqFlags, setEqFlags] = useState<EqFlags>(EQ_FLAGS)
  const [probe, setProbe] = useState<[number, number] | null>(null)
  const [seis, setSeis] = useState<Seismogram | null>(null)
  const [seisErr, setSeisErr] = useState<string | null>(null)
  const [ready, setReady] = useState<MapHandle | null>(null)
  // building-level reference: OSM footprints + Google Photorealistic 3D Tiles
  const integ = useFetch(() => api.get<Integrations>('/integrations'), [])
  const [userKey, setUserKey] = useState<string | null>(() => storedKey())
  const [keyDraft, setKeyDraft] = useState('')
  const [googleOn, setGoogleOn] = useState(false)
  const [googleErr, setGoogleErr] = useState<string | null>(null)
  const [credits, setCredits] = useState('')
  const [streetMap, setStreetMap] = useState(true)
  const [zoom, setZoom] = useState(4)
  const [match, setMatch] = useState<MatchResult>(EMPTY_MATCH)
  const [sel, setSel] = useState<number | null>(null)
  const [groundZ, setGroundZ] = useState<number | null>(null)
  const googleMode = useMemo<GoogleMode | null>(() => (integ.data?.google_3d_tiles.available ? { kind: 'proxy' } : userKey ? { kind: 'key', key: userKey } : null), [integ.data, userKey])
  // the Google 3D Tiles stack (loaders.gl, geo-layers) is fetched only when photoreal mode is first switched on
  const [g3d, setG3d] = useState<typeof import('../dev/google3d') | null>(null)
  useEffect(() => { if (googleOn && !g3d) import('../dev/google3d').then(setG3d).catch((e) => setGoogleErr(String(e))) }, [googleOn, g3d])
  // a failed Google load falls back to the OSM rendering (the error stays on screen until toggled)
  const google = googleOn && googleMode != null && g3d != null && !googleErr
  const detail = google || zoom >= DETAIL_ZOOM

  const scene = useMemo<Scene | null>(() => (data ? (data.peril === 'TC' ? new TcScene(data, mode) : new EqScene(data, mode)) : null), [data, mode])
  const sites = useMemo(() => scene?.siteAccess() ?? null, [scene])
  const tRange = useMemo<[number, number]>(() => (data ? (data.peril === 'TC' ? [data.t0, data.t1] : [0, data.t_max]) : [0, 1]), [data])
  const tRef = useRef(0)
  const playRef = useRef(false)
  const speedRef = useRef(3)
  const hoverRef = useRef<number | null>(null)
  const dirty = useRef(true)
  useEffect(() => { playRef.current = playing }, [playing])
  useEffect(() => { speedRef.current = Number(speed) }, [speed])

  // new event: start just before landfall / at origin time; reset probes
  useEffect(() => {
    if (!data) return
    const t = data.peril === 'TC' ? Math.max(data.t0, -12) : 0
    tRef.current = t
    setTUi(t)
    setSpeed(data.peril === 'TC' ? '3' : '1')
    setProbe(null)
    setSeis(null)
    setSel(null)
    setMatch(EMPTY_MATCH)
    dirty.current = true
  }, [data])

  const run = useCallback(async () => {
    if (!portfolioId) return
    setErr(null)
    const body = source === 'analog' ? { portfolio_id: portfolioId, analog, seed, surge }
      : { portfolio_id: portfolioId, peril, event_id: Number(eventId), seed, surge }
    const key = JSON.stringify(body)
    if (LAST?.key === key) { setData(LAST.data); return }
    try {
      setPlaying(false)
      const j = await api.post<Job>('/develop', body)
      setJob(j)
      const done = await api.waitJob<DevPayload>(j, setJob, 300)
      LAST = { key, data: done.result }
      setData(done.result)
    } catch (e) { setErr(String(e)) }
  }, [portfolioId, source, analog, seed, surge, peril, eventId])

  // first visit: develop a default event so the view is never empty
  const booted = useRef(false)
  useEffect(() => {
    if (booted.current || data || !portfolioId || !meta) return
    booted.current = true
    if (!meta.analogs[analog]) setAnalog(Object.keys(meta.analogs)[0])
    else run()
  }, [portfolioId, meta, data, analog, run])

  const drapes = useMemo<Drape[]>(() => {
    if (!scene) return []
    if (scene instanceof TcScene) {
      const f = tcFlags
      const out: Drape[] = []
      const on = !google && zoom < 14 // km-scale fields say nothing at street scale; the flood plane does
      if (scene.rainC) out.push({ id: 'drape-rain', canvas: scene.rainC.canvas, bbox: scene.rainC.bbox, opacity: 0.9, visible: on && f.rain })
      out.push({ id: 'drape-wind', canvas: scene.wind.canvas, bbox: scene.wind.bbox, opacity: 0.88, visible: on && f.wind })
      if (scene.surgeC) out.push({ id: 'drape-surge', canvas: scene.surgeC.canvas, bbox: scene.surgeC.bbox, opacity: 1, visible: on && f.surge })
      out.push({ id: 'drape-particles', canvas: scene.parts.canvas, bbox: scene.parts.bbox, opacity: 0.9, visible: on && f.particles })
      return out
    }
    return [{ id: 'drape-shake', canvas: scene.c.canvas, bbox: scene.c.bbox, opacity: zoom < 14 ? 1 : 0.35, visible: !google }]
  }, [scene, tcFlags, google, zoom])

  // camera focus: the landfall window of the track, or the rupture plus the strong-shaking reach
  const focus = useMemo<[number, number, number, number] | null>(() => {
    if (!data) return null
    if (data.peril === 'TC') {
      const tr = data.track
      const idx = tr.t.map((t, i) => [t, i]).filter(([t]) => t >= -9 && t <= 9).map(([, i]) => i)
      const la = idx.map((i) => tr.lat[i])
      const lo = idx.map((i) => tr.lon[i])
      if (!la.length) return data.bbox
      return [Math.min(...lo) - 2.6, Math.min(...la) - 2.0, Math.max(...lo) + 2.6, Math.max(...la) + 2.0]
    }
    const r = data.fault.ring
    const pad = 0.35 + 0.25 * Math.max(0, data.fault.Mw - 6)
    return [Math.min(...r.lon) - pad * 1.3, Math.min(...r.lat) - pad, Math.max(...r.lon) + pad * 1.3, Math.max(...r.lat) + pad]
  }, [data])

  const onHover = useCallback((i: number) => { hoverRef.current = i >= 0 ? i : null }, [])

  // ------------------------------------------------------------------ render loop
  useEffect(() => { dirty.current = true }, [tcFlags, eqFlags, exag, probe, depthScale, ready, google, detail, match, sel, groundZ])
  const selectSite = useCallback((i: number) => { setSel(i); setGroundZ(null) }, [])
  useEffect(() => {
    if (!scene || !ready) return
    let raf = 0
    let last = performance.now()
    let lastUi = 0
    let lastT = NaN
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      if (now - last < 30) return // ≈30 fps is plenty for the physics; halves texture-upload bandwidth
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      if (playRef.current) {
        tRef.current += dt * speedRef.current
        if (tRef.current >= tRange[1]) { tRef.current = tRange[1]; playRef.current = false; setPlaying(false) }
      }
      const t = tRef.current
      const changed = t !== lastT || dirty.current
      const tm = (window as unknown as { catforge3d?: { timings: Record<string, number> } }).catforge3d?.timings ?? {}
      const time = (k: string, fn: () => void) => { const a = performance.now(); fn(); tm[k] = 0.8 * (tm[k] ?? 0) + 0.2 * (performance.now() - a) }
      const touched: string[] = []
      if (scene instanceof TcScene) {
        if (changed) {
          if (tcFlags.wind) { time('wind', () => scene.renderWind(t, tcFlags.footprint)); touched.push('drape-wind') }
          if (tcFlags.surge) { time('surge', () => scene.renderSurge(t)); touched.push('drape-surge') }
          if (tcFlags.rain) { time('rain', () => scene.renderRain(t)); touched.push('drape-rain') }
        }
        if (tcFlags.particles) { time('particles', () => scene.stepParticles(t)); touched.push('drape-particles') }
      } else if (changed) { time('ground', () => scene.render(t, eqFlags)); touched.push('drape-shake') }
      if (touched.length) refreshDrapes(ready, touched)
      if (changed) {
        const ex = google ? 1 : exag
        const regional = scene instanceof TcScene
          ? scene.layers(t, { ...tcFlags, buildings: tcFlags.buildings && !detail }, ex, probe, onHover)
          : scene.layers(t, { ...eqFlags, buildings: eqFlags.buildings && !detail }, ex, probe, onHover, depthScale)
        const meshDrapes = !google || !g3d ? [] : scene instanceof TcScene
          ? [...(tcFlags.rain && scene.rainC ? [g3d.drapeOnMesh('mesh-rain', scene.rainC, 0.75)] : []),
            ...(tcFlags.wind ? [g3d.drapeOnMesh('mesh-wind', scene.wind, 0.55)] : []),
            ...(tcFlags.surge && scene.surgeC ? [g3d.drapeOnMesh('mesh-surge', scene.surgeC, 0.8)] : [])]
          : [g3d.drapeOnMesh('mesh-shake', scene.c, 0.6)]
        const bldg = detail && sites && ((scene instanceof TcScene && tcFlags.buildings) || (scene instanceof EqScene && eqFlags.buildings))
          ? buildingLayers({ sites, t, mode, google, match, selected: sel, groundZ, exag: ex, onSelect: selectSite }) : []
        ready.overlay.setProps({
          layers: [...(google && googleMode && g3d ? [g3d.googleTilesLayer(googleMode, setCredits, setGoogleErr)] : []), ...meshDrapes, ...regional, ...bldg],
          onError: (e: Error, layer?: { id: string } | null) => { if (layer?.id === 'google-3d') setGoogleErr(String(e?.message ?? e)) },
          onClick: (info: { layer?: { id: string } | null; index: number }) => { if (info.layer?.id === 'sites' && info.index >= 0) selectSite(info.index) },
          getTooltip: ({ layer, index, object }: { layer?: { id: string } | null; index: number; object?: unknown }) => {
            if (!layer || index < 0) return null
            const tt = tRef.current
            if (layer.id === 'sites' && scene instanceof TcScene && scene.p.sites) {
              const S = scene.p.sites
              const v = scene.siteAt(index, tt)
              return { html: `<b>${S.loc_id[index]}</b> · ${S.construction[index]}<br/>TIV ${money(S.tiv[index])}<br/>Gust now ${v.gust.toFixed(1)} m/s` +
                `<br/>Inundation ${v.depth.toFixed(2)} m<br/>Damage ratio ${(100 * v.damage).toFixed(1)}%`, className: 'deck-tip' }
            }
            if (layer.id === 'sites' && scene instanceof EqScene && scene.p.sites) {
              const S = scene.p.sites
              return { html: `<b>${S.loc_id[index]}</b> · ${S.construction[index]} · Vs30 ${S.vs30[index]}<br/>PGA ${S.pga[index].toFixed(3)} g (median ${S.pga_median[index].toFixed(3)})` +
                `<br/>S arrival ${S.t_s[index].toFixed(1)} s<br/>Damage ${(100 * scene.siteDamage(index, tt)).toFixed(1)}% → ${(100 * S.damage[index]).toFixed(1)}%<br/>GU loss ${money(S.gu[index])}`, className: 'deck-tip' }
            }
            if ((layer.id === 'exposed-footprints' || layer.id === 'exposed-points') && sites) {
              const i = layer.id === 'exposed-points' ? (object as number) : (object as { i: number }).i
              const f = match.byIndex.get(i)
              return { html: `<b>${sites.id[i]}</b> · ${sites.cls[i]}<br/>TIV ${money(sites.tiv[i])}<br/>${sites.intensity(i, tt).text}` +
                `<br/>Damage now ${(100 * sites.damage(i, tt)).toFixed(1)}% · final loss ${money(sites.finalLoss[i])}` +
                (f ? `<br/>Mapped footprint: ${f.inside ? 'inside' : `${f.snapM.toFixed(0)} m away`} · ${f.height.toFixed(0)} m tall` : '<br/>No mapped footprint within 35 m'), className: 'deck-tip' }
            }
            if (layer.id === 'fault' && scene instanceof EqScene) {
              const s = scene.subs[index]
              return { html: `Sub-fault (${s.i}, ${s.j})<br/>Slip ${s.slip.toFixed(2)} m<br/>Rupture time ${s.tr.toFixed(2)} s`, className: 'deck-tip' }
            }
            return null
          },
        } as never)
        tm.layers_ms = performance.now() - now
      }
      lastT = t
      dirty.current = false
      if (now - lastUi > 110) { lastUi = now; setTUi(t) }
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [scene, ready, tcFlags, eqFlags, exag, probe, depthScale, tRange, onHover, google, googleMode, g3d, detail, sites, match, sel, groundZ, mode, selectSite])

  // ------------------------------------------------------------------ buildings: footprint matching, street view
  const matchSig = useRef('')
  const onSettled = useCallback(() => {
    const m = ready?.map
    if (!m || !sites) return
    const b = m.getBounds()
    const sig = [m.getZoom().toFixed(2), b.getWest().toFixed(4), b.getSouth().toFixed(4), b.getEast().toFixed(4), b.getNorth().toFixed(4), streetMap].join()
    if (sig === matchSig.current) return
    matchSig.current = sig
    setMatch(streetMap ? matchFootprints(m, sites) : EMPTY_MATCH)
  }, [ready, sites, streetMap])
  useEffect(() => { matchSig.current = '' }, [sites, streetMap])
  const onView = useCallback((z: number) => setZoom(Math.round(z * 4) / 4), [])

  const flyToSite = useCallback((i: number) => {
    if (!ready || !sites) return
    selectSite(i)
    if (!google) setExag(1)
    ready.map.flyTo({ center: [sites.lon[i], sites.lat[i]], zoom: google ? 17.6 : 16.4, pitch: 62, bearing: ready.map.getBearing(), duration: 2800, essential: true })
  }, [ready, sites, google, selectSite])
  const backToEvent = useCallback(() => {
    if (!ready || !focus) return
    ready.map.fitBounds([[focus[0], focus[1]], [focus[2], focus[3]]], { padding: 40, pitch: 52, bearing: -10, duration: 2200 })
  }, [ready, focus])

  // ground under the selected building on the photoreal mesh: re-sampled as finer tiles stream in
  useEffect(() => {
    if (!google || !g3d || sel == null || !ready || !sites || zoom < 15) return
    const timers = [900, 2500, 5000, 9000].map((ms) => window.setTimeout(() => {
      const z = g3d.sampleGround(ready.overlay, ready.map, sites.lon[sel], sites.lat[sel])
      if (z != null) setGroundZ(z)
    }, ms))
    return () => timers.forEach((x) => window.clearTimeout(x))
  }, [google, g3d, sel, ready, sites, zoom])
  useEffect(() => { setGoogleErr(null); setCredits('') }, [googleMode, googleOn])

  const exposed = useMemo(() => {
    if (!sites) return []
    return Array.from({ length: sites.n }, (_, i) => i).sort((a, b) => sites.finalLoss[b] - sites.finalLoss[a]).slice(0, 40)
      .map((i) => ({ i, id: sites.id[i], cls: sites.cls[i], tiv: sites.tiv[i], dmg: sites.finalDamage[i], loss: sites.finalLoss[i] }))
  }, [sites])

  const seek = (t: number) => { tRef.current = t; setTUi(t) }
  const togglePlay = () => {
    if (!playing && tRef.current >= tRange[1] - 1e-6) seek(tRange[0])
    setPlaying(!playing)
  }

  // ------------------------------------------------------------------ probes
  const onMapClick = useCallback((lat: number, lon: number) => {
    setProbe([lat, lon])
    if (data?.peril === 'EQ') {
      setSeis(null)
      setSeisErr(null)
      const body = source === 'analog' ? { analog, seed, lat, lon } : { peril: 'EQ', event_id: Number(eventId), seed, lat, lon }
      api.post<Seismogram>('/develop/seismogram', body).then(setSeis).catch((e) => setSeisErr(String(e)))
    }
  }, [data, source, analog, eventId, seed])

  const meteo = useMemo(() => (probe && scene instanceof TcScene ? scene.meteogram(probe[0], probe[1]) : null), [probe, scene])
  const parity = useMemo(() => (scene instanceof TcScene ? scene.parity() : null), [scene])

  // ------------------------------------------------------------------ charts (time cursor at ~9 Hz)
  const tKey = Math.round(tUi * (data?.peril === 'TC' ? 4 : 5)) / (data?.peril === 'TC' ? 4 : 5)
  const cursor = useMemo(() => ({ silent: true, symbol: 'none', lineStyle: { color: tok.text2, type: 'solid', width: 1 }, label: { show: false }, data: [{ xAxis: tKey }] }), [tKey, tok])

  const lossOpt = useMemo((): EChartsOption | null => {
    if (!data?.loss) return null
    const L = data.loss
    const series = [lineSeries('Ground-up loss', tok.series[0], L.t.map((x, i) => [x, L.gu[i]]), { markLine: cursor })]
    if (data.peril === 'TC') series.push(lineSeries('Wind only', tok.series[2], L.t.map((x, i) => [x, (data as TcDev).loss!.gu_wind_only[i]]), { lineStyle: { width: 1.5, type: 'dashed', color: tok.series[2] } }))
    return {
      legend: data.peril === 'TC' ? { data: ['Ground-up loss', 'Wind only'] } : undefined,
      tooltip: { trigger: 'axis', valueFormatter: (v: unknown) => money(Number(v)) },
      grid: { left: 12, right: 18, top: data.peril === 'TC' ? 34 : 14, bottom: 26, containLabel: true },
      xAxis: valueAxis(mode, { name: data.peril === 'TC' ? 'Hours from landfall' : 'Seconds from origin', nameLocation: 'middle', nameGap: 26,
        min: L.t[0], max: data.peril === 'TC' ? L.t[L.t.length - 1] : Math.ceil(L.t[L.t.length - 1] / 10) * 10, axisLabel: { color: tok.text3 } }),
      yAxis: valueAxis(mode),
      series,
    }
  }, [data, tok, mode, cursor])

  const stfOpt = useMemo((): EChartsOption | null => {
    if (!(scene instanceof EqScene)) return null
    const s = scene.stf
    const ex = Math.floor(Math.log10(Math.max(...s.rate, 1)))
    const u = 10 ** ex
    return {
      tooltip: { trigger: 'axis', valueFormatter: (v: unknown) => `${Number(v).toFixed(2)} × ${pow10(ex)} N·m/s` },
      grid: { left: 12, right: 18, top: 26, bottom: 26, containLabel: true },
      xAxis: valueAxis(mode, { name: 'Seconds from origin', nameLocation: 'middle', nameGap: 26, max: Math.ceil(s.t[s.t.length - 1]), axisLabel: { color: tok.text3 } }),
      yAxis: valueAxis(mode, { name: `Moment rate (× ${pow10(ex)} N·m/s)`, nameTextStyle: { color: tok.text3, align: 'left' }, splitNumber: 3, axisLabel: { color: tok.text3, formatter: (v: number) => `${v}` } }),
      series: [lineSeries('Moment rate', tok.series[6], s.t.map((x, i) => [x, +(s.rate[i] / u).toFixed(3)]), { areaStyle: { color: tok.series[6], opacity: 0.15 }, markLine: cursor })],
    }
  }, [scene, tok, mode, cursor])

  const meteoOpt = useMemo((): EChartsOption | null => {
    if (!meteo) return null
    const hasS = !!meteo.surge
    const hasR = !!meteo.rain
    const panels = 2 + (hasS ? 1 : 0) + (hasR ? 1 : 0)
    const h = 100 / panels
    const grid = Array.from({ length: panels }, (_, k) => ({ left: 56, right: 14, top: `${k * h + 4}%`, height: `${h - 10}%` }))
    const xa = grid.map((_, k) => ({ ...valueAxis(mode, { axisLabel: { color: tok.text3, show: k === panels - 1 } }), gridIndex: k, min: meteo.t[0], max: meteo.t[meteo.t.length - 1] }))
    const names = ['Wind (m/s)', 'Pressure (hPa)', ...(hasS ? ['Water level (m)'] : []), ...(hasR ? ['Rain (mm)'] : [])]
    const ya = names.map((n, k) => ({ ...valueAxis(mode, { axisLabel: { color: tok.text3, formatter: (v: number) => `${v}` }, scale: true, splitNumber: 2 }), gridIndex: k, name: n, nameTextStyle: { color: tok.text2, align: 'left' }, nameGap: 6 }))
    const series = [
      lineSeries('3-s gust', tok.series[1], meteo.t.map((x, i) => [x, +meteo.gust[i].toFixed(1)]), { xAxisIndex: 0, yAxisIndex: 0, markLine: cursor }),
      lineSeries('1-min sustained', tok.series[0], meteo.t.map((x, i) => [x, +meteo.sustained[i].toFixed(1)]), { xAxisIndex: 0, yAxisIndex: 0, lineStyle: { width: 1.5, color: tok.series[0] } }),
      lineSeries('Surface pressure', tok.series[4], meteo.t.map((x, i) => [x, +meteo.pressure[i].toFixed(1)]), { xAxisIndex: 1, yAxisIndex: 1, markLine: cursor }),
    ]
    let k = 2
    if (hasS) {
      series.push(lineSeries('Water level', tok.series[0], meteo.surge!.t.map((x, i) => [x, Number.isFinite(meteo.surge!.level[i]) ? +meteo.surge!.level[i].toFixed(2) : null]),
        { xAxisIndex: k, yAxisIndex: k, connectNulls: false, markLine: cursor, areaStyle: { color: tok.series[0], opacity: 0.12 } }))
      k++
    }
    if (hasR) series.push(lineSeries('Rain accumulation', tok.series[2], meteo.rain!.t.map((x, i) => [x, +(meteo.rain!.mm[i] || 0).toFixed(0)]), { xAxisIndex: k, yAxisIndex: k, markLine: cursor }))
    return { legend: { show: false }, tooltip: { trigger: 'axis' }, axisPointer: { link: [{ xAxisIndex: 'all' }] }, grid, xAxis: xa, yAxis: ya, series } as EChartsOption
  }, [meteo, tok, mode, cursor])

  const seisOpt = useMemo((): EChartsOption | null => {
    if (!seis) return null
    const t = seis.acc_g.map((_, i) => +(i * seis.dt).toFixed(3))
    const mk = { silent: true, symbol: 'none', label: { show: true, color: tok.text2, formatter: '{b}', position: 'insideEndTop' },
      data: [{ name: 'P', xAxis: seis.t_p, lineStyle: { color: tok.series[0], type: 'dashed' } }, { name: 'S', xAxis: seis.t_s, lineStyle: { color: tok.series[1], type: 'dashed' } },
        { name: '', xAxis: tKey, lineStyle: { color: tok.text2, type: 'solid' } }] }
    const grid = [{ left: 56, right: 14, top: '6%', height: '38%' }, { left: 56, right: 14, top: '56%', height: '34%' }]
    return {
      legend: { show: false }, tooltip: { trigger: 'axis' }, axisPointer: { link: [{ xAxisIndex: 'all' }] }, grid,
      xAxis: [0, 1].map((k) => ({ ...valueAxis(mode, { axisLabel: { color: tok.text3, show: k === 1 } }), gridIndex: k, max: t[t.length - 1] })),
      yAxis: ['Acceleration (g)', 'Velocity (cm/s)'].map((n, k) => ({ ...valueAxis(mode, { axisLabel: { color: tok.text3, formatter: (v: number) => `${v}` }, splitNumber: 2 }), gridIndex: k, name: n, nameTextStyle: { color: tok.text2, align: 'left' }, nameGap: 6 })),
      series: [
        lineSeries('Acceleration', tok.series[1], t.map((x, i) => [x, seis.acc_g[i]]), { xAxisIndex: 0, yAxisIndex: 0, lineStyle: { width: 1, color: tok.series[1] }, markLine: mk, sampling: 'lttb' }),
        lineSeries('Velocity', tok.series[0], t.map((x, i) => [x, seis.vel_cms[i]]), { xAxisIndex: 1, yAxisIndex: 1, lineStyle: { width: 1, color: tok.series[0] }, markLine: mk, sampling: 'lttb' }),
      ],
    } as EChartsOption
  }, [seis, tok, mode, tKey])

  const psaOpt = useMemo((): EChartsOption | null => {
    if (!seis) return null
    return {
      legend: { data: ['Simulated PSA (5%)', 'GMPE PGA ±1σ'] },
      tooltip: { trigger: 'item', formatter: (p: unknown) => { const q = p as { value: number[] }; return `T = ${q.value[0]} s<br/><b>${q.value[1].toFixed(3)} g</b>` } },
      grid: { left: 12, right: 18, top: 34, bottom: 30, containLabel: true },
      xAxis: logAxis(mode, { name: 'Period (s)', min: 0.01, max: 10 }),
      yAxis: logAxis(mode, { nameLocation: 'end', name: 'g', nameGap: 8 }),
      series: [
        lineSeries('Simulated PSA (5%)', tok.series[1], seis.periods.map((p, i) => [p, seis.psa_g[i]])),
        { name: 'GMPE PGA ±1σ', type: 'line', data: [[0.01, seis.gmpe_lo_g], [0.01, seis.gmpe_hi_g]], lineStyle: { color: tok.series[0], width: 3 }, symbol: 'none' },
        { name: 'GMPE PGA ±1σ', type: 'scatter', data: [[0.01, seis.gmpe_median_g]], symbolSize: 9, itemStyle: { color: tok.series[0], borderColor: tok.surface, borderWidth: 2 } },
        { name: 'Simulated PGA', type: 'scatter', data: [[0.01, seis.pga_g]], symbol: 'diamond', symbolSize: 10, itemStyle: { color: tok.series[1], borderColor: tok.surface, borderWidth: 2 } },
      ],
    } as EChartsOption
  }, [seis, tok, mode])

  // ------------------------------------------------------------------ HUD
  const hud = useMemo(() => {
    if (!scene) return null
    if (scene instanceof TcScene) {
      const s = scene.storm(tUi)
      const cat = saffirSimpson(s.vmax)
      const radii = tcFlags.radii ? scene.windRadii(tUi) : []
      return (
        <>
          <div className="hud-time num">{hoursLabel(tUi)}</div>
          <div className="hud-line num">{cat ? `Category ${cat}` : s.vmax >= 17.5 ? 'Tropical storm' : 'Depression'} · V<sub>max</sub> {s.vmax.toFixed(0)} m/s · p<sub>c</sub> {(1013 - s.dp / 100).toFixed(0)} hPa</div>
          <div className="hud-line num">R<sub>max</sub> {s.rm.toFixed(0)} km · B {s.b.toFixed(2)} · heading {s.hdg.toFixed(0)}° at {(s.vt * 1.944).toFixed(0)} kt</div>
          {radii.length > 0 && (
            <table className="hud-radii num">
              <thead><tr><th>kt</th><th>NE</th><th>SE</th><th>SW</th><th>NW</th></tr></thead>
              <tbody>{radii.map((r) => <tr key={r.kt}><td>{r.kt}</td>{r.nm.map((v, i) => <td key={i}>{v > 0 ? v.toFixed(0) : '—'}</td>)}</tr>)}</tbody>
            </table>
          )}
          {radii.length > 0 && <div className="hud-note">Wind radii (nm), NHC quadrant convention</div>}
        </>
      )
    }
    const F = scene.p.fault
    const m0 = scene.momentAt(tUi)
    const mw = m0 > 0 ? (Math.log10(m0) - 9.05) / 1.5 : null
    return (
      <>
        <div className="hud-time num">t = {tUi.toFixed(1)} s</div>
        <div className="hud-line num">Rupture {(100 * scene.rupturedFraction(tUi)).toFixed(0)}% of {F.L_km.toFixed(0)}×{F.W_km.toFixed(0)} km · V<sub>r</sub> {F.vr_kms.toFixed(1)} km/s</div>
        <div className="hud-line num">Moment released {(100 * m0 / F.M0).toFixed(0)}%{mw ? ` · M${' '}w(t) ${mw.toFixed(2)}` : ''} of {F.Mw.toFixed(1)}</div>
        <div className="hud-line num">S front ≈ {(scene.p.waves.beta_kms * tUi).toFixed(0)} km from hypocentre · P ≈ {(scene.p.waves.alpha_kms * tUi).toFixed(0)} km</div>
      </>
    )
  }, [scene, tUi, tcFlags.radii])

  const flags = data?.peril === 'EQ' ? eqFlags : tcFlags
  const setFlag = (k: string, v: boolean) => {
    if (data?.peril === 'EQ') setEqFlags({ ...eqFlags, [k]: v })
    else setTcFlags({ ...tcFlags, [k]: v })
  }
  const running = job && (job.status === 'running' || job.status === 'queued')
  const tc = data?.peril === 'TC' ? (data as TcDev) : null
  const eq = data?.peril === 'EQ' ? (data as EqDev) : null
  const surgeOk = tc?.surge && !('error' in tc.surge) ? tc.surge : null

  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>Event development · 3-D</h1>
          <p>One physically consistent realization of an event, resolved in time over ETOPO1 terrain and bathymetry: the
            analytic Holland wind field (evaluated in your browser with the engine&apos;s own equations), a 2-D shallow-water
            storm surge with wetting/drying, R-CLIPER rain — or a kinematic finite-fault rupture with P/S wavefronts — and
            every building&apos;s damage as the hazard reaches it.</p>
        </div>
        <label className="row small sub">Portfolio
          <select value={portfolioId ?? ''} onChange={(e) => setPortfolioId(e.target.value)}>
            {portfolios.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select></label>
      </div>

      <Card>
        <div className="row wrap" style={{ gap: 12, alignItems: 'flex-end' }}>
          <Seg value={source} onChange={setSource} ariaLabel="Event source"
            options={[{ value: 'analog', label: 'Historical analog' }, { value: 'catalog', label: 'Stochastic event' }]} />
          {source === 'analog' ? (
            <label className="field" style={{ minWidth: 320 }}><span>Analog</span>
              <select value={analog} onChange={(e) => setAnalog(e.target.value)}>
                {['TC', 'EQ'].map((p) => (
                  <optgroup key={p} label={p === 'TC' ? 'Hurricanes' : 'Earthquakes'}>
                    {analogs.filter(([, a]) => a.peril === p).map(([k, a]) => <option key={k} value={k}>{a.label}</option>)}
                  </optgroup>
                ))}
              </select></label>
          ) : (
            <>
              <label className="field"><span>Peril</span>
                <select value={peril} onChange={(e) => setPeril(e.target.value as 'TC' | 'EQ')}>
                  <option value="TC">Hurricane</option><option value="EQ">Earthquake</option>
                </select></label>
              <label className="field"><span>Catalog event id</span>
                <input value={eventId} onChange={(e) => setEventId(e.target.value.replace(/[^0-9]/g, ''))} placeholder="e.g. from Hazard page" /></label>
            </>
          )}
          <label className="field" style={{ width: 110 }}><span>Realization seed</span>
            <input type="number" value={seed} min={1} onChange={(e) => setSeed(Math.max(1, Number(e.target.value) || 1))} /></label>
          <label className="row small" style={{ gap: 6, paddingBottom: 8 }}>
            <input type="checkbox" checked={surge} onChange={(e) => setSurge(e.target.checked)} /> 2-D surge model</label>
          <button className="btn primary" onClick={run} disabled={!!running || !portfolioId || (source === 'catalog' && !eventId)}>▶ Develop event</button>
          <div style={{ flex: 1, minWidth: 220 }}><JobStatus job={running ? job : null} /></div>
        </div>
        {err && <div className="error small mt">{err}</div>}
      </Card>

      <div className="grid g3">
        <Card className="span2" title={data?.name ?? 'Event'} desc={data ? (data.peril === 'TC'
          ? 'Drag to pan · right-drag / ctrl-drag to tilt and rotate · click the map for a site meteogram'
          : 'Fault plane shown below ground (x-ray) · click the map for a synthetic seismogram at that site') : undefined}
          tools={
            <div className="row small wrap" style={{ gap: 14 }}>
              <label className={`chip ${streetMap ? 'on' : ''}`} title="OpenStreetMap roads, water and building footprints (OpenFreeMap)">
                <input type="checkbox" checked={streetMap} onChange={(e) => setStreetMap(e.target.checked)} />Street map</label>
              <label className={`chip ${google ? 'on' : ''}`} title={googleMode ? 'Google Photorealistic 3D Tiles' : 'Configure a Google Maps key in the Building view panel'}
                style={{ opacity: googleMode ? 1 : 0.55 }}>
                <input type="checkbox" checked={googleOn && googleMode != null} disabled={!googleMode} onChange={(e) => setGoogleOn(e.target.checked)} />Google 3D</label>
              <label className="row" style={{ gap: 6, opacity: google ? 0.45 : 1 }}>Terrain ×{exag.toFixed(1)}
                <input type="range" min={1} max={6} step={0.5} value={exag} disabled={google} onChange={(e) => setExag(Number(e.target.value))} aria-label="Terrain exaggeration" /></label>
              {eq && <label className="row" style={{ gap: 6 }}>Depth ×{depthScale.toFixed(1)}
                <input type="range" min={1} max={4} step={0.5} value={depthScale} onChange={(e) => setDepthScale(Number(e.target.value))} aria-label="Fault depth scale" /></label>}
            </div>
          }>
          <div className="row wrap small" style={{ gap: 6, marginBottom: 10 }}>
            {Object.keys(flags).map((k) => (
              <label key={k} className={`chip ${(flags as unknown as Record<string, boolean>)[k] ? 'on' : ''}`}>
                <input type="checkbox" checked={(flags as unknown as Record<string, boolean>)[k]} onChange={(e) => setFlag(k, e.target.checked)} />
                {LAYER_LABELS[k] ?? k}
              </label>
            ))}
          </div>
          <Map3D mode={mode} bbox={focus} fitKey={data ? `${data.name}-${data.seed}` : ''} exaggeration={exag} terrain={!google}
            streetMap={streetMap} drapes={drapes} onReady={setReady} onClick={onMapClick} onView={onView} onSettled={onSettled} height={620}>
            {google && <div className="g3d-attrib"><b>Google</b>{credits ? ` · ${credits}` : ''}</div>}
            {googleOn && googleErr && <div className="g3d-error">Google 3D Tiles failed: {googleErr} — showing OpenStreetMap buildings instead.</div>}
            {scene && <div className="hud">{hud}</div>}
            {data && (
              <div className="legends">
                {tc && tcFlags.wind && <Legend title={tcFlags.footprint ? 'Max 3-s gust so far (m/s)' : '3-s gust, this realization (m/s)'} ramp={HEAT} lo="15" hi="80" />}
                {tc && tcFlags.surge && surgeOk && <Legend title="Surge / inundation depth (m)" ramp={WATER} lo="0.2" hi="5" />}
                {tc && tcFlags.rain && <Legend title="Rain accumulation (mm, log)" ramp={RAIN} lo="5" hi="700" />}
                {eq && (eqFlags.shaking || eqFlags.footprint) && <Legend title="PGA (g, log) · instantaneous / peak" ramp={HEAT} lo="0.01" hi="1.5" />}
                {eq && eqFlags.fault && <Legend title="Coseismic slip (m)" ramp={SLIP} lo="0" hi={eq.fault.D_max.toFixed(1)} />}
                {data.sites && ((tc && tcFlags.buildings) || (eq && eqFlags.buildings)) && <Legend title="Building damage ratio (√ scale)" ramp={DAMAGE} lo="0%" hi="100%" />}
                {eq && eqFlags.fronts && <div className="leg row small" style={{ gap: 10 }}><span><i className="sw" style={{ background: mode === 'dark' ? '#78d2ff' : '#006ec8' }} /> P front</span><span><i className="sw" style={{ background: mode === 'dark' ? '#fff' : '#1e1e1e' }} /> S front</span></div>}
              </div>
            )}
            {data && (
              <div className="timebar">
                <button className="btn sm" onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>{playing ? '❚❚' : '▶'}</button>
                <input type="range" min={tRange[0]} max={tRange[1]} step={data.peril === 'TC' ? 0.05 : 0.05} value={tUi}
                  onChange={(e) => { setPlaying(false); seek(Number(e.target.value)) }} aria-label="Time" style={{ flex: 1 }} />
                <span className="num small" style={{ minWidth: 76, textAlign: 'right' }}>{data.peril === 'TC' ? hoursLabel(tUi) : `${tUi.toFixed(1)} s`}</span>
                <Seg value={speed} onChange={setSpeed} options={data.peril === 'TC' ? TC_SPEEDS : EQ_SPEEDS} ariaLabel="Playback speed" />
              </div>
            )}
            {!data && <div className="map-empty">{running ? 'Simulating…' : 'Choose an event and press Develop.'}</div>}
          </Map3D>
          {parity && (
            <div className="small muted mt">
              Physics parity — browser wind field vs server at {parity.n} probe points: max |Δgust| = {parity.maxAbs.toFixed(3)} m/s.
              Realization: η = {data!.realization.eta.toFixed(3)}, σ<sub>w</sub> = {data!.realization.sigma_w.toFixed(2)}, Matérn ν = {data!.realization.grf.nu},
              range {data!.realization.grf.range_km} km.
            </div>
          )}
          {eq && <div className="small muted mt">Realization: η = {eq.realization.eta.toFixed(3)}, σ<sub>w</sub> = {eq.realization.sigma_w.toFixed(2)},
            Matérn ν = {eq.realization.grf.nu}, range {eq.realization.grf.range_km} km · wave speeds α = {eq.waves.alpha_kms} km/s, β = {eq.waves.beta_kms} km/s.
            The wavefield mesh is a vertically exaggerated, phase-locked envelope (amplitude ∝ realized PGA), not a full-waveform solution.</div>}
        </Card>

        <div className="col">
          <Card title="Event">
            {tc && (
              <div className="grid g2">
                <Tile label="Landfall intensity" value={`${tc.stats.vmax_landfall.toFixed(0)} m/s`} foot={`Category ${saffirSimpson(tc.stats.vmax_landfall)} · ${tc.stats.min_pressure_hpa.toFixed(0)} hPa min`} />
                <Tile label="Peak surge" value={surgeOk ? `${surgeOk.peak_m.toFixed(1)} m` : '—'} foot={surgeOk ? `${surgeOk.peak_lat.toFixed(2)}°, ${surgeOk.peak_lon.toFixed(2)}° · ${num(surgeOk.inundated_km2)} km² flooded` : (tc.surge && 'error' in tc.surge ? 'outside DEM coverage' : 'surge model off')} />
                <Tile label="Max rain" value={tc.stats.max_rain_mm ? `${tc.stats.max_rain_mm.toFixed(0)} mm` : '—'} foot="R-CLIPER, storm total" />
                <Tile label="Ground-up loss" value={tc.stats.final_gu != null ? money(tc.stats.final_gu) : '—'} foot={`${num(tc.stats.n_sites)} buildings in footprint`} />
              </div>
            )}
            {eq && (
              <div className="grid g2">
                <Tile label="Magnitude" value={`M${eq.fault.Mw.toFixed(1)}`} foot={`M₀ ${eq.fault.M0.toExponential(2)} N·m · ${eq.fault.mech}`} />
                <Tile label="Rupture" value={`${eq.fault.L_km.toFixed(0)} × ${eq.fault.W_km.toFixed(0)} km`} foot={`dip ${eq.fault.dip.toFixed(0)}° · ${eq.fault.duration_s.toFixed(1)} s · D̄ ${eq.fault.D_mean.toFixed(1)} m, max ${eq.fault.D_max.toFixed(1)} m`} />
                <Tile label="Max PGA (realized)" value={`${eq.stats.max_pga_g.toFixed(2)} g`} foot="Vs30 = 400 m/s reference grid" />
                <Tile label="Ground-up loss" value={eq.stats.final_gu != null ? money(eq.stats.final_gu) : '—'} foot={`${num(eq.stats.n_sites)} buildings shaken ≥ 0.02 g`} />
              </div>
            )}
            {!data && <Empty>—</Empty>}
          </Card>
          <Card title="Loss as the event unfolds" desc={tc ? 'Damage follows the running-max gust and inundation depth at each building' : eq ? 'Each building is damaged when its strong shaking peaks' : undefined}>
            {lossOpt ? <Chart option={lossOpt} height={200} ariaLabel="Loss timeline" /> : <Empty>No portfolio buildings in the footprint.</Empty>}
          </Card>
          {eq && stfOpt && (
            <Card title="Source time function" desc={`Σ sub-fault triangles, rise time ${scene instanceof EqScene ? scene.stf.riseTime.toFixed(2) : ''} s (Somerville et al. 1999)`}>
              <Chart option={stfOpt} height={170} ariaLabel="Moment rate function" />
            </Card>
          )}
          {tc && (
            <Card title="Site meteogram" desc={meteo ? `${meteo.lat.toFixed(3)}°, ${meteo.lon.toFixed(3)}° · gust factor × residual ${meteo.kfac.toFixed(2)}${meteo.surge ? ` · ground ${meteo.surge.z.toFixed(1)} m` : ''}` : 'Click anywhere on the map'}>
              {meteoOpt ? <Chart option={meteoOpt} height={330} ariaLabel="Site meteogram" /> : <Empty>Click the map to probe wind, pressure, water level and rain at any point.</Empty>}
            </Card>
          )}
        </div>
      </div>

      {data && sites && (
        <div className="grid g3">
          <Card className="span2" title="Exposed buildings — mapped reference"
            desc="Portfolio buildings in this event's footprint, ranked by final ground-up loss. Pick one to fly to street level: it is matched to its mapped building footprint, tinted by live damage and, with Google 3D, draped onto the photorealistic city mesh."
            tools={<span className="small muted num">{match.zoomOk
              ? `In view: ${match.matched}/${match.inView} matched to mapped footprints (${match.inView ? Math.round((100 * match.matched) / match.inView) : 0}%) · ${match.inside} inside a footprint${match.medianSnap != null ? ` · median snap ${match.medianSnap.toFixed(0)} m` : ''}`
              : `Zoom in past ${MIN_MATCH_ZOOM} (now ${zoom.toFixed(1)}) to match footprints`}</span>}>
            <DataTable rows={exposed} maxHeight={300} onRowClick={(r) => flyToSite(r.i)} selected={(r) => r.i === sel}
              columns={[
                { key: 'id', label: 'Location' }, { key: 'cls', label: 'Construction' },
                { key: 'tiv', label: 'TIV', align: 'r', render: (r) => money(r.tiv) },
                { key: 'now', label: 'Damage now', align: 'r', value: (r) => sites.damage(r.i, tUi), render: (r) => `${(100 * sites.damage(r.i, tUi)).toFixed(1)}%` },
                { key: 'dmg', label: 'Final damage', align: 'r', render: (r) => `${(100 * r.dmg).toFixed(1)}%` },
                { key: 'loss', label: 'Final GU loss', align: 'r', render: (r) => money(r.loss) },
                { key: 'fp', label: 'Footprint', value: (r) => (match.byIndex.get(r.i)?.snapM ?? 999), render: (r) => {
                  const f = match.byIndex.get(r.i)
                  return f ? (f.inside ? 'inside' : `${f.snapM.toFixed(0)} m`) : (match.zoomOk && ready?.map.getBounds().contains([sites.lon[r.i], sites.lat[r.i]]) ? 'none ≤ 35 m' : '—')
                } },
              ]} />
          </Card>
          <Card title="Building view" desc={googleMode?.kind === 'proxy' ? 'Google Photorealistic 3D Tiles via this server (the key never reaches the browser)'
            : googleMode ? 'Google Photorealistic 3D Tiles with your browser key' : 'OpenStreetMap footprints; add a Google Maps key for photorealistic 3D'}>
            {sel != null && sel < sites.n ? (
              <div className="col" style={{ gap: 10 }}>
                <div className="row wrap" style={{ gap: 8 }}>
                  <b>{sites.id[sel]}</b><span className="pill">{sites.cls[sel]}</span><span className="small muted num">{sites.lat[sel].toFixed(5)}°, {sites.lon[sel].toFixed(5)}°</span>
                </div>
                <div className="grid g2">
                  <Tile label="Damage now" value={`${(100 * sites.damage(sel, tUi)).toFixed(1)}%`} foot={sites.intensity(sel, tUi).text} />
                  <Tile label="Final GU loss" value={money(sites.finalLoss[sel])} foot={`TIV ${money(sites.tiv[sel])} · final damage ${(100 * sites.finalDamage[sel]).toFixed(1)}%`} />
                </div>
                {tc && <div className="small num">Water at the building now: <b>{sites.depth(sel, tUi).toFixed(2)} m</b> above ground (sub-grid surge depth that drives the surge damage).</div>}
                <div className="small muted">{(() => {
                  const f = match.byIndex.get(sel)
                  if (f) return `Mapped footprint (OSM): ${f.inside ? 'location inside the footprint' : `nearest footprint ${f.snapM.toFixed(0)} m away`} · height ${f.height.toFixed(0)} m.`
                  return match.zoomOk ? 'No mapped footprint within 35 m of this location — check its geocode.' : 'Fly to street level to match the mapped footprint.'
                })()}{google ? ` Ground on the photoreal mesh: ${groundZ != null ? `${groundZ.toFixed(1)} m (ellipsoidal)` : 'sampling…'}.` : ''}</div>
                <div className="row" style={{ gap: 8 }}>
                  <button className="btn primary sm" onClick={() => flyToSite(sel)}>Street view</button>
                  <button className="btn sm" onClick={backToEvent}>Back to event</button>
                  <button className="btn sm ghost" onClick={() => setSel(null)}>Clear</button>
                </div>
              </div>
            ) : <Empty>Select a building in the table or click one on the map.</Empty>}
            <div className="mt small col" style={{ gap: 8, borderTop: '1px solid var(--border)', paddingTop: 10 }}>
              {googleMode?.kind === 'proxy' && <div className="muted">Server proxy active (<code>GOOGLE_MAPS_API_KEY</code>). Toggle <b>Google 3D</b> above the map.</div>}
              {googleMode?.kind === 'key' && (
                <div className="row wrap" style={{ gap: 8 }}><span className="muted">Using a key stored in this browser only.</span>
                  <button className="btn sm ghost" onClick={() => { storeKey(null); setUserKey(null); setGoogleOn(false) }}>Forget key</button></div>
              )}
              {!googleMode && (
                <>
                  <div className="muted">{integ.data?.google_3d_tiles.setup ?? 'Google 3D Tiles need a Google Maps Platform key with the Map Tiles API enabled.'}</div>
                  <form className="row" style={{ gap: 8 }} onSubmit={(e) => { e.preventDefault(); if (keyDraft.trim()) { storeKey(keyDraft.trim()); setUserKey(keyDraft.trim()); setKeyDraft(''); setGoogleOn(true) } }}>
                    <input type="password" value={keyDraft} onChange={(e) => setKeyDraft(e.target.value)} placeholder="Google Maps API key (browser)" aria-label="Google Maps API key" style={{ flex: 1 }} autoComplete="off" />
                    <button className="btn sm" type="submit" disabled={!keyDraft.trim()}>Use key</button>
                  </form>
                </>
              )}
            </div>
          </Card>
        </div>
      )}
      {eq && (
        <div className="grid g2">
          <Card title="Synthetic seismogram" desc={seis ? `${seis.lat.toFixed(3)}°, ${seis.lon.toFixed(3)}° · ${seis.epicentral_km.toFixed(0)} km from epicentre · ${seis.n_subfaults} sub-faults · ${seis.method} · horizontal S-wave synthesis (no P energy by construction)` : 'Click the map to simulate ground motion at a site'}>
            {seisErr && <div className="error small">{seisErr}</div>}
            {seisOpt ? <Chart option={seisOpt} height={300} ariaLabel="Seismogram" /> : <Empty>{probe ? 'Simulating…' : 'Click a point on the map.'}</Empty>}
          </Card>
          <Card title="Response spectrum vs GMPE" desc={seis ? `PGA ${seis.pga_g.toFixed(3)} g · PGV ${seis.pgv_cms.toFixed(1)} cm/s · GMPE median ${seis.gmpe_median_g.toFixed(3)} g` : undefined}>
            {psaOpt ? <Chart option={psaOpt} height={300} ariaLabel="Response spectrum" /> : <Empty>—</Empty>}
          </Card>
        </div>
      )}
      {data && <div className="small muted">{data.peril === 'TC'
        ? `${surgeOk ? surgeOk.model + ' · ' : ''}${tc?.rain?.model ?? ''} · wind: Holland (2008) B_s profile, Sobey inflow, 0.55·Vt asymmetry, Kaplan–DeMaria decay.`
        : 'Kinematic rupture: von Kármán slip, Vr = 0.8β; P/S first arrivals from the finite fault; realized PGA = GMPE median × exp(η + σ_w·W(s)).'}{' '}Loss axis: {moneyAxis(data.loss?.tiv_affected ?? 0)} TIV in footprint.</div>}
    </div>
  )
}

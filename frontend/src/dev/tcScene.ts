// Hurricane development scene: evaluates the analytic wind field per frame on Mercator-exact drapes,
// advects tracer particles through it, reconstructs surge/rain frames and drives the deck.gl layers.
import type { Layer, Position } from '@deck.gl/core'
import { ColumnLayer, LineLayer, PathLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers'
import type { Mode } from '../theme'
import { DAMAGE, HEAT, RAIN, TOKENS, WATER, rampColor } from '../theme'
import type { GridSpec, Storm } from './physics'
import { decode, destination, invMercY, mercY, pressureDeficit, sample, sampleNan, trackState, windUV } from './physics'
import type { Canvas2D, Stencil } from './raster'
import { at, declutter, hexRgba, lut, makeCanvas, stencil } from './raster'
import type { TcDev } from './types'

export interface TcFlags { wind: boolean; footprint: boolean; particles: boolean; surge: boolean; rain: boolean; buildings: boolean; radii: boolean; labels: boolean }
export const TC_FLAGS: TcFlags = { wind: true, footprint: false, particles: true, surge: true, rain: false, buildings: true, radii: true, labels: true }

const KT = 0.514444
const RADII = [34, 50, 64]
const G_LO = 15
const G_HI = 80

type Surge = Extract<NonNullable<TcDev['surge']>, { frames_cm: unknown }>

export interface WindRadii { kt: number; nm: [number, number, number, number]; path: [number, number][] }

export class TcScene {
  readonly p: TcDev
  readonly mode: Mode
  readonly kOpen: number
  readonly kMarine: number
  private readonly field: GridSpec & { values: Float32Array }
  private readonly land: Float32Array
  // wind
  readonly wind: Canvas2D
  private readonly kmult: Float32Array
  private readonly g: Float32Array
  private readonly gmax: Float32Array
  private maxT = -Infinity
  // particles
  readonly parts: Canvas2D
  private readonly P: Float32Array
  private readonly N: number
  // surge / rain
  readonly surgeC: Canvas2D | null = null
  private surge: (Surge & { fr: Float32Array; zv: Float32Array; st: Stencil; zpix: Float32Array; n: number }) | null = null
  readonly rainC: Canvas2D | null = null
  private rain: { fr: Float32Array; st: Stencil; n: number; t: number[] } | null = null
  // sites
  readonly sites: { gust: Float32Array; damage: Float32Array; depth: Float32Array; nf: number } | null = null
  private readonly L = { heat: lut(HEAT), water: lut(WATER), rain: lut(RAIN), dmg: lut(DAMAGE) }
  private readonly cities: TcDev['cities']
  private radiiMemo: { t: number; r: WindRadii[] } | null = null

  constructor(p: TcDev, mode: Mode) {
    this.p = p
    this.mode = mode
    const c = p.constants as { gust_open: number; gust_marine: number }
    this.kOpen = c.gust_open
    this.kMarine = c.gust_marine
    const f = p.realization.field
    this.field = { ...f, values: decode(f.values) }
    this.land = decode(p.land)
    this.cities = declutter(p.cities, p.bbox)

    // ---- wind drape: per-pixel K_terrain(land/sea) × exp(η + σ_w W(s)) precomputed once
    this.wind = makeCanvas(p.bbox, 460, 240_000)
    const n = this.wind.W * this.wind.H
    this.kmult = new Float32Array(n)
    this.g = new Float32Array(n)
    this.gmax = new Float32Array(n)
    for (let r = 0; r < this.wind.H; r++) {
      for (let q = 0; q < this.wind.W; q++) this.kmult[r * this.wind.W + q] = this.gustFactor(this.wind.lat[r], this.wind.lon[q])
    }

    // ---- particles
    this.parts = makeCanvas(p.bbox, 1000, 560_000)
    this.N = Math.round(Math.min(3600, 2600 * Math.sqrt(this.parts.W * this.parts.H / 500_000)))
    this.P = new Float32Array(this.N * 3)
    for (let i = 0; i < this.N; i++) this.P[3 * i + 2] = -1 // spawn on first step

    // ---- surge
    if (p.surge && !('error' in p.surge)) {
      const s = p.surge
      const bb: [number, number, number, number] = [s.lon0 - s.dlon / 2, s.lat0 - s.dlat / 2, s.lon0 + (s.nx - 0.5) * s.dlon, s.lat0 + (s.ny - 0.5) * s.dlat]
      this.surgeC = makeCanvas(bb, s.nx * 4, 400_000)
      const st = stencil(this.surgeC, s)
      const zv = decode(s.z)
      const zpix = new Float32Array(this.surgeC.W * this.surgeC.H)
      for (let q = 0; q < zpix.length; q++) zpix[q] = at(st, zv, q)
      this.surge = { ...s, fr: decode(s.frames_cm), zv, st, zpix, n: s.ny * s.nx }
    }
    // ---- rain
    if (p.rain) {
      const r = p.rain
      const bb: [number, number, number, number] = [r.lon0 - r.dlon / 2, r.lat0 - r.dlat / 2, r.lon0 + (r.nx - 0.5) * r.dlon, r.lat0 + (r.ny - 0.5) * r.dlat]
      this.rainC = makeCanvas(bb, r.nx * 3, 300_000)
      this.rain = { fr: decode(r.frames), st: stencil(this.rainC, r), n: r.ny * r.nx, t: r.t }
    }
    // ---- sites
    if (p.sites) {
      const nf = p.frames.length
      this.sites = { gust: decode(p.sites.gust), damage: decode(p.sites.damage), depth: decode(p.sites.surge_depth), nf }
    }
  }

  /** K_terrain (marine over water, open over land — land mask on the realization grid) × realization multiplier. */
  gustFactor(lat: number, lon: number): number {
    const f = this.field
    const iy = Math.min(Math.max(Math.round((lat - f.lat0) / f.dlat), 0), f.ny - 1)
    const ix = Math.min(Math.max(Math.round((lon - f.lon0) / f.dlon), 0), f.nx - 1)
    const k = this.land[iy * f.nx + ix] > 0.5 ? this.kOpen : this.kMarine
    return k * Math.exp(this.p.realization.eta + this.p.realization.sigma_w * sample(f, f.values, lat, lon))
  }

  storm(t: number): Storm { return trackState(this.p.track, t) }

  /** Max |client − server| gust at the server probes (open terrain, no residual): proves both sides run the same field. */
  parity(): { maxAbs: number; n: number } {
    let m = 0
    for (const pr of this.p.probes) {
      const [u, v] = windUV(pr.lat, pr.lon, this.storm(pr.t))
      m = Math.max(m, Math.abs(this.kOpen * Math.hypot(u, v) - pr.gust))
    }
    return { maxAbs: m, n: this.p.probes.length }
  }

  // --------------------------------------------------------------------------- wind field
  private gustField(t: number, out: Float32Array, cutKm = 1300) {
    const s = this.storm(t)
    const { W, H, lat, lon } = this.wind
    const cut2 = cutKm * cutKm
    for (let r = 0; r < H; r++) {
      const la = lat[r]
      const dy = (la - s.lat) * 110.574
      const cx = 111.32 * Math.cos(0.5 * (la + s.lat) * Math.PI / 180)
      const row = r * W
      for (let q = 0; q < W; q++) {
        const dx = (lon[q] - s.lon) * cx
        if (dx * dx + dy * dy > cut2) { out[row + q] = 0; continue }
        const [u, v] = windUV(la, lon[q], s)
        out[row + q] = this.kmult[row + q] * Math.hypot(u, v)
      }
    }
  }

  private foldMax(src: Float32Array) {
    const m = this.gmax
    for (let i = 0; i < m.length; i++) if (src[i] > m[i]) m[i] = src[i]
  }

  /** Running-max footprint up to t (incremental while playing; rebuilt at 0.5 h steps after a scrub). */
  private ensureMax(t: number) {
    const t0 = this.p.t0
    const tmp = new Float32Array(this.gmax.length)
    if (t < this.maxT - 1e-9 || !Number.isFinite(this.maxT) || t - this.maxT > 3) {
      this.gmax.fill(0)
      for (let tt = t0; tt < t; tt += 0.5) { this.gustField(tt, tmp, 900); this.foldMax(tmp) }
    } else {
      for (let tt = this.maxT + 0.25; tt < t; tt += 0.25) { this.gustField(tt, tmp, 900); this.foldMax(tmp) }
    }
    this.maxT = t
  }

  renderWind(t: number, footprint: boolean) {
    this.gustField(t, this.g)
    let src = this.g
    if (footprint) { this.ensureMax(t); this.foldMax(this.g); src = this.gmax }
    const d = this.wind.img.data
    const L = this.L.heat
    for (let i = 0; i < src.length; i++) {
      const v = src[i]
      const o = 4 * i
      if (v < G_LO) { d[o + 3] = 0; continue }
      const k = Math.min(255, Math.max(0, Math.round(((v - G_LO) / (G_HI - G_LO)) * 255)))
      d[o] = L[3 * k]; d[o + 1] = L[3 * k + 1]; d[o + 2] = L[3 * k + 2]
      d[o + 3] = Math.round(Math.min(1, (v - G_LO) / 10) * 215)
    }
    this.wind.ctx.putImageData(this.wind.img, 0, 0)
  }

  // --------------------------------------------------------------------------- particles
  stepParticles(t: number) {
    const c = this.parts
    const ctx = c.ctx
    const s = this.storm(t)
    const [lo0, la0, lo1, la1] = c.bbox
    const y0 = mercY(la1)
    const yr = y0 - mercY(la0)
    const sc = 0.05 * (c.W / 1100)
    ctx.globalCompositeOperation = 'destination-out'
    ctx.fillStyle = 'rgba(0,0,0,0.09)'
    ctx.fillRect(0, 0, c.W, c.H)
    ctx.globalCompositeOperation = 'source-over'
    ctx.lineWidth = 1.5 * Math.max(1, c.W / 1400)
    ctx.lineCap = 'round'
    const bins: number[][] = [[], [], [], []]
    const P = this.P
    for (let i = 0; i < this.N; i++) {
      let x = P[3 * i]
      let y = P[3 * i + 1]
      let age = P[3 * i + 2]
      if (age < 0) {
        // spawn uniformly by area within 900 km of the centre, where the flow is organised
        const [la, lo] = destination(s.lat, s.lon, Math.random() * 360, 750 * Math.sqrt(Math.random()))
        x = ((lo - lo0) / (lo1 - lo0)) * c.W
        y = ((y0 - mercY(la)) / yr) * c.H
        age = 30 + Math.random() * 110
      }
      const lon = lo0 + (x / c.W) * (lo1 - lo0)
      const lat = invMercY(y0 - (y / c.H) * yr)
      const [u, v] = windUV(lat, lon, s)
      const sp = Math.hypot(u, v)
      const nx = x + u * sc
      const ny = y - v * sc
      age -= 1
      if (nx < 0 || ny < 0 || nx >= c.W || ny >= c.H || (sp < 13 && Math.random() < 0.25)) age = -1
      else bins[Math.min(3, Math.max(0, Math.floor((sp - 10) / 15)))].push(x, y, nx, ny)
      P[3 * i] = nx; P[3 * i + 1] = ny; P[3 * i + 2] = age
    }
    const base = this.mode === 'dark' ? '255,255,255' : '20,20,20'
    const alpha = [0.22, 0.5, 0.78, 0.95]
    bins.forEach((b, k) => {
      if (!b.length) return
      ctx.strokeStyle = `rgba(${base},${alpha[k]})`
      ctx.beginPath()
      for (let j = 0; j < b.length; j += 4) { ctx.moveTo(b[j], b[j + 1]); ctx.lineTo(b[j + 2], b[j + 3]) }
      ctx.stroke()
    })
  }

  // --------------------------------------------------------------------------- surge / rain
  private frameIndex(ts: number[], t: number): [number, number] | null {
    if (t < ts[0] || t > ts[ts.length - 1]) return null
    let k = 0
    while (k < ts.length - 2 && ts[k + 1] <= t) k++
    return [k, ts.length > 1 ? Math.min(1, (t - ts[k]) / Math.max(ts[k + 1] - ts[k], 1e-9)) : 0]
  }

  renderSurge(t: number): boolean {
    const s = this.surge
    const c = this.surgeC
    if (!s || !c) return false
    const d = c.img.data
    const fi = this.frameIndex(s.t, t)
    if (!fi) { d.fill(0); c.ctx.putImageData(c.img, 0, 0); return false }
    const [k, w] = fi
    const k1 = Math.min(k + 1, s.t.length - 1)
    const L = this.L.water
    for (let q = 0; q < s.zpix.length; q++) {
      const o = 4 * q
      const e0 = at(s.st, s.fr, q, k * s.n)
      const e1 = at(s.st, s.fr, q, k1 * s.n)
      const eta = e0 === e0 && e1 === e1 ? e0 + w * (e1 - e0) : e0 === e0 ? (w < 0.5 ? e0 : NaN) : e1 === e1 ? (w >= 0.5 ? e1 : NaN) : NaN
      const z = s.zpix[q]
      if (!(eta === eta) || !(z === z)) { d[o + 3] = 0; continue }
      if (z > 0) {
        const depth = eta - z
        if (depth <= 0.05) { d[o + 3] = 0; continue }
        const ki = Math.min(255, Math.round(96 + (depth / 4) * 159))
        d[o] = L[3 * ki]; d[o + 1] = L[3 * ki + 1]; d[o + 2] = L[3 * ki + 2]; d[o + 3] = 240
      } else {
        if (eta < 0.2) { d[o + 3] = 0; continue }
        const ki = Math.min(255, Math.round((eta / 5) * 255))
        d[o] = L[3 * ki]; d[o + 1] = L[3 * ki + 1]; d[o + 2] = L[3 * ki + 2]
        d[o + 3] = Math.round(Math.min(1, eta / 1.2) * 190)
      }
    }
    c.ctx.putImageData(c.img, 0, 0)
    return true
  }

  renderRain(t: number) {
    const r = this.rain
    const c = this.rainC
    if (!r || !c) return
    const d = c.img.data
    const tt = Math.min(Math.max(t, r.t[0]), r.t[r.t.length - 1])
    const [k, w] = this.frameIndex(r.t, tt) ?? [0, 0]
    const k1 = Math.min(k + 1, r.t.length - 1)
    const L = this.L.rain
    const lmax = Math.log(700 / 5)
    for (let q = 0; q < d.length / 4; q++) {
      const o = 4 * q
      const a = at(r.st, r.fr, q, k * r.n)
      const b = at(r.st, r.fr, q, k1 * r.n)
      const mm = a + w * (b - a)
      if (!(mm >= 5)) { d[o + 3] = 0; continue }
      const ki = Math.min(255, Math.round((Math.log(mm / 5) / lmax) * 255))
      d[o] = L[3 * ki]; d[o + 1] = L[3 * ki + 1]; d[o + 2] = L[3 * ki + 2]
      d[o + 3] = Math.round(Math.min(1, (mm - 5) / 40) * 200)
    }
    c.ctx.putImageData(c.img, 0, 0)
  }

  // --------------------------------------------------------------------------- structure diagnostics
  /** NHC-style wind radii (sustained 1-min marine, per quadrant NE/SE/SW/NW), from the same analytic field. */
  windRadii(t: number): WindRadii[] {
    if (this.radiiMemo && this.radiiMemo.t === t) return this.radiiMemo.r
    const s = this.storm(t)
    const speed = (az: number, r: number) => {
      const [la, lo] = destination(s.lat, s.lon, az, r)
      const [u, v] = windUV(la, lo, s)
      return Math.hypot(u, v)
    }
    // per ray: coarse scan for the eyewall peak, then bisection on the monotone outer profile
    const AZ = Array.from({ length: 72 }, (_, k) => k * 5)
    const peak = AZ.map((az) => {
      let best = 0
      let rb = 0
      for (let r = 2; r <= Math.max(120, 4 * s.rm); r += 3) { const v = speed(az, r); if (v > best) { best = v; rb = r } }
      return { rb, best }
    })
    const out: WindRadii[] = []
    for (const kt of RADII) {
      const thr = kt * KT
      const quad: [number, number, number, number] = [0, 0, 0, 0]
      AZ.forEach((az, k) => {
        const { rb, best } = peak[k]
        if (best < thr) return
        let lo = rb
        let hi = 1200
        if (speed(az, hi) >= thr) lo = hi
        else for (let it = 0; it < 16; it++) { const m = 0.5 * (lo + hi); if (speed(az, m) >= thr) lo = m; else hi = m }
        const qd = Math.floor(az / 90)
        quad[qd] = Math.max(quad[qd], lo)
      })
      if (Math.max(...quad) <= 0) continue
      // NHC convention: each quadrant drawn as a circular arc at its maximum extent
      const path: [number, number][] = []
      for (let az = 0; az <= 360; az += 3) {
        const r = quad[Math.min(3, Math.floor((az % 360) / 90))]
        const [la, lo] = r > 0 ? destination(s.lat, s.lon, az, r) : [s.lat, s.lon]
        path.push([lo, la])
      }
      out.push({ kt, nm: quad.map((r) => r / 1.852) as [number, number, number, number], path })
    }
    this.radiiMemo = { t, r: out }
    return out
  }

  // --------------------------------------------------------------------------- sites
  siteDamage(i: number, t: number): number {
    const S = this.sites
    if (!S) return 0
    const x = Math.min(Math.max((t - this.p.t0) / this.p.frame_h, 0), S.nf - 1)
    const k = Math.min(Math.floor(x), S.nf - 2)
    const w = x - k
    return S.damage[i * S.nf + k] * (1 - w) + S.damage[i * S.nf + k + 1] * w
  }

  siteAt(i: number, t: number): { gust: number; damage: number; depth: number } {
    const S = this.sites!
    const k = Math.round(Math.min(Math.max((t - this.p.t0) / this.p.frame_h, 0), S.nf - 1))
    return { gust: S.gust[i * S.nf + k], damage: this.siteDamage(i, t), depth: S.depth[i * S.nf + k] }
  }

  // --------------------------------------------------------------------------- probe meteogram
  meteogram(lat: number, lon: number) {
    const tr = this.p.track
    const t0 = this.p.t0
    const t1 = this.p.t1
    const kf = this.gustFactor(lat, lon)
    const T: number[] = []; const gust: number[] = []; const sus: number[] = []; const pres: number[] = []; const dir: number[] = []
    for (let t = t0; t <= t1 + 1e-9; t += 0.25) {
      const s = trackState(tr, t)
      const [u, v] = windUV(lat, lon, s)
      T.push(+t.toFixed(2))
      sus.push(Math.hypot(u, v))
      gust.push(kf * Math.hypot(u, v))
      pres.push(1013 - pressureDeficit(lat, lon, s) / 100)
      dir.push(((Math.atan2(-u, -v) * 180) / Math.PI + 360) % 360) // meteorological: direction wind comes FROM
    }
    let surge: { t: number[]; level: number[]; z: number } | null = null
    if (this.surge) {
      const s = this.surge
      const z = sample(s, s.zv, lat, lon)
      surge = { t: s.t, level: s.t.map((_, k) => sampleNan(s, s.fr, lat, lon, k * s.n)), z }
    }
    let rain: { t: number[]; mm: number[] } | null = null
    if (this.rain && this.p.rain) {
      const r = this.rain
      rain = { t: r.t, mm: r.t.map((_, k) => sampleNan(this.p.rain!, r.fr, lat, lon, k * r.n)) }
    }
    return { lat, lon, kfac: kf, t: T, gust, sustained: sus, pressure: pres, dir, surge, rain }
  }

  // --------------------------------------------------------------------------- deck layers
  layers(t: number, f: TcFlags, exag: number, probe: [number, number] | null, hover: (i: number) => void): Layer[] {
    const tr = this.p.track
    const tok = TOKENS[this.mode]
    const s = this.storm(t)
    const seg = tr.t.map((_, i) => i).slice(0, -1)
    const cur = seg.findIndex((i) => tr.t[i + 1] > t)
    const out: Layer[] = [
      new LineLayer({
        id: 'track', data: seg, widthUnits: 'pixels', getWidth: 3.5,
        getSourcePosition: (i: number) => [tr.lon[i], tr.lat[i], 300],
        getTargetPosition: (i: number) => [tr.lon[i + 1], tr.lat[i + 1], 300],
        getColor: (i: number) => {
          const c = rampColor(HEAT, (tr.vmax[i] - 17) / 58)
          return [c[0], c[1], c[2], tr.t[i] <= t ? 255 : 90]
        },
        updateTriggers: { getColor: cur },
      }),
    ]
    if (f.radii) {
      const radii = this.windRadii(t)
      const circle: [number, number][] = []
      for (let az = 0; az <= 360; az += 5) {
        const [la, lo] = destination(s.lat, s.lon, az, s.rm)
        circle.push([lo, la])
      }
      out.push(new PathLayer({
        id: 'radii', data: [...radii.map((r) => ({ path: r.path, kt: r.kt })), { path: circle, kt: 0 }],
        widthUnits: 'pixels', getWidth: (d: { kt: number }) => (d.kt === 0 ? 1.5 : 2),
        getPath: (d: { path: [number, number][] }) => d.path.map(([x, y]) => [x, y, 500]) as Position[],
        getColor: (d: { kt: number }) => {
          if (d.kt === 0) return hexRgba(tok.text1, 200)
          const c = rampColor(HEAT, d.kt === 34 ? 0.15 : d.kt === 50 ? 0.5 : 0.85)
          return [c[0], c[1], c[2], 230]
        },
        getDashArray: [4, 3], updateTriggers: { getPath: t },
      }))
    }
    out.push(new ScatterplotLayer({
      id: 'eye', data: [s], getPosition: () => [s.lon, s.lat, 600], radiusUnits: 'pixels', getRadius: 5, stroked: true,
      getFillColor: hexRgba(tok.text1), getLineColor: hexRgba(tok.surface), lineWidthUnits: 'pixels', getLineWidth: 2,
      updateTriggers: { getPosition: t },
    }))
    if (f.buildings && this.p.sites) {
      const S = this.p.sites
      const tk = Math.round(t * 12)
      const dmgL = this.L.dmg
      out.push(new ColumnLayer({
        id: 'sites', data: S.lat.map((_, i) => i), diskResolution: 8, radius: 1800, extruded: true, pickable: true,
        getPosition: (i: number) => [S.lon[i], S.lat[i], Math.max(0, S.elev[i]) * exag],
        getElevation: (i: number) => 150 + this.siteDamage(i, t) * 30000,
        getFillColor: (i: number) => {
          const d = this.siteDamage(i, t)
          if (d < 0.01) return [140, 140, 140, 150]
          const k = Math.min(255, Math.round(Math.sqrt(d) * 255))
          return [dmgL[3 * k], dmgL[3 * k + 1], dmgL[3 * k + 2], 235]
        },
        updateTriggers: { getElevation: tk, getFillColor: tk, getPosition: exag },
        onHover: (info: { index: number }) => hover(info.index),
        material: { ambient: 0.55, diffuse: 0.6, shininess: 24, specularColor: [60, 60, 60] },
      }))
    }
    if (f.labels && this.p.cities.length) {
      const cities = this.cities
      out.push(new ScatterplotLayer({
        id: 'city-dots', data: cities, getPosition: (c: { lon: number; lat: number }) => [c.lon, c.lat, 800],
        radiusUnits: 'pixels', getRadius: 2.5, getFillColor: hexRgba(tok.text1, 230),
      }))
      out.push(new TextLayer({
        id: 'cities', data: cities, getPosition: (c: { lon: number; lat: number }) => [c.lon, c.lat, 800],
        getText: (c: { name: string }) => c.name, getSize: 12, getColor: hexRgba(tok.text1, 235), getPixelOffset: [0, -12],
        fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', fontWeight: 600,
        fontSettings: { sdf: true }, outlineWidth: 3, outlineColor: hexRgba(this.mode === 'dark' ? '#000000' : '#ffffff', 200),
      }))
    }
    if (probe) {
      out.push(new ScatterplotLayer({
        id: 'probe', data: [probe], getPosition: () => [probe[1], probe[0], 900], radiusUnits: 'pixels', getRadius: 7,
        stroked: true, filled: false, lineWidthUnits: 'pixels', getLineWidth: 2.5, getLineColor: hexRgba(tok.text1),
        updateTriggers: { getPosition: probe.join(',') },
      }))
    }
    return out
  }
}

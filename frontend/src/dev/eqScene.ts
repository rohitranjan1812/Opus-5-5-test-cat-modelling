// Earthquake development scene: kinematic rupture on the 3-D fault plane, P/S isochrones and the
// shaking envelope on a Mercator-exact drape, an exaggerated wavefield mesh and building response.
import type { Layer, Position } from '@deck.gl/core'
import { ColumnLayer, LineLayer, PathLayer, ScatterplotLayer, SolidPolygonLayer, TextLayer } from '@deck.gl/layers'
import type { Mode } from '../theme'
import { DAMAGE, HEAT, SLIP, TOKENS } from '../theme'
import { decode, shakingEnvelope } from './physics'
import type { Canvas2D } from './raster'
import { at, declutter, hexRgba, lut, makeCanvas, stencil } from './raster'
import type { SiteAccess } from './sites'
import type { EqDev } from './types'

export interface EqFlags { shaking: boolean; fronts: boolean; footprint: boolean; mesh: boolean; fault: boolean; buildings: boolean; labels: boolean }
export const EQ_FLAGS: EqFlags = { shaking: true, fronts: true, footprint: true, mesh: true, fault: true, buildings: true, labels: true }

const PGA_LO = 0.01
const PGA_HI = 1.5
const MESH_STEP = 3
const MESH_MIN_PGA = 0.04 // mesh covers the strongly shaken area only
const MESH_M_PER_G = 9000
const MESH_F0 = 0.6 // Hz, display frequency of the phase-locked wavefield

interface Sub { poly: number[][]; tr: number; slip: number; i: number; j: number }

export class EqScene {
  readonly p: EqDev
  readonly mode: Mode
  readonly c: Canvas2D
  private readonly px: { tp: Float32Array; ts: Float32Array; tk: Float32Array; te: Float32Array; pga: Float32Array }
  private readonly bP: number
  private readonly bS: number
  private readonly grid: { tp: Float32Array; ts: Float32Array; tk: Float32Array; te: Float32Array; pga: Float32Array; elev: Float32Array }
  private readonly L = { heat: lut(HEAT), slip: lut(SLIP), dmg: lut(DAMAGE) }
  readonly subs: Sub[]
  private readonly cities: EqDev['cities']
  readonly stf: { t: number[]; rate: number[]; cum: number[]; riseTime: number; mu: number }

  constructor(p: EqDev, mode: Mode) {
    this.p = p
    this.mode = mode
    this.cities = declutter(p.cities, p.bbox, 16)
    const g = p.grid
    this.grid = { tp: decode(g.t_p), ts: decode(g.t_s), tk: decode(g.t_peak), te: decode(g.t_end), pga: decode(g.pga), elev: decode(g.elev) }
    const bb: [number, number, number, number] = [g.lon0, g.lat0, g.lon0 + (g.nx - 1) * g.dlon, g.lat0 + (g.ny - 1) * g.dlat]
    this.c = makeCanvas(bb, 520, 300_000)
    const st = stencil(this.c, g)
    const n = this.c.W * this.c.H
    const mk = () => new Float32Array(n)
    this.px = { tp: mk(), ts: mk(), tk: mk(), te: mk(), pga: mk() }
    for (let q = 0; q < n; q++) {
      this.px.tp[q] = at(st, this.grid.tp, q); this.px.ts[q] = at(st, this.grid.ts, q)
      this.px.tk[q] = at(st, this.grid.tk, q); this.px.te[q] = at(st, this.grid.te, q)
      this.px.pga[q] = at(st, this.grid.pga, q)
    }
    const kmPerPx = ((bb[2] - bb[0]) * 111.32 * Math.cos(((bb[1] + bb[3]) / 2) * Math.PI / 180)) / this.c.W
    this.bP = Math.max(0.25, (1.3 * kmPerPx) / p.waves.alpha_kms)
    this.bS = Math.max(0.3, (1.3 * kmPerPx) / p.waves.beta_kms)

    // sub-faults (corner grids are (nW+1)×(nL+1))
    const F = p.fault
    const subs: Sub[] = []
    for (let i = 0; i < F.nW; i++) {
      for (let j = 0; j < F.nL; j++) {
        const cn = [[i, j], [i, j + 1], [i + 1, j + 1], [i + 1, j]]
        subs.push({ poly: cn.map(([a, b]) => [F.corner_lon[a][b], F.corner_lat[a][b], F.corner_depth_km[a][b]]), tr: F.t_rupture_s[i][j], slip: F.slip_m[i][j], i, j })
      }
    }
    this.subs = subs
    this.stf = this.momentRate()
  }

  /** Moment-rate function: Σ sub-fault triangles (rise time from Somerville et al. 1999) started at the rupture times. */
  private momentRate() {
    const F = this.p.fault
    const mu = F.mech === 'SUB' ? 4.0e10 : 3.3e10
    const area = (F.L_km * F.W_km * 1e6) / (F.nL * F.nW)
    const riseTime = Math.max(0.3, 2.03e-9 * Math.cbrt(F.M0 * 1e7))
    const tEnd = F.duration_s + riseTime * 1.2 + 1
    const nT = Math.min(1200, Math.max(200, Math.ceil(tEnd / 0.05)))
    const dt = tEnd / nT
    const rate = new Float64Array(nT + 1)
    for (const s of this.subs) {
      const m0 = mu * area * s.slip
      const h = (2 * m0) / riseTime
      const a = Math.floor(s.tr / dt)
      const b = Math.min(nT, Math.ceil((s.tr + riseTime) / dt))
      for (let k = a; k <= b; k++) {
        const x = (k * dt - s.tr) / riseTime
        if (x > 0 && x < 1) rate[k] += h * (x < 0.5 ? 2 * x : 2 * (1 - x))
      }
    }
    const t: number[] = []; const r: number[] = []; const cum: number[] = []
    let acc = 0
    for (let k = 0; k <= nT; k++) { acc += rate[k] * dt; t.push(+(k * dt).toFixed(3)); r.push(rate[k]); cum.push(acc) }
    const scale = F.M0 / Math.max(acc, 1) // discretisation → exact M0
    return { t, rate: r.map((v) => v * scale), cum: cum.map((v) => v * scale), riseTime, mu }
  }

  momentAt(t: number): number {
    const s = this.stf
    if (t <= 0) return 0
    if (t >= s.t[s.t.length - 1]) return s.cum[s.cum.length - 1]
    const k = Math.floor((t / s.t[s.t.length - 1]) * (s.t.length - 1))
    return s.cum[k]
  }

  rupturedFraction(t: number): number {
    let n = 0
    for (const s of this.subs) if (s.tr <= t) n++
    return n / this.subs.length
  }

  // --------------------------------------------------------------------------- ground drape
  render(t: number, f: EqFlags) {
    const d = this.c.img.data
    const { tp, ts, tk, te, pga } = this.px
    const L = this.L.heat
    const lspan = Math.log(PGA_HI / PGA_LO)
    const pCol = this.mode === 'dark' ? [120, 210, 255] : [0, 110, 200]
    const sCol = this.mode === 'dark' ? [255, 255, 255] : [30, 30, 30]
    for (let q = 0; q < tp.length; q++) {
      const o = 4 * q
      d[o + 3] = 0
      const a = pga[q]
      if (!(a === a)) continue
      if (f.footprint && t >= tk[q] && a > 0.02) {
        const k = Math.min(255, Math.max(0, Math.round((Math.log(a / PGA_LO) / lspan) * 255)))
        d[o] = L[3 * k]; d[o + 1] = L[3 * k + 1]; d[o + 2] = L[3 * k + 2]; d[o + 3] = 105
      }
      if (f.shaking) {
        const inst = a * shakingEnvelope(t, tp[q], ts[q], tk[q], te[q])
        if (inst > PGA_LO) {
          const x = Math.log(inst / PGA_LO) / lspan
          const k = Math.min(255, Math.round(x * 255))
          d[o] = L[3 * k]; d[o + 1] = L[3 * k + 1]; d[o + 2] = L[3 * k + 2]
          d[o + 3] = Math.max(d[o + 3], Math.round(Math.min(1, 0.35 + 1.6 * x) * 230))
        }
      }
      if (f.fronts) {
        if (Math.abs(t - tp[q]) < this.bP) { d[o] = pCol[0]; d[o + 1] = pCol[1]; d[o + 2] = pCol[2]; d[o + 3] = 235 }
        if (Math.abs(t - ts[q]) < this.bS) { d[o] = sCol[0]; d[o + 1] = sCol[1]; d[o + 2] = sCol[2]; d[o + 3] = 245 }
      }
    }
    this.c.ctx.putImageData(this.c.img, 0, 0)
  }

  // --------------------------------------------------------------------------- wavefield mesh (binary attributes)
  private meshSegs: { a: Int32Array; b: Int32Array } | null = null

  /** Grid-line segments (every MESH_STEP nodes) whose endpoints both reach MESH_MIN_PGA; built once. */
  private segments() {
    if (this.meshSegs) return this.meshSegs
    const g = this.p.grid
    const G = this.grid
    const A: number[] = []
    const B: number[] = []
    const add = (a: number, b: number) => { if (G.pga[a] >= MESH_MIN_PGA && G.pga[b] >= MESH_MIN_PGA) { A.push(a); B.push(b) } }
    for (let i = 0; i < g.ny; i += MESH_STEP) for (let j = 0; j < g.nx - 1; j++) add(i * g.nx + j, i * g.nx + j + 1)
    for (let j = 0; j < g.nx; j += MESH_STEP) for (let i = 0; i < g.ny - 1; i++) add(i * g.nx + j, (i + 1) * g.nx + j)
    this.meshSegs = { a: Int32Array.from(A), b: Int32Array.from(B) }
    return this.meshSegs
  }

  private mesh(t: number, exag: number) {
    const g = this.p.grid
    const G = this.grid
    const { a: SA, b: SB } = this.segments()
    const nseg = SA.length
    const src = new Float32Array(nseg * 3)
    const dst = new Float32Array(nseg * 3)
    const col = new Uint8Array(nseg * 4)
    const lspan = Math.log(PGA_HI / PGA_LO)
    const tok = this.mode === 'dark' ? [200, 200, 195] : [70, 70, 66]
    const node = (k: number, out: Float32Array, o: number) => {
      const env = shakingEnvelope(t, G.tp[k], G.ts[k], G.tk[k], G.te[k])
      const a = G.pga[k] * env
      const phase = t >= G.ts[k] ? t - G.ts[k] : t - G.tp[k]
      out[o] = g.lon0 + (k % g.nx) * g.dlon
      out[o + 1] = g.lat0 + Math.floor(k / g.nx) * g.dlat
      out[o + 2] = Math.max(0, G.elev[k]) * exag + 1500 + MESH_M_PER_G * a * Math.cos(2 * Math.PI * MESH_F0 * phase)
      return a
    }
    for (let s = 0; s < nseg; s++) {
      const inten = 0.5 * (node(SA[s], src, 3 * s) + node(SB[s], dst, 3 * s))
      if (inten > PGA_LO) {
        const k = Math.min(255, Math.round((Math.log(inten / PGA_LO) / lspan) * 255))
        col[4 * s] = this.L.heat[3 * k]; col[4 * s + 1] = this.L.heat[3 * k + 1]; col[4 * s + 2] = this.L.heat[3 * k + 2]; col[4 * s + 3] = 235
      } else {
        col[4 * s] = tok[0]; col[4 * s + 1] = tok[1]; col[4 * s + 2] = tok[2]; col[4 * s + 3] = 45
      }
    }
    return new LineLayer({
      id: 'wave-mesh', widthUnits: 'pixels', getWidth: 1.2,
      data: { length: nseg, attributes: { getSourcePosition: { value: src, size: 3 }, getTargetPosition: { value: dst, size: 3 }, getColor: { value: col, size: 4, normalized: true } } },
    } as never)
  }

  siteAccess(): SiteAccess | null {
    const S = this.p.sites
    if (!S) return null
    return {
      n: S.n, lat: S.lat, lon: S.lon, elev: S.elev, id: S.loc_id, cls: S.construction, tiv: S.tiv,
      finalLoss: S.gu, finalDamage: S.damage,
      damage: (i, t) => this.siteDamage(i, t),
      depth: () => 0,
      intensity: (i, t) => {
        const tE = S.t_peak[i] + 2.5 * Math.max(S.t_peak[i] - S.t_s[i], 2) + 5
        const a = S.pga[i] * shakingEnvelope(t, S.t_p[i], S.t_s[i], S.t_peak[i], tE)
        return { text: `PGA now ${a.toFixed(2)} g (peak ${S.pga[i].toFixed(2)} g)`, x: Math.min(1, a / 0.6) }
      },
    }
  }

  siteDamage(i: number, t: number): number {
    const S = this.p.sites!
    const grow = Math.min(1, Math.max(0, (t - S.t_s[i]) / Math.max(S.t_peak[i] - S.t_s[i] + 0.5, 0.5)))
    return S.damage[i] * grow
  }

  // --------------------------------------------------------------------------- deck layers
  layers(t: number, f: EqFlags, exag: number, probe: [number, number] | null, hover: (i: number) => void, depthScale = 1): Layer[] {
    const tok = TOKENS[this.mode]
    const F = this.p.fault
    const zs = -1000 * depthScale
    const out: Layer[] = []
    if (f.fault) {
      const Dmax = Math.max(F.D_max, 1e-6)
      const tk = Math.round(t * 20)
      out.push(new SolidPolygonLayer({
        id: 'fault', data: this.subs, _full3d: true, pickable: true,
        getPolygon: (s: Sub) => s.poly.map(([x, y, z]) => [x, y, z * zs]) as Position[],
        getFillColor: (s: Sub) => {
          if (t < s.tr) return [150, 150, 150, 55]
          if (t - s.tr < 1.2) return [255, 248, 214, 255]
          const k = Math.min(255, Math.round((s.slip / Dmax) * 255))
          return [this.L.slip[3 * k], this.L.slip[3 * k + 1], this.L.slip[3 * k + 2], 235]
        },
        updateTriggers: { getFillColor: tk, getPolygon: depthScale },
      }))
      const ring = (a: number[][], b: number[][], dd: number[][]) => {
        const nW = F.nW
        const nL = F.nL
        const pts: number[][] = []
        for (let j = 0; j <= nL; j++) pts.push([b[0][j], a[0][j], dd[0][j]])
        for (let i = 1; i <= nW; i++) pts.push([b[i][nL], a[i][nL], dd[i][nL]])
        for (let j = nL - 1; j >= 0; j--) pts.push([b[nW][j], a[nW][j], dd[nW][j]])
        for (let i = nW - 1; i >= 0; i--) pts.push([b[i][0], a[i][0], dd[i][0]])
        return pts
      }
      const outline = ring(F.corner_lat, F.corner_lon, F.corner_depth_km).map(([x, y, z]) => [x, y, z * zs])
      const top = F.corner_lat[0].map((_, j) => [F.corner_lon[0][j], F.corner_lat[0][j], 0])
      out.push(new PathLayer({
        id: 'fault-outline', data: [{ p: outline, w: 1.6, c: hexRgba(tok.text1, 220) }, { p: top, w: 2, c: hexRgba(tok.text1, 160) },
          { p: F.ring.lon.map((x, i) => [x, F.ring.lat[i], 0]), w: 1.2, c: hexRgba(tok.text2, 150) }],
        widthUnits: 'pixels', getPath: (d: { p: number[][] }) => d.p as unknown as Position[], getWidth: (d: { w: number }) => d.w,
        getColor: (d: { c: number[] }) => d.c as [number, number, number, number], getDashArray: [5, 3],
        updateTriggers: { getPath: depthScale },
      }))
      const H = F.hypo
      out.push(new LineLayer({
        id: 'hypo-drop', data: [H], widthUnits: 'pixels', getWidth: 1.5, getColor: hexRgba(tok.text1, 180),
        getSourcePosition: () => [H.lon, H.lat, H.depth * zs], getTargetPosition: () => [H.lon, H.lat, 0],
        updateTriggers: { getSourcePosition: depthScale },
      }))
      out.push(new ScatterplotLayer({
        id: 'hypo', data: [H, { ...H, depth: 0 }], radiusUnits: 'pixels', getRadius: (d: { depth: number }) => (d.depth > 0 ? 7 : 4),
        getPosition: (d: { lon: number; lat: number; depth: number }) => [d.lon, d.lat, d.depth * zs], stroked: true,
        getFillColor: (d: { depth: number }) => (d.depth > 0 ? [255, 214, 10, 255] : hexRgba(tok.text1)),
        getLineColor: [20, 20, 20, 255], lineWidthUnits: 'pixels', getLineWidth: 1.5,
        updateTriggers: { getPosition: depthScale },
      }))
    }
    if (f.mesh) out.push(this.mesh(t, exag))
    if (f.buildings && this.p.sites) {
      const S = this.p.sites
      const tk = Math.round(t * 10)
      out.push(new ColumnLayer({
        id: 'sites', data: S.lat.map((_, i) => i), diskResolution: 8, radius: 700, extruded: true, pickable: true,
        getPosition: (i: number) => [S.lon[i], S.lat[i], Math.max(0, S.elev[i]) * exag],
        getElevation: (i: number) => 120 + this.siteDamage(i, t) * 20000,
        getFillColor: (i: number) => {
          if (t < S.t_s[i]) return [140, 140, 140, 140]
          const d = this.siteDamage(i, t)
          const k = Math.min(255, Math.round(Math.sqrt(d) * 255))
          return [this.L.dmg[3 * k], this.L.dmg[3 * k + 1], this.L.dmg[3 * k + 2], 235]
        },
        updateTriggers: { getElevation: tk, getFillColor: tk, getPosition: exag },
        onHover: (info: { index: number }) => hover(info.index),
        material: { ambient: 0.55, diffuse: 0.6, shininess: 24, specularColor: [60, 60, 60] },
      }))
    }
    if (f.labels && this.p.cities.length) {
      const cities = this.cities
      out.push(new ScatterplotLayer({ id: 'city-dots', data: cities, getPosition: (c: { lon: number; lat: number }) => [c.lon, c.lat, 600],
        radiusUnits: 'pixels', getRadius: 2.5, getFillColor: hexRgba(tok.text1, 230) }))
      out.push(new TextLayer({
        id: 'cities', data: cities, getPosition: (c: { lon: number; lat: number }) => [c.lon, c.lat, 600],
        getText: (c: { name: string }) => c.name, getSize: 12, getColor: hexRgba(tok.text1, 235), getPixelOffset: [0, -12],
        fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', fontWeight: 600,
        fontSettings: { sdf: true }, outlineWidth: 3, outlineColor: hexRgba(this.mode === 'dark' ? '#000000' : '#ffffff', 200),
      }))
    }
    if (probe) {
      out.push(new ScatterplotLayer({
        id: 'probe', data: [probe], getPosition: () => [probe[1], probe[0], 700], radiusUnits: 'pixels', getRadius: 7,
        stroked: true, filled: false, lineWidthUnits: 'pixels', getLineWidth: 2.5, getLineColor: hexRgba(tok.text1),
        updateTriggers: { getPosition: probe.join(',') },
      }))
    }
    return out
  }
}

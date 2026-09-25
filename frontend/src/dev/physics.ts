// Client-side mirror of the server's analytic hazard physics (catforge/hazard/tropical_cyclone.py,
// catforge/physics/tc_dynamics.py, catforge/geo.py). Same equations, same constants, so the wind field
// the browser animates is the field the engine prices; server "probes" verify parity at load time.

export interface B64 { b64: string; dtype: 'int16' | 'uint8' | 'float32'; scale: number; shape: number[]; nan: number | null }
export interface GridSpec { lat0: number; lon0: number; dlat: number; dlon: number; ny: number; nx: number }

const DEG = Math.PI / 180
const KM_LAT = 110.574
const KM_LON_EQ = 111.32

/** Decode a base64 little-endian array to Float32 (value / scale; NaN for the int16 sentinel). */
export function decode(g: B64): Float32Array {
  const bin = atob(g.b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  const inv = 1 / g.scale
  if (g.dtype === 'uint8') return Float32Array.from(bytes, (v) => v * inv)
  if (g.dtype === 'float32') return new Float32Array(bytes.buffer)
  const src = new Int16Array(bytes.buffer)
  const out = new Float32Array(src.length)
  for (let i = 0; i < src.length; i++) out[i] = src[i] === g.nan ? NaN : src[i] * inv
  return out
}

/** Bilinear sample of a regular lat/lon grid (row-major, lat rows), clamped like the server's _sample_grid. */
export function sample(g: GridSpec, v: ArrayLike<number>, lat: number, lon: number, offset = 0): number {
  const y = Math.min(Math.max((lat - g.lat0) / g.dlat, 0), g.ny - 1.0001)
  const x = Math.min(Math.max((lon - g.lon0) / g.dlon, 0), g.nx - 1.0001)
  const i = Math.floor(y)
  const j = Math.floor(x)
  const fy = y - i
  const fx = x - j
  const a = offset + i * g.nx + j
  return (1 - fy) * ((1 - fx) * v[a] + fx * v[a + 1]) + fy * ((1 - fx) * v[a + g.nx] + fx * v[a + g.nx + 1])
}

/** NaN-aware bilinear sample: renormalises weights over finite corners (used for wet/dry surge frames). */
export function sampleNan(g: GridSpec, v: ArrayLike<number>, lat: number, lon: number, offset = 0): number {
  const y = (lat - g.lat0) / g.dlat
  const x = (lon - g.lon0) / g.dlon
  if (y < 0 || x < 0 || y > g.ny - 1 || x > g.nx - 1) return NaN
  const i = Math.min(Math.floor(y), g.ny - 2)
  const j = Math.min(Math.floor(x), g.nx - 2)
  const fy = y - i
  const fx = x - j
  const a = offset + i * g.nx + j
  const w = [(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx]
  const c = [v[a], v[a + 1], v[a + g.nx], v[a + g.nx + 1]]
  let s = 0
  let ws = 0
  for (let k = 0; k < 4; k++) if (Number.isFinite(c[k])) { s += w[k] * c[k]; ws += w[k] }
  return ws > 0.25 ? s / ws : NaN
}

export function localXYkm(lat: number, lon: number, lat0: number, lon0: number): [number, number] {
  return [(lon - lon0) * KM_LON_EQ * Math.cos(0.5 * (lat + lat0) * DEG), (lat - lat0) * KM_LAT]
}

export function destination(lat: number, lon: number, bearingDeg: number, km: number): [number, number] {
  const R = 6371.0088
  const d = km / R
  const b = bearingDeg * DEG
  const p1 = lat * DEG
  const p2 = Math.asin(Math.sin(p1) * Math.cos(d) + Math.cos(p1) * Math.sin(d) * Math.cos(b))
  const l2 = lon * DEG + Math.atan2(Math.sin(b) * Math.sin(d) * Math.cos(p1), Math.cos(d) - Math.sin(p1) * Math.sin(p2))
  return [p2 / DEG, l2 / DEG]
}

// ------------------------------------------------------------------------------------------ hurricane
export interface Track {
  t: number[]; lat: number[]; lon: number[]; dp_pa: number[]; rmax_km: number[]; b: number[]; heading: number[]
  vt: number[]; vmax: number[]; category: number[]
}
export interface Storm { lat: number; lon: number; dp: number; rm: number; b: number; hdg: number; vt: number; vmax: number }

export const TC = { RHO_AIR: 1.15, E: Math.E, OMEGA: 7.292e-5, ASYM: 0.55, SURFACE: 1.0 }

/** Linear interpolation of the hourly track state (heading on the circle) — track_state(). */
export function trackState(tr: Track, t: number): Storm {
  const n = tr.t.length
  let i: number
  let w: number
  if (t <= tr.t[0]) { i = 0; w = 0 } else if (t >= tr.t[n - 1]) { i = n - 2; w = 1 } else {
    let lo = 0
    let hi = n - 1
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (tr.t[m] < t) lo = m; else hi = m }
    i = Math.min(Math.max(lo, 0), n - 2)
    w = (t - tr.t[i]) / Math.max(tr.t[i + 1] - tr.t[i], 1e-9)
  }
  const L = (a: number[]) => a[i] + w * (a[i + 1] - a[i])
  const dh = ((((tr.heading[i + 1] - tr.heading[i] + 540) % 360) + 360) % 360) - 180
  return { lat: L(tr.lat), lon: L(tr.lon), dp: L(tr.dp_pa), rm: L(tr.rmax_km), b: L(tr.b), hdg: tr.heading[i] + w * dh,
    vt: L(tr.vt), vmax: L(tr.vmax) }
}

/** Sobey, Harper & Stark (1977) inflow angle. */
export function inflowDeg(rOverRm: number): number {
  if (rOverRm < 1) return 10
  if (rOverRm < 1.2) return 10 + 75 * (rOverRm - 1)
  return 25
}

/** Surface (10 m, 1-min, marine) wind vector [east, north] m/s — exact port of wind_uv(). */
export function windUV(lat: number, lon: number, s: Storm): [number, number] {
  const [x, y] = localXYkm(lat, lon, s.lat, s.lon)
  const r = Math.max(Math.hypot(x, y), 0.5)
  const f = 2 * TC.OMEGA * Math.abs(Math.sin(s.lat * DEG))
  const rr = (s.rm / r) ** s.b
  const rf = r * 1000 * f * 0.5
  const vg = Math.sqrt(s.b * s.dp / TC.RHO_AIR * rr * Math.exp(-rr) + rf * rf) - rf
  const vs = TC.SURFACE * vg
  const vsm = TC.SURFACE * Math.sqrt(s.b * s.dp / (TC.RHO_AIR * TC.E))
  const beta = inflowDeg(r / s.rm) * DEG
  const cb = Math.cos(beta)
  const sb = Math.sin(beta)
  const ux = x / r
  const uy = y / r
  let wx = vs * (cb * -uy - sb * ux)
  let wy = vs * (cb * ux - sb * uy)
  const sc = TC.ASYM * vs / Math.max(vsm, 1)
  wx += sc * s.vt * Math.sin(s.hdg * DEG)
  wy += sc * s.vt * Math.cos(s.hdg * DEG)
  return [wx, wy]
}

/** Holland pressure deficit p_env − p(r) in Pa. */
export function pressureDeficit(lat: number, lon: number, s: Storm): number {
  const [x, y] = localXYkm(lat, lon, s.lat, s.lon)
  const r = Math.max(Math.hypot(x, y), 0.5)
  return s.dp * (1 - Math.exp(-((s.rm / r) ** s.b)))
}

/** Hazard realization: ln-residual field W(s) on a grid, event term η, within-event σ. */
export interface Realization { eta: number; sigma_w: number; field: GridSpec & { values: Float32Array } }

export function residualMultiplier(r: Realization, lat: number, lon: number): number {
  return Math.exp(r.eta + r.sigma_w * sample(r.field, r.field.values, lat, lon))
}

export function saffirSimpson(v: number): number {
  return v >= 70 ? 5 : v >= 58 ? 4 : v >= 50 ? 3 : v >= 43 ? 2 : v >= 33 ? 1 : 0
}

// ------------------------------------------------------------------------------------------ mercator helpers
export const mercY = (lat: number) => Math.log(Math.tan(Math.PI / 4 + (lat * DEG) / 2))
export const invMercY = (y: number) => (2 * Math.atan(Math.exp(y)) - Math.PI / 2) / DEG

/** Latitude of each pixel row centre for a canvas draped over [la0, la1] (MapLibre maps image rows linearly in Mercator). */
export function mercatorRows(la0: number, la1: number, h: number): Float64Array {
  const y0 = mercY(la1)
  const y1 = mercY(la0)
  const out = new Float64Array(h)
  for (let r = 0; r < h; r++) out[r] = invMercY(y0 + ((r + 0.5) / h) * (y1 - y0))
  return out
}

// ------------------------------------------------------------------------------------------ earthquake
/** Normalised shaking envelope at time t for a site with S arrival tS, peak tPk and end tE (Saragoni–Hart-like). */
export function shakingEnvelope(t: number, tP: number, tS: number, tPk: number, tE: number): number {
  if (!(t >= tP)) return 0
  if (t < tS) return 0.3 * Math.min(1, (t - tP) / Math.max(0.6, 0.25 * (tS - tP))) // P coda (vertical-dominant, ~1/3 of S)
  const rise = Math.max(tPk - tS, 0.4)
  if (t < tPk) return 0.3 + 0.7 * ((t - tS) / rise) ** 0.7
  const decay = Math.max((tE - tPk) / 2.5, 1.0)
  return Math.exp(-(t - tPk) / decay)
}

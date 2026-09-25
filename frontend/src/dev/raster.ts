// Canvas drapes: pixel rows are uniform in Web-Mercator y (MapLibre warps image sources linearly in
// Mercator space), so every pixel is evaluated at its true latitude — no geolocation drift across tall domains.
import { rampColor } from '../theme'
import type { GridSpec } from './physics'
import { mercatorRows, mercY } from './physics'

export interface Canvas2D {
  canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; img: ImageData; W: number; H: number
  bbox: [number, number, number, number]; lat: Float64Array; lon: Float64Array
}

export function makeCanvas(bbox: [number, number, number, number], width: number, maxPixels = 600_000): Canvas2D {
  const [lo0, la0, lo1, la1] = bbox
  const wm = (lo1 - lo0) * Math.PI / 180
  const hm = mercY(la1) - mercY(la0)
  let W = Math.round(width)
  let H = Math.max(8, Math.round(W * hm / wm))
  const s = Math.sqrt(Math.min(1, maxPixels / (W * H)))
  W = Math.max(8, Math.round(W * s))
  H = Math.max(8, Math.round(H * s))
  const canvas = document.createElement('canvas')
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d', { willReadFrequently: false })!
  const lon = new Float64Array(W)
  for (let c = 0; c < W; c++) lon[c] = lo0 + ((c + 0.5) / W) * (lo1 - lo0)
  return { canvas, ctx, img: ctx.createImageData(W, H), W, H, bbox, lat: mercatorRows(la0, la1, H), lon }
}

/** Pixel position of a lon/lat on a drape canvas. */
export function toPixel(c: Canvas2D, lat: number, lon: number): [number, number] {
  const [lo0, la0, lo1, la1] = c.bbox
  const y0 = mercY(la1)
  return [((lon - lo0) / (lo1 - lo0)) * c.W, ((y0 - mercY(lat)) / (y0 - mercY(la0))) * c.H]
}

/** 256-entry RGB lookup table for a colour ramp. */
export function lut(ramp: string[]): Uint8Array {
  const out = new Uint8Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    const c = rampColor(ramp, i / 255)
    out[3 * i] = c[0]; out[3 * i + 1] = c[1]; out[3 * i + 2] = c[2]
  }
  return out
}

/** Precomputed bilinear stencil (corner index + weights) of every canvas pixel on a lat/lon grid; idx −1 = outside. */
export interface Stencil { idx: Int32Array; fx: Float32Array; fy: Float32Array; nx: number }

export function stencil(c: Canvas2D, g: GridSpec): Stencil {
  const n = c.W * c.H
  const idx = new Int32Array(n).fill(-1)
  const fx = new Float32Array(n)
  const fy = new Float32Array(n)
  for (let r = 0; r < c.H; r++) {
    const y = (c.lat[r] - g.lat0) / g.dlat
    if (y < 0 || y > g.ny - 1) continue
    const i = Math.min(Math.floor(y), g.ny - 2)
    for (let q = 0; q < c.W; q++) {
      const x = (c.lon[q] - g.lon0) / g.dlon
      if (x < 0 || x > g.nx - 1) continue
      const j = Math.min(Math.floor(x), g.nx - 2)
      const p = r * c.W + q
      idx[p] = i * g.nx + j
      fx[p] = x - j
      fy[p] = y - i
    }
  }
  return { idx, fx, fy, nx: g.nx }
}

/** Bilinear value at pixel p (NaN-aware: weights renormalised over finite corners). */
export function at(s: Stencil, v: ArrayLike<number>, p: number, offset = 0): number {
  const a = s.idx[p]
  if (a < 0) return NaN
  const fx = s.fx[p]
  const fy = s.fy[p]
  const b = offset + a
  const c00 = v[b]; const c01 = v[b + 1]; const c10 = v[b + s.nx]; const c11 = v[b + s.nx + 1]
  const w00 = (1 - fy) * (1 - fx); const w01 = (1 - fy) * fx; const w10 = fy * (1 - fx); const w11 = fy * fx
  if (c00 === c00 && c01 === c01 && c10 === c10 && c11 === c11) return w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11
  let sum = 0
  let ws = 0
  if (c00 === c00) { sum += w00 * c00; ws += w00 }
  if (c01 === c01) { sum += w01 * c01; ws += w01 }
  if (c10 === c10) { sum += w10 * c10; ws += w10 }
  if (c11 === c11) { sum += w11 * c11; ws += w11 }
  return ws > 0.3 ? sum / ws : NaN
}

export const hexRgba = (hex: string, a = 255): [number, number, number, number] => {
  const v = parseInt(hex.slice(1), 16)
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255, a]
}

/** Greedy label declutter: keep the heaviest cities, drop any within minKm of one already kept. */
export function declutter<T extends { lat: number; lon: number; weight: number }>(items: T[], bbox: [number, number, number, number], max = 18): T[] {
  const [lo0, la0, lo1, la1] = bbox
  const diag = Math.hypot((lo1 - lo0) * 111.32 * Math.cos(((la0 + la1) / 2) * Math.PI / 180), (la1 - la0) * 110.574)
  const minKm = 0.075 * diag
  const kept: T[] = []
  for (const c of [...items].sort((a, b) => b.weight - a.weight)) {
    const ok = kept.every((k) => Math.hypot((k.lon - c.lon) * 111.32 * Math.cos(k.lat * Math.PI / 180), (k.lat - c.lat) * 110.574) > minKm)
    if (ok) kept.push(c)
    if (kept.length >= max) break
  }
  return kept
}

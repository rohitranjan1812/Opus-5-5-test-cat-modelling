// Snap exposed portfolio locations to mapped building footprints (OpenStreetMap via the OpenFreeMap
// vector tiles already loaded by the map). A location inside a footprint matches it exactly; otherwise
// the nearest footprint edge within `maxSnapM` is taken. The match rate and snap distances double as a
// geocoding-quality check for the portfolio.
import type { Map as MLMap } from 'maplibre-gl'
import type { SiteAccess } from './sites'

export interface Footprint {
  i: number; ring: [number, number][]; height: number; base: number; snapM: number; inside: boolean; ground: number | null
}
export interface MatchResult {
  byIndex: Map<number, Footprint>; inView: number; matched: number; inside: number; medianSnap: number | null; zoomOk: boolean
}

export const EMPTY_MATCH: MatchResult = { byIndex: new Map(), inView: 0, matched: 0, inside: 0, medianSnap: null, zoomOk: false }
export const MIN_MATCH_ZOOM = 13.5

interface Poly { ring: [number, number][]; bb: [number, number, number, number]; h: number; base: number; area: number }

const M_LAT = 110_574
const mLon = (lat: number) => 111_320 * Math.cos((lat * Math.PI) / 180)

function pip(ring: [number, number][], x: number, y: number): boolean {
  let inside = false
  for (let a = 0, b = ring.length - 1; a < ring.length; b = a++) {
    const [xa, ya] = ring[a]
    const [xb, yb] = ring[b]
    if ((ya > y) !== (yb > y) && x < ((xb - xa) * (y - ya)) / (yb - ya) + xa) inside = !inside
  }
  return inside
}

/** Distance (m) from a point to a ring's edges, in a local equirectangular frame. */
function ringDistM(ring: [number, number][], lon: number, lat: number): number {
  const kx = mLon(lat)
  let best = Infinity
  for (let a = 0; a < ring.length - 1; a++) {
    const ax = (ring[a][0] - lon) * kx; const ay = (ring[a][1] - lat) * M_LAT
    const bx = (ring[a + 1][0] - lon) * kx; const by = (ring[a + 1][1] - lat) * M_LAT
    const dx = bx - ax; const dy = by - ay
    const L2 = dx * dx + dy * dy
    const u = L2 > 0 ? Math.min(1, Math.max(0, -(ax * dx + ay * dy) / L2)) : 0
    best = Math.min(best, Math.hypot(ax + u * dx, ay + u * dy))
  }
  return best
}

export function matchFootprints(map: MLMap, sites: SiteAccess, maxSnapM = 35): MatchResult {
  if (map.getZoom() < MIN_MATCH_ZOOM || !map.getSource('osm')) return EMPTY_MATCH
  let feats: ReturnType<MLMap['querySourceFeatures']> = []
  try { feats = map.querySourceFeatures('osm', { sourceLayer: 'building' }) } catch { return EMPTY_MATCH }
  const polys: Poly[] = []
  for (const f of feats) {
    const g = f.geometry
    const rings = g.type === 'Polygon' ? [g.coordinates[0]] : g.type === 'MultiPolygon' ? g.coordinates.map((p) => p[0]) : []
    const p = f.properties as { render_height?: number; render_min_height?: number }
    for (const r of rings) {
      const ring = r as [number, number][]
      let x0 = Infinity; let y0 = Infinity; let x1 = -Infinity; let y1 = -Infinity
      for (const [x, y] of ring) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y) }
      polys.push({ ring, bb: [x0, y0, x1, y1], h: Math.max(3, Number(p.render_height ?? 6)), base: Number(p.render_min_height ?? 0), area: (x1 - x0) * (y1 - y0) })
    }
  }
  // uniform grid index (~110 m cells) over footprint bounding boxes
  const C = 0.001
  const grid = new Map<string, number[]>()
  polys.forEach((p, k) => {
    for (let gx = Math.floor(p.bb[0] / C); gx <= Math.floor(p.bb[2] / C); gx++) {
      for (let gy = Math.floor(p.bb[1] / C); gy <= Math.floor(p.bb[3] / C); gy++) {
        const key = `${gx}:${gy}`
        const a = grid.get(key)
        if (a) a.push(k); else grid.set(key, [k])
      }
    }
  })
  const bounds = map.getBounds()
  const byIndex = new Map<number, Footprint>()
  const snaps: number[] = []
  let inView = 0
  let inside = 0
  for (let i = 0; i < sites.n; i++) {
    const lon = sites.lon[i]
    const lat = sites.lat[i]
    if (!bounds.contains([lon, lat])) continue
    inView++
    const cand = new Set<number>()
    const gx = Math.floor(lon / C)
    const gy = Math.floor(lat / C)
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (const k of grid.get(`${gx + dx}:${gy + dy}`) ?? []) cand.add(k)
    let best: { k: number; d: number; inside: boolean } | null = null
    for (const k of cand) {
      const p = polys[k]
      if (lon >= p.bb[0] && lon <= p.bb[2] && lat >= p.bb[1] && lat <= p.bb[3] && pip(p.ring, lon, lat)) {
        if (!best || !best.inside || p.area < polys[best.k].area) best = { k, d: 0, inside: true }
        continue
      }
      if (best?.inside) continue
      const d = ringDistM(p.ring, lon, lat)
      if (d <= maxSnapM && (!best || d < best.d)) best = { k, d, inside: false }
    }
    if (!best) continue
    const p = polys[best.k]
    let ground: number | null = null
    try { ground = map.queryTerrainElevation([lon, lat]) } catch { ground = null }
    byIndex.set(i, { i, ring: p.ring, height: p.h, base: p.base, snapM: best.d, inside: best.inside, ground })
    snaps.push(best.d)
    if (best.inside) inside++
  }
  snaps.sort((a, b) => a - b)
  return { byIndex, inView, matched: byIndex.size, inside, medianSnap: snaps.length ? snaps[Math.floor(snaps.length / 2)] : null, zoomOk: true }
}

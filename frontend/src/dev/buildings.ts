// Building-level layers for the development view: exposed buildings as their real mapped footprints
// (tinted by live damage), unmatched locations as markers, and the selected building's street-level
// annotations — a label, a halo that pulses with the hazard, and (hurricanes) the flood-water plane at
// ground + modelled inundation depth, which intersects the photoreal / OSM facades.
import type { Layer, Position } from '@deck.gl/core'
import { _TerrainExtension as TerrainExtension } from '@deck.gl/extensions'
import { LineLayer, ScatterplotLayer, SolidPolygonLayer, TextLayer } from '@deck.gl/layers'
import type { Mode } from '../theme'
import { DAMAGE, TOKENS, WATER } from '../theme'
import type { Footprint, MatchResult } from './footprints'
import { hexRgba, lut } from './raster'
import type { SiteAccess } from './sites'

const DMG = lut(DAMAGE)
const WAT = lut(WATER)
const terrain = new TerrainExtension()

function dmgColor(d: number, alpha: number): [number, number, number, number] {
  if (d < 0.01) return [150, 150, 150, Math.round(alpha * 0.8)]
  const k = Math.min(255, Math.round(Math.sqrt(d) * 255))
  return [DMG[3 * k], DMG[3 * k + 1], DMG[3 * k + 2], alpha]
}

export interface BuildingCtx {
  sites: SiteAccess; t: number; mode: Mode; google: boolean; match: MatchResult; selected: number | null
  /** ground height under the selected building in the scene's vertical frame (photoreal mesh or terrain) */
  groundZ: number | null; exag: number
  onSelect: (i: number) => void
}

export function buildingLayers(c: BuildingCtx): Layer[] {
  const { sites: S, t, google, match } = c
  const tok = TOKENS[c.mode]
  const tk = Math.round(t * 12)
  const out: Layer[] = []
  const matched = [...match.byIndex.values()]
  const unmatched: number[] = []
  for (let i = 0; i < S.n; i++) if (!match.byIndex.has(i)) unmatched.push(i)
  const groundOf = (i: number) => {
    const f = match.byIndex.get(i)
    return f?.ground ?? Math.max(0, S.elev[i]) * c.exag
  }

  if (matched.length) {
    out.push(google
      ? new SolidPolygonLayer<Footprint>({
        id: 'exposed-footprints', data: matched, pickable: true, extensions: [terrain], terrainDrawMode: 'drape',
        getPolygon: (f: Footprint) => f.ring as Position[], getFillColor: (f: Footprint) => dmgColor(c.sites.damage(f.i, t), 190),
        updateTriggers: { getFillColor: tk }, onClick: (info: { object?: Footprint }) => { if (info.object) c.onSelect(info.object.i) },
      } as never)
      : new SolidPolygonLayer<Footprint>({
        id: 'exposed-footprints', data: matched, pickable: true, extruded: true,
        getPolygon: (f) => f.ring.map(([x, y]) => [x, y, (f.ground ?? 0) + f.base]) as Position[],
        getElevation: (f) => Math.max(3, f.height - f.base) + 0.6, getFillColor: (f) => dmgColor(c.sites.damage(f.i, t), 245),
        material: { ambient: 0.5, diffuse: 0.7, shininess: 16, specularColor: [40, 40, 40] },
        updateTriggers: { getFillColor: tk }, onClick: (info) => { if (info.object) c.onSelect(info.object.i) },
      }))
  }
  // geocode → matched footprint leaders where the location sits outside its building (geocode offset)
  const snapped = matched.filter((f) => !f.inside)
  if (snapped.length) {
    const cen = (f: Footprint): [number, number] => {
      let x = 0; let y = 0
      for (const [a, b] of f.ring) { x += a; y += b }
      return [x / f.ring.length, y / f.ring.length]
    }
    out.push(new LineLayer<Footprint>({
      id: 'geocode-leaders', data: snapped, widthUnits: 'pixels', getWidth: 1.5, getColor: hexRgba(tok.text2, 200),
      getSourcePosition: (f: Footprint) => [S.lon[f.i], S.lat[f.i], google ? (c.groundZ ?? 0) + 1 : (f.ground ?? 0) + 1],
      getTargetPosition: (f: Footprint) => { const [x, y] = cen(f); return [x, y, google ? (c.groundZ ?? 0) + 1 : (f.ground ?? 0) + 1] },
    } as never))
    out.push(new ScatterplotLayer<Footprint>({
      id: 'geocode-points', data: snapped, radiusUnits: 'pixels', getRadius: 3, getFillColor: hexRgba(tok.text1, 230),
      getPosition: (f: Footprint) => [S.lon[f.i], S.lat[f.i], google ? 0 : (f.ground ?? 0) + 1],
      ...(google ? { extensions: [terrain], terrainDrawMode: 'drape' } : {}),
    } as never))
  }
  // locations without a mapped footprint within the snap radius
  out.push(new ScatterplotLayer<number>({
    id: 'exposed-points', data: unmatched, pickable: true, radiusUnits: 'meters', getRadius: 9, radiusMinPixels: 3.5,
    stroked: true, lineWidthUnits: 'pixels', getLineWidth: 1, getLineColor: hexRgba(tok.surface, 220),
    getPosition: (i: number) => [S.lon[i], S.lat[i], google ? 0 : groundOf(i) + 2],
    getFillColor: (i: number) => dmgColor(S.damage(i, t), 235),
    ...(google ? { extensions: [terrain], terrainDrawMode: 'drape' } : {}),
    updateTriggers: { getFillColor: tk, getPosition: c.exag }, onClick: (info: { object?: number }) => { if (info.object != null) c.onSelect(info.object) },
  } as never))

  const i = c.selected
  if (i == null || i >= S.n) return out
  const f = match.byIndex.get(i)
  const z0 = google ? c.groundZ : groundOf(i)
  const x = S.intensity(i, t)
  // halo pulses with the live hazard intensity
  out.push(new ScatterplotLayer({
    id: 'selected-halo', data: [i], getPosition: () => [S.lon[i], S.lat[i], google ? 0 : (z0 ?? 0) + 1],
    radiusUnits: 'meters', getRadius: 28 + 55 * Math.min(1, x.x), radiusMinPixels: 10, stroked: true, filled: false,
    lineWidthUnits: 'pixels', getLineWidth: 3, getLineColor: hexRgba(tok.text1, 235),
    ...(google ? { extensions: [terrain], terrainDrawMode: 'drape' } : {}),
    updateTriggers: { getRadius: tk, getPosition: [z0, google] },
  } as never))
  // flood water: a level plane at ground + modelled depth (the same depth that drives surge damage)
  const depth = S.depth(i, t)
  if (depth > 0.02 && z0 != null) {
    const zw = z0 + depth
    const dLat = 260 / 110_574
    const dLon = 260 / (111_320 * Math.cos((S.lat[i] * Math.PI) / 180))
    const k = Math.min(255, Math.round(80 + (depth / 5) * 175))
    out.push(new SolidPolygonLayer({
      id: 'flood-plane', data: [i], _full3d: true,
      getPolygon: () => [[S.lon[i] - dLon, S.lat[i] - dLat, zw], [S.lon[i] + dLon, S.lat[i] - dLat, zw],
        [S.lon[i] + dLon, S.lat[i] + dLat, zw], [S.lon[i] - dLon, S.lat[i] + dLat, zw]],
      getFillColor: [WAT[3 * k], WAT[3 * k + 1], WAT[3 * k + 2], 150], updateTriggers: { getPolygon: zw.toFixed(2) },
    } as never))
    out.push(new LineLayer({
      id: 'depth-gauge', data: [i], widthUnits: 'pixels', getWidth: 3, getColor: [WAT[3 * 200], WAT[3 * 200 + 1], WAT[3 * 200 + 2], 255],
      getSourcePosition: () => [S.lon[i], S.lat[i], z0], getTargetPosition: () => [S.lon[i], S.lat[i], zw],
      updateTriggers: { getTargetPosition: zw.toFixed(2), getSourcePosition: z0 },
    }))
  }
  if (z0 != null || !google) {
    const top = (z0 ?? 0) + (f ? f.height : 10) + 25
    const d = S.damage(i, t)
    const lines = [`${S.id[i]} · ${S.cls[i]}`, x.text, ...(depth > 0.02 ? [`water ${depth.toFixed(2)} m above ground`] : []),
      `damage ${(100 * d).toFixed(0)}%`]
    out.push(new TextLayer({
      id: 'selected-label', data: [i], getPosition: () => [S.lon[i], S.lat[i], top], getText: () => lines.join('\n'),
      getSize: 13, lineHeight: 1.25, getColor: hexRgba(tok.text1), fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
      fontWeight: 600, background: true, getBackgroundColor: hexRgba(tok.surface, 225), backgroundPadding: [8, 5],
      getBorderColor: hexRgba(tok.text3, 160), getBorderWidth: 1, getTextAnchor: 'middle', getAlignmentBaseline: 'bottom',
      updateTriggers: { getText: tk, getPosition: top.toFixed(1) },
    }))
  }
  return out
}

// Google Photorealistic 3D Tiles in the deck.gl overlay.
// - server-proxy mode: the tileset root is this server's /v1/3dtiles/root.json (key stays server-side)
// - browser-key mode:  Google directly, key in the X-GOOG-API-KEY header (key lives only in this browser)
// The layer is created with operation 'terrain+draw' so hazard drapes and footprints can be draped onto
// the photoreal mesh with the TerrainExtension. Attributions from the visible tiles must be displayed.
import type { Layer } from '@deck.gl/core'
import { _TerrainExtension as TerrainExtension } from '@deck.gl/extensions'
import { Tile3DLayer } from '@deck.gl/geo-layers'
import { BitmapLayer } from '@deck.gl/layers'
import type { MapboxOverlay } from '@deck.gl/mapbox'
import { Tiles3DLoader } from '@loaders.gl/3d-tiles'
import type { Map as MLMap } from 'maplibre-gl'
import type { Canvas2D } from './raster'

import type { GoogleMode } from './googleKey'

export const GOOGLE_LAYER_ID = 'google-3d'
const meshDrape = new TerrainExtension()

// one props object per mode: deck diffs props shallowly, so a stable object keeps the tileset (and the
// Google session) alive across the 30 fps layer rebuilds
const cache = new Map<string, Record<string, unknown>>()

// an empty but valid tileset: returned when the root request fails so the layer idles instead of throwing
const EMPTY_TILESET = JSON.stringify({ asset: { version: '1.0' }, geometricError: 0,
  root: { boundingVolume: { sphere: [0, 0, 0, 1] }, geometricError: 0, refine: 'ADD' } })

/** loaders.gl fetch hook: adds the browser key (key mode) and turns Google errors into a readable message. */
function tileFetch(mode: GoogleMode, onError: (msg: string) => void) {
  return async (url: string, init?: RequestInit): Promise<Response> => {
    const headers = new Headers(init?.headers)
    if (mode.kind === 'key') headers.set('X-GOOG-API-KEY', mode.key)
    const r = await fetch(url, { ...init, headers })
    if (r.ok) return r
    let msg = `${r.status} ${r.statusText}`
    try {
      const j = await r.clone().json()
      msg = `${r.status}: ${j?.error?.message ?? j?.detail ?? msg}`
    } catch { /* not JSON */ }
    onError(msg)
    if (!url.includes('root.json')) return r
    const empty = new Response(EMPTY_TILESET, { headers: { 'content-type': 'application/json' } })
    Object.defineProperty(empty, 'url', { value: url }) // loaders.gl picks the tileset parser from the URL
    return empty
  }
}

export function googleTilesLayer(mode: GoogleMode, onCredits: (credits: string) => void, onError: (msg: string) => void = () => {}): Layer {
  const k = mode.kind === 'proxy' ? 'proxy' : `key:${mode.key}`
  let props = cache.get(k)
  if (!props) {
    let last = ''
    props = {
      id: GOOGLE_LAYER_ID,
      data: mode.kind === 'proxy' ? `${window.location.origin}/v1/3dtiles/root.json` : 'https://tile.googleapis.com/v1/3dtiles/root.json',
      loader: Tiles3DLoader,
      loadOptions: { fetch: tileFetch(mode, onError) },
      operation: 'terrain+draw',
      pickable: true,
      onTilesetLoad: (tileset: { options: { onTraversalComplete: (sel: unknown[]) => unknown[] } }) => {
        tileset.options.onTraversalComplete = (selected: unknown[]) => {
          const credits = new Set<string>()
          for (const t of selected as { content?: { gltf?: { asset?: { copyright?: string } } } }[]) {
            for (const c of (t.content?.gltf?.asset?.copyright ?? '').split(';')) if (c.trim()) credits.add(c.trim())
          }
          const s = [...credits].join('; ')
          if (s !== last) { last = s; onCredits(s) }
          return selected
        }
      },
    }
    cache.set(k, props)
  }
  return new Tile3DLayer(props as never)
}

/**
 * Ground height (m, same vertical datum as the photoreal mesh) around a building: unproject a spiral of
 * screen samples onto the mesh and take a low percentile, so streets and yards win over roofs and facades.
 */
export function sampleGround(overlay: MapboxOverlay, map: MLMap, lon: number, lat: number): number | null {
  const c = map.project([lon, lat])
  const zs: number[] = []
  for (let k = 0; k < 36; k++) {
    const r = 6 + 2.2 * k
    const a = k * 2.399963 // golden angle
    const info = overlay.pickObject({ x: c.x + r * Math.cos(a), y: c.y + r * Math.sin(a), radius: 0, layerIds: [GOOGLE_LAYER_ID], unproject3D: true })
    const z = info?.coordinate?.[2]
    if (typeof z === 'number' && Number.isFinite(z)) zs.push(z)
  }
  if (zs.length < 6) return null
  zs.sort((a, b) => a - b)
  return zs[Math.floor(0.12 * zs.length)]
}

/**
 * Hazard canvases re-draped onto the photoreal mesh (MapLibre drapes would sit under it). Bitmap bounds
 * are lon/lat and deck maps them linearly in Web-Mercator, matching the Mercator-uniform canvas rows.
 * A fresh ImageData view per update (same pixel buffer) is what makes deck re-upload the texture.
 */
export function drapeOnMesh(id: string, c: Canvas2D, opacity = 0.85): Layer {
  return new BitmapLayer({
    id, image: new ImageData(c.img.data, c.W, c.H), bounds: c.bbox, opacity, extensions: [meshDrape], terrainDrawMode: 'drape',
  } as never)
}

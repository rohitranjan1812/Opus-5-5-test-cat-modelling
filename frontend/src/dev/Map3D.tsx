import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'
import * as maplibregl from 'maplibre-gl'
import type { StyleSpecification } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
// MapLibre v6 locates its worker next to its own module URL, which a bundler rewrites; bundle the worker
// (with its shared chunk) explicitly and hand MapLibre the emitted URL.
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { MapboxOverlay } from '@deck.gl/mapbox'
import type { Mode } from '../theme'

maplibregl.setWorkerUrl(workerUrl)

/** A canvas draped on the terrain (MapLibre canvas source + raster layer). Pixel rows must be Mercator-uniform. */
export interface Drape { id: string; canvas: HTMLCanvasElement; bbox: [number, number, number, number]; opacity: number; visible: boolean }
export interface MapHandle { map: maplibregl.Map; overlay: MapboxOverlay }

// Hypsometric tint from the same ETOPO1 grid the surge model runs on: bathymetry and relief are data, not decoration.
const RELIEF: Record<Mode, (number | string)[]> = {
  dark: [-6000, '#070d18', -2000, '#0b1626', -200, '#10233a', -20, '#173352', -0.5, '#1d3d5f', 0, '#2b2f2a', 50, '#30342d',
    300, '#3a3c33', 1000, '#4a4a40', 2500, '#6a675c', 4000, '#8e8b82'],
  light: [-6000, '#8fb1d3', -2000, '#a6c3df', -200, '#bcd3ea', -20, '#cfe0f1', -0.5, '#dbe8f5', 0, '#e9e7dc', 50, '#e3e0d2',
    300, '#d8d3c1', 1000, '#c9c1aa', 2500, '#b4aa93', 4000, '#9d937e'],
}

// OpenStreetMap vector tiles (OpenFreeMap, keyless): crisp water/roads at city scale, and the building
// footprints (with heights) that exposed locations are matched against.
export const OSM_TILEJSON = 'https://tiles.openfreemap.org/planet'
const OSM_COLORS: Record<Mode, { water: string; road: string; minor: string; building: string }> = {
  dark: { water: '#10233a', road: '#5b5a53', minor: '#403f3a', building: '#6b6a64' },
  light: { water: '#bcd3ea', road: '#ffffff', minor: '#efece4', building: '#d6d1c4' },
}
const OSM_LAYERS = ['osm-water', 'osm-roads', 'osm-buildings']

function style(mode: Mode, exaggeration: number, terrain: boolean, streetMap: boolean): StyleSpecification {
  const tiles = [`${window.location.origin}/api/terrain/{z}/{x}/{y}.png`]
  const c = OSM_COLORS[mode]
  const vis = streetMap ? 'visible' : 'none'
  const major = ['motorway', 'trunk', 'primary', 'secondary']
  return {
    version: 8,
    sources: {
      dem: { type: 'raster-dem', tiles, tileSize: 256, encoding: 'terrarium', maxzoom: 8 },
      demHs: { type: 'raster-dem', tiles, tileSize: 256, encoding: 'terrarium', maxzoom: 8 },
      osm: { type: 'vector', url: OSM_TILEJSON },
    },
    layers: [
      { id: 'bg', type: 'background', paint: { 'background-color': mode === 'dark' ? '#0b1626' : '#a6c3df' } },
      { id: 'relief', type: 'color-relief', source: 'demHs',
        paint: { 'color-relief-color': ['interpolate', ['linear'], ['elevation'], ...RELIEF[mode]] as never } },
      { id: 'hillshade', type: 'hillshade', source: 'demHs',
        paint: { 'hillshade-method': 'multidirectional', 'hillshade-exaggeration': 0.55,
          'hillshade-shadow-color': mode === 'dark' ? '#000000' : '#5b5646', 'hillshade-highlight-color': mode === 'dark' ? '#8a8a80' : '#ffffff' } },
      { id: 'osm-water', type: 'fill', source: 'osm', 'source-layer': 'water', minzoom: 7, layout: { visibility: vis },
        paint: { 'fill-color': c.water, 'fill-opacity': ['interpolate', ['linear'], ['zoom'], 7, 0, 9.5, 0.9] } },
      { id: 'osm-roads', type: 'line', source: 'osm', 'source-layer': 'transportation', minzoom: 8, layout: { visibility: vis, 'line-cap': 'round' },
        filter: ['in', ['get', 'class'], ['literal', [...major, 'tertiary', 'minor', 'service']]],
        paint: {
          'line-color': ['match', ['get', 'class'], major, c.road, c.minor],
          'line-width': ['interpolate', ['exponential', 1.6], ['zoom'], 8, ['match', ['get', 'class'], major, 0.8, 0.2], 17, ['match', ['get', 'class'], major, 12, 5]],
          'line-opacity': ['interpolate', ['linear'], ['zoom'], 8, 0.4, 12, 0.9],
        } },
      { id: 'osm-buildings', type: 'fill-extrusion', source: 'osm', 'source-layer': 'building', minzoom: 13, layout: { visibility: vis },
        paint: { 'fill-extrusion-color': c.building, 'fill-extrusion-height': ['coalesce', ['get', 'render_height'], 6],
          'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], 0], 'fill-extrusion-opacity': 0.92 } },
    ],
    ...(terrain ? { terrain: { source: 'dem', exaggeration } } : {}),
    sky: mode === 'dark'
      ? { 'sky-color': '#0b1120', 'horizon-color': '#1b2638', 'fog-color': '#0b1626', 'sky-horizon-blend': 0.6, 'horizon-fog-blend': 0.6, 'fog-ground-blend': 0.85 }
      : { 'sky-color': '#bcd6f2', 'horizon-color': '#eef3f8', 'fog-color': '#e6edf5', 'sky-horizon-blend': 0.6, 'horizon-fog-blend': 0.6, 'fog-ground-blend': 0.85 },
  }
}

let drapeRevision = 0
/**
 * Tell MapLibre that drape canvases were redrawn. With terrain on, draped layers are rendered to cached
 * per-tile textures keyed by a source fingerprint that ignores canvas content; a feature-state write
 * bumps the source revision in that fingerprint, so exactly the tiles under the changed drapes re-render.
 */
export function refreshDrapes(h: MapHandle, ids: string[]) {
  drapeRevision++
  for (const id of ids) if (h.map.getSource(id)) h.map.setFeatureState({ source: id, id: 1 }, { r: drapeRevision })
}

const corners = (b: [number, number, number, number]): [[number, number], [number, number], [number, number], [number, number]] =>
  [[b[0], b[3]], [b[2], b[3]], [b[2], b[1]], [b[0], b[1]]]

export default function Map3D({ mode, bbox, fitKey, exaggeration, terrain = true, streetMap = true, drapes, onReady, onClick, onView,
  onSettled, height = 640, children }: {
  mode: Mode; bbox: [number, number, number, number] | null; fitKey: string; exaggeration: number; terrain?: boolean; streetMap?: boolean
  drapes: Drape[]; onReady: (h: MapHandle | null) => void; onClick?: (lat: number, lon: number) => void
  onView?: (zoom: number) => void; onSettled?: () => void; height?: number | string; children?: ReactNode
}) {
  const el = useRef<HTMLDivElement>(null)
  const h = useRef<MapHandle | null>(null)
  const loaded = useRef(false)
  const clickRef = useRef(onClick)
  const drapesRef = useRef<Drape[]>(drapes)
  const readyRef = useRef(onReady)
  const viewRef = useRef(onView)
  const idleRef = useRef(onSettled)
  const opts = useRef({ terrain, streetMap, exaggeration })
  useEffect(() => { clickRef.current = onClick }, [onClick])
  useEffect(() => { readyRef.current = onReady }, [onReady])
  useEffect(() => { viewRef.current = onView }, [onView])
  useEffect(() => { idleRef.current = onSettled }, [onSettled])
  useEffect(() => { opts.current = { terrain, streetMap, exaggeration } }, [terrain, streetMap, exaggeration])

  const syncDrapes = () => {
    const m = h.current?.map
    if (!m || !loaded.current) return
    for (const d of drapesRef.current) {
      const src = m.getSource(d.id) as maplibregl.CanvasSource | undefined
      const cur = src as unknown as { canvas?: HTMLCanvasElement } | undefined
      if (src && cur?.canvas !== d.canvas) {
        m.removeLayer(d.id)
        m.removeSource(d.id)
      }
      if (!m.getSource(d.id)) {
        m.addSource(d.id, { type: 'canvas', canvas: d.canvas, coordinates: corners(d.bbox), animate: true })
        m.addLayer({ id: d.id, type: 'raster', source: d.id, paint: { 'raster-opacity': d.opacity, 'raster-fade-duration': 0, 'raster-resampling': 'linear' } })
      } else {
        (m.getSource(d.id) as maplibregl.CanvasSource).setCoordinates(corners(d.bbox))
        m.setPaintProperty(d.id, 'raster-opacity', d.opacity)
      }
      m.setLayoutProperty(d.id, 'visibility', d.visible ? 'visible' : 'none')
    }
    // drop drapes no longer requested
    for (const l of m.getStyle().layers ?? []) {
      if (l.id.startsWith('drape-') && !drapesRef.current.some((d) => d.id === l.id)) {
        m.removeLayer(l.id)
        m.removeSource(l.id)
      }
    }
  }

  useEffect(() => {
    if (!el.current) return
    const map = new maplibregl.Map({
      container: el.current, style: style(mode, exaggeration, terrain, streetMap), center: [-88, 30], zoom: 4.3, pitch: 50, bearing: -8,
      maxPitch: 80, attributionControl: false, canvasContextAttributes: { antialias: true },
    })
    // deck.gl's MapboxOverlay reads map.transform (terrain centre elevation, viewport height); MapLibre v6 keeps
    // the transform on its camera, so expose it where deck expects it.
    const anyMap = map as unknown as { transform?: unknown; _camera?: { transform?: unknown } }
    if (anyMap.transform === undefined && anyMap._camera) {
      Object.defineProperty(map, 'transform', { configurable: true, get: () => anyMap._camera?.transform })
    }
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right')
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right')
    map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution: 'Relief: NOAA ETOPO1 · Streets & footprints: © OpenStreetMap contributors, OpenMapTiles, OpenFreeMap · Physics: CatForge' }))
    const overlay = new MapboxOverlay({ interleaved: false, layers: [] })
    map.addControl(overlay as unknown as maplibregl.IControl)
    h.current = { map, overlay }
    // programmatic access to the live scene (automation, notebooks driving the page, debugging)
    ;(window as unknown as { catforge3d?: unknown }).catforge3d = { map, overlay, timings: {} as Record<string, number> }
    map.on('load', () => {
      loaded.current = true
      el.current?.querySelector('.maplibregl-ctrl-attrib')?.classList.remove('maplibregl-compact-show')
      syncDrapes()
      readyRef.current(h.current)
    })
    map.on('click', (e) => clickRef.current?.(e.lngLat.lat, e.lngLat.lng))
    map.on('moveend', () => viewRef.current?.(map.getZoom()))
    // 'idle' never fires while canvas drapes animate, so settle on camera stops and street-tile arrivals
    let settle: number | undefined
    const kick = () => { window.clearTimeout(settle); settle = window.setTimeout(() => idleRef.current?.(), 350) }
    map.on('moveend', kick)
    map.on('sourcedata', (e) => { if (e.sourceId === 'osm' && e.tile) kick() })
    const ro = new ResizeObserver(() => map.resize())
    ro.observe(el.current)
    return () => {
      ro.disconnect()
      readyRef.current(null)
      map.remove()
      h.current = null
      loaded.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // theme switch: swap style, keep camera; re-add drapes once the new style is in
  useEffect(() => {
    const m = h.current?.map
    if (!m || !loaded.current) return
    const o = opts.current
    m.setStyle(style(mode, o.exaggeration, o.terrain, o.streetMap), { diff: false })
    m.once('style.load', () => syncDrapes())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode])

  useEffect(() => {
    const m = h.current?.map
    if (m && loaded.current) m.setTerrain(terrain ? { source: 'dem', exaggeration } : null)
  }, [exaggeration, terrain])

  useEffect(() => {
    const m = h.current?.map
    if (!m || !loaded.current) return
    for (const id of OSM_LAYERS) if (m.getLayer(id)) m.setLayoutProperty(id, 'visibility', streetMap ? 'visible' : 'none')
  }, [streetMap])

  useEffect(() => {
    drapesRef.current = drapes
    syncDrapes()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drapes])

  useEffect(() => {
    const m = h.current?.map
    if (!m || !bbox) return
    const go = () => m.fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 40, pitch: 52, bearing: -10, duration: 900 })
    if (loaded.current) go(); else m.once('load', go)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey])

  return (
    <div className="map3d" style={{ height }}>
      <div ref={el} className="map3d-canvas" />
      {children}
    </div>
  )
}

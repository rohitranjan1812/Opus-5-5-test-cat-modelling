import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useApp } from '../state'
import type { ReactNode } from 'react'

export interface PointsLayer {
  kind: 'points'
  lat: number[]
  lon: number[]
  color: (i: number) => string
  radius: (i: number) => number
  tooltip?: (i: number) => string
  onClick?: (i: number) => void
  opacity?: number
}
export interface LinesLayer {
  kind: 'lines'
  items: { lat: number[]; lon: number[]; color: string; weight?: number; opacity?: number; tooltip?: string; onClick?: () => void }[]
}
export interface GridLayer {
  kind: 'grid'
  lat0: number
  lon0: number
  res: number
  ny: number
  nx: number
  value: (iy: number, ix: number) => number | null // NaN/null → transparent
  color: (v: number) => [number, number, number]
  opacity?: number
}
export type Layer = PointsLayer | LinesLayer | GridLayer

interface Props {
  layers: Layer[]
  height?: number
  bounds?: [[number, number], [number, number]] | null
  onMapClick?: (lat: number, lon: number) => void
  children?: ReactNode // overlay (legend)
  fitKey?: string // change to re-fit bounds
}

const TILES = {
  dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  light: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
}

function mercY(lat: number) {
  const r = (lat * Math.PI) / 180
  return Math.log(Math.tan(Math.PI / 4 + r / 2))
}
function invMercY(y: number) {
  return (2 * Math.atan(Math.exp(y)) - Math.PI / 2) * (180 / Math.PI)
}

/** Rasterise a regular lat/lon grid to a Mercator-correct PNG (row-wise inverse projection). */
function gridImage(g: GridLayer): string {
  const W = Math.min(g.nx, 1600)
  const latTop = g.lat0 + g.ny * g.res
  const H = Math.min(Math.max(g.ny, 2), 1600)
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const ctx = c.getContext('2d')!
  const img = ctx.createImageData(W, H)
  const yTop = mercY(latTop)
  const yBot = mercY(g.lat0)
  const a = Math.round(255 * (g.opacity ?? 0.75))
  for (let py = 0; py < H; py++) {
    const lat = invMercY(yTop + ((py + 0.5) / H) * (yBot - yTop))
    const iy = Math.floor((lat - g.lat0) / g.res)
    if (iy < 0 || iy >= g.ny) continue
    for (let px = 0; px < W; px++) {
      const ix = Math.floor(((px + 0.5) / W) * g.nx)
      const v = g.value(iy, ix)
      if (v === null || !Number.isFinite(v)) continue
      const col = g.color(v)
      const o = (py * W + px) * 4
      img.data[o] = col[0]
      img.data[o + 1] = col[1]
      img.data[o + 2] = col[2]
      img.data[o + 3] = a
    }
  }
  ctx.putImageData(img, 0, 0)
  return c.toDataURL()
}

export default function MapView({ layers, height = 460, bounds, onMapClick, children, fitKey }: Props) {
  const el = useRef<HTMLDivElement>(null)
  const map = useRef<L.Map | null>(null)
  const tiles = useRef<L.TileLayer | null>(null)
  const group = useRef<L.LayerGroup | null>(null)
  const renderer = useRef<L.Canvas | null>(null)
  const clickRef = useRef(onMapClick)
  const { mode } = useApp()
  useEffect(() => { clickRef.current = onMapClick }, [onMapClick])

  useEffect(() => {
    if (!el.current) return
    const m = L.map(el.current, { zoomControl: true, preferCanvas: true, worldCopyJump: true, attributionControl: true })
      .setView([34, -95], 4)
    renderer.current = L.canvas({ padding: 0.3 })
    group.current = L.layerGroup().addTo(m)
    m.on('click', (e: L.LeafletMouseEvent) => clickRef.current?.(e.latlng.lat, e.latlng.lng))
    map.current = m
    const ro = new ResizeObserver(() => m.invalidateSize())
    ro.observe(el.current)
    return () => { ro.disconnect(); m.remove(); map.current = null }
  }, [])

  useEffect(() => {
    const m = map.current
    if (!m) return
    tiles.current?.remove()
    tiles.current = L.tileLayer(TILES[mode], {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO', subdomains: 'abcd', maxZoom: 18,
    }).addTo(m)
    tiles.current.bringToBack()
  }, [mode])

  useEffect(() => {
    const g = group.current
    const r = renderer.current
    if (!g || !r) return
    g.clearLayers()
    for (const layer of layers) {
      if (layer.kind === 'grid') {
        if (layer.nx <= 0 || layer.ny <= 0) continue
        const url = gridImage(layer)
        L.imageOverlay(url, [[layer.lat0, layer.lon0], [layer.lat0 + layer.ny * layer.res, layer.lon0 + layer.nx * layer.res]],
          { interactive: false }).addTo(g)
      } else if (layer.kind === 'lines') {
        for (const it of layer.items) {
          const pl = L.polyline(it.lat.map((la, i) => [la, it.lon[i]] as [number, number]), {
            color: it.color, weight: it.weight ?? 1.5, opacity: it.opacity ?? 0.8, renderer: r, interactive: !!(it.tooltip || it.onClick),
          }).addTo(g)
          if (it.tooltip) pl.bindTooltip(it.tooltip, { sticky: true })
          if (it.onClick) pl.on('click', (e) => { L.DomEvent.stopPropagation(e); it.onClick?.() })
        }
      } else {
        const n = layer.lat.length
        for (let i = 0; i < n; i++) {
          const mk = L.circleMarker([layer.lat[i], layer.lon[i]], {
            renderer: r, radius: layer.radius(i), color: layer.color(i), weight: 0, fillColor: layer.color(i),
            fillOpacity: layer.opacity ?? 0.8, interactive: !!(layer.tooltip || layer.onClick),
          }).addTo(g)
          if (layer.tooltip) {
            const tip = layer.tooltip
            mk.bindTooltip(() => tip(i), { direction: 'top' })
          }
          if (layer.onClick) {
            const cb = layer.onClick
            mk.on('click', (e) => { L.DomEvent.stopPropagation(e); cb(i) })
          }
        }
      }
    }
  }, [layers])

  useEffect(() => {
    if (map.current && bounds) map.current.fitBounds(bounds, { padding: [20, 20], maxZoom: 9 })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey])

  return (
    <div className="map-wrap">
      <div ref={el} className="map" style={{ height }} />
      {children}
    </div>
  )
}

export function MapLegend({ title, ramp, lo, hi }: { title: string; ramp: string; lo: string; hi: string }) {
  return (
    <div className="map-legend">
      <div>{title}</div>
      <div className="ramp" style={{ background: ramp }} />
      <div className="ends"><span>{lo}</span><span>{hi}</span></div>
    </div>
  )
}

// Chart tokens: the validated reference palette (see dataviz skill palette.md), per mode.
import type { EChartsOption } from 'echarts'
import { moneyAxis } from './format'

export type Mode = 'dark' | 'light'
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Frag = any // ECharts option fragment (its literal-typed unions don't compose well with helpers)

export const TOKENS = {
  dark: {
    surface: '#1a1a19', text1: '#ffffff', text2: '#c3c2b7', text3: '#898781', grid: '#2c2c2a', axis: '#383835',
    series: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
  },
  light: {
    surface: '#fcfcfb', text1: '#0b0b0b', text2: '#52514e', text3: '#898781', grid: '#e1e0d9', axis: '#c3c2b7',
    series: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
  },
}
export const STATUS = { good: '#0ca30c', warning: '#fab219', serious: '#ec835a', critical: '#d03b3b' }

// Fixed identities — colour follows the entity everywhere in the app.
export const SLOT = { gross: 0, net: 1, gu: 2, analytic: 6, TC: 0, EQ: 1, aep: 0, oep: 1 } as const

/** Sequential blue ramp, steps 100 → 700. */
export const BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
  '#1c5cab', '#184f95', '#104281', '#0d366b']
/** Semantic heat ramp (analogous yellow→red) for hazard intensity — always shown with a scale legend. */
export const HEAT = ['#fff2b2', '#fde08a', '#f9c75b', '#f3a93a', '#eb8a2f', '#e0692c', '#cf4a2f', '#b3302f', '#8c1f2c']
/** Accumulated rainfall (light green → deep teal). */
export const RAIN = ['#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#1b7a6e', '#16607a', '#0f4868']
/** Building damage ratio (pink → deep magenta): distinct from hazard heat and water blues on the 3-D map. */
export const DAMAGE = ['#fde0ef', '#f7c3de', '#f1a1ca', '#e377ae', '#d44f94', '#c51b7d', '#a5116a', '#8e0152']
/** Coseismic slip on the fault plane (lavender → deep violet). */
export const SLIP = ['#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
/** Water level / inundation depth (pale cyan → deep blue). */
export const WATER = ['#d7f5fb', '#aee8f5', '#7fd3ee', '#4fb8e3', '#2a96d4', '#1a74bd', '#15559e', '#0f3b7a']

function hexToRgb(h: string): [number, number, number] {
  const v = parseInt(h.slice(1), 16)
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255]
}

export function rampColor(ramp: string[], t: number): [number, number, number] {
  const x = Math.min(Math.max(t, 0), 1) * (ramp.length - 1)
  const i = Math.min(Math.floor(x), ramp.length - 2)
  const f = x - i
  const a = hexToRgb(ramp[i])
  const b = hexToRgb(ramp[i + 1])
  return [0, 1, 2].map((k) => Math.round(a[k] + f * (b[k] - a[k]))) as [number, number, number]
}

export function rgb(c: [number, number, number], alpha = 1): string {
  return alpha >= 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${alpha})`
}

/** Sequential magnitude colour. In dark mode the anchor flips so high values are the light end. */
export function seqColor(t: number, mode: Mode): string {
  return rgb(rampColor(mode === 'dark' ? [...BLUE].reverse().slice(0, 11) : BLUE.slice(2), t))
}

export function rampCss(ramp: string[]): string {
  return `linear-gradient(90deg, ${ramp.join(', ')})`
}

// ---------------------------------------------------------------- echarts helpers
export function base(mode: Mode): EChartsOption {
  const t = TOKENS[mode]
  return {
    backgroundColor: 'transparent',
    color: t.series,
    textStyle: { fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: t.text2 },
    animationDuration: 300,
    grid: { left: 12, right: 20, top: 36, bottom: 8, containLabel: true },
    legend: { top: 0, left: 0, textStyle: { color: t.text2 }, icon: 'roundRect', itemWidth: 12, itemHeight: 4, itemGap: 16 },
    tooltip: {
      backgroundColor: t.surface, borderColor: t.axis, borderWidth: 1, textStyle: { color: t.text1, fontSize: 12 },
      extraCssText: 'box-shadow: 0 2px 8px rgba(0,0,0,0.25); border-radius: 8px;',
    },
  }
}

export function valueAxis(mode: Mode, opts: Record<string, unknown> = {}): Frag {
  const t = TOKENS[mode]
  return {
    type: 'value', axisLine: { show: false }, axisTick: { show: false },
    splitLine: { lineStyle: { color: t.grid, width: 1, type: 'solid' } },
    axisLabel: { color: t.text3, formatter: (v: number) => moneyAxis(v) },
    nameTextStyle: { color: t.text3 }, ...opts,
  }
}

export function logAxis(mode: Mode, opts: Record<string, unknown> = {}): Frag {
  const t = TOKENS[mode]
  return {
    type: 'log', logBase: 10, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false },
    splitLine: { lineStyle: { color: t.grid, width: 1, type: 'solid' } }, minorSplitLine: { show: false },
    axisLabel: { color: t.text3 }, nameTextStyle: { color: t.text3 }, nameLocation: 'middle', nameGap: 26, ...opts,
  }
}

export function catAxis(mode: Mode, data: string[], opts: Record<string, unknown> = {}): Frag {
  const t = TOKENS[mode]
  return {
    type: 'category', data, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false },
    axisLabel: { color: t.text3 }, ...opts,
  }
}

export const barItem = (color: string, horizontal = false): Frag => ({
  color, borderRadius: horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0],
})

export const lineSeries = (name: string, color: string, data: unknown[], extra: Record<string, unknown> = {}): Frag => ({
  name, type: 'line', data, showSymbol: false, symbolSize: 8, smooth: false,
  lineStyle: { width: 2, color, cap: 'round', join: 'round' }, itemStyle: { color }, emphasis: { focus: 'series' },
  ...extra,
})

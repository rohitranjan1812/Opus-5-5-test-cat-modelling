import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import type { EChartsOption } from 'echarts'
import { useApp } from '../state'
import { base } from '../theme'

interface Props {
  option: EChartsOption
  height?: number | string
  onClick?: (p: echarts.ECElementEvent) => void
  ariaLabel?: string
}

function merge(a: Record<string, unknown>, b: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...a }
  for (const [k, v] of Object.entries(b)) {
    const av = a[k]
    if (v && av && typeof v === 'object' && typeof av === 'object' && !Array.isArray(v) && !Array.isArray(av)) {
      out[k] = merge(av as Record<string, unknown>, v as Record<string, unknown>)
    } else {
      out[k] = v
    }
  }
  return out
}

/** ECharts wrapper: theme-aware base options, resize-observed, click passthrough. */
export default function Chart({ option, height = 320, onClick, ariaLabel }: Props) {
  const el = useRef<HTMLDivElement>(null)
  const inst = useRef<echarts.ECharts | null>(null)
  const clickRef = useRef(onClick)
  const { mode } = useApp()
  useEffect(() => { clickRef.current = onClick }, [onClick])

  useEffect(() => {
    if (!el.current) return
    const c = echarts.init(el.current, undefined, { renderer: 'canvas' })
    inst.current = c
    c.on('click', (p) => clickRef.current?.(p as echarts.ECElementEvent))
    const ro = new ResizeObserver(() => c.resize())
    ro.observe(el.current)
    return () => { ro.disconnect(); c.dispose(); inst.current = null }
  }, [])

  useEffect(() => {
    const o = option as Record<string, unknown>
    const b = base(mode) as Record<string, unknown>
    if (!('legend' in o)) b.legend = { show: false } // single-series charts: the title names the series
    inst.current?.setOption(merge(b, o) as EChartsOption, { notMerge: true })
  }, [option, mode])

  return <div ref={el} style={{ width: '100%', height }} role="img" aria-label={ariaLabel} />
}

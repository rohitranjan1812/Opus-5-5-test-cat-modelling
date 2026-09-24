import type { EChartsOption } from 'echarts'
import type { Curve, EpRow } from '../api'
import { money, rpLabel } from '../format'
import type { Mode } from '../theme'
import { TOKENS, lineSeries, logAxis, valueAxis } from '../theme'

export interface EpSeries {
  name: string
  curve: Curve
  slot: number
  band?: EpRow[] // CI band (loss_lo / loss_hi)
}

/** Exceedance-probability curve: return period (log x) vs loss, one y-axis, crosshair tooltip. */
export function epOption(mode: Mode, series: EpSeries[], opts: { minRp?: number; maxRp?: number; points?: { name: string; slot: number; data: [number, number][] }[] } = {}): EChartsOption {
  const t = TOKENS[mode]
  const minRp = opts.minRp ?? 2
  const out: Record<string, unknown>[] = []
  for (const s of series) {
    const color = t.series[s.slot]
    const data = s.curve.rp.map((rp, i) => [rp, s.curve.loss[i]]).filter((d) => d[0] >= minRp && (!opts.maxRp || d[0] <= opts.maxRp))
    out.push(lineSeries(s.name, color, data))
    if (s.band && s.band.length) {
      const b = s.band.filter((r) => r.loss_lo !== undefined && r.rp >= minRp)
      // band = stacked area between lo and hi (wash at ~10%)
      out.push({ name: `${s.name} 95% CI`, type: 'line', data: b.map((r) => [r.rp, r.loss_lo]), stack: `band-${s.name}`,
        lineStyle: { opacity: 0 }, symbol: 'none', silent: true, tooltip: { show: false } })
      out.push({ name: `${s.name} 95% CI`, type: 'line', data: b.map((r) => [r.rp, (r.loss_hi ?? 0) - (r.loss_lo ?? 0)]),
        stack: `band-${s.name}`, lineStyle: { opacity: 0 }, symbol: 'none', areaStyle: { color, opacity: 0.12 }, silent: true,
        tooltip: { show: false } })
    }
  }
  for (const p of opts.points ?? []) {
    const color = t.series[p.slot]
    out.push({ name: p.name, type: 'scatter', data: p.data.filter((d) => d[0] >= minRp), symbolSize: 9,
      itemStyle: { color, borderColor: t.surface, borderWidth: 2 } })
  }
  const legendNames = [...series.map((s) => s.name), ...(opts.points ?? []).map((p) => p.name)]
  return {
    legend: { data: legendNames },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: t.axis } },
      formatter: (params: unknown) => {
        const ps = (params as { seriesName: string; value: [number, number]; color: string }[])
          .filter((p) => !p.seriesName.endsWith('CI'))
        if (!ps.length) return ''
        const rp = ps[0].value[0]
        return `<b>${rpLabel(rp)}</b> · ${(100 / rp).toFixed(2)}% annual prob.<br/>` +
          ps.map((p) => `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${p.color};margin-right:6px"></span>${p.seriesName}: <b>${money(p.value[1])}</b>`).join('<br/>')
      },
    },
    xAxis: logAxis(mode, { name: 'Return period (years)', min: minRp, max: opts.maxRp,
      axisLabel: { color: t.text3, formatter: (v: number) => `${v}` } }),
    yAxis: valueAxis(mode),
    grid: { left: 12, right: 24, top: 36, bottom: 30, containLabel: true },
    series: out as EChartsOption['series'],
  }
}

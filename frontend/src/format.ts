export function money(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—'
  const a = Math.abs(x)
  const s = x < 0 ? '-' : ''
  if (a >= 1e9) return `${s}$${(a / 1e9).toFixed(digits + 1)}bn`
  if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(digits)}m`
  if (a >= 1e3) return `${s}$${(a / 1e3).toFixed(0)}k`
  return `${s}$${a.toFixed(0)}`
}

/** Compact axis label (no decimals beyond need). */
export function moneyAxis(x: number): string {
  const a = Math.abs(x)
  if (a >= 1e9) return `$${+(x / 1e9).toFixed(1)}bn`
  if (a >= 1e6) return `$${+(x / 1e6).toFixed(0)}m`
  if (a >= 1e3) return `$${+(x / 1e3).toFixed(0)}k`
  return `$${x.toFixed(0)}`
}

export function pct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—'
  return `${(100 * x).toFixed(digits)}%`
}

export function num(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—'
  return x.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

export function compact(x: number): string {
  return Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(x)
}

export function rpLabel(rp: number): string {
  return `1-in-${rp >= 1000 ? compact(rp) : Math.round(rp)}`
}

export function when(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  } catch {
    return iso
  }
}

import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { Insight, Job } from '../api'
import { STATUS } from '../theme'

export function Card({ title, desc, tools, children, className = '' }: {
  title?: ReactNode; desc?: ReactNode; tools?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <section className={`card ${className}`}>
      {(title || tools) && (
        <div className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {desc && <div className="desc">{desc}</div>}
          </div>
          {tools && <div className="card-tools">{tools}</div>}
        </div>
      )}
      {children}
    </section>
  )
}

export function Tile({ label, value, foot, hero = false }: { label: string; value: ReactNode; foot?: ReactNode; hero?: boolean }) {
  return (
    <div className={`tile ${hero ? 'hero' : ''}`}>
      <div className="tile-label">{label}</div>
      <div className="tile-value" title={typeof value === 'string' ? value : undefined}>{value}</div>
      {foot && <div className="tile-foot">{foot}</div>}
    </div>
  )
}

export function Seg<T extends string>({ value, options, onChange, ariaLabel }: {
  value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; ariaLabel?: string
}) {
  return (
    <div className="seg" role="radiogroup" aria-label={ariaLabel}>
      {options.map((o) => (
        <button key={o.value} className={o.value === value ? 'on' : ''} role="radio" aria-checked={o.value === value}
          onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  )
}

export function Tabs<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="tabs" role="tablist">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={o.value === value} className={o.value === value ? 'on' : ''}
          onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  )
}

export function Progress({ value, label }: { value: number; label?: string }) {
  return (
    <div className="col" style={{ gap: 6 }}>
      <div className="progress" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div style={{ width: `${Math.max(2, value * 100)}%` }} />
      </div>
      {label && <div className="small muted">{label}</div>}
    </div>
  )
}

export function JobStatus({ job }: { job: Job | null }) {
  if (!job) return null
  if (job.status === 'error') return <div className="error small">Job failed: {job.error}</div>
  return <Progress value={job.progress} label={`${job.message || job.status} · ${job.elapsed_s.toFixed(1)}s`} />
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}

const SEV: Record<Insight['severity'], { color: string; glyph: string; label: string }> = {
  critical: { color: STATUS.critical, glyph: '!', label: 'Critical' },
  warning: { color: STATUS.warning, glyph: '!', label: 'Warning' },
  info: { color: '#86b6ef', glyph: 'i', label: 'Info' },
  positive: { color: STATUS.good, glyph: '✓', label: 'Good' },
}

export function InsightList({ items, limit }: { items: Insight[]; limit?: number }) {
  const shown = limit ? items.slice(0, limit) : items
  return (
    <div>
      {shown.map((it) => {
        const s = SEV[it.severity]
        return (
          <div className="insight" key={it.id}>
            <div className="insight-icon" style={{ background: s.color }} aria-label={s.label} title={s.label}>{s.glyph}</div>
            <div>
              <div className="row" style={{ gap: 8 }}>
                <span className="pill">{s.label} · {it.category}</span>
              </div>
              <div className="insight-title" style={{ marginTop: 4 }}>{it.title}</div>
              <div className="insight-detail">{it.detail}</div>
              {it.recommendation && <div className="insight-rec">→ {it.recommendation}</div>}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export interface Column<T> {
  key: string
  label: string
  render?: (row: T) => ReactNode
  value?: (row: T) => number | string
  align?: 'r'
}

export function DataTable<T>({ rows, columns, onRowClick, selected, maxHeight, initialSort }: {
  rows: T[]; columns: Column<T>[]; onRowClick?: (r: T) => void; selected?: (r: T) => boolean; maxHeight?: number
  initialSort?: { key: string; desc: boolean }
}) {
  const [sort, setSort] = useState(initialSort ?? null)
  const sorted = useMemo(() => {
    if (!sort) return rows
    const col = columns.find((c) => c.key === sort.key)
    if (!col) return rows
    const get = col.value ?? ((r: T) => (r as Record<string, unknown>)[col.key] as number | string)
    return [...rows].sort((a, b) => {
      const va = get(a)
      const vb = get(b)
      const c = typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb))
      return sort.desc ? -c : c
    })
  }, [rows, sort, columns])
  return (
    <div className="table-wrap" style={maxHeight ? { maxHeight } : undefined}>
      <table className="data">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={`sortable ${c.align ?? ''}`} aria-sort={sort?.key === c.key ? (sort.desc ? 'descending' : 'ascending') : 'none'}
                onClick={() => setSort((s) => ({ key: c.key, desc: s?.key === c.key ? !s.desc : true }))}>
                {c.label}{sort?.key === c.key ? (sort.desc ? ' ↓' : ' ↑') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={i} className={`${onRowClick ? 'click' : ''} ${selected?.(r) ? 'sel' : ''}`} onClick={() => onRowClick?.(r)}>
              {columns.map((c) => (
                <td key={c.key} className={c.align ?? ''}>{c.render ? c.render(r) : String((r as Record<string, unknown>)[c.key] ?? '')}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Swatch({ color }: { color: string }) {
  return <span className="dot" style={{ background: color }} />
}

export function NumberField({ label, value, onChange, step, min, max, suffix }: {
  label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; suffix?: string
}) {
  return (
    <label className="field">
      <span>{label}{suffix ? <span className="muted"> ({suffix})</span> : null}</span>
      <input type="number" value={Number.isFinite(value) ? value : ''} step={step} min={min} max={max}
        onChange={(e) => onChange(e.target.value === '' ? 0 : Number(e.target.value))} />
    </label>
  )
}

export function Download({ href, children }: { href: string; children: ReactNode }) {
  return <a className="btn sm" href={href} download>{children}</a>
}

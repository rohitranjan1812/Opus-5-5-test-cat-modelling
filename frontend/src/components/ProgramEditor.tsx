import type { Contract, Program } from '../api'

export const DEFAULT_LAYER: Contract = {
  name: 'Cat XL', type: 'cat_xl', stage: 2, attachment: 1e8, limit: 1.5e8, reinstatements: 1, reinstatement_rate: 1,
  aad: 0, cession: 0, placed: 1, perils: null,
}

export function defaultProgram(): Program {
  return {
    pricing_method: 'stdev', pricing_load: 0.3, contracts: [
      { ...DEFAULT_LAYER, name: 'Cat XL 150m xs 100m' },
      { ...DEFAULT_LAYER, name: 'Cat XL 250m xs 250m', attachment: 2.5e8, limit: 2.5e8 },
    ],
  }
}

const M = 1e6

function MoneyIn({ value, onChange, label }: { value: number; onChange: (v: number) => void; label: string }) {
  return <input type="number" aria-label={label} value={+(value / M).toFixed(3)} step={10} min={0} style={{ width: 96 }}
    onChange={(e) => onChange(Number(e.target.value) * M)} />
}

export default function ProgramEditor({ program, onChange }: { program: Program; onChange: (p: Program) => void }) {
  const set = (i: number, patch: Partial<Contract>) =>
    onChange({ ...program, contracts: program.contracts.map((c, j) => (i === j ? { ...c, ...patch } : c)) })
  const add = (c: Contract) => onChange({ ...program, contracts: [...program.contracts, c] })
  const del = (i: number) => onChange({ ...program, contracts: program.contracts.filter((_, j) => j !== i) })
  return (
    <div className="col">
      <div className="table-wrap" style={{ maxHeight: 'none' }}>
        <table className="data">
          <thead>
            <tr>
              <th>Name</th><th>Type</th><th title="Inuring stage: same-stage contracts share the subject loss">Stage</th>
              <th className="r">Attachment ($m)</th><th className="r">Limit ($m)</th><th className="r">Reinst.</th>
              <th className="r">AAD ($m)</th><th className="r">Cession %</th><th className="r">Placed %</th><th>Perils</th><th />
            </tr>
          </thead>
          <tbody>
            {program.contracts.map((c, i) => (
              <tr key={i}>
                <td><input type="text" value={c.name} aria-label="Contract name" onChange={(e) => set(i, { name: e.target.value })} style={{ width: 170 }} /></td>
                <td>
                  <select value={c.type} aria-label="Contract type" onChange={(e) => set(i, { type: e.target.value as Contract['type'] })}>
                    <option value="cat_xl">Cat XL</option><option value="quota_share">Quota share</option><option value="agg_xl">Stop loss</option>
                  </select>
                </td>
                <td><input type="number" aria-label="Stage" value={c.stage} min={1} max={10} style={{ width: 56 }} onChange={(e) => set(i, { stage: Number(e.target.value) })} /></td>
                <td className="r">{c.type !== 'quota_share' ? <MoneyIn label="Attachment" value={c.attachment} onChange={(v) => set(i, { attachment: v })} /> : <span className="muted">—</span>}</td>
                <td className="r">{c.type !== 'quota_share' ? <MoneyIn label="Limit" value={c.limit} onChange={(v) => set(i, { limit: v })} /> : <span className="muted">—</span>}</td>
                <td className="r">{c.type === 'cat_xl' ? <input type="number" aria-label="Reinstatements" value={c.reinstatements} min={0} max={20} style={{ width: 56 }} onChange={(e) => set(i, { reinstatements: Number(e.target.value) })} /> : <span className="muted">—</span>}</td>
                <td className="r">{c.type === 'cat_xl' ? <MoneyIn label="AAD" value={c.aad} onChange={(v) => set(i, { aad: v })} /> : <span className="muted">—</span>}</td>
                <td className="r">{c.type === 'quota_share' ? <input type="number" aria-label="Cession" value={Math.round(c.cession * 100)} min={0} max={100} style={{ width: 64 }} onChange={(e) => set(i, { cession: Number(e.target.value) / 100 })} /> : <span className="muted">—</span>}</td>
                <td className="r"><input type="number" aria-label="Placed" value={Math.round(c.placed * 100)} min={1} max={100} style={{ width: 64 }} onChange={(e) => set(i, { placed: Math.max(0.01, Number(e.target.value) / 100) })} /></td>
                <td>
                  <select value={c.perils?.[0] ?? ''} aria-label="Perils" onChange={(e) => set(i, { perils: e.target.value ? [e.target.value] : null })}>
                    <option value="">All</option><option value="TC">TC</option><option value="EQ">EQ</option>
                  </select>
                </td>
                <td><button className="btn sm ghost danger" onClick={() => del(i)} aria-label="Remove contract">✕</button></td>
              </tr>
            ))}
            {!program.contracts.length && <tr><td colSpan={11} className="muted">No contracts — gross only.</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="row">
        <button className="btn sm" onClick={() => add({ ...DEFAULT_LAYER, name: `Cat XL ${program.contracts.length + 1}` })}>+ Cat XL layer</button>
        <button className="btn sm" onClick={() => add({ ...DEFAULT_LAYER, name: 'Quota share 20%', type: 'quota_share', stage: 1, cession: 0.2, attachment: 0, limit: 0 })}>+ Quota share</button>
        <button className="btn sm" onClick={() => add({ ...DEFAULT_LAYER, name: 'Stop loss', type: 'agg_xl', stage: 3, attachment: 3e8, limit: 2e8 })}>+ Aggregate stop loss</button>
        <div style={{ flex: 1 }} />
        <label className="row small sub">Pricing
          <select value={program.pricing_method} aria-label="Pricing method" onChange={(e) => onChange({ ...program, pricing_method: e.target.value as Program['pricing_method'] })}>
            <option value="stdev">EL + k·σ</option><option value="coc">EL + CoC·(TVaR99 − EL)</option>
          </select>
          <input type="number" aria-label="Pricing load" value={program.pricing_load} step={0.05} min={0} style={{ width: 70 }}
            onChange={(e) => onChange({ ...program, pricing_load: Number(e.target.value) })} />
        </label>
      </div>
    </div>
  )
}

// Typed client for the CatForge REST API.

export interface RpRow {
  rp: number; gross_aep: number; gross_aep_lo?: number; gross_aep_hi?: number; gross_aep_tvar: number;
  gross_oep: number; gross_oep_tvar: number; gu_aep: number; net_aep?: number; net_oep?: number
}
export interface Insight {
  id: string; category: string; severity: 'critical' | 'warning' | 'info' | 'positive'; title: string;
  detail: string; recommendation?: string; metric?: unknown
}
export interface LayerMetric {
  name: string; type: string; stage: number; attachment: number; limit: number; placed: number; reinstatements: number;
  expected_loss: number; sd: number; prob_attach: number; prob_exhaust: number; premium: number;
  expected_reinstatement_premium: number; rate_on_line: number | null; loss_on_line: number | null; expected_margin: number
}
export interface CapitalView { aal: number; sd: number; var_99_5: number; var_99: number; tvar_99: number }
export interface ReinsMetrics {
  contracts: LayerMetric[]; gross: CapitalView; net: CapitalView; total_premium: number; total_expected_recovery: number;
  net_cost: number; capital_relief_var995: number; relief_per_cost: number | null
}
export interface Contract {
  name: string; type: 'cat_xl' | 'quota_share' | 'agg_xl'; stage: number; attachment: number; limit: number;
  reinstatements: number; reinstatement_rate: number; aad: number; aal?: number | null; cession: number;
  event_limit?: number | null; placed: number; perils?: string[] | null; premium?: number | null
}
export interface Program { contracts: Contract[]; pricing_method: 'stdev' | 'coc'; pricing_load: number; coc_alpha?: number }
export interface AnalysisSummary {
  id: string; name: string; portfolio_id: string; portfolio_name: string; created: string; config: Record<string, unknown>
  timings: Record<string, number>; n_years: number; n_occurrences: number; n_events_relevant: number; n_pairs: number
  tiv: number; n_locations: number; aal: { gu: number; gross: number; net?: number }; aal_by_peril: Record<string, number>
  sd: number; loss_cost_per_mille: number; elt_aal: number; location_aal_sum: number; rp_table: RpRow[]
  analytic: { rp: number; oep: number; aep: number }[]; analytic_aal: number
  tail_check: { tvar_direct: number; tvar_alloc_sum: number; rp: number; n_tail_years: number; max_abs_occ_diff: number }
  reinsurance: { program: Program; metrics: ReinsMetrics } | null; insights: Insight[]
}
export interface AnalysisBrief {
  id: string; name: string; portfolio_id: string; portfolio_name: string; created: string; n_years: number
  perils: string[]; aal: { gu: number; gross: number; net?: number }; aep_100: number; aep_250: number
  has_reinsurance: boolean; total_s: number
}
export interface EpRow { rp: number; prob: number; loss: number; tvar: number; loss_lo?: number; loss_hi?: number; tvar_lo?: number; tvar_hi?: number }
export interface Curve { rp: number[]; prob: number[]; loss: number[] }
export interface EpResponse {
  basis: string; peril: string; aep: EpRow[]; oep: EpRow[]
  stats: { mean: number; sd: number; cov: number | null; se_mean: number; p_nonzero: number; max: number }
  aep_curve: Curve; oep_curve: Curve
}
export interface Job {
  id: string; kind: string; status: 'queued' | 'running' | 'done' | 'error'; progress: number; message: string
  result_id: string | null; error: string | null; elapsed_s: number; result?: unknown
}
export interface Breakdown { key: string; n: number; tiv: number }
export interface PortfolioSummary {
  id: string; name: string; created: string; n_locations: number; n_accounts: number; tiv_total: number
  tiv_by_coverage: Record<string, number>; bbox: number[]; by_state: Breakdown[]; by_construction: Breakdown[]
  by_occupancy: Breakdown[]; by_lob: Breakdown[]; by_year_band: Breakdown[]; meta: Record<string, unknown>; warnings: string[]
}
export interface PortfolioBrief { id: string; name: string; created: string; n_locations: number; tiv: number; source: string }
export interface Meta {
  version: string; perils: { code: string; name: string }[]; construction_classes: Record<string, string>
  occupancies: Record<string, string>; terrains: string[]; roof_shapes: string[]; dimensions: string[]
  mitigation_presets: Record<string, string>; analogs: Record<string, { label: string; peril: string }>
  default_config: Record<string, unknown>
}
export interface Grid { lat0: number; lon0: number; res_deg: number; ny: number; nx: number; values: number[][]; unit: string }

export class ApiError extends Error {}

async function req<T>(method: string, path: string, body?: unknown, params?: Record<string, unknown>): Promise<T> {
  const q = params
    ? '?' + Object.entries(params).filter(([, v]) => v !== undefined && v !== null)
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`).join('&')
    : ''
  const r = await fetch(`/api${path}${q}`, {
    method, headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) {
    let msg = r.statusText
    try {
      const j = await r.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch { /* not json */ }
    throw new ApiError(`${r.status}: ${msg}`)
  }
  return r.json() as Promise<T>
}

export const api = {
  get: <T,>(path: string, params?: Record<string, unknown>) => req<T>('GET', path, undefined, params),
  post: <T,>(path: string, body?: unknown, params?: Record<string, unknown>) => req<T>('POST', path, body ?? {}, params),
  del: <T,>(path: string) => req<T>('DELETE', path),
  async upload(file: File, name: string): Promise<PortfolioSummary> {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('name', name)
    const r = await fetch('/api/portfolios/upload', { method: 'POST', body: fd })
    const j = await r.json()
    if (!r.ok) throw new ApiError(`${r.status}: ${j.detail}`)
    return j
  },
  /** Poll a background job until it finishes. */
  async waitJob<T = unknown>(job: Job, onUpdate?: (j: Job) => void, intervalMs = 400): Promise<Job & { result: T }> {
    let j = job
    for (;;) {
      j = await api.get<Job>(`/jobs/${j.id}`)
      onUpdate?.(j)
      if (j.status === 'done') return j as Job & { result: T }
      if (j.status === 'error') throw new ApiError(j.error ?? 'job failed')
      await new Promise((res) => setTimeout(res, intervalMs))
    }
  },
}

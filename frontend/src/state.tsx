import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { api } from './api'
import type { AnalysisBrief, Job, Meta, PortfolioBrief } from './api'
import type { Mode } from './theme'

export type Page = 'overview' | 'exposure' | 'hazard' | 'run' | 'results' | 'reinsurance' | 'risklab' | 'scenarios' | 'develop' | 'api'

interface AppState {
  mode: Mode
  setMode: (m: Mode) => void
  meta: Meta | null
  portfolios: PortfolioBrief[]
  analyses: AnalysisBrief[]
  portfolioId: string | null
  setPortfolioId: (id: string | null) => void
  analysisId: string | null
  setAnalysisId: (id: string | null) => void
  refresh: () => Promise<void>
  page: Page
  go: (p: Page) => void
  bootJob: Job | null
  error: string | null
}

const Ctx = createContext<AppState | null>(null)

function readPage(): Page {
  const h = window.location.hash.replace(/^#\/?/, '')
  const valid: Page[] = ['overview', 'exposure', 'hazard', 'run', 'results', 'reinsurance', 'risklab', 'scenarios', 'develop', 'api']
  return (valid.includes(h as Page) ? h : 'overview') as Page
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<Mode>(() => {
    try {
      const s = localStorage.getItem('catforge.mode')
      if (s === 'light' || s === 'dark') return s
    } catch { /* storage unavailable */ }
    return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
  })
  const [meta, setMeta] = useState<Meta | null>(null)
  const [portfolios, setPortfolios] = useState<PortfolioBrief[]>([])
  const [analyses, setAnalyses] = useState<AnalysisBrief[]>([])
  const [portfolioId, setPortfolioId] = useState<string | null>(null)
  const [analysisId, setAnalysisIdState] = useState<string | null>(null)
  const [page, setPage] = useState<Page>(readPage)
  const [bootJob, setBootJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)

  const setMode = (m: Mode) => {
    setModeState(m)
    try { localStorage.setItem('catforge.mode', m) } catch { /* ignore */ }
  }
  useEffect(() => { document.documentElement.dataset.theme = mode }, [mode])
  useEffect(() => {
    const onHash = () => setPage(readPage())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const go = (p: Page) => { window.location.hash = `/${p}` }

  const setAnalysisId = useCallback((id: string | null) => {
    setAnalysisIdState(id)
    if (id) {
      const a = analyses.find((x) => x.id === id)
      if (a) setPortfolioId(a.portfolio_id)
    }
  }, [analyses])

  const refresh = useCallback(async () => {
    try {
      const [p, a] = await Promise.all([api.get<PortfolioBrief[]>('/portfolios'), api.get<AnalysisBrief[]>('/analyses')])
      setPortfolios(p)
      setAnalyses(a)
      setPortfolioId((cur) => cur && p.some((x) => x.id === cur) ? cur : (a[0]?.portfolio_id ?? p[0]?.id ?? null))
      setAnalysisIdState((cur) => cur && a.some((x) => x.id === cur) ? cur : (a[0]?.id ?? null))
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => {
    api.get<Meta>('/meta').then(setMeta).catch((e) => setError(String(e)))
    refresh()
    // follow the server's demo bootstrap job (if any) so the UI populates itself
    api.get<Job[]>('/jobs').then(async (jobs) => {
      const boot = jobs.find((j) => j.kind === 'bootstrap' && (j.status === 'running' || j.status === 'queued'))
      if (boot) {
        setBootJob(boot)
        try {
          await api.waitJob(boot, setBootJob, 800)
        } finally {
          setBootJob(null)
          refresh()
        }
      }
    }).catch(() => undefined)
  }, [refresh])

  const value = useMemo(() => ({
    mode, setMode, meta, portfolios, analyses, portfolioId, setPortfolioId, analysisId, setAnalysisId, refresh, page, go,
    bootJob, error,
  }), [mode, meta, portfolios, analyses, portfolioId, analysisId, setAnalysisId, refresh, page, bootJob, error])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp(): AppState {
  const c = useContext(Ctx)
  if (!c) throw new Error('useApp outside provider')
  return c
}

/** Fetch helper hook: re-runs when deps change (dropping stale data), exposes loading/error; reload() keeps data. */
export function useFetch<T>(fn: (() => Promise<T>) | null, deps: unknown[]): { data: T | null; loading: boolean; error: string | null; reload: () => void } {
  const [state, setState] = useState<{ key: string; data: T | null }>({ key: '', data: null })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const key = JSON.stringify(deps)
  useEffect(() => {
    if (!fn) { setState({ key, data: null }); return }
    let alive = true
    setLoading(true)
    setError(null)
    fn().then((d) => { if (alive) setState({ key, data: d }) }).catch((e) => { if (alive) setError(String(e)) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, tick])
  // never hand out a response that belongs to different inputs
  const data = state.key === key ? state.data : null
  return { data, loading, error, reload: () => setTick((t) => t + 1) }
}

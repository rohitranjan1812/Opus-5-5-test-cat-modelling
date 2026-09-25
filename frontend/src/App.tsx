import { Suspense, lazy } from 'react'
import type { ReactNode } from 'react'
import { AppProvider, useApp } from './state'
import type { Page } from './state'
import { JobStatus } from './components/ui'
import { money, when } from './format'
import Overview from './pages/Overview'
import Exposure from './pages/Exposure'
import Hazard from './pages/Hazard'
import RunAnalysis from './pages/RunAnalysis'
import Results from './pages/Results'
import Reinsurance from './pages/Reinsurance'
import RiskLab from './pages/RiskLab'
import Scenarios from './pages/Scenarios'
import ApiPage from './pages/ApiPage'

const Develop = lazy(() => import('./pages/Develop'))

const I = (d: string) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d={d} />
  </svg>
)
const NAV: { page: Page; label: string; icon: ReactNode; section?: string }[] = [
  { page: 'overview', label: 'Overview', icon: I('M3 12l9-8 9 8M5 10v10h14V10'), section: 'Portfolio' },
  { page: 'exposure', label: 'Exposure', icon: I('M12 21s-7-6.2-7-11a7 7 0 0 1 14 0c0 4.8-7 11-7 11zM12 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z') },
  { page: 'hazard', label: 'Hazard & vulnerability', icon: I('M4 14a8 8 0 0 1 16 0M8 14a4 4 0 0 1 8 0M12 14v7') },
  { page: 'run', label: 'Run analysis', icon: I('M6 4l14 8-14 8z'), section: 'Modelling' },
  { page: 'results', label: 'Results', icon: I('M4 20V10M10 20V4M16 20v-7M22 20H2') },
  { page: 'reinsurance', label: 'Reinsurance', icon: I('M12 3l8 4v5c0 5-3.5 8-8 9-4.5-1-8-4-8-9V7z') },
  { page: 'risklab', label: 'Risk lab', icon: I('M9 3h6M10 3v6L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2L14 9V3') },
  { page: 'scenarios', label: 'Scenarios', icon: I('M3 12h4l3-8 4 16 3-8h4') },
  { page: 'develop', label: 'Event development 3-D', icon: I('M12 3l9 5-9 5-9-5zM3 13l9 5 9-5M3 17.5l9 5 9-5') },
  { page: 'api', label: 'API & SDK', icon: I('M8 6l-6 6 6 6M16 6l6 6-6 6'), section: 'Developers' },
]

function Shell() {
  const app = useApp()
  const { page, go, analyses, analysisId, setAnalysisId, mode, setMode, bootJob, error } = app
  const current = analyses.find((a) => a.id === analysisId)
  const pages: Record<Page, ReactNode> = {
    overview: <Overview />, exposure: <Exposure />, hazard: <Hazard />, run: <RunAnalysis />, results: <Results />,
    reinsurance: <Reinsurance />, risklab: <RiskLab />, scenarios: <Scenarios />,
    develop: <Suspense fallback={<div className="empty">Loading 3-D engine…</div>}><Develop /></Suspense>, api: <ApiPage />,
  }
  return (
    <div className="shell">
      <nav className="sidebar" aria-label="Main">
        <div className="brand">
          <div className="brand-mark">C</div>
          <div>
            <div className="brand-name">CatForge</div>
            <div className="brand-sub">Catastrophe modelling platform</div>
          </div>
        </div>
        {NAV.map((n) => (
          <div key={n.page}>
            {n.section && <div className="nav-section">{n.section}</div>}
            <button className={`nav-item ${page === n.page ? 'active' : ''}`} onClick={() => go(n.page)}
              aria-current={page === n.page ? 'page' : undefined}>
              {n.icon}{n.label}
            </button>
          </div>
        ))}
        <div className="sidebar-foot">
          <a href="/docs" target="_blank" rel="noreferrer">OpenAPI docs ↗</a>
          <span>v{app.meta?.version ?? '…'} · TC + EQ</span>
        </div>
      </nav>
      <div className="main">
        <header className="topbar">
          <label className="row small sub" style={{ gap: 8 }}>
            Analysis
            <select value={analysisId ?? ''} onChange={(e) => setAnalysisId(e.target.value || null)} style={{ minWidth: 320 }}
              aria-label="Select analysis">
              {!analyses.length && <option value="">No analyses yet</option>}
              {analyses.map((a) => (
                <option key={a.id} value={a.id}>{a.name} — {a.portfolio_name} · {when(a.created)}</option>
              ))}
            </select>
          </label>
          {current && <span className="pill num">AAL {money(current.aal.gross)} · 1-in-250 {money(current.aep_250)}</span>}
          <div className="spacer" />
          {bootJob && <div style={{ width: 260 }}><JobStatus job={bootJob} /></div>}
          {error && <span className="error small">{error}</span>}
          <button className="btn sm ghost" onClick={() => setMode(mode === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme">
            {mode === 'dark' ? '☀ Light' : '☾ Dark'}
          </button>
        </header>
        <main className="content">{pages[page]}</main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  )
}

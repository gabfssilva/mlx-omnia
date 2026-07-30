import { type JSX, useEffect, useState } from 'react'
import { json } from './api/http'
import { Overview } from './views/overview/Overview'
import { Models } from './views/models/Models'
import { Chat } from './views/chat/Chat'
import { Download } from './views/download/Download'
import { Quantize } from './views/quantize/Quantize'

export type View = 'overview' | 'chat' | 'models' | 'download' | 'quantize'

/* A view that wants to hand off to another one says so on the window; the rail is the
   only thing that owns which view is on. */
declare global {
  interface WindowEventMap {
    'sideros:view': CustomEvent<View>
  }
}

const VIEWS: { id: View; title: string; icon: JSX.Element; render: () => JSX.Element }[] = [
  {
    id: 'overview',
    title: 'Overview',
    render: () => <Overview />,
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="2" width="20" height="8" rx="2" />
        <rect x="2" y="14" width="20" height="8" rx="2" />
        <line x1="6" y1="6" x2="6.01" y2="6" />
        <line x1="6" y1="18" x2="6.01" y2="18" />
      </svg>
    )
  },
  {
    id: 'chat',
    title: 'Chat',
    render: () => <Chat />,
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
      </svg>
    )
  },
  {
    id: 'models',
    title: 'Models',
    render: () => <Models />,
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2" />
        <polyline points="2 17 12 22 22 17" />
        <polyline points="2 12 12 17 22 12" />
      </svg>
    )
  },
  {
    id: 'download',
    title: 'Download',
    render: () => <Download />,
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
      </svg>
    )
  },
  {
    id: 'quantize',
    title: 'Quantize',
    render: () => <Quantize />,
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <line x1="4" y1="21" x2="4" y2="14" />
        <line x1="4" y1="10" x2="4" y2="3" />
        <line x1="12" y1="21" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12" y2="3" />
        <line x1="20" y1="21" x2="20" y2="16" />
        <line x1="20" y1="12" x2="20" y2="3" />
        <line x1="1" y1="14" x2="7" y2="14" />
        <line x1="9" y1="8" x2="15" y2="8" />
        <line x1="17" y1="16" x2="23" y2="16" />
      </svg>
    )
  }
]

/* Only what the rail draws. Everything else about these two routes belongs to the
   view that asks for it. */
interface RailState {
  resident_bytes: number
  kv_bytes: number
}

interface RailSystem {
  memory_bytes: number
}

interface Health {
  status: string
  pid: number
}

const GIB = 1024 ** 3
const POLL_MS = 2000

export function App(): JSX.Element {
  const [view, setView] = useState<View>('overview')
  const [state, setState] = useState<RailState | null>(null)
  const [system, setSystem] = useState<RailSystem | null>(null)
  const [health, setHealth] = useState<Health | null>(null)

  useEffect(() => {
    json<RailSystem>('/admin/system').then(setSystem).catch(() => setSystem(null))
  }, [])

  useEffect(() => {
    const jump = (event: WindowEventMap['sideros:view']) => {
      if (VIEWS.some((entry) => entry.id === event.detail)) setView(event.detail)
    }
    window.addEventListener('sideros:view', jump)
    return () => window.removeEventListener('sideros:view', jump)
  }, [])

  useEffect(() => {
    let live = true
    const beat = async () => {
      const [next, up] = await Promise.all([
        json<RailState>('/admin/state').catch(() => null),
        json<Health>('/admin/health').catch(() => null)
      ])
      if (!live) return
      setState(next)
      setHealth(up)
    }
    void beat()
    const timer = setInterval(beat, POLL_MS)
    return () => {
      live = false
      clearInterval(timer)
    }
  }, [])

  const used = state === null ? null : state.resident_bytes + state.kv_bytes
  const total = system?.memory_bytes ?? null
  const share = used !== null && total ? Math.round((used / total) * 100) : null
  const memTitle =
    used !== null && total
      ? `Memory · ${Math.round(used / GIB)} of ${Math.round(total / GIB)} GB`
      : 'Memory · unknown'

  return (
    <div className="window">
      <aside className="side">
        <div className="railpanel">
          {/* The real traffic lights land here; see theme.css and desktop/macos.ts. */}
          <div className="lights" />
          <div className="mark">S</div>
          <div className="nav">
            {VIEWS.map((entry) => (
              <button
                key={entry.id}
                className={entry.id === view ? 'on' : undefined}
                title={entry.title}
                aria-label={entry.title}
                aria-current={entry.id === view}
                onClick={() => setView(entry.id)}
              >
                {entry.icon}
              </button>
            ))}
          </div>
          <div className="railfoot">
            <div className="memfoot" title={memTitle}>
              <b>
                {share ?? '—'}
                <small>%</small>
              </b>
              <span className="eyebrow">mem</span>
            </div>
            <span
              className={health === null ? 'dot down' : 'dot'}
              title={health === null ? 'Daemon not answering' : `Daemon answering · pid ${health.pid}`}
            />
          </div>
        </div>
      </aside>

      {/* All views stay mounted — a switch is a display toggle, never a remount with
          its refetch; re-adding `on` still replays the enter animation. */}
      <main className="main">
        {VIEWS.map((entry) => (
          <section className={entry.id === view ? 'view on' : 'view'} key={entry.id}>
            {entry.render()}
          </section>
        ))}
      </main>
    </div>
  )
}

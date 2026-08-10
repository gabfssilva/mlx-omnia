import { type JSX, useEffect, useState } from 'react'
import { EngineProvider, useEngine, whole } from './api/engine'
import { DownloadsProvider, useDownloads, type Pull } from './api/downloads'
import { Meter, scale } from './Memory'
import { Resident } from './views/resident/Resident'
import { Library } from './views/library/Library'
import { Benchmark } from './views/benchmark/Benchmark'
import { Chat } from './views/chat/Chat'
import { Settings } from './views/settings/Settings'

export type View = 'resident' | 'library' | 'benchmark' | 'chat' | 'settings'
export type Theme = 'light' | 'dark' | 'system'

/* A view that wants to hand off to another one says so on the window; the title bar is
   the only thing that owns which view is on. `sideros:ghost` is Library telling the
   meter what the checkpoint under the cursor would cost, and `sideros:theme` is
   Settings telling the document what to stamp. */
declare global {
  interface WindowEventMap {
    'sideros:view': CustomEvent<View>
    'sideros:ghost': CustomEvent<number | null>
    'sideros:theme': CustomEvent<Theme>
  }
}

const VIEWS: { id: View; title: string; render: () => JSX.Element }[] = [
  { id: 'resident', title: 'Resident', render: () => <Resident /> },
  { id: 'library', title: 'Library', render: () => <Library /> },
  { id: 'benchmark', title: 'Benchmark', render: () => <Benchmark /> },
  { id: 'chat', title: 'Chat', render: () => <Chat /> },
  { id: 'settings', title: 'Settings', render: () => <Settings /> }
]

const KEY = 'sideros-theme'

export function readTheme(): Theme {
  const saved = localStorage.getItem(KEY)
  return saved === 'light' || saved === 'dark' ? saved : 'system'
}

/* Nothing stamped is the system following itself, which is what the stylesheet's
   prefers-color-scheme block is for. */
export function writeTheme(theme: Theme): void {
  if (theme === 'system') {
    localStorage.removeItem(KEY)
    document.documentElement.removeAttribute('data-theme')
  } else {
    localStorage.setItem(KEY, theme)
    document.documentElement.setAttribute('data-theme', theme)
  }
}

/* The ring is one reading over every download at once — bytes, not jobs, so a 17 GB
   checkpoint beside a 400 MB one does not report itself as half done. Absent totals are
   absent from both sides of the division rather than counted as zero. */
function Ring({ pulls }: { pulls: Pull[] }): JSX.Element {
  const priced = pulls.filter((pull) => pull.total !== null)
  const total = priced.reduce((sum, pull) => sum + (pull.total ?? 0), 0)
  const done = priced.reduce((sum, pull) => sum + pull.completed, 0)
  const circumference = 2 * Math.PI * 5
  const arc = total === 0 ? 0 : (done / total) * circumference
  return (
    <svg viewBox="0 0 13 13" fill="none" aria-hidden="true">
      <circle className="rail" cx="6.5" cy="6.5" r="5" strokeWidth="2" />
      <circle
        className="arc"
        cx="6.5"
        cy="6.5"
        r="5"
        strokeWidth="2"
        strokeDasharray={`${arc} ${circumference}`}
      />
    </svg>
  )
}

export function App(): JSX.Element {
  const [view, setView] = useState<View>('resident')
  const [ghost, setGhost] = useState<number | null>(null)
  const engine = useEngine()
  const downloads = useDownloads(engine.refresh)

  useEffect(() => {
    writeTheme(readTheme())
  }, [])

  useEffect(() => {
    const jump = (event: WindowEventMap['sideros:view']) => {
      if (VIEWS.some((entry) => entry.id === event.detail)) setView(event.detail)
    }
    const shadow = (event: WindowEventMap['sideros:ghost']) => setGhost(event.detail)
    const paint = (event: WindowEventMap['sideros:theme']) => writeTheme(event.detail)
    window.addEventListener('sideros:view', jump)
    window.addEventListener('sideros:ghost', shadow)
    window.addEventListener('sideros:theme', paint)
    return () => {
      window.removeEventListener('sideros:view', jump)
      window.removeEventListener('sideros:ghost', shadow)
      window.removeEventListener('sideros:theme', paint)
    }
  }, [])

  const ceiling = whole(engine.config?.['memory_limit_bytes'])
  const s = scale(engine.system, engine.state, ceiling)
  const down = engine.health === null

  const pulls = [...downloads.active.values()].filter((pull) => pull.state !== 'error')

  return (
    <EngineProvider value={engine}>
    <DownloadsProvider value={downloads}>
    <div className="window">
      <div className="tb">
        {/* The real traffic lights land here; see theme.css and desktop/main.ts. */}
        <div className="lights" />
        <div className="tabs" role="tablist">
          {VIEWS.map((entry) => (
            <button
              key={entry.id}
              role="tab"
              aria-selected={entry.id === view}
              className={entry.id === view ? 'on' : undefined}
              onClick={() => setView(entry.id)}
            >
              {entry.title}
            </button>
          ))}
        </div>
        <div className="tbr">
          {pulls.length > 0 && (
            <button
              className="chip pulls"
              onClick={() => setView('library')}
              title="Show the downloads in Library"
            >
              <Ring pulls={pulls} />
              Downloading <b>{pulls.length}</b>
            </button>
          )}
          {s !== null && view !== 'resident' && <Meter scale={s} ghost={ghost} />}
          <button
            className="chip"
            onClick={() => setView('settings')}
            title={down ? 'The engine is not answering' : `Engine · pid ${engine.health?.pid ?? '—'}`}
          >
            <i className={down ? 'dot bad' : 'dot ok'} />
            Engine <b className="mono">:{engine.system?.constants['port'] ?? '8642'}</b>
          </button>
        </div>
      </div>

      {/* Every view stays mounted — switching is a display toggle, never a remount with
          its refetch, so a half-typed message survives a trip to Library. */}
      {VIEWS.map((entry) => (
        <div key={entry.id} className="view" hidden={entry.id !== view}>
          {entry.render()}
        </div>
      ))}
    </div>
    </DownloadsProvider>
    </EngineProvider>
  )
}

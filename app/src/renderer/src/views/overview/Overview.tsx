import { type JSX, useCallback, useEffect, useState } from 'react'
import './overview.css'
import {
  type CatalogEntry,
  type ConfigPatch,
  type ConfigView,
  type DaemonState,
  type Health,
  type Policy,
  type Snapshot,
  type SystemInfo,
  daemonAction,
  getCatalog,
  getConfig,
  getHealth,
  getOwnership,
  getState,
  getSystem,
  metricEvents,
  patchConfig,
  policy,
  whole
} from './api'

const GIB = 1024 ** 3
const POLL_MS = 2000
const RETRY_MS = 2000
/* Floor between two `/admin/state` reads driven by a metrics frame. */
const RESTATE_MS = 500

/* The daemon serves no stop or restart route; both act through the desktop shell
   (/desktop/daemon/*), which only controls the process it spawned. The two cases
   where the buttons stay locked, told apart by their titles: */
const NO_SHELL_TITLE = 'No desktop shell reachable to control the daemon'
const FOREIGN_TITLE = 'The daemon was started outside the app; stop it where it was started'

const DIALECTS: { label: string; path: string }[] = [
  { label: 'OpenAI', path: '/api/openai/v1' },
  { label: 'Anthropic', path: '/api/anthropic' },
  { label: 'Gemini', path: '/api/gemini' }
]

const TTLS: { label: string; seconds: number | null }[] = [
  { label: '10 min', seconds: 600 },
  { label: '30 min', seconds: 1800 },
  { label: '2 h', seconds: 7200 },
  { label: 'Never', seconds: null }
]

const POLICIES: { label: string; value: Policy }[] = [
  { label: 'Load, then answer', value: 'load' },
  { label: 'Fail', value: 'fail' }
]

const CLIENTS = ['Claude Code', 'Zed', 'aider'] as const
type Client = (typeof CLIENTS)[number]

function snippet(client: Client, base: string, model: string): string {
  switch (client) {
    case 'Claude Code':
      return `export ANTHROPIC_BASE_URL=\\\n  ${base}/api/anthropic\nexport ANTHROPIC_MODEL=\\\n  ${model}`
    case 'Zed':
      return `"language_models": {\n  "openai": {\n    "api_url":\n      "${base}/api/openai/v1"\n  }\n}`
    case 'aider':
      return `aider \\\n  --openai-api-base \\\n    ${base}/api/openai/v1 \\\n  --model ${model}`
  }
}

/* ── theme ─────────────────────────────────────────────────────────────── */

type ThemeChoice = 'system' | 'light' | 'dark'

/* The key index.html's pre-paint script reads; absent means follow the system. */
const THEME_KEY = 'sideros-theme'

function readTheme(): ThemeChoice {
  const saved = localStorage.getItem(THEME_KEY)
  return saved === 'light' || saved === 'dark' ? saved : 'system'
}

function stampTheme(choice: ThemeChoice): void {
  if (choice === 'system') {
    localStorage.removeItem(THEME_KEY)
    document.documentElement.removeAttribute('data-theme')
    return
  }
  localStorage.setItem(THEME_KEY, choice)
  document.documentElement.setAttribute('data-theme', choice)
}

/* ── formatting ────────────────────────────────────────────────────────── */

const gigabytes = (bytes: number): number => Math.round(bytes / GIB)

function elapsed(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  if (hours > 0) return `${hours}h ${minutes}m`
  if (minutes > 0) return `${minutes}m`
  return `${total}s`
}

/* `sysctl` answers "Apple M5 Max"; the machine is what is left after the vendor. */
const machine = (chip: string): string => chip.replace(/^Apple\s+/, '')

const message = (failure: unknown): string =>
  failure instanceof Error ? failure.message : String(failure)

function CopyIcon({ done }: { done: boolean }): JSX.Element {
  return done ? (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ) : (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  )
}

function Hstat({ label, value, unit }: { label: string; value: string; unit?: string }): JSX.Element {
  return (
    <div className="hstat">
      <span className="eyebrow">{label}</span>
      <b>
        {value}
        {unit === undefined ? null : <small>{unit}</small>}
      </b>
    </div>
  )
}

/* ── view ──────────────────────────────────────────────────────────────── */

export function Overview(): JSX.Element {
  const [system, setSystem] = useState<SystemInfo | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [state, setState] = useState<DaemonState | null>(null)
  const [metrics, setMetrics] = useState<Snapshot | null>(null)
  const [config, setConfig] = useState<ConfigView | null>(null)
  const [catalog, setCatalog] = useState<CatalogEntry[] | null>(null)
  const [failure, setFailure] = useState<string | null>(null)

  /* What the two editable controls show while they are being moved; the config the
     daemon answers with is the truth the moment the PATCH lands. */
  const [draftLimit, setDraftLimit] = useState<number | null>(null)
  const [draftAddress, setDraftAddress] = useState<string | null>(null)

  const [theme, setTheme] = useState<ThemeChoice>(readTheme)
  const [client, setClient] = useState<Client>('Claude Code')
  const [copied, setCopied] = useState<string | null>(null)
  /* null: no shell answered — the buttons stay locked rather than guessing. */
  const [owned, setOwned] = useState<boolean | null>(null)
  const [acting, setActing] = useState(false)

  useEffect(() => {
    void getSystem().then(setSystem).catch(() => setSystem(null))
    void getConfig().then(setConfig).catch(() => setConfig(null))
    void getCatalog().then(setCatalog).catch(() => setCatalog(null))
    void getOwnership().then((it) => setOwned(it.owned)).catch(() => setOwned(null))
  }, [])

  useEffect(() => {
    let live = true
    const beat = async (): Promise<void> => {
      const [next, up] = await Promise.all([
        getState().catch(() => null),
        getHealth().catch(() => null)
      ])
      if (!live) return
      setState(next)
      setHealth(up)
    }
    void beat()
    const timer = setInterval(() => void beat(), POLL_MS)
    return () => {
      live = false
      clearInterval(timer)
    }
  }, [])

  /* The register publishes at both boundaries of a request and resamples on the silent
     tick, so a frame is the live numbers and the cue that residency and the queue
     moved — which is what `/admin/state` answers. */
  useEffect(() => {
    const abort = new AbortController()
    let live = true
    let read = 0
    const watch = async (): Promise<void> => {
      while (live) {
        try {
          for await (const frame of metricEvents(abort.signal)) {
            if (!live) return
            setMetrics(frame)
            /* A generation publishes twice a second; the residency behind it does not
               move that fast, and the poll alongside covers the rest. */
            if (Date.now() - read < RESTATE_MS) continue
            read = Date.now()
            setState(await getState().catch(() => null))
          }
        } catch {
          /* the daemon went away; the poll above keeps the screen honest */
        }
        if (!live) return
        setMetrics(null)
        await new Promise((wake) => setTimeout(wake, RETRY_MS))
      }
    }
    void watch()
    return () => {
      live = false
      abort.abort()
    }
  }, [])

  const commit = useCallback(async (patch: ConfigPatch): Promise<void> => {
    try {
      setConfig(await patchConfig(patch))
      setFailure(null)
    } catch (error) {
      setFailure(message(error))
    }
    setDraftLimit(null)
    setDraftAddress(null)
  }, [])

  const act = useCallback(async (action: 'stop' | 'restart'): Promise<void> => {
    setActing(true)
    try {
      await daemonAction(action)
      setFailure(null)
    } catch (error) {
      setFailure(message(error))
    }
    setOwned(await getOwnership().then((it) => it.owned).catch(() => null))
    setActing(false)
  }, [])

  const copy = useCallback(async (label: string, text: string): Promise<void> => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(label)
      setTimeout(() => setCopied(null), 1200)
    } catch (error) {
      setFailure(message(error))
    }
  }, [])

  const used = state === null ? null : state.resident_bytes + state.kv_bytes
  const limitBytes = config === null ? null : whole(config.memory_limit_bytes)
  const limitGb = limitBytes === null ? null : gigabytes(limitBytes)
  const installedGb = system === null ? null : gigabytes(system.memory_bytes)
  const sliderGb = draftLimit ?? limitGb
  const ttl = config === null ? null : whole(config.idle_ttl_seconds)
  const notResident = config === null ? null : policy(config.not_resident)
  const port = config === null ? null : whole(config.port)
  const host = port === null ? null : `127.0.0.1:${port}`
  const base = host === null ? null : `http://${host}`
  const running = metrics === null ? (state?.queue.running ?? 0) : metrics.live === null ? 0 : 1
  const catalogBytes =
    catalog === null ? null : catalog.reduce((sum, entry) => sum + entry.bytes_on_disk, 0)
  const served = state?.models[0]?.id ?? catalog?.[0]?.id ?? '<model-id>'
  /* Stop needs a daemon of ours; Restart also stands in for "start" when nothing
     answers at all — a stopped daemon has no owner to ask. */
  const canStop = !acting && owned === true
  const canRestart = !acting && (owned === true || (owned === false && health === null))
  const lockTitle =
    owned === null ? NO_SHELL_TITLE : owned === false && health !== null ? FOREIGN_TITLE : undefined

  return (
    <>
      <div className="vhead">
        <h1>Sideros</h1>
        {health === null ? (
          <span className="chip idle ov-run">
            <span className="dot" />
            not answering
          </span>
        ) : (
          <span className="chip res ov-run">
            <span className="dot" style={{ background: 'var(--ok)' }} />
            running · {elapsed(health.uptime)}
          </span>
        )}
        <div className="hstats">
          <Hstat label="Machine" value={system === null ? '—' : machine(system.chip)} />
          <Hstat label="GPU" value={system === null ? '—' : String(system.gpu_cores)} unit="cores" />
          <Hstat
            label="Bandwidth"
            value={system === null ? '—' : String(Math.round(system.bandwidth_sustained_gbs))}
            unit="GB/s"
          />
          <Hstat
            label="Memory"
            value={used === null ? '—' : String(gigabytes(used))}
            unit={limitGb === null ? 'GB' : `/ ${limitGb} GB`}
          />
          <Hstat
            label="Catalog"
            value={catalogBytes === null ? '—' : String(gigabytes(catalogBytes))}
            unit={catalog === null ? 'GB' : `GB · ${catalog.length} models`}
          />
          <Hstat label="Queue" value={String(running)} unit="running" />
        </div>
        <div className="trail">
          <button
            className={`btn solid${canRestart ? '' : ' ov-locked'}`}
            disabled={!canRestart}
            title={lockTitle}
            onClick={() => void act('restart')}
          >
            Restart
          </button>
          <button
            className={`btn danger${canStop ? '' : ' ov-locked'}`}
            disabled={!canStop}
            title={lockTitle}
            onClick={() => void act('stop')}
          >
            Stop
          </button>
        </div>
      </div>

      <div className="vbody">
        {failure === null ? null : <div className="ov-failure eyebrow">{failure}</div>}

        <div className="cols2">
          <div className="colstack">
            <div className="card">
              <h3>Residency</h3>
              <div className="fieldcol">
                <div className="ov-fieldhead">
                  <span className="eyebrow">Memory ceiling</span>
                  <b className="tn">{sliderGb === null ? '—' : `${sliderGb} GB`}</b>
                </div>
                <input
                  type="range"
                  className="ov-ceiling"
                  min={8}
                  max={installedGb ?? Math.max(8, sliderGb ?? 8)}
                  value={sliderGb ?? 8}
                  disabled={sliderGb === null || installedGb === null}
                  aria-label="Memory ceiling in gigabytes"
                  onChange={(event) => setDraftLimit(Number(event.target.value))}
                  onPointerUp={() => {
                    if (draftLimit !== null) void commit({ memory_limit_bytes: draftLimit * GIB })
                  }}
                  onKeyUp={() => {
                    if (draftLimit !== null) void commit({ memory_limit_bytes: draftLimit * GIB })
                  }}
                />
              </div>
              <div className="fieldcol">
                <span className="eyebrow">Idle unload</span>
                <div className="seg">
                  {TTLS.map((option) => (
                    <button
                      key={option.label}
                      className={config !== null && ttl === option.seconds ? 'on' : undefined}
                      disabled={config === null}
                      onClick={() => void commit({ idle_ttl_seconds: option.seconds })}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="fieldcol">
                <span className="eyebrow">Model not resident</span>
                <div className="seg">
                  {POLICIES.map((option) => (
                    <button
                      key={option.value}
                      className={notResident === option.value ? 'on' : undefined}
                      disabled={config === null}
                      onClick={() => void commit({ not_resident: option.value })}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="card">
              <h3>
                Engine <span>{health === null ? 'not answering' : `pid ${health.pid}`}</span>
              </h3>
              <div className="switchrow">
                <div className="lab">
                  <b>Address</b>
                  <span>
                    {config?.port.effect === 'restart'
                      ? 'The port is fixed when the process comes up: a change counts from the next start'
                      : 'Where every dialect answers'}
                  </span>
                </div>
                <input
                  className="input mono ov-address"
                  value={draftAddress ?? base ?? ''}
                  disabled={base === null}
                  aria-label="Address the daemon is bound to"
                  onChange={(event) => setDraftAddress(event.target.value)}
                  onBlur={() => {
                    if (draftAddress === null) return
                    const next = Number(draftAddress.split(':').pop())
                    if (!Number.isInteger(next) || next < 1 || next > 65535) {
                      setFailure(`port: ${draftAddress} names no port between 1 and 65535`)
                      setDraftAddress(null)
                      return
                    }
                    if (next === port) {
                      setDraftAddress(null)
                      return
                    }
                    void commit({ port: next })
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') event.currentTarget.blur()
                    if (event.key === 'Escape') setDraftAddress(null)
                  }}
                />
              </div>
              <div className="switchrow">
                <div className="lab">
                  <b>Start with the app</b>
                  <span>The engine follows the window's lifecycle</span>
                </div>
                <div
                  className="seg ov-locked"
                  title="Not configurable yet: the daemon the app starts dies with it"
                >
                  <button className="on" disabled>
                    On
                  </button>
                  <button disabled>Off</button>
                </div>
              </div>
              <div className="switchrow">
                <div className="lab">
                  <b>Version</b>
                </div>
                <code className="mono ov-path">
                  {health === null ? (system?.version ?? '—') : health.version}
                </code>
              </div>
            </div>
          </div>

          <div className="colstack">
            <div className="card">
              <h3>
                Endpoints <span>{host ?? '—'}</span>
              </h3>
              {DIALECTS.map((dialect) => (
                <div className="copyrow" key={dialect.label}>
                  <span className="eyebrow">{dialect.label}</span>
                  <code>{dialect.path}</code>
                  <button
                    className="iconbtn ov-copy"
                    title="Copy full URL"
                    aria-label={`Copy the ${dialect.label} base URL`}
                    disabled={base === null}
                    onClick={() => {
                      if (base !== null) void copy(dialect.label, `${base}${dialect.path}`)
                    }}
                  >
                    <CopyIcon done={copied === dialect.label} />
                  </button>
                </div>
              ))}
              <div className="ov-clients">
                <div className="seg">
                  {CLIENTS.map((name) => (
                    <button
                      key={name}
                      className={name === client ? 'on' : undefined}
                      onClick={() => setClient(name)}
                    >
                      {name}
                    </button>
                  ))}
                </div>
                <pre className="code">{snippet(client, base ?? 'http://127.0.0.1:8642', served)}</pre>
              </div>
            </div>

            <div className="card">
              <h3>Appearance</h3>
              <div className="switchrow">
                <div className="lab">
                  <b>Theme</b>
                  <span>Follows the system by default</span>
                </div>
                <div className="seg">
                  {(['system', 'light', 'dark'] as const).map((choice) => (
                    <button
                      key={choice}
                      className={choice === theme ? 'on' : undefined}
                      onClick={() => {
                        stampTheme(choice)
                        setTheme(choice)
                      }}
                    >
                      {choice === 'system' ? 'System' : choice === 'light' ? 'Light' : 'Dark'}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

/* Where what is neither a block nor a checkpoint lives: the three dialects, the port,
   the ceiling, idle unloading, the engine process and the theme. */

import { type JSX, useCallback, useEffect, useState } from 'react'
import { GIB, gb, useEngineContext, whole } from '../../api/engine'
import { readTheme, writeTheme, type Theme } from '../../App'
import {
  type CatalogEntry,
  confirmSheet,
  daemonAction,
  getOwnership,
  patchConfig,
  type Policy
} from '../overview/api'
import { displayName, homeless } from '../models/format'

/* The daemon serves no stop or restart route; both act through the desktop shell
   (/desktop/daemon/*), which only controls the process it spawned. */
const NO_SHELL = 'No desktop shell is reachable to control the engine.'
const FOREIGN = 'The engine was started outside the app. Stop it where it was started.'

const DIALECTS = [
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

const CLIENTS = ['Claude Code', 'Zed', 'aider'] as const
type Client = (typeof CLIENTS)[number]

/* Claude Code picks a model per alias: ANTHROPIC_MODEL is what it starts on, and each
   ANTHROPIC_DEFAULT_<TIER>_MODEL is what that alias resolves to when /model switches. */
const TIERS = ['opus', 'sonnet', 'haiku', 'fable'] as const
type Tier = (typeof TIERS)[number]
type Picks = { default: string } & Record<Tier, string>

const EMPTY: Picks = { default: '', opus: '', sonnet: '', haiku: '', fable: '' }

/* Zed's snippet names no model; aider's line names one; Claude Code names one per alias. */
function slots(client: Client): readonly (keyof Picks)[] {
  switch (client) {
    case 'Claude Code':
      return ['default', ...TIERS]
    case 'Zed':
      return []
    case 'aider':
      return ['default']
  }
}

function line(client: Client, base: string, key: string, picks: Picks, fallback: string): string {
  const model = picks.default === '' ? fallback : picks.default
  switch (client) {
    case 'Claude Code': {
      const env = [`ANTHROPIC_BASE_URL=${base}/api/anthropic`]
      if (key !== '') env.push(`ANTHROPIC_AUTH_TOKEN=${key}`)
      if (picks.default !== '') env.push(`ANTHROPIC_MODEL=${picks.default}`)
      for (const tier of TIERS) {
        if (picks[tier] !== '') env.push(`ANTHROPIC_DEFAULT_${tier.toUpperCase()}_MODEL=${picks[tier]}`)
      }
      /* A local model answers slower than the API and has nothing to report home about:
         the default timeout would cut a long generation short. */
      env.push('API_TIMEOUT_MS=3000000', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1')
      return `${env.join(' ')} claude`
    }
    case 'Zed':
      return `"language_models": { "openai": { "api_url": "${base}/api/openai/v1" } }`
    case 'aider':
      return `aider --openai-api-base ${base}/api/openai/v1 --model ${model}`
  }
}

/* What the prefix budget buys, read on a model that is actually loaded. A byte count on
   its own says nothing here: the trie holds conversations, and how many tokens fit in a
   gigabyte is the checkpoint's `kv_bytes_per_token` away. The costliest resident cache is
   the one quoted — it is the model that runs out first, and a screen that quoted the
   cheapest would promise a length the expensive one never reaches. */
function holds(
  budget: number | null,
  catalog: readonly CatalogEntry[] | null
): { id: string; tokens: number } | null {
  if (budget === null || catalog === null) return null
  let worst: { id: string; tokens: number } | null = null
  for (const entry of catalog) {
    const perToken = entry.kv_bytes_per_token
    if (!entry.resident || perToken === null || perToken <= 0) continue
    const tokens = Math.floor(budget / perToken)
    if (worst === null || tokens < worst.tokens) worst = { id: entry.id, tokens }
  }
  return worst
}

const thousands = (tokens: number): string =>
  tokens >= 1000 ? `${Math.round(tokens / 1000)}k` : String(tokens)

const elapsed = (seconds: number): string => {
  const total = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  if (hours > 0) return `${hours} h ${minutes} min`
  if (minutes > 0) return `${minutes} min`
  return `${total} s`
}

export function Settings(): JSX.Element {
  const engine = useEngineContext()
  const [theme, setTheme] = useState<Theme>(readTheme)
  const [owned, setOwned] = useState<boolean | null>(null)
  const [client, setClient] = useState<Client>('Claude Code')
  const [picks, setPicks] = useState<Picks>(EMPTY)
  const [copied, setCopied] = useState<string | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const [ceilingGb, setCeilingGb] = useState<number | null>(null)

  useEffect(() => {
    getOwnership()
      .then((answer) => setOwned(answer.owned))
      .catch(() => setOwned(null))
  }, [])

  const configured = whole(engine.config?.['memory_limit_bytes'])
  const totalBytes = engine.system?.memory_bytes ?? null
  const ceiling = ceilingGb ?? (configured === null ? null : Math.round(configured / GIB))
  const prefixBytes = whole(engine.config?.['prefix_cache_bytes'])
  const conversation = holds(prefixBytes, engine.catalog)
  const diskBytes = whole(engine.config?.['prefix_disk_bytes'])
  const ttl = engine.config?.['idle_ttl_seconds']
  const port = engine.config?.['port']?.value ?? 8642
  const base = `http://127.0.0.1:${String(port)}`
  const firstModel = engine.catalog?.[0]?.id ?? 'model-id'
  const apiKey = engine.config?.['api_key']?.value
  const command = line(client, base, typeof apiKey === 'string' ? apiKey : '', picks, firstModel)

  const copy = useCallback((key: string, text: string) => {
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(key)
      setTimeout(() => setCopied((current) => (current === key ? null : current)), 1400)
    })
  }, [])

  const commit = (patch: Parameters<typeof patchConfig>[0]) => {
    setFailure(null)
    void patchConfig(patch)
      .then(() => engine.refresh())
      .catch((error: unknown) => setFailure(error instanceof Error ? error.message : String(error)))
  }

  const control = (action: 'stop' | 'restart') => {
    setFailure(null)
    void (async () => {
      if (action === 'stop' && !(await confirmSheet('Stop the engine?', 'Stop', 'Loaded models are unloaded and clients stop being answered.'))) return
      await daemonAction(action)
      engine.refresh()
    })().catch((error: unknown) => setFailure(error instanceof Error ? error.message : String(error)))
  }

  return (
    <div className="wk">
      <main className="pane mid">
        <div className="ph">
          <h2>Settings</h2>
          {engine.system !== null && (
            <span className="meta">
              {engine.system.chip} · {engine.system.gpu_cores} GPU cores ·{' '}
              {gb(engine.system.memory_bytes, 0)} GB · {engine.system.bandwidth_sustained_gbs} GB/s
            </span>
          )}
        </div>
        <div className="pbody scroll">
          {failure !== null && <div className="err" style={{ marginBottom: 9 }}>{failure}</div>}
          <div className="cols">
            <div className="colstack">
              <section className="blk">
                <h3>Endpoints</h3>
                {DIALECTS.map((dialect) => {
                  const url = `${base}${dialect.path}`
                  return (
                    <div className="urlrow" key={dialect.path}>
                      <span className="u">{url}</span>
                      <button className="btn quiet" onClick={() => copy(dialect.path, url)}>
                        {copied === dialect.path ? 'Copied' : 'Copy'}
                      </button>
                    </div>
                  )
                })}
                <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                  <span className="note">Port</span>
                  <input
                    className="input mono"
                    style={{ width: 82 }}
                    defaultValue={String(port)}
                    onBlur={(event) => {
                      const next = Number(event.currentTarget.value)
                      if (Number.isInteger(next) && next !== Number(port)) commit({ port: next })
                    }}
                  />
                  <span className="note">Changing the port restarts the engine.</span>
                </div>
              </section>

              <section className="blk">
                <h3>Clients</h3>
                <div className="urlrow">
                  <span className="u">{command}</span>
                  <button className="btn quiet" onClick={() => copy(client, command)}>
                    {copied === client ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  {CLIENTS.map((name) => (
                    <button
                      key={name}
                      className={name === client ? 'fchip on' : 'fchip'}
                      onClick={() => setClient(name)}
                    >
                      {name}
                    </button>
                  ))}
                </div>
                <div className="kvs">
                  {slots(client).map((slot) => (
                    <div className="kvr" key={slot}>
                      <span className="k" style={{ textTransform: 'capitalize' }}>{slot}</span>
                      <select
                        className="input mono"
                        style={{ marginLeft: 'auto', maxWidth: 260 }}
                        value={picks[slot]}
                        onChange={(event) => {
                          const id = event.currentTarget.value
                          setPicks((current) => ({ ...current, [slot]: id }))
                        }}
                      >
                        <option value="">{slot === 'default' ? '—' : 'follows the default'}</option>
                        {(engine.catalog ?? []).map((entry) => (
                          <option key={entry.id} value={entry.id}>
                            {entry.id}
                          </option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
                <p className="note">
                  {client === 'Claude Code'
                    ? 'Default is what Claude Code starts on; each alias is what /model resolves to.'
                    : 'Copy writes the whole line, ready to paste.'}
                </p>
              </section>

              {engine.system !== null && (
                <section className="blk">
                  <h3>Storage</h3>
                  <div className="kvs">
                    <div className="kvr">
                      <span className="k">Checkpoints</span>
                      <span className="v">
                        {gb(engine.catalog?.reduce((sum, item) => sum + item.bytes_on_disk, 0) ?? 0)} GB{' '}
                        <em>· {engine.catalog?.length ?? 0}</em>
                      </span>
                    </div>
                    <div className="kvr">
                      <span className="k">Folder</span>
                      <span className="v">{homeless(engine.system.catalog)}</span>
                    </div>
                    <div className="kvr">
                      <span className="k">Disk free</span>
                      <span className="v">{gb(engine.system.disk_free_bytes, 0)} GB</span>
                    </div>
                  </div>
                </section>
              )}
            </div>

            <div className="colstack">
              <section className="blk">
                <h3>Memory</h3>
                {totalBytes !== null && ceiling !== null ? (
                  <>
                    <div className="sublab">
                      <span>ceiling {ceiling} GB</span>
                      <span>{gb(totalBytes, 0)} GB</span>
                    </div>
                    <input
                      type="range"
                      min={4}
                      max={Math.round(totalBytes / GIB)}
                      step={1}
                      value={ceiling}
                      onChange={(event) => setCeilingGb(Number(event.currentTarget.value))}
                      onMouseUp={() => commit({ memory_limit_bytes: ceiling * GIB })}
                      onKeyUp={() => commit({ memory_limit_bytes: ceiling * GIB })}
                    />
                    <p className="note">
                      Above the ceiling the engine refuses to load rather than letting macOS swap.{' '}
                      {gb((engine.state?.resident_bytes ?? 0) + (engine.state?.kv_bytes ?? 0))} GB in use
                      right now.
                    </p>
                  </>
                ) : (
                  <p className="note">Waiting for the engine.</p>
                )}
                {prefixBytes !== null && (
                  <p className="note">
                    Conversation cache: {gb(prefixBytes)} GB across every loaded model, inside the
                    ceiling above.
                    {conversation !== null && (
                      <>
                        {' '}
                        About {thousands(conversation.tokens)} tokens on{' '}
                        {displayName(conversation.id)} — a turn that continues one of those is
                        answered without reading the whole conversation again.
                      </>
                    )}
                  </p>
                )}
                {diskBytes !== null && (
                  <p className="note">
                    On disk: {gb(engine.state?.prefix_disk_bytes ?? 0)} GB of {gb(diskBytes)} GB.
                    What is written there outlives a restart, which the one above does not.
                  </p>
                )}
                <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                  <span className="note" style={{ flex: 1 }}>
                    Unload a model after
                  </span>
                  <div className="seg">
                    {TTLS.map((option) => (
                      <button
                        key={option.label}
                        className={whole(ttl) === option.seconds || (option.seconds === null && ttl?.value === null) ? 'on' : undefined}
                        onClick={() => commit({ idle_ttl_seconds: option.seconds })}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                  <span className="note" style={{ flex: 1 }}>
                    A request for a model that is not loaded
                  </span>
                  <div className="seg">
                    {(['load', 'fail'] as Policy[]).map((value) => (
                      <button
                        key={value}
                        className={engine.config?.['not_resident']?.value === value ? 'on' : undefined}
                        onClick={() => commit({ not_resident: value })}
                      >
                        {value === 'load' ? 'Loads it' : 'Fails'}
                      </button>
                    ))}
                  </div>
                </div>
              </section>

              <section className="blk">
                <h3>Engine</h3>
                <div className="kvs">
                  <div className="kvr">
                    <span className="k">Status</span>
                    <span className="v">
                      {engine.health === null ? 'not answering' : 'running'}
                      {engine.health !== null && <em> · {elapsed(engine.health.uptime)}</em>}
                    </span>
                  </div>
                  <div className="kvr">
                    <span className="k">Process</span>
                    <span className="v">
                      {owned === true ? 'started by this app' : owned === false ? 'started elsewhere' : 'unknown'}
                      {engine.health !== null && <em> · pid {engine.health.pid}</em>}
                    </span>
                  </div>
                  <div className="kvr">
                    <span className="k">Version</span>
                    <span className="v">
                      sideros {engine.system?.version ?? '—'}
                      {engine.system?.constants['mlx'] !== undefined && <em> · mlx {engine.system.constants['mlx']}</em>}
                    </span>
                  </div>
                </div>
                {owned === true ? (
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button className="btn" onClick={() => control('restart')}>
                      Restart
                    </button>
                    <button className="btn danger" onClick={() => control('stop')}>
                      Stop
                    </button>
                  </div>
                ) : (
                  <p className="note">{owned === false ? FOREIGN : NO_SHELL}</p>
                )}
              </section>

              <section className="blk">
                <h3>Appearance</h3>
                <div className="seg">
                  {(['light', 'dark', 'system'] as Theme[]).map((option) => (
                    <button
                      key={option}
                      className={option === theme ? 'on' : undefined}
                      onClick={() => {
                        setTheme(option)
                        writeTheme(option)
                      }}
                    >
                      {option === 'light' ? 'Light' : option === 'dark' ? 'Dark' : 'System'}
                    </button>
                  ))}
                </div>
              </section>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

/* Every checkpoint, wherever it is. What used to be two screens — Models and Download —
   differed by something the person does not control: whether the bytes had already come
   down. Here it is one shelf and a filter, and the same card answers "load it" and
   "fetch it, then load it". */

import { type JSX, useCallback, useEffect, useMemo, useState } from 'react'
import './library.css'
import { gb, useEngineContext, whole, type CatalogEntry } from '../../api/engine'
import { CardBlock } from '../models/Card'
import { Profiles } from '../models/Profiles'
import { Speculation } from '../models/Speculation'
import { loadModel, reason, removeModel, unloadModel } from '../models/api'
import { dropStaged, hubFiles, listStaged, pull, searchHub, type HubModel, type Staged } from '../download/api'
import { contextTokens, displayName, owner } from '../models/format'
import { Quantize } from '../quantize/Quantize'
import { eta, useDownloadsContext, type Pull } from '../../api/downloads'
import { left, rate, size, sizePair } from '../download/format'

type Filter = 'all' | 'resident' | 'disk' | 'incomplete' | 'hub'
type Tab = 'card' | 'files' | 'profiles' | 'source'

interface Chosen {
  id: string
  hub: boolean
}

const FILTERS: { id: Filter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'resident', label: 'Resident' },
  { id: 'disk', label: 'On disk' },
  { id: 'incomplete', label: 'Incomplete' },
  { id: 'hub', label: 'Hub' }
]

/* Null while the total is unknown, which is what keeps a bar from being drawn at all: the
   download learns its size from the Hub's first answer. */
const fraction = (entry: Pull): number | null =>
  entry.total === null || entry.total === 0 ? null : entry.completed / entry.total

const share = (entry: Pull): string => {
  const part = fraction(entry)
  return part === null ? '—' : `${Math.round(part * 100)}%`
}

function Magnifier(): JSX.Element {
  return (
    <svg viewBox="0 0 11 11" fill="none" aria-hidden="true">
      <circle cx="4.6" cy="4.6" r="3.4" stroke="currentColor" strokeWidth="1.2" />
      <path d="M7.3 7.3 10 10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  )
}

export function Library(): JSX.Element {
  const engine = useEngineContext()
  const downloads = useDownloadsContext()
  const [staged, setStaged] = useState<Staged[]>([])
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [hub, setHub] = useState<HubModel[]>([])
  const [chosen, setChosen] = useState<Chosen | null>(null)
  const [tab, setTab] = useState<Tab>('card')
  const [price, setPrice] = useState<number | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [quantizing, setQuantizing] = useState<string | null>(null)

  const disk = engine.catalog ?? []

  /* Bytes on disk with no model to show for them. Read on the same beat as the catalog
     because that is when they appear and disappear: a download landing turns staging into a
     checkpoint, and a discard turns it into free space. */
  useEffect(() => {
    void listStaged()
      .then(setStaged)
      .catch(() => setStaged([]))
  }, [engine.catalog, downloads.active])

  const materials = useMemo(
    () => new Map((engine.state?.models ?? []).map((model, index) => [model.id, `m${(index % 6) + 1}`])),
    [engine.state]
  )

  /* The Hub is only searched once there is something to search for. */
  useEffect(() => {
    const term = query.trim()
    if (term.length < 2 || filter === 'resident' || filter === 'disk' || filter === 'incomplete') {
      setHub([])
      return
    }
    const timer = setTimeout(() => {
      void searchHub(term)
        .then(setHub)
        .catch(() => setHub([]))
    }, 350)
    return () => clearTimeout(timer)
  }, [query, filter])

  const term = query.trim().toLowerCase()
  const matches = (id: string): boolean => term === '' || id.toLowerCase().includes(term)

  const shown = useMemo(() => {
    const wanted = query.trim().toLowerCase()
    return disk
      .filter((entry) => (filter === 'resident' ? entry.resident : filter === 'all' || filter === 'disk'))
      .filter((entry) => wanted === '' || entry.id.toLowerCase().includes(wanted))
  }, [disk, filter, query])

  const onDisk = new Set(disk.map((entry) => entry.id))
  const hubShown = hub.filter((model) => !onDisk.has(model.id))

  /* A repository is in exactly one of these: the daemon is fetching it, it has bytes from a
     download that stopped, or it is neither. A failed pull counts as incomplete — it is the
     same bytes, and it is the one that can also say why it stopped. */
  const pulls = [...downloads.active.values()].filter((entry) => !onDisk.has(entry.repo))
  const running = pulls.filter((entry) => entry.state !== 'error')
  const failed = pulls.filter((entry) => entry.state === 'error')
  const abandoned = staged.filter(
    (entry) => !downloads.active.has(entry.repo) && !onDisk.has(entry.repo)
  )

  const incomplete = filter === 'all' || filter === 'incomplete'
  const pullsShown = (filter === 'all' ? running : []).filter((entry) => matches(entry.repo))
  const failedShown = (incomplete ? failed : []).filter((entry) => matches(entry.repo))
  const abandonedShown = (incomplete ? abandoned : []).filter((entry) => matches(entry.repo))

  const entry = chosen !== null && !chosen.hub ? disk.find((item) => item.id === chosen.id) ?? null : null

  /* The inspector's blocks each arrive on their own roundtrip. Showing them as they land is
     what reads as a flicker, so the body is held blank until every block the current tab
     mounts has reported in. The flags are stamped with the selection they belong to, which
     is what keeps a late answer from unblanking the next checkpoint. */
  const picked = chosen === null ? '' : `${chosen.hub ? 'hub' : 'disk'}:${chosen.id}`
  const [settled, setSettled] = useState({ picked: '', card: false, spec: false })
  const flags = settled.picked === picked ? settled : { picked, card: false, spec: false }
  const settle = useCallback(
    (which: 'card' | 'spec') =>
      setSettled((prev) => {
        const base = prev.picked === picked ? prev : { picked, card: false, spec: false }
        return which === 'card' ? { ...base, card: true } : { ...base, spec: true }
      }),
    [picked]
  )

  /* What the selection would cost, drawn as a ghost on the title bar's meter. */
  useEffect(() => {
    if (chosen === null) {
      setPrice(null)
      window.dispatchEvent(new CustomEvent('sideros:ghost', { detail: null }))
      return
    }
    if (entry !== null) {
      const cost = entry.resident ? null : entry.bytes_on_disk
      setPrice(cost)
      window.dispatchEvent(new CustomEvent('sideros:ghost', { detail: cost }))
      return
    }
    let live = true
    void hubFiles(chosen.id)
      .then((files) => {
        if (!live) return
        const total = files.reduce((sum, file) => sum + file.size, 0)
        setPrice(total)
        window.dispatchEvent(new CustomEvent('sideros:ghost', { detail: total }))
      })
      .catch(() => {
        if (live) setPrice(null)
      })
    return () => {
      live = false
      window.dispatchEvent(new CustomEvent('sideros:ghost', { detail: null }))
    }
  }, [chosen, entry])

  const act = useCallback(
    async (run: () => Promise<unknown>) => {
      setBusy(true)
      setFailure(null)
      try {
        await run()
        engine.refresh()
      } catch (error: unknown) {
        setFailure(reason(error))
      } finally {
        setBusy(false)
      }
    },
    [engine]
  )

  /* The selection's own bytes-in-flight, if it has any. A failed pull and a staged
     repository are the same 7 GB on disk; the pull is the one that also says why it stopped. */
  const flight = chosen === null ? null : downloads.active.get(chosen.id) ?? null
  const halted = chosen === null ? null : staged.find((item) => item.repo === chosen.id) ?? null
  const stalled = flight !== null && flight.state === 'error'
  const fetching = flight !== null && !stalled

  const start = useCallback(
    (repo: string) => act(() => pull(repo).then(downloads.follow)),
    [act, downloads]
  )

  const ceiling = whole(engine.config?.['memory_limit_bytes'])
  const used = (engine.state?.resident_bytes ?? 0) + (engine.state?.kv_bytes ?? 0)
  const free = ceiling === null ? null : ceiling - used
  const tooBig = price !== null && free !== null && price > free

  /* Which of the two asynchronous blocks the current tab puts on screen. */
  const speculates = entry !== null && tab === 'card'
  const carded = tab !== 'profiles'

  const counts = {
    all: disk.length + running.length + failed.length + abandoned.length + hubShown.length,
    resident: disk.filter((item) => item.resident).length,
    disk: disk.length,
    incomplete: failed.length + abandoned.length,
    hub: hubShown.length
  }

  return (
    <div className="wk">
      <main className="pane mid">
        <div className="ph">
          <div className="search">
            <Magnifier />
            <input
              value={query}
              placeholder="Search this machine and the Hub"
              onChange={(event) => setQuery(event.currentTarget.value)}
            />
          </div>
          <button className="iconbtn" title="Refresh" onClick={() => engine.refresh()}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12a9 9 0 1 1-3-6.7" />
              <polyline points="21 3 21 9 15 9" />
            </svg>
          </button>
        </div>
        <div className="pbody scroll">
          <div className="filters">
            {FILTERS.map((option) => (
              <button
                key={option.id}
                className={option.id === filter ? 'fchip on' : 'fchip'}
                onClick={() => setFilter(option.id)}
              >
                {option.label} {counts[option.id]}
              </button>
            ))}
          </div>

          {shown.length === 0 &&
          hubShown.length === 0 &&
          pullsShown.length === 0 &&
          failedShown.length === 0 &&
          abandonedShown.length === 0 ? (
            <div className="empty">
              <b>{query.trim() === '' ? 'Nothing on this machine yet' : 'No match'}</b>
              <p>
                {query.trim() === ''
                  ? 'Search for a repository — anything on the Hub can be pulled straight into the catalog.'
                  : 'Two characters search the Hub as well. Try the organisation, or part of the name.'}
              </p>
            </div>
          ) : (
            <div className="cards">
              {pullsShown.map((item) => (
                <Tile
                  key={item.repo}
                  id={item.repo}
                  org={owner(item.repo)}
                  size={item.total === null ? '—' : size(item.completed)}
                  meta={item.total === null ? 'starting' : `of ${size(item.total)}`}
                  tag={item.state === 'pending' ? 'Queued' : share(item)}
                  material=""
                  kind="pull"
                  progress={fraction(item)}
                  on={chosen?.id === item.repo}
                  onPick={() => {
                    setChosen({ id: item.repo, hub: true })
                    setTab('files')
                    setConfirming(false)
                  }}
                />
              ))}
              {failedShown.map((item) => (
                <Tile
                  key={item.repo}
                  id={item.repo}
                  org={owner(item.repo)}
                  size={size(item.completed)}
                  meta={item.total === null ? 'on disk' : `of ${size(item.total)}`}
                  tag="Failed"
                  material=""
                  kind="fail"
                  progress={fraction(item)}
                  on={chosen?.id === item.repo}
                  onPick={() => {
                    setChosen({ id: item.repo, hub: true })
                    setTab('files')
                    setConfirming(false)
                  }}
                />
              ))}
              {abandonedShown.map((item) => (
                <Tile
                  key={item.repo}
                  id={item.repo}
                  org={owner(item.repo)}
                  size={size(item.bytes_on_disk)}
                  meta="on disk"
                  tag="Stopped"
                  material=""
                  kind="stop"
                  progress={null}
                  on={chosen?.id === item.repo}
                  onPick={() => {
                    setChosen({ id: item.repo, hub: true })
                    setTab('files')
                    setConfirming(false)
                  }}
                />
              ))}
              {shown.map((item) => (
                <Tile
                  key={item.id}
                  id={item.id}
                  org={owner(item.id)}
                  size={`${gb(item.bytes_on_disk)} GB`}
                  meta={`${item.quantization ?? 'dense'}${item.context ? ` · ${contextTokens(item.context)}` : ''}`}
                  tag={item.resident ? 'Resident' : ''}
                  material={materials.get(item.id) ?? ''}
                  on={chosen?.id === item.id && !chosen.hub}
                  onPick={() => {
                    setChosen({ id: item.id, hub: false })
                    setTab('card')
                    setConfirming(false)
                  }}
                />
              ))}
              {hubShown.map((item) => (
                <Tile
                  key={item.id}
                  id={item.id}
                  org={`${owner(item.id)} · Hub`}
                  size={item.downloads === null ? '—' : `${Math.round(item.downloads / 1000)}k`}
                  meta={item.downloads === null ? 'on the Hub' : 'downloads / month'}
                  tag="Hub"
                  material=""
                  on={chosen?.id === item.id && chosen.hub}
                  onPick={() => {
                    setChosen({ id: item.id, hub: true })
                    setTab('card')
                    setConfirming(false)
                  }}
                />
              ))}
            </div>
          )}
        </div>
      </main>

      <aside className={`pane insp ${entry === null ? '' : materials.get(entry.id) ?? ''}`}>
        {chosen === null ? (
          <div className="empty">
            <b>Nothing selected</b>
            <p>Pick a checkpoint to read its card, price it against the ceiling, and load it.</p>
          </div>
        ) : (
          <>
            <div className={`insph${fetching ? ' pull' : stalled ? ' fail' : halted !== null ? ' stop' : ''}`}>
              <b>{displayName(chosen.id)}</b>
              <span>
                {chosen.id}
                {price !== null && ` · ${gb(price)} GB`}
              </span>
            </div>
            <div className="itabs">
              <button className={tab === 'card' ? 'on' : undefined} onClick={() => setTab('card')}>
                Card
              </button>
              <button className={tab === 'files' ? 'on' : undefined} onClick={() => setTab('files')}>
                Files
              </button>
              {entry !== null && (
                <button className={tab === 'profiles' ? 'on' : undefined} onClick={() => setTab('profiles')}>
                  Profiles
                </button>
              )}
              <button className={tab === 'source' ? 'on' : undefined} onClick={() => setTab('source')}>
                Source
              </button>
            </div>
            <div className={`ibody scroll${(speculates && !flags.spec) || (carded && !flags.card) ? ' waiting' : ''}`}>
              {failure !== null && <div className="err" style={{ marginBottom: 9 }}>{failure}</div>}
              {stalled && flight !== null && flight.error !== null && (
                <div className="err" style={{ marginBottom: 9 }}>
                  {flight.error} What was fetched is on disk; resuming continues from there.
                </div>
              )}
              {tooBig && (
                <p className="note bad" style={{ marginBottom: 9 }}>
                  {gb(price ?? 0)} GB needed, {gb(free ?? 0)} GB free below the ceiling. Raise the ceiling
                  in Settings, or unload something first.
                </p>
              )}
              {entry !== null && tab === 'card' && <Facts entry={entry} />}
              {speculates && entry !== null && <Speculation model={entry.id} onReady={() => settle('spec')} />}
              {tab === 'profiles' && entry !== null && <Profiles model={entry.id} />}
              {carded && (
                <CardBlock
                  id={chosen.id}
                  hub={chosen.hub}
                  shown={tab}
                  store={entry?.store}
                  onReady={() => settle('card')}
                />
              )}
            </div>
            {fetching && flight !== null ? (
              <PullBox pull={flight} onCancel={() => void act(() => downloads.cancel(flight.job))} />
            ) : (
            <div className="acts">
              {stalled || halted !== null ? (
                <>
                  <button className="btn pri" disabled={busy} onClick={() => void start(chosen.id)}>
                    Resume
                  </button>
                  <button
                    className="btn quiet danger"
                    style={{ marginLeft: 'auto' }}
                    disabled={busy}
                    onClick={() =>
                      void act(() => dropStaged(chosen.id)).then(() => {
                        downloads.forget(chosen.id)
                        setChosen(null)
                      })
                    }
                  >
                    Discard {size(flight?.completed ?? halted?.bytes_on_disk ?? 0)}
                  </button>
                </>
              ) : entry === null ? (
                <button className="btn pri wide" disabled={busy} onClick={() => void start(chosen.id)}>
                  Download {price === null ? '' : `${gb(price)} GB`}
                </button>
              ) : entry.resident ? (
                <>
                  <button
                    className="btn pri"
                    onClick={() => window.dispatchEvent(new CustomEvent('sideros:view', { detail: 'chat' }))}
                  >
                    Chat
                  </button>
                  <button className="btn" disabled={busy} onClick={() => void act(() => unloadModel(entry.id))}>
                    Unload
                  </button>
                </>
              ) : (
                <>
                  <button className="btn pri" disabled={busy} onClick={() => void act(() => loadModel(entry.id))}>
                    Load {gb(entry.bytes_on_disk)} GB
                  </button>
                  <button className="btn" onClick={() => setQuantizing(entry.id)}>
                    Quantize
                  </button>
                </>
              )}
              {entry !== null && (
                <button
                  className={confirming ? 'btn danger' : 'btn quiet danger'}
                  style={{ marginLeft: 'auto' }}
                  disabled={busy}
                  onClick={() => {
                    if (!confirming) {
                      setConfirming(true)
                      return
                    }
                    void act(() => removeModel(entry.id)).then(() => {
                      setConfirming(false)
                      setChosen(null)
                    })
                  }}
                >
                  {confirming ? 'Delete from disk' : 'Remove'}
                </button>
              )}
            </div>
            )}
          </>
        )}
      </aside>

      {quantizing !== null && <Quantize initial={quantizing} onClose={() => setQuantizing(null)} />}
    </div>
  )
}

/* Where the Download button was, doing the job the Download button no longer has. Cancelling
   takes the staging with it, so it asks twice — the same two steps as Remove, and for the
   same reason: the second press is the one that throws bytes away. */
function PullBox({ pull: entry, onCancel }: { pull: Pull; onCancel: () => void }): JSX.Element {
  const [confirming, setConfirming] = useState(false)
  const part = fraction(entry)
  const remaining = eta(entry)
  return (
    <div className="pullbox">
      <div className="pullhead">
        <b>{entry.state === 'pending' ? 'Queued' : 'Downloading'}</b>
        <em>{entry.message}</em>
        <span className="pct">{share(entry)}</span>
      </div>
      <div className="pullbar">
        <i style={{ width: `${part === null ? 0 : Math.round(part * 100)}%` }} />
        <u className="hatch" />
      </div>
      <div className="pullfoot">
        {entry.total !== null && <span>{sizePair(entry.completed, entry.total)}</span>}
        {entry.rate !== null && <span className="sep">·</span>}
        {entry.rate !== null && <span>{rate(entry.rate)}</span>}
        {remaining !== null && <span className="sep">·</span>}
        {remaining !== null && <span>{left(remaining)}</span>}
        <button
          className={confirming ? 'btn danger' : 'btn quiet danger'}
          onClick={() => (confirming ? onCancel() : setConfirming(true))}
        >
          {confirming ? `Cancel and discard ${size(entry.completed)}` : 'Cancel'}
        </button>
      </div>
    </div>
  )
}

function Facts({ entry }: { entry: CatalogEntry }): JSX.Element {
  return (
    <>
      <div className="grouplab">Checkpoint</div>
      <div className="kvs">
        <div className="kvr">
          <span className="k">Architecture</span>
          <span className="v">{entry.architecture}</span>
        </div>
        <div className="kvr">
          <span className="k">Quantization</span>
          <span className="v">
            {entry.quantization ?? 'dense'}
            {entry.dtype !== null && <em> · {entry.dtype}</em>}
          </span>
        </div>
        <div className="kvr">
          <span className="k">Context</span>
          <span className="v">{entry.context === null ? '—' : `${contextTokens(entry.context)} tokens`}</span>
        </div>
        <div className="kvr">
          <span className="k">Bytes per token</span>
          <span className="v">
            {entry.bytes_per_token === null ? '—' : `${(entry.bytes_per_token / 1e9).toFixed(2)} GB`}
          </span>
        </div>
        <div className="kvr">
          <span className="k">On disk</span>
          <span className="v">{gb(entry.bytes_on_disk)} GB</span>
        </div>
      </div>
    </>
  )
}

function Tile({
  id,
  org,
  size,
  meta,
  tag,
  material,
  kind,
  progress,
  on,
  onPick
}: {
  id: string
  org: string
  size: string
  meta: string
  tag: string
  material: string
  /* A checkpoint that is not on disk yet wears no material: the colour is identity, and
     bytes that do not exist have none to wear. */
  kind?: 'pull' | 'stop' | 'fail'
  progress?: number | null
  on: boolean
  onPick: () => void
}): JSX.Element {
  return (
    <button
      className={`card ${material}${material === '' ? '' : ' res'}${kind === undefined ? '' : ` ${kind}`}${on ? ' on' : ''}`}
      onClick={onPick}
      aria-pressed={on}
    >
      <span className="nm">{displayName(id)}</span>
      <span className="og">{org}</span>
      <span className="ft">
        <span className="g">{size}</span>
        <span className="q">{meta}</span>
        {tag !== '' && <span className="tag">{tag}</span>}
      </span>
      {progress !== null && progress !== undefined && (
        <span className="prog">
          <i style={{ width: `${Math.round(progress * 100)}%` }} />
        </span>
      )}
    </button>
  )
}

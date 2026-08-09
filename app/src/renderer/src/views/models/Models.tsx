/* The catalog on disk, with what the engine holds of it laid over the top: the list is
   `/admin/models`, residency is `/admin/state`, the rate and the % of ceiling are the
   daemon's own benches, and "Generating" is whatever `/admin/metrics` says is live.

   Load is a job and is followed to its last frame; unload is a 204. Both are offered from
   the row and from the detail, so both live in the hook the two share. */

import { type JSX, useCallback, useEffect, useState } from 'react'
import { cancelJob, isFinished, jobEvents, type JobView } from '../../api/http'
import * as api from './api'
import { ceilingRate, displayName, gib, homeless, owner, percent } from './format'
import { ModelDetail } from './ModelDetail'
import './models.css'

const POLL_MS = 2000

export interface Fleet {
  entries: api.CatalogEntry[]
  residents: Map<string, api.Resident>
  benches: Map<string, api.Bench>
  system: api.SystemInfo | null
  /** The model generating right now, which is the only thing this screen reads of the meter. */
  live: string | null
  /** Residency jobs in flight, by model id. */
  jobs: Record<string, JobView>
  error: string | null
  say: (message: string | null) => void
  rescan: () => void
  refresh: () => Promise<void>
  load: (model: string) => void
  unload: (model: string) => void
  cancel: (model: string) => void
}

function useFleet(): Fleet {
  const [entries, setEntries] = useState<api.CatalogEntry[]>([])
  const [state, setState] = useState<api.EngineState | null>(null)
  const [system, setSystem] = useState<api.SystemInfo | null>(null)
  const [benches, setBenches] = useState<api.Bench[]>([])
  const [live, setLive] = useState<string | null>(null)
  const [jobs, setJobs] = useState<Record<string, JobView>>({})
  const [error, setError] = useState<string | null>(null)

  /* The scan stats a few hundred files, so it is asked for once and again when the
     screen asks — never on the beat below. */
  const rescan = useCallback(() => {
    void (async () => {
      try {
        const [listed, history] = await Promise.all([api.listModels(), api.getBenches()])
        setEntries(listed)
        setBenches(history.benches)
      } catch (failure) {
        setError(api.reason(failure))
      }
    })()
  }, [])

  const refresh = useCallback(async () => {
    const [next, metrics] = await Promise.all([
      api.getState().catch(() => null),
      api.getMetrics().catch(() => null)
    ])
    setState(next)
    const sample = metrics?.live ?? null
    setLive(sample === null || sample.state !== 'running' ? null : sample.model)
  }, [])

  useEffect(rescan, [rescan])

  useEffect(() => {
    api.getSystem().then(setSystem).catch(() => setSystem(null))
  }, [])

  useEffect(() => {
    let alive = true
    const beat = () => {
      if (alive) void refresh()
    }
    beat()
    const timer = setInterval(beat, POLL_MS)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [refresh])

  /* Every frame is the whole state and the stream opens with the state as it is now, so
     nothing is lost by subscribing after the 202 came back. */
  const follow = useCallback(
    async (model: string, started: JobView) => {
      setJobs((current) => ({ ...current, [model]: started }))
      try {
        for await (const frame of jobEvents(started.id)) {
          setJobs((current) => ({ ...current, [model]: frame }))
          if (isFinished(frame)) {
            if (frame.error !== null) setError(frame.error)
            break
          }
        }
      } catch (failure) {
        setError(api.reason(failure))
      } finally {
        setJobs((current) =>
          Object.fromEntries(Object.entries(current).filter(([held]) => held !== model))
        )
        await refresh()
        rescan()
      }
    },
    [refresh, rescan]
  )

  const load = useCallback(
    (model: string) => {
      setError(null)
      void (async () => {
        try {
          await follow(model, await api.loadModel(model))
        } catch (failure) {
          setError(api.reason(failure))
        }
      })()
    },
    [follow]
  )

  const unload = useCallback(
    (model: string) => {
      setError(null)
      void (async () => {
        try {
          await api.unloadModel(model)
          await refresh()
        } catch (failure) {
          setError(api.reason(failure))
        }
      })()
    },
    [refresh]
  )

  /* Cancelling a load means "do not keep this": the weights are given back when they
     land, so the row goes on saying `loading` until the job's own last frame. */
  const cancel = useCallback(
    (model: string) => {
      const job = jobs[model]
      if (job === undefined) return
      void cancelJob(job.id).catch((failure: unknown) => setError(api.reason(failure)))
    },
    [jobs]
  )

  return {
    entries,
    residents: new Map((state?.models ?? []).map((resident) => [resident.id, resident])),
    benches: new Map(benches.map((bench) => [bench.model, bench])),
    system,
    live,
    jobs,
    error,
    say: setError,
    rescan,
    refresh,
    load,
    unload,
    cancel
  }
}

interface RowProps {
  entry: api.CatalogEntry
  fleet: Fleet
  onOpen: () => void
}

function Row({ entry, fleet, onOpen }: RowProps): JSX.Element {
  const resident = fleet.residents.get(entry.id)
  const bench = fleet.benches.get(entry.id)
  const job = fleet.jobs[entry.id]
  const generating = fleet.live === entry.id
  const memory = fleet.system?.memory_bytes ?? null

  const ceiling =
    entry.bytes_per_token === null || fleet.system === null
      ? null
      : ceilingRate(entry.bytes_per_token, fleet.system.bandwidth_sustained_gbs)
  const fraction =
    bench === undefined
      ? null
      : (bench.ceiling_fraction ?? (ceiling === null ? null : bench.tokens_per_second / ceiling))
  const share =
    resident === undefined || memory === null
      ? null
      : percent((resident.weights_bytes + resident.kv_bytes) / memory)

  const meta = [
    owner(entry.id),
    entry.architecture,
    entry.quantization ?? 'dense',
    `${gib(entry.bytes_on_disk)} GB`
  ].join(' · ')

  return (
    <div
      className="row"
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onOpen()
        }
      }}
    >
      <div className="id">
        <b>{displayName(entry.id)}</b>
        <span>{meta}</span>
      </div>

      {job === undefined ? (
        <div className={resident === undefined ? 'gauge quiet' : 'gauge'}>
          <div className="bar">
            {fraction !== null && <div className="fill" style={{ width: `${percent(fraction)}%` }} />}
          </div>
          <div className="sub">
            <span>{fraction === null ? 'not benched' : `${percent(fraction)}%`}</span>
            <span>{ceiling === null ? '—' : Math.round(ceiling)}</span>
          </div>
        </div>
      ) : (
        <div className="gauge">
          <div className="bar">
            {job.progress.total === null || job.progress.total === 0 ? (
              <div className="fill creep" />
            ) : (
              <div
                className="fill"
                style={{ width: `${percent(job.progress.completed / job.progress.total)}%` }}
              />
            )}
          </div>
          <div className="sub">
            <span className="msg">{job.progress.message || job.state}</span>
          </div>
        </div>
      )}

      <div className="rate">
        <b>{bench === undefined ? '—' : bench.tokens_per_second.toFixed(1)}</b>
        <small>{bench === undefined ? '' : 'tok/s'}</small>
      </div>
      <div className="rate memshare">
        <b>{share === null ? '—' : `${share}%`}</b>
        <small>{share === null ? '' : 'of memory'}</small>
      </div>

      {job !== undefined ? (
        <span className="chip hot">
          <span className="dot" />
          Loading
        </span>
      ) : generating ? (
        <span className="chip hot">
          <span className="dot" />
          Generating
        </span>
      ) : resident !== undefined ? (
        <span className="chip res">
          <span className="dot" />
          Resident
        </span>
      ) : (
        <span className="chip idle">
          <span className="dot" />
          On disk
        </span>
      )}

      {/* the row itself opens the detail; the button must not do both */}
      <div
        className="act"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => event.stopPropagation()}
      >
        {job !== undefined ? (
          <button className="btn" onClick={() => fleet.cancel(entry.id)}>
            Cancel
          </button>
        ) : resident !== undefined ? (
          <button className="btn solid" onClick={() => fleet.unload(entry.id)}>
            Unload
          </button>
        ) : (
          <button className="btn solid" onClick={() => fleet.load(entry.id)}>
            Load
          </button>
        )}
      </div>
    </div>
  )
}

export function Models(): JSX.Element {
  const fleet = useFleet()
  const [selected, setSelected] = useState<string | null>(null)
  const [filter, setFilter] = useState('')

  if (selected !== null) {
    return <ModelDetail id={selected} fleet={fleet} onBack={() => setSelected(null)} />
  }

  const needle = filter.trim().toLowerCase()
  const listed =
    needle === ''
      ? fleet.entries
      : fleet.entries.filter((entry) =>
          `${entry.id} ${entry.architecture} ${entry.quantization ?? 'dense'}`
            .toLowerCase()
            .includes(needle)
        )
  /* Held models float to the top; unloading drops the row back into catalog order. */
  const held = (entry: api.CatalogEntry): number =>
    fleet.residents.has(entry.id) || fleet.jobs[entry.id] !== undefined ? 0 : 1
  const ordered = [...listed].sort((a, b) => held(a) - held(b))
  const bytes = fleet.entries.reduce((sum, entry) => sum + entry.bytes_on_disk, 0)

  return (
    <>
      <div className="vhead">
        <h1>Models</h1>
        <div className="trail">
          <input
            className="input filter"
            placeholder="Filter…"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          />
          {/* The pull itself is the Download view's; the shell owns which view is on. */}
          <button
            className="btn ember"
            onClick={() => window.dispatchEvent(new CustomEvent('sideros:view', { detail: 'download' }))}
          >
            Pull model
          </button>
        </div>
      </div>

      <div className="vbody">
        {fleet.error !== null && (
          <div className="notice">
            {fleet.error}
            <button className="iconbtn" aria-label="Dismiss" onClick={() => fleet.say(null)}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        )}

        <div className="list">
          {ordered.length === 0 ? (
            <div className="empty">
              {fleet.entries.length === 0
                ? 'Nothing in the catalog yet.'
                : `Nothing matches “${filter.trim()}”.`}
            </div>
          ) : (
            ordered.map((entry) => (
              <Row key={entry.id} entry={entry} fleet={fleet} onOpen={() => setSelected(entry.id)} />
            ))
          )}
        </div>

        <div className="listfoot">
          <span className="tn">
            {fleet.entries.length} models · {gib(bytes, 0)} GB
          </span>
          ·<span className="mono">{fleet.system === null ? '—' : homeless(fleet.system.catalog)}</span>
          <button className="btn" onClick={fleet.rescan}>
            Rescan
          </button>
        </div>
      </div>
    </>
  )
}

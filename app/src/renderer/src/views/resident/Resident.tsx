/* What the machine is doing, in the two dimensions that bound it: the band is space,
   the list below it is time. Nothing here is a form — the only control on the screen
   is the ceiling, and it lives on the band. */

import { type JSX, useEffect, useRef, useState } from 'react'
import { gb, useEngineContext, whole, type CatalogEntry, type Resident as Slot } from '../../api/engine'
import { Band, scale } from '../../Memory'
import { useTypeAhead } from '../../keys'
import { ceilingRate, contextTokens, displayName, percent } from '../models/format'
import { type Sample, type Snapshot } from '../overview/api'
import { unloadModel } from '../models/api'
import { patchConfig } from '../overview/api'

/* One point per frame of the metrics stream, so the trace is what actually happened
   while the window was open — the daemon keeps no history to ask for. */
const TRACE = 56

function useTrace(snapshot: Snapshot | null): Map<string, number[]> {
  const [trace, setTrace] = useState<Map<string, number[]>>(new Map())
  const held = useRef(new Map<string, number[]>())

  useEffect(() => {
    if (snapshot === null) return
    const next = new Map(held.current)
    for (const model of snapshot.models) {
      const live = snapshot.live?.model === model.model ? snapshot.live : null
      const point = live?.ceiling_fraction ?? 0
      const series = [...(next.get(model.model) ?? []), point * 100].slice(-TRACE)
      next.set(model.model, series)
    }
    held.current = next
    setTrace(next)
  }, [snapshot])

  return trace
}

const W = 84
const H = 24

/* 20–75 % of the ceiling is the window a decode run lives in; outside it the trace
   says nothing worth the pixels. */
function Spark({ points }: { points: number[] }): JSX.Element | null {
  if (points.length < 2) return null
  const d = points
    .map((value, index) => {
      const x = 1 + (index / (points.length - 1)) * (W - 2)
      const y = H - 2 - Math.max(0, Math.min(1, (value - 20) / 55)) * (H - 4)
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`
    })
    .join('')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <path d={d} fill="none" stroke="var(--m)" strokeWidth="1.2" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

const seconds = (value: number): string => {
  if (value < 60) return `${Math.round(value)} s`
  if (value < 3600) return `${Math.round(value / 60)} min`
  return `${(value / 3600).toFixed(1)} h`
}

const clock = (epoch: number): string =>
  new Date(epoch * 1000).toLocaleTimeString('en-GB', { hour12: false })

export function Resident(): JSX.Element {
  const engine = useEngineContext()
  const snapshot = engine.metrics
  const trace = useTrace(snapshot)
  const jobs = engine.jobs ?? []
  const [chosen, setChosen] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const runs = useTypeAhead()

  const ceiling = whole(engine.config?.['memory_limit_bytes'])
  const s = scale(engine.system, engine.state, ceiling)
  const loading = jobs.find((job) => job.kind === 'load' || job.kind === 'download')
  const sustained = engine.system?.bandwidth_sustained_gbs ?? null

  const models = s?.models ?? []
  const selected = models.find((model) => model.id === chosen) ?? models[0] ?? null
  const entry = engine.catalog?.find((item) => item.id === selected?.id) ?? null
  const aggregate = snapshot?.models.find((item) => item.model === selected?.id) ?? null

  const queued = (snapshot?.requests ?? []).filter(
    (sample) => sample.state === 'queued' || sample.state === 'running'
  )
  const past = (snapshot?.requests ?? []).filter(
    (sample) => sample.state !== 'queued' && sample.state !== 'running'
  )

  const unload = async (id: string) => {
    setBusy(true)
    try {
      await unloadModel(id)
      engine.refresh()
    } finally {
      setBusy(false)
    }
  }

  const setCeiling = (bytes: number) => {
    void patchConfig({ memory_limit_bytes: bytes }).then(engine.refresh)
  }

  return (
    <>
      {s === null ? (
        <div className="band">
          <div className="bandh">
            <b>Memory</b>
            <span>waiting for the engine</span>
          </div>
          <div className="track" />
        </div>
      ) : (
        <Band
          scale={s}
          loading={
            loading === undefined
              ? null
              : {
                  id: loading.progress.message,
                  done: loading.progress.completed,
                  total: loading.progress.total
                }
          }
          onCeiling={setCeiling}
        />
      )}

      <div className="wk">
        <main className="pane mid">
          <div className="ph">
            <h2>Running</h2>
            <span className="meta">
              {models.length} resident · {engine.state?.queue.waiting ?? 0} waiting
            </span>
          </div>
          <div className="pbody scroll">
            {/* The empty state is full height, so anything under it starts below the fold.
                A job in flight is the answer to "nothing is loaded" — and it is also what
                the prose would be telling the person to go and do. */}
            {models.length === 0 && jobs.length === 0 && (
              <div className="empty">
                <b>Nothing is loaded</b>
                <p>
                  Pick a checkpoint in Library and load it, or point a client at the engine — the first
                  request loads what it asks for.
                </p>
                <button
                  className="btn"
                  onClick={() => window.dispatchEvent(new CustomEvent('sideros:view', { detail: 'library' }))}
                >
                  Open Library
                </button>
              </div>
            )}
            {models.length > 0 && (
              <ul className="runs" ref={runs} role="listbox" aria-label="Resident models">
                {models.map((model) => (
                  <Run
                    key={model.id}
                    model={model}
                    material={s?.material.get(model.id) ?? 'm1'}
                    entry={engine.catalog?.find((item) => item.id === model.id) ?? null}
                    live={snapshot?.live?.model === model.id ? snapshot.live : null}
                    points={trace.get(model.id) ?? []}
                    sustained={sustained}
                    idleFor={model.last_used === null ? null : Date.now() / 1000 - model.last_used}
                    on={model.id === selected?.id}
                    onPick={() => setChosen(model.id)}
                  />
                ))}
              </ul>
            )}

            {jobs.map((job) => (
              <div key={job.id} className={job.kind === 'download' ? 'job pull' : 'job m6'}>
                <div className="t">
                  {'model' in job.subject
                    ? displayName(job.subject.model)
                    : job.subject.models.map(displayName).join(', ')}{' '}
                  <em>· {job.progress.message}</em>
                </div>
                <div className="r">
                  {job.progress.total === null
                    ? gb(job.progress.completed) + ' GB'
                    : `${gb(job.progress.completed)} of ${gb(job.progress.total)} GB`}
                </div>
                <div className="bar">
                  <i
                    style={{
                      width:
                        job.progress.total === null || job.progress.total === 0
                          ? '30%'
                          : `${Math.min(100, (job.progress.completed / job.progress.total) * 100)}%`
                    }}
                  />
                </div>
              </div>
            ))}

            {queued.length > 0 && (
              <>
                <div className="qhead">
                  <b>Queue</b>
                  <span>first come, first served · {queued.length} in flight</span>
                </div>
                {queued.map((sample, index) => (
                  <div key={`${sample.model}-${sample.started_at}`} className={`qrow ${s?.material.get(sample.model) ?? 'm1'}`}>
                    <span className="ix">{index + 1}</span>
                    <span className="ep">{sample.state === 'running' ? 'generating' : 'waiting'}</span>
                    <span className="mdl">
                      <i />
                      {displayName(sample.model)}
                    </span>
                    <span className="n">{sample.prompt_tokens} in</span>
                    <span className="n">{seconds(Date.now() / 1000 - sample.started_at)}</span>
                  </div>
                ))}
              </>
            )}

            {past.length > 0 && (
              <>
                <div className="qhead">
                  <b>Recent</b>
                  <span>this session</span>
                </div>
                {past.slice(0, 8).map((sample) => (
                  <Past key={`${sample.model}-${sample.started_at}`} sample={sample} />
                ))}
              </>
            )}
          </div>
          {engine.system !== null && (
            <div className="pfoot">
              <span>
                {engine.system.chip} · {engine.system.gpu_cores} GPU cores ·{' '}
                {gb(engine.system.memory_bytes, 0)} GB
              </span>
              <span className="mono" style={{ marginLeft: 'auto' }}>
                {engine.system.bandwidth_sustained_gbs} GB/s
              </span>
            </div>
          )}
        </main>

        <aside className={`pane insp ${selected === null ? '' : (s?.material.get(selected.id) ?? 'm1')}`}>
          {selected === null || s === null ? (
            <div className="empty">
              <b>No block selected</b>
              <p>Load a model to see what it costs and what it can reach.</p>
            </div>
          ) : (
            <>
              <div className="insph">
                <b>{displayName(selected.id)}</b>
                <span>{selected.id}</span>
              </div>
              <div className="ibody scroll">
                <div className="grouplab">Throughput</div>
                <div className="kvs">
                  <Row k="Decode" v={aggregate?.tokens_per_second === null || aggregate === null ? '—' : `${aggregate.tokens_per_second.toFixed(1)} tok/s`} />
                  <Row
                    k="Ceiling"
                    v={
                      entry?.bytes_per_token && sustained !== null
                        ? `${Math.round(ceilingRate(entry.bytes_per_token, sustained))} tok/s`
                        : '—'
                    }
                    em={aggregate?.ceiling_fraction ? `· ${percent(aggregate.ceiling_fraction)} %` : undefined}
                  />
                  <Row
                    k="Bytes per token"
                    v={entry?.bytes_per_token ? `${(entry.bytes_per_token / 1e9).toFixed(2)} GB` : '—'}
                  />
                  <Row k="Requests" v={aggregate === null ? '0' : String(aggregate.requests)} />
                </div>

                <div className="grouplab">Memory</div>
                <div className="kvs">
                  <Row k="Weights resident" v={`${gb(selected.weights_bytes)} GB`} />
                  <Row k="KV cache" v={`${gb(selected.kv_bytes)} GB`} />
                  <Row
                    k="Share of the machine"
                    v={
                      engine.system === null
                        ? '—'
                        : `${(((selected.weights_bytes + selected.kv_bytes) / engine.system.memory_bytes) * 100).toFixed(1)} %`
                    }
                  />
                  <Row
                    k="Last used"
                    v={selected.last_used === null ? 'never' : `${seconds(Date.now() / 1000 - selected.last_used)} ago`}
                  />
                </div>

                {entry !== null && (
                  <>
                    <div className="grouplab">Checkpoint</div>
                    <div className="kvs">
                      <Row k="Architecture" v={entry.architecture} />
                      <Row k="Quantization" v={entry.quantization ?? 'dense'} em={entry.dtype ? `· ${entry.dtype}` : undefined} />
                      <Row k="Context" v={entry.context === null ? '—' : `${contextTokens(entry.context)} tokens`} />
                      <Row k="On disk" v={`${gb(entry.bytes_on_disk)} GB`} />
                    </div>
                  </>
                )}
              </div>
              <div className="acts">
                <button
                  className="btn pri"
                  onClick={() => window.dispatchEvent(new CustomEvent('sideros:view', { detail: 'chat' }))}
                >
                  Chat
                </button>
                <button className="btn" disabled={busy} onClick={() => void unload(selected.id)}>
                  Unload
                </button>
                <button
                  className="btn quiet"
                  style={{ marginLeft: 'auto' }}
                  onClick={() => window.dispatchEvent(new CustomEvent('sideros:view', { detail: 'library' }))}
                >
                  In Library
                </button>
              </div>
            </>
          )}
        </aside>
      </div>
    </>
  )
}

function Row({ k, v, em }: { k: string; v: string; em?: string }): JSX.Element {
  return (
    <div className="kvr">
      <span className="k">{k}</span>
      <span className="v">
        {v}
        {em !== undefined && <em> {em}</em>}
      </span>
    </div>
  )
}

function Past({ sample }: { sample: Sample }): JSX.Element {
  const failed = sample.state === 'error'
  return (
    <div className={failed ? 'logrow err' : 'logrow'}>
      <span className="ts">{clock(sample.started_at)}</span>
      <span className="tx">
        <b>{displayName(sample.model)}</b> · {sample.prompt_tokens} in
        {sample.reused_tokens > 0 && ` (${sample.reused_tokens} cached)`}
        {sample.reused_tokens === 0 && !sample.kept_prefix && ' (keeps no prefix)'},{' '}
        {sample.completion_tokens} out
        {sample.state === 'cancelled' && ' · cancelled'}
        {failed && ' · failed'}
      </span>
      <span className="rt">
        {sample.tokens_per_second === null ? '—' : `${sample.tokens_per_second.toFixed(1)} tok/s`}
      </span>
    </div>
  )
}

function Run({
  model,
  material,
  entry,
  live,
  points,
  sustained,
  idleFor,
  on,
  onPick
}: {
  model: Slot
  material: string
  entry: CatalogEntry | null
  live: Sample | null
  points: number[]
  sustained: number | null
  idleFor: number | null
  on: boolean
  onPick: () => void
}): JSX.Element {
  const rate = live?.tokens_per_second ?? null
  const fraction = live?.ceiling_fraction ?? null
  const ceiling =
    entry?.bytes_per_token && sustained !== null ? ceilingRate(entry.bytes_per_token, sustained) : null

  /* A row of a single-select roster, which is what makes it a keyboard target at all: Tab
     reaches it, focus is the selection, and type-ahead matches on `data-key`. */
  return (
    <li
      className={`run ${material}${on ? ' on' : ''}`}
      role="option"
      aria-selected={on}
      tabIndex={0}
      data-key={displayName(model.id)}
      onFocus={onPick}
      onClick={onPick}
    >
      <div className="id">
        <b>{displayName(model.id)}</b>
        <span>
          {entry?.quantization ?? 'dense'}
          {entry?.context ? ` · ${contextTokens(entry.context)} ctx` : ''}
          {ceiling !== null ? ` · ceiling ${Math.round(ceiling)}` : ''}
        </span>
      </div>
      <div className={rate === null ? 'num dim' : 'num'}>
        <b>{rate === null ? '—' : rate.toFixed(1)}</b>
        <span>tok/s</span>
      </div>
      <div className={fraction === null ? 'num dim' : 'num'}>
        <b>{fraction === null ? '—' : `${percent(fraction)} %`}</b>
        {fraction !== null && (
          <div className="pctbar">
            <i style={{ width: `${percent(fraction)}%` }} />
          </div>
        )}
        <span>of ceiling</span>
      </div>
      <div className="spark">
        <Spark points={points} />
      </div>
      <div className="state">
        <span className="l1">
          <i className={live === null ? 'dot' : 'dot ok'} />
          {live === null ? 'Idle' : 'Generating'}
        </span>
        {live === null && idleFor !== null && <span className="l2">for {seconds(idleFor)}</span>}
      </div>
    </li>
  )
}

/* The Benchmark tab: three views that never share a chart.

   Speed, Fidelity and Quality answer questions that are not traded against each other in a
   single run, and concurrency only exists on one of the three — so the band follows the
   view switch and the knobs in its header follow with it. What is shared is the comparison:
   the set of checkpoints drawn, which is `auto ∪ pinned − dropped` and survives a change of
   knobs.

   The view stays mounted when you leave it, like every other one, so a comparison built by
   hand does not evaporate on a trip to Chat. */

import { type JSX, useCallback, useEffect, useMemo, useState } from 'react'
import './benchmark.css'
import { useEngineContext } from '../../api/engine'
import { jobEvents, type JobView } from '../../api/http'
import { autofill, Band, type Line } from './Band'
import { fidelityAxes, fidelityValues, FidelityResult, FidelityRows, split } from './Fidelity'
import { DatasetForm, qualityAxes, QualityResult, QualityRows, qualityValues } from './Quality'
import { Sheet } from './Sheet'
import { speedAxes, SpeedResult, SpeedRows, SpeedSetup, speedValues } from './Speed'
import {
  CONCURRENCIES,
  CONTEXTS,
  GENERATES,
  datasets as fetchDatasets,
  exportUrl,
  forget,
  human,
  runs as fetchRuns,
  speedKey,
  speedPrefix,
  type DatasetDefinition,
  type Kind,
  type Run
} from './api'

const TITLES: Record<Kind, string> = { speed: 'Speed', fidelity: 'Fidelity', quality: 'Quality' }

type Tab = 'result' | 'setup' | 'raw'

function Magnifier(): JSX.Element {
  return (
    <svg viewBox="0 0 11 11" fill="none" aria-hidden="true">
      <circle cx="4.6" cy="4.6" r="3.4" stroke="currentColor" strokeWidth="1.2" />
      <path d="M7.3 7.3 10 10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  )
}

export function Benchmark(): JSX.Element {
  const engine = useEngineContext()
  const [view, setView] = useState<Kind>('speed')
  const [context, setContext] = useState(4096)
  const [generate, setGenerate] = useState(256)
  const [concurrency, setConcurrency] = useState(1)
  /* The rest of the key — rounds, sampler, page cache, stream source. Held as the whole key
     rather than as its parts because it is only ever chosen from keys that exist. */
  const [variant, setVariant] = useState<string | null>(null)
  const [reference, setReference] = useState<string | null>(null)
  const [all, setAll] = useState<Run[]>([])
  const [scoped, setScoped] = useState<Run[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [dataset, setDataset] = useState<string | null>(null)
  const [pinned, setPinned] = useState<string[]>([])
  const [dropped, setDropped] = useState<string[]>([])
  const [query, setQuery] = useState('')
  const [tab, setTab] = useState<Tab>('result')
  const [sheet, setSheet] = useState(false)
  const [declaring, setDeclaring] = useState(false)
  const [definitions, setDefinitions] = useState<DatasetDefinition[]>([])
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<JobView | null>(null)
  const [failure, setFailure] = useState<string | null>(null)

  const catalog = useMemo(() => engine.catalog ?? [], [engine.catalog])
  const corpus = 'wikitext103'

  const prefix = useMemo(
    () => speedPrefix(context, generate, concurrency),
    [context, generate, concurrency]
  )

  /* Every spelling of the rest of the key that has actually been measured under these three
     knobs, newest first. Assuming one instead — the defaults of the day — is what made a
     benchmark run under `r8+2` and land in a list scoped to `r3+1`, which reads from the
     outside as the run having disappeared. */
  const variants = useMemo(() => {
    if (view !== 'speed') return []
    const newest = new Map<string, number>()
    for (const run of all) {
      if (!run.key.startsWith(prefix)) continue
      newest.set(run.key, Math.max(newest.get(run.key) ?? 0, run.created_at))
    }
    return [...newest.entries()].sort((left, right) => right[1] - left[1]).map(([entry]) => entry)
  }, [view, all, prefix])

  const key = useMemo(() => {
    if (view !== 'speed') return undefined
    if (variant !== null && variant.startsWith(prefix)) return variant
    return variants[0] ?? speedKey(context, generate, concurrency, 3, 1)
  }, [view, variant, variants, prefix, context, generate, concurrency])

  const reload = useCallback(() => {
    void fetchRuns(view)
      .then((found) => {
        setAll(found)
        setFailure(null)
      })
      .catch((error: unknown) => setFailure(error instanceof Error ? error.message : String(error)))
    void fetchRuns(view, key)
      .then(setScoped)
      .catch(() => setScoped([]))
  }, [view, key])

  useEffect(() => {
    void fetchDatasets()
      .then(setDefinitions)
      .catch(() => setDefinitions([]))
  }, [])

  useEffect(() => {
    if (reference === null && catalog.length > 0) setReference(catalog[0].id)
  }, [catalog, reference])

  /* The batch reports through the SSE every job already has. One subscription per job —
     the dependency is the id and not the job, because a frame changes the job and a
     dependency on it tore the stream down and reopened it once per frame: a connection
     storm on the daemon that is measuring, which is the one process that must be left
     alone. */
  useEffect(() => {
    if (jobId === null) return
    const stop = new AbortController()
    void (async () => {
      try {
        for await (const frame of jobEvents(jobId, stop.signal)) setJob(frame)
      } catch {
        /* the stream closing with the job is not a failure */
      }
    })()
    return () => stop.abort()
  }, [jobId])

  const running =
    job !== null && job.state !== 'ok' && job.state !== 'error' && job.state !== 'cancelled'

  /* The list follows the batch: one refetch per shape the batch finishes, and one when it
     ends. Rows are refetched and never patched — what a row says is the server's. */
  const done = job === null ? 0 : job.progress.completed
  useEffect(() => {
    reload()
  }, [reload, done, running])

  const fidelity = useMemo(
    () => split(catalog, reference ?? '', all.filter((run) => run.fidelity?.reference === reference)),
    [catalog, reference, all]
  )

  const forView = view === 'speed' ? scoped : all

  const models = useMemo(() => {
    if (view === 'fidelity') return fidelity.measured.map((entry) => entry.entry.id)
    return [...new Set(forView.map((run) => run.model))]
  }, [view, fidelity, forView])

  const score = useCallback(
    (model: string): number | null => {
      const run = forView.find((entry) => entry.model === model)
      if (view === 'speed') return run?.speed?.decode_tps ?? null
      if (view === 'fidelity') {
        const found = run?.fidelity?.kl_mean
        return found === null || found === undefined ? null : -found
      }
      return run?.quality?.accuracy ?? null
    },
    [forView, view]
  )

  const drawn = useMemo(() => {
    const union = [...new Set([...autofill(models, score), ...pinned])]
    return union.filter((model) => !dropped.includes(model)).slice(0, 6)
  }, [models, score, pinned, dropped])

  const materials = useMemo(
    () => new Map(drawn.map((model, index) => [model, (index % 6) + 1])),
    [drawn]
  )

  const axes = useMemo(() => {
    if (view === 'speed') return speedAxes(context, generate, concurrency)
    if (view === 'fidelity') return fidelityAxes(corpus)
    const seen = new Map<string, string>()
    for (const run of all) if (run.quality !== null) seen.set(run.quality.dataset, run.key)
    return qualityAxes([...seen.entries()].map(([id, entryKey]) => ({ id, key: entryKey })))
  }, [view, context, generate, concurrency, all])

  const lines: Line[] = useMemo(
    () =>
      drawn.map((model, index) => ({
        model,
        material: (index % 6) + 1,
        values:
          view === 'speed'
            ? speedValues(forView.find((entry) => entry.model === model))
            : view === 'fidelity'
              ? fidelityValues(forView.find((entry) => entry.model === model))
              : qualityValues(all.filter((entry) => entry.model === model))
      })),
    [drawn, forView, view, all]
  )

  const chosen = useMemo(() => {
    if (selected === null) return null
    if (view === 'quality') {
      return (
        all.find((run) => run.model === selected && run.quality?.dataset === dataset) ??
        all.find((run) => run.model === selected) ??
        null
      )
    }
    return forView.find((run) => run.model === selected) ?? null
  }, [selected, view, all, dataset, forView])

  /* The same checkpoint under every stream count, which is what shows the number falling off
     with context long before it falls off with streams. */
  const series = useMemo(
    () =>
      CONCURRENCIES.map((value) => ({
        concurrency: value,
        decode:
          all.find((entry) => entry.model === selected && entry.speed?.concurrency === value)?.speed
            ?.decode_tps ?? null
      })),
    [all, selected]
  )

  const pick = (model: string, ds?: string): void => {
    setSelected(model)
    if (ds !== undefined) setDataset(ds)
    if (!drawn.includes(model)) {
      setPinned([...new Set([...pinned, model])])
      setDropped(dropped.filter((entry) => entry !== model))
    }
  }

  const visible = (run: Run): boolean =>
    query === '' || run.model.toLowerCase().includes(query.toLowerCase())

  const rest = models.length - drawn.length

  return (
    <div className="bench">
      <div className="band tradeoff m1">
        <div className="bandh">
          <b>{TITLES[view]}</b>
          <div className="seg">
            {(['speed', 'fidelity', 'quality'] as const).map((entry) => (
              <button
                key={entry}
                type="button"
                className={view === entry ? 'on' : undefined}
                onClick={() => setView(entry)}
              >
                {TITLES[entry]}
              </button>
            ))}
          </div>

          <div className="knob" id="ref" hidden={view !== 'fidelity'}>
            <span>Reference</span>
            <select
              className="input"
              value={reference ?? ''}
              onChange={(event) => setReference(event.target.value)}
              aria-label="Reference checkpoint"
            >
              {catalog.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.id}
                </option>
              ))}
            </select>
          </div>
          <div className="knob" id="corp" hidden={view !== 'fidelity'}>
            <span>Corpus</span>
            <select className="input mono" value={corpus} disabled aria-label="Corpus">
              <option value="wikitext103">wikitext103 · test</option>
            </select>
          </div>

          <div className="knob" id="ctx" hidden={view !== 'speed'}>
            <span>Context</span>
            <select
              className="input mono"
              value={context}
              onChange={(event) => setContext(Number(event.target.value))}
              aria-label="Context length"
            >
              {CONTEXTS.map((value) => (
                <option key={value} value={value}>
                  {human(value)}
                </option>
              ))}
            </select>
          </div>
          <div className="knob" id="gen" hidden={view !== 'speed'}>
            <span>Generate</span>
            <select
              className="input mono"
              value={generate}
              onChange={(event) => setGenerate(Number(event.target.value))}
              aria-label="Generated tokens"
            >
              {GENERATES.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <div className="knob" id="conc" hidden={view !== 'speed'}>
            <span>Concurrency</span>
            <div className="seg">
              {CONCURRENCIES.map((value) => (
                <button
                  key={value}
                  type="button"
                  className={concurrency === value ? 'on' : undefined}
                  onClick={() => setConcurrency(value)}
                >
                  {value}
                </button>
              ))}
            </div>
          </div>
          <div className="knob" id="var" hidden={view !== 'speed'}>
            <span>Variant</span>
            <select
              className="input mono"
              value={key ?? ''}
              onChange={(event) => setVariant(event.target.value)}
              aria-label="Rounds, sampler and cache"
              disabled={variants.length < 2}
            >
              {(variants.length === 0 && key !== undefined ? [key] : variants).map((entry) => (
                <option key={entry} value={entry}>
                  {entry.slice(prefix.length)}
                </option>
              ))}
            </select>
          </div>
          <span className="seed">
            {running && job !== null
              ? (job.error ?? job.progress.message)
              : `${models.length} measured`}
          </span>
        </div>

        <Band axes={axes} lines={lines} selected={selected} onSelect={setSelected} />

        <div className="tfoot">
          {drawn.map((model, index) => (
            <button
              key={model}
              type="button"
              className={`tkey m${(index % 6) + 1}`}
              title="Remove from the comparison"
              onClick={() => {
                setDropped([...new Set([...dropped, model])])
                setPinned(pinned.filter((entry) => entry !== model))
              }}
            >
              <i />
              <b>{model}</b> ×
            </button>
          ))}
          <span className="note" style={{ marginLeft: 'auto' }}>
            {rest > 0
              ? `${rest} more measured under this key — click a run to add`
              : 'every run under this key is here'}
          </span>
        </div>
      </div>

      <div className="wk">
        <div className="pane mid">
          <div className="ph">
            <h2>{TITLES[view]} runs</h2>
            <span className="meta">{forView.length} runs</span>
          </div>

          <div className="pbody scroll">
            <div className="headrow">
              <div className="search">
                <Magnifier />
                <input
                  placeholder="Filter by checkpoint"
                  aria-label="Filter"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </div>
              <button
                type="button"
                className="fchip"
                onClick={() => {
                  setPinned([])
                  setDropped([])
                }}
              >
                Clear comparison
              </button>
            </div>

            {failure !== null && <p className="note bad">{failure}</p>}

            {/* What is running, in the pane the rows land in. The band's header says the
                same thing in one line; here it says how much of the batch is left. */}
            {running && job !== null && (
              <div className="cost" style={{ marginBottom: 10 }}>
                <span>Running</span>
                <b>{Math.round(job.progress.completed)}</b>
                <span>of {job.progress.total === null ? '—' : Math.round(job.progress.total)}</span>
                <span className="r">{job.progress.message}</span>
              </div>
            )}

            {view === 'speed' && (
              <SpeedRows
                runs={scoped.filter(visible)}
                context={context}
                generate={generate}
                concurrency={concurrency}
                selected={selected}
                materials={materials}
                onSelect={pick}
              />
            )}
            {view === 'quality' && (
              <QualityRows
                runs={all.filter(visible)}
                selected={selected}
                materials={materials}
                onSelect={pick}
              />
            )}
            {view === 'fidelity' && reference !== null && (
              <FidelityRows
                reference={reference}
                corpus={corpus}
                groups={fidelity}
                selected={selected}
                materials={materials}
                onSelect={pick}
              />
            )}
          </div>

          <div className="pfoot">
            <button type="button" className="btn pri" onClick={() => setSheet(true)}>
              New benchmark
            </button>
            <a className="btn" href={exportUrl(view, key)} download>
              Export CSV
            </a>
          </div>
        </div>

        <div className="pane insp m1">
          <div className="insph">
            <b>{selected ?? 'Nothing selected'}</b>
            <span>{chosen === null ? 'pick a run' : `run ${chosen.id.slice(0, 6)} · ${chosen.key}`}</span>
          </div>
          <div className="itabs">
            {(['result', 'setup', 'raw'] as const).map((entry) => (
              <button
                key={entry}
                type="button"
                className={tab === entry ? 'on' : undefined}
                onClick={() => setTab(entry)}
              >
                {entry === 'result' ? 'Result' : entry === 'setup' ? 'Setup' : 'Raw'}
              </button>
            ))}
          </div>

          <div className="ibody scroll">
            {chosen === null ? (
              <p className="note">
                The band draws the comparison; the inspector explains one line of it.
              </p>
            ) : tab === 'raw' ? (
              <pre className="code">{JSON.stringify(chosen, null, 2)}</pre>
            ) : tab === 'setup' ? (
              <SpeedSetup run={chosen} />
            ) : view === 'speed' ? (
              <SpeedResult
                run={chosen}
                context={context}
                concurrency={concurrency}
                series={series}
                onConcurrency={setConcurrency}
              />
            ) : view === 'quality' ? (
              <QualityResult
                run={chosen}
                others={all.filter((run) => run.model === chosen.model && run.id !== chosen.id)}
                onDataset={setDataset}
              />
            ) : (
              <FidelityResult run={chosen} reference={reference ?? ''} />
            )}
          </div>

          <div className="acts">
            <button
              type="button"
              className="btn wide"
              disabled={chosen === null}
              onClick={() => setSheet(true)}
            >
              Run again
            </button>
            <button
              type="button"
              className="btn danger"
              disabled={chosen === null}
              onClick={() => {
                if (chosen !== null) void forget(chosen.id).then(reload)
              }}
            >
              Delete
            </button>
          </div>
        </div>
      </div>

      {sheet && (
        <Sheet
          kind={view}
          catalog={catalog}
          datasets={definitions}
          onClose={() => setSheet(false)}
          onStarted={(id, request) => {
            setJobId(id)
            setJob({
              id,
              kind: 'benchmark',
              subject: { kind: request.kind, models: [] },
              state: 'pending',
              progress: { message: 'queued', completed: 0, total: null },
              created_at: Date.now() / 1000,
              updated_at: Date.now() / 1000,
              error: null
            })
            setView(request.kind)
            /* The view moves to what was just asked for. A batch launched at 8k landing in a
               list scoped to 4k is a run that, from here, did not happen. */
            if (request.kind === 'speed') {
              const [first] = request.contexts
              const [tokens] = request.generates
              const [streams] = request.concurrencies
              setContext(first)
              setGenerate(tokens)
              setConcurrency(streams)
              setVariant(
                speedKey(
                  first,
                  tokens,
                  streams,
                  request.rounds,
                  request.warmup,
                  request.sampling,
                  request.page_cache
                )
              )
            }
          }}
          onDeclare={() => setDeclaring(true)}
        />
      )}
      {declaring && (
        <DatasetForm
          onClose={() => setDeclaring(false)}
          onSaved={() => void fetchDatasets().then(setDefinitions)}
        />
      )}
    </div>
  )
}

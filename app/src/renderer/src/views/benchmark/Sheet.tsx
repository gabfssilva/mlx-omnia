/* New benchmark: what to measure, on which checkpoints, and what it will cost.

   Two requirements that came out of the mockup review and are not cosmetic:

   - Nothing scrolls to decide. Everything chosen is on screen when it opens. The one list
     that scrolls is the model library, because it is the only one that grows without bound.
   - The same box in all three views. `.sheetb` inherits `flex: 1` from the house
     stylesheet, and with `min-height: auto` the taller panel pushes the sheet and it changes
     size when the view changes — so the body states its height outright and opts out of the
     flex.

   The cost line — how many shapes will run, how many were skipped and why, how many are
   already answered — comes from `POST /admin/benchmarks/plan` and is never recomputed here.
   Two copies of that arithmetic is two answers to "does this fit". */

import { type JSX, useCallback, useEffect, useState } from 'react'
import { gb, type CatalogEntry } from '../../api/engine'
import { useEscape } from '../../keys'
import { getProfile, listProfileNames } from '../models/api'
import {
  CONCURRENCIES,
  CONTEXTS,
  GATES,
  GENERATES,
  GREEDY,
  human,
  plan as pricePlan,
  speedKey,
  start,
  type DatasetDefinition,
  type Kind,
  type Plan,
  type Request,
  type Sampling
} from './api'

const toggle = (values: number[], value: number): number[] =>
  values.includes(value)
    ? values.filter((entry) => entry !== value)
    : [...values, value].sort((a, b) => a - b)

function Chips({
  values,
  chosen,
  label,
  onToggle
}: {
  values: readonly number[]
  chosen: number[]
  label: (value: number) => string
  onToggle: (value: number) => void
}): JSX.Element {
  return (
    <div className="chips">
      {values.map((value) => (
        <button
          key={value}
          type="button"
          className={`fchip${chosen.includes(value) ? ' on' : ''}`}
          onClick={() => onToggle(value)}
        >
          {label(value)}
        </button>
      ))}
    </div>
  )
}

export function Sheet({
  kind,
  catalog,
  datasets,
  onClose,
  onStarted,
  onDeclare
}: {
  kind: Kind
  catalog: CatalogEntry[]
  datasets: DatasetDefinition[]
  onClose: () => void
  onStarted: (jobId: string, request: Request) => void
  onDeclare: () => void
}): JSX.Element {
  const [view, setView] = useState<Kind>(kind)
  const [models, setModels] = useState<string[]>([])
  const [contexts, setContexts] = useState<number[]>([4096])
  const [generates, setGenerates] = useState<number[]>([256])
  const [concurrencies, setConcurrencies] = useState<number[]>([1])
  const [rounds, setRounds] = useState(3)
  const [sampling, setSampling] = useState<Sampling>(GREEDY)
  const [profiles, setProfiles] = useState<string[]>([])
  const [profile, setProfile] = useState('')
  const [page, setPage] = useState<'warm' | 'cold'>('warm')
  const [gate, setGate] = useState<number | null>(45)
  const [skip, setSkip] = useState(true)
  const [pairs, setPairs] = useState<Record<string, string>>({})
  const [chosenDatasets, setChosenDatasets] = useState<string[]>(['mmlu'])
  const [items, setItems] = useState(1400)
  const [seed, setSeed] = useState(42)
  const [shots, setShots] = useState(5)
  const [priced, setPriced] = useState<Plan | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEscape(onClose)

  const body = useCallback((): Request | null => {
    if (view === 'speed') {
      if (models.length === 0) return null
      return {
        kind: 'speed',
        models,
        contexts,
        generates,
        concurrencies,
        rounds,
        sampling,
        page_cache: page,
        thermal_gate_c: gate,
        skip_if_measured: skip
      }
    }
    if (view === 'quality') {
      if (models.length === 0 || chosenDatasets.length === 0) return null
      return {
        kind: 'quality',
        models,
        datasets: chosenDatasets,
        items,
        seed,
        shots,
        skip_if_measured: skip
      }
    }
    const declared = Object.entries(pairs).filter(([, reference]) => reference !== '')
    if (declared.length === 0) return null
    return {
      kind: 'fidelity',
      pairs: declared.map(([model, reference]) => ({ model, reference })),
      corpus: 'wikitext103',
      tokens: 10000,
      seed,
      skip_if_measured: skip
    }
  }, [
    view,
    models,
    contexts,
    generates,
    concurrencies,
    rounds,
    sampling,
    page,
    gate,
    skip,
    pairs,
    chosenDatasets,
    items,
    seed,
    shots
  ])

  useEffect(() => {
    const request = body()
    if (request === null) {
      setPriced(null)
      return
    }
    let live = true
    void pricePlan(request)
      .then((answer) => {
        if (live) {
          setPriced(answer)
          setFailure(null)
        }
      })
      .catch((error: unknown) => {
        if (live) {
          setPriced(null)
          setFailure(error instanceof Error ? error.message : String(error))
        }
      })
    return () => {
      live = false
    }
  }, [body])

  /* The profiles of the first checkpoint chosen. A profile is per model, and the key may not
     name one — so what is taken from it is the numbers, and those go in the key. Two models
     under the same numbers stay on the same axis, which is the whole point of the key. */
  const anchor: string | undefined = models[0]

  useEffect(() => {
    if (anchor === undefined) {
      setProfiles([])
      return
    }
    void listProfileNames(anchor)
      .then(setProfiles)
      .catch(() => setProfiles([]))
  }, [anchor])

  const fill = async (name: string): Promise<void> => {
    setProfile(name)
    if (name === '' || anchor === undefined) {
      setSampling(GREEDY)
      return
    }
    try {
      const found = await getProfile(anchor, name)
      /* A knob the profile leaves unset is one it does not opine on, and what stands under
         it here is this section's own neutral value — not the dialect's, which would make
         an unset temperature 1.0 and turn "fill from profile" into a drawn run nobody
         asked for. */
      setSampling({
        temperature: found.sampling.temperature ?? 0,
        top_p: found.sampling.top_p ?? 1,
        top_k: found.sampling.top_k,
        min_p: found.sampling.min_p ?? 0,
        repetition_penalty: found.sampling.repetition_penalty ?? 1,
        seed: found.sampling.seed
      })
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error))
    }
  }

  const knob = (name: 'temperature' | 'top_p' | 'top_k' | 'repetition_penalty', raw: string): void => {
    const parsed = raw.trim() === '' ? null : Number(raw)
    if (parsed !== null && !Number.isFinite(parsed)) return
    setProfile('')
    setSampling((current) => {
      switch (name) {
        case 'temperature':
          return { ...current, temperature: parsed ?? 0 }
        case 'top_p':
          return { ...current, top_p: parsed ?? 1 }
        case 'top_k':
          return { ...current, top_k: parsed }
        default:
          return { ...current, repetition_penalty: parsed ?? 1 }
      }
    })
  }

  const run = async (): Promise<void> => {
    const request = body()
    if (request === null) return
    setBusy(true)
    try {
      const job = await start(request)
      onStarted(job.id, request)
      onClose()
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  /* Which checkpoints can be a reference for this one. The vocabulary decides — a different
     one has no vector to subtract — and a different architecture is marked and allowed. */
  const referencesFor = (entry: CatalogEntry): CatalogEntry[] =>
    catalog.filter(
      (other) =>
        other.id !== entry.id &&
        (other.vocab_size === null ||
          entry.vocab_size === null ||
          other.vocab_size === entry.vocab_size)
    )

  const orphans = view === 'fidelity' ? catalog.filter((entry) => referencesFor(entry).length === 0) : []

  const shapes = priced?.shapes.length ?? 0
  const estimate =
    priced?.estimate_seconds === null || priced === null
      ? '—'
      : `≈ ${Math.max(1, Math.round(priced.estimate_seconds / 60))} min`

  const models_ = (
    <div className="pickscroll">
      {catalog.map((entry) => (
        <label className="mrow" key={entry.id}>
          <input
            type="checkbox"
            checked={models.includes(entry.id)}
            onChange={() =>
              setModels(
                models.includes(entry.id)
                  ? models.filter((id) => id !== entry.id)
                  : [...models, entry.id]
              )
            }
          />
          <span>
            <b>{entry.id}</b>
            <br />
            <span>{entry.quantization ?? 'dense'}</span>
          </span>
          <span className="g">{gb(entry.bytes_on_disk, 1)} GB</span>
        </label>
      ))}
    </div>
  )

  return (
    <div className="sheet" onClick={onClose}>
      <div className="sheetc m1" role="dialog" aria-label="New benchmark" onClick={(event) => event.stopPropagation()}>
        <div className="sheeth">
          <b>New benchmark</b>
          <span className="arrow">the key is what lets two checkpoints share an axis</span>
          <button type="button" className="btn quiet" style={{ marginLeft: 'auto' }} onClick={onClose}>
            Close
          </button>
        </div>

        <div className="sheetb">
          <div className="sheetl">
            <div className="field">
              <label>Measure</label>
              <div className="seg" style={{ flexDirection: 'column' }}>
                {(['speed', 'fidelity', 'quality'] as const).map((entry) => (
                  <button
                    key={entry}
                    type="button"
                    className={view === entry ? 'on' : undefined}
                    onClick={() => setView(entry)}
                  >
                    {entry === 'speed' ? 'Speed' : entry === 'fidelity' ? 'Fidelity' : 'Quality'}
                  </button>
                ))}
              </div>
            </div>

            {view === 'speed' && (
              <>
                <div className="field">
                  <label>Sampling</label>
                  <select
                    className="input"
                    value={profile}
                    onChange={(event) => void fill(event.target.value)}
                    aria-label="Fill from a profile"
                  >
                    <option value="">Greedy — argmax</option>
                    {profiles.map((name) => (
                      <option key={name} value={name}>
                        from profile {name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="cols">
                  <div className="field">
                    <label>Temperature</label>
                    <input
                      className="input mono"
                      value={sampling.temperature}
                      onChange={(event) => knob('temperature', event.target.value)}
                    />
                  </div>
                  <div className="field">
                    <label>Top-p</label>
                    <input
                      className="input mono"
                      value={sampling.top_p}
                      onChange={(event) => knob('top_p', event.target.value)}
                    />
                  </div>
                  <div className="field">
                    <label>Top-k</label>
                    <input
                      className="input mono"
                      value={sampling.top_k ?? ''}
                      placeholder="off"
                      onChange={(event) => knob('top_k', event.target.value)}
                    />
                  </div>
                  <div className="field">
                    <label>Repetition</label>
                    <input
                      className="input mono"
                      value={sampling.repetition_penalty}
                      onChange={(event) => knob('repetition_penalty', event.target.value)}
                    />
                  </div>
                </div>
                {/* The sampler is in the key: a drawn token pays for filters an argmax does
                    not, so two numbers taken under different ones are not one axis. */}
                <p className="note">
                  {sampling.temperature === 0
                    ? 'Argmax. The four measurements — load, prefill, TTFT, decode — are taken on every shape.'
                    : 'Drawn. The filters run per step and the key says so, so this does not compare with a greedy run.'}
                </p>
              </>
            )}

            {failure !== null && <p className="note bad">{failure}</p>}
            {priced !== null && priced.skipped.length > 0 && (
              <p className="note">
                {priced.skipped.length} shape{priced.skipped.length === 1 ? '' : 's'} skipped —{' '}
                {priced.skipped[0].reason === 'kv_over_budget' && priced.skipped[0].needed_bytes !== null
                  ? `the largest needed ${gb(priced.skipped[0].needed_bytes, 0)} GB`
                  : priced.skipped[0].reason.replace(/_/g, ' ')}
                .
              </p>
            )}
          </div>

          <div className="sheetr">
            {view === 'speed' && (
              <div className="two">
                <div>
                  <div className="grouplab" style={{ marginTop: 0 }}>
                    Models
                  </div>
                  {models_}
                </div>
                <div>
                  <div className="field">
                    <label>Context</label>
                    <Chips
                      values={CONTEXTS}
                      chosen={contexts}
                      label={human}
                      onToggle={(value) => setContexts(toggle(contexts, value))}
                    />
                  </div>
                  <div className="field" style={{ marginTop: 10 }}>
                    <label>Generate</label>
                    <Chips
                      values={GENERATES}
                      chosen={generates}
                      label={String}
                      onToggle={(value) => setGenerates(toggle(generates, value))}
                    />
                  </div>
                  <div className="field" style={{ marginTop: 10 }}>
                    <label>Concurrency</label>
                    <Chips
                      values={CONCURRENCIES}
                      chosen={concurrencies}
                      label={String}
                      onToggle={(value) => setConcurrencies(toggle(concurrencies, value))}
                    />
                  </div>
                  <div className="cols" style={{ marginTop: 12 }}>
                    <div className="field">
                      <label>Rounds</label>
                      <input
                        className="input mono"
                        value={rounds}
                        onChange={(event) => setRounds(Number(event.target.value) || 1)}
                      />
                    </div>
                    <div className="field">
                      <label>Page cache on load</label>
                      <select
                        className="input"
                        value={page}
                        onChange={(event) => setPage(event.target.value === 'cold' ? 'cold' : 'warm')}
                      >
                        <option value="warm">Warm — as in use</option>
                        <option value="cold">Cold — purge first</option>
                      </select>
                    </div>
                    <div className="field">
                      <label>Start below</label>
                      <select
                        className="input"
                        value={gate === null ? 'none' : String(gate)}
                        onChange={(event) =>
                          setGate(event.target.value === 'none' ? null : Number(event.target.value))
                        }
                      >
                        {GATES.map((value) => (
                          <option key={String(value)} value={value === null ? 'none' : String(value)}>
                            {value === null ? 'No gate' : `${value} °C`}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <label className="check" style={{ marginTop: 13 }}>
                    <input type="checkbox" checked={skip} onChange={() => setSkip(!skip)} />
                    <b>Skip if already run</b>
                  </label>
                  <div className="keyline" style={{ marginTop: 11 }}>
                    key{' '}
                    <b>
                      {speedKey(
                        contexts[0] ?? 4096,
                        generates[0] ?? 256,
                        concurrencies[0] ?? 1,
                        rounds,
                        sampling,
                        page
                      )}
                    </b>
                  </div>
                  <div className="cost" style={{ marginTop: 8 }}>
                    <span>Will run</span>
                    <b>{shapes}</b>
                    <span>shapes</span>
                    <span className="r">{estimate}</span>
                  </div>
                  <p className="note" style={{ marginTop: 9 }}>
                    {priced === null
                      ? 'Pick at least one checkpoint.'
                      : `${priced.already.length} already answered under this key.`}
                  </p>
                </div>
              </div>
            )}

            {view === 'fidelity' && (
              <div className="two">
                <div>
                  <div className="grouplab" style={{ marginTop: 0 }}>
                    Model and its reference
                  </div>
                  <div className="pickscroll">
                    {catalog
                      .filter((entry) => referencesFor(entry).length > 0)
                      .map((entry) => (
                        <div className="pair" key={entry.id}>
                          <input
                            type="checkbox"
                            checked={(pairs[entry.id] ?? '') !== ''}
                            onChange={() =>
                              setPairs({
                                ...pairs,
                                [entry.id]:
                                  (pairs[entry.id] ?? '') === '' ? referencesFor(entry)[0].id : ''
                              })
                            }
                          />
                          <span className="t">
                            <b>{entry.id}</b>
                            <span>{entry.shape ?? '—'}</span>
                          </span>
                          <select
                            className="input"
                            value={pairs[entry.id] ?? ''}
                            onChange={(event) => setPairs({ ...pairs, [entry.id]: event.target.value })}
                          >
                            <option value="">no reference</option>
                            {referencesFor(entry).map((other) => (
                              <option key={other.id} value={other.id}>
                                {other.id}
                                {other.shape !== entry.shape ? ' · other shape' : ''}
                              </option>
                            ))}
                          </select>
                        </div>
                      ))}
                  </div>
                  {orphans.length > 0 && (
                    <p className="note" style={{ marginTop: 9 }}>
                      No reference shares a vocabulary with {orphans.map((entry) => entry.id).join(', ')}.
                    </p>
                  )}
                </div>
                <div>
                  <div className="cols">
                    <div className="field">
                      <label>Corpus</label>
                      <select className="input" value="wikitext103" disabled>
                        <option value="wikitext103">wikitext-103 · test</option>
                      </select>
                    </div>
                    <div className="field">
                      <label>Tokens</label>
                      <input className="input mono" value="10000" readOnly />
                    </div>
                    <div className="field">
                      <label>Seed</label>
                      <input
                        className="input mono"
                        value={seed}
                        onChange={(event) => setSeed(Number(event.target.value) || 0)}
                      />
                    </div>
                    <div className="field">
                      <label>Reference logits</label>
                      <select className="input" value="cache" disabled>
                        <option value="cache">Cache top-64</option>
                      </select>
                    </div>
                  </div>
                  <label className="check" style={{ marginTop: 13 }}>
                    <input type="checkbox" checked={skip} onChange={() => setSkip(!skip)} />
                    <b>Skip if already run</b>
                  </label>
                  <div className="keyline" style={{ marginTop: 11 }}>
                    key <b>wikitext103 · n10000 · s{seed} · ref per model</b>
                  </div>
                  <div className="cost" style={{ marginTop: 8 }}>
                    <span>Will run</span>
                    <b>{shapes}</b>
                    <span>pairs</span>
                    <span className="r">{estimate}</span>
                  </div>
                  <p className="note" style={{ marginTop: 9 }}>
                    Each reference is read once and its top-64 cached, so two candidates sharing a
                    reference pay for it once. A model whose vocabulary nobody else shares has no
                    pair to make.
                  </p>
                </div>
              </div>
            )}

            {view === 'quality' && (
              <div className="two">
                <div>
                  <div className="grouplab" style={{ marginTop: 0 }}>
                    Models
                  </div>
                  {models_}
                </div>
                <div>
                  <div className="grouplab" style={{ marginTop: 0 }}>
                    Datasets
                  </div>
                  <div>
                    {datasets
                      .filter((entry) => entry.use !== 'corpus')
                      .map((entry) => (
                        <label className={`dsrow${entry.builtin ? '' : ' custom'}`} key={entry.id}>
                          <input
                            type="checkbox"
                            checked={chosenDatasets.includes(entry.id)}
                            onChange={() =>
                              setChosenDatasets(
                                chosenDatasets.includes(entry.id)
                                  ? chosenDatasets.filter((id) => id !== entry.id)
                                  : [...chosenDatasets, entry.id]
                              )
                            }
                          />
                          <span className="t">
                            <b>{entry.id}</b>
                            <span>{entry.repo}</span>
                          </span>
                          <span className="k">{entry.use.replace('_', ' ')}</span>
                          <span className="n">{entry.size ?? '—'}</span>
                        </label>
                      ))}
                  </div>
                  <div className="addrow">
                    <button type="button" className="btn" onClick={onDeclare}>
                      Add from the Hub
                    </button>
                  </div>
                  <div className="cols" style={{ marginTop: 12 }}>
                    <div className="field">
                      <label>Items</label>
                      <input
                        className="input mono"
                        value={items}
                        onChange={(event) => setItems(Number(event.target.value) || 1)}
                      />
                    </div>
                    <div className="field">
                      <label>Seed</label>
                      <input
                        className="input mono"
                        value={seed}
                        onChange={(event) => setSeed(Number(event.target.value) || 0)}
                      />
                    </div>
                    <div className="field">
                      <label>Shots</label>
                      <input
                        className="input mono"
                        value={shots}
                        onChange={(event) => setShots(Number(event.target.value) || 0)}
                      />
                    </div>
                  </div>
                  <label className="check" style={{ marginTop: 13 }}>
                    <input type="checkbox" checked={skip} onChange={() => setSkip(!skip)} />
                    <b>Skip if already run</b>
                  </label>
                  <div className="keyline" style={{ marginTop: 11 }}>
                    key <b>n{items} · s{seed} · {shots}-shot</b>
                  </div>
                  <div className="cost" style={{ marginTop: 8 }}>
                    <span>Will run</span>
                    <b>{shapes}</b>
                    <span>runs</span>
                    <span className="r">{estimate}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="sheetf">
          <button type="button" className="btn pri" disabled={busy || shapes === 0} onClick={() => void run()}>
            Run {shapes} shape{shapes === 1 ? '' : 's'}
          </button>
          <button type="button" className="btn quiet" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}

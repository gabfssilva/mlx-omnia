import { type JSX, useCallback, useEffect, useRef, useState } from 'react'
import { cancelJob, isFinished, jobEvents } from '../../api/http'
import type { JobView } from '../../api/http'
import {
  BITS,
  GROUP_SIZES,
  METHODS,
  REPO,
  create,
  detail,
  listModels,
  machine,
  price,
  residency
} from './api'
import type { CatalogEntry, Machine, Method, PricedPlan, Residency } from './api'
import { group } from './leaves'
import type { LeafGroup } from './leaves'
import './quantize.css'

const GIB = 1024 ** 3
const PRICE_DEBOUNCE_MS = 180

const gb = (bytes: number): string => (bytes / GIB).toFixed(1)

/* What the checkpoint calls itself, the way the screen writes it. */
const width = (dtype: string | null): string => {
  switch (dtype) {
    case null:
      return ''
    case 'bfloat16':
      return 'bf16'
    case 'float16':
      return 'fp16'
    case 'float32':
      return 'fp32'
    default:
      return dtype
  }
}

const describe = (entry: CatalogEntry): string =>
  [entry.id, [entry.quantization ?? 'dense', width(entry.dtype)].filter(Boolean).join(' '), `${gb(entry.bytes_on_disk)} GB`].join(' · ')

/* The id the screen suggests: `local/` because a load by id asks the Hub first, and a
   name that also exists up there resolves to somebody else's weights. */
const suggest = (source: string, bits: number): string =>
  source === '' ? '' : `local/${source.split('/').pop() ?? source}-${bits}bit`

export function Quantize(): JSX.Element {
  const [entries, setEntries] = useState<CatalogEntry[] | null>(null)
  const [system, setSystem] = useState<Machine | null>(null)
  const [state, setState] = useState<Residency | null>(null)

  const [source, setSource] = useState('')
  const [method, setMethod] = useState<Method>('rtn')
  const [bits, setBits] = useState(4)
  const [groupSize, setGroupSize] = useState(64)
  const [overrides, setOverrides] = useState<Record<string, number | null>>({})
  const [repo, setRepo] = useState('')
  const [named, setNamed] = useState(false)

  const [plan, setPlan] = useState<PricedPlan | null>(null)
  const [refusal, setRefusal] = useState<string | null>(null)
  const [pricing, setPricing] = useState(false)

  const [job, setJob] = useState<JobView | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const stream = useRef<AbortController | null>(null)

  const load = useCallback(async () => {
    const [catalog, resident] = await Promise.all([
      listModels().catch(() => null),
      residency().catch(() => null)
    ])
    if (catalog !== null) setEntries(catalog)
    setState(resident)
    return catalog
  }, [])

  useEffect(() => {
    machine().then(setSystem).catch(() => setSystem(null))
    void load().then((catalog) => {
      if (catalog === null || catalog.length === 0) return
      const dense = catalog.find((entry) => entry.quantization === null)
      setSource((current) => (current === '' ? (dense ?? catalog[0]).id : current))
    })
  }, [load])

  useEffect(() => {
    if (!named) setRepo(suggest(source, bits))
  }, [source, bits, named])

  useEffect(() => () => stream.current?.abort(), [])

  /* Every change of the selection is a new price, and the one in flight is dropped:
     the panel never shows the answer to a question that is no longer on screen. */
  useEffect(() => {
    if (source === '') return
    const controller = new AbortController()
    const timer = setTimeout(() => {
      setPricing(true)
      price({ source, bits, group_size: groupSize, overrides, method }, controller.signal)
        .then((priced) => {
          setPlan(priced)
          setRefusal(null)
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return
          setPlan(null)
          setRefusal(detail(error))
        })
        .finally(() => {
          if (!controller.signal.aborted) setPricing(false)
        })
    }, PRICE_DEBOUNCE_MS)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [source, bits, groupSize, overrides, method])

  const follow = useCallback(
    async (id: string) => {
      stream.current?.abort()
      const controller = new AbortController()
      stream.current = controller
      try {
        for await (const frame of jobEvents(id, controller.signal)) setJob(frame)
      } catch {
        /* the stream ends with the view; the job outlives it either way */
      }
      if (!controller.signal.aborted) await load()
    },
    [load]
  )

  const launch = async (): Promise<void> => {
    setFailure(null)
    try {
      const started = await create({
        source,
        bits,
        group_size: groupSize,
        overrides,
        method,
        repo
      })
      setJob(started)
      await follow(started.id)
    } catch (error) {
      setFailure(detail(error))
    }
  }

  const stop = async (): Promise<void> => {
    if (job === null) return
    try {
      setJob(await cancelJob(job.id))
    } catch (error) {
      setFailure(detail(error))
    }
  }

  const pick = (pattern: string, choice: string): void =>
    setOverrides((current) => {
      const next = { ...current }
      if (choice === '') delete next[pattern]
      else next[pattern] = choice === 'dense' ? null : Number(choice)
      return next
    })

  const chosen = entries?.find((entry) => entry.id === source) ?? null
  const groups = plan === null ? [] : group(plan.leaves)
  const running = job !== null && !isFinished(job)
  const ready = plan !== null && refusal === null && REPO.test(repo) && !running
  const held = state === null ? null : state.resident_bytes + state.kv_bytes
  const packed = plan === null ? 0 : plan.leaves.filter((leaf) => leaf.bits !== null).length

  return (
    <>
      <div className="vhead qz">
        <h1>Quantize</h1>
        <div className="trail">
          {job !== null && (
            <span className={running ? 'chip hot' : 'chip res'} title={job.error ?? job.progress.message}>
              <span className="dot" />
              <span className="what">
                {job.progress.total !== null && job.progress.total > 0 && running
                  ? `${Math.round((job.progress.completed / job.progress.total) * 100)}% · `
                  : ''}
                {job.error ?? job.progress.message}
              </span>
            </span>
          )}
          {running ? (
            <button className="btn danger" onClick={() => void stop()}>
              Cancel
            </button>
          ) : (
            <button className="btn ember" disabled={!ready} onClick={() => void launch()}>
              Quantize
            </button>
          )}
        </div>
      </div>

      <div className="vbody qz">
        {failure !== null && <p className="refusal">{failure}</p>}

        <div className="qgrid">
          <div className="card">
            <div className="fieldcol">
              <span className="eyebrow">Source</span>
              <select
                className="input mono"
                value={source}
                onChange={(event) => {
                  setSource(event.target.value)
                  /* The patterns belong to the tree they were read off: carried over,
                     they match no leaf and the next price fails for the wrong reason. */
                  setOverrides({})
                }}
              >
                {entries === null && <option value="">reading the catalog…</option>}
                {entries !== null && entries.length === 0 && (
                  <option value="">nothing on this disk to quantize</option>
                )}
                {entries?.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {describe(entry)}
                  </option>
                ))}
              </select>
            </div>

            <div className="fieldcol">
              <span className="eyebrow">Method</span>
              <div className="seg">
                {METHODS.map((entry) => (
                  <button
                    key={entry.id}
                    className={entry.id === method ? 'on' : undefined}
                    onClick={() => setMethod(entry.id)}
                  >
                    {entry.label}
                  </button>
                ))}
              </div>
              <span className="eyebrow" style={{ fontWeight: 400 }}>
                AWQ, GPTQ and oQ run a calibration pass before writing.
              </span>
            </div>

            <div className="fieldcol">
              <span className="eyebrow">Width</span>
              <div className="seg">
                {BITS.map((value) => (
                  <button
                    key={value}
                    className={value === bits ? 'on' : undefined}
                    onClick={() => setBits(value)}
                  >
                    {value}
                  </button>
                ))}
              </div>
            </div>

            <div className="fieldcol">
              <span className="eyebrow">Group size</span>
              <div className="seg">
                {GROUP_SIZES.map((value) => (
                  <button
                    key={value}
                    className={value === groupSize ? 'on' : undefined}
                    onClick={() => setGroupSize(value)}
                  >
                    {value}
                  </button>
                ))}
              </div>
            </div>

            <div className="fieldcol">
              <span className="eyebrow">Output id</span>
              <input
                className="input mono"
                value={repo}
                spellCheck={false}
                onChange={(event) => {
                  setNamed(true)
                  setRepo(event.target.value)
                }}
              />
              {repo !== '' && !REPO.test(repo) && (
                <span className="refusal">
                  Two segments, and nothing that could name a directory outside the cache:
                  org/name.
                </span>
              )}
            </div>
          </div>

          <div className={pricing ? 'card projected pricing' : 'card projected'}>
            <span className="eyebrow">Projected</span>
            {plan === null ? (
              <>
                <div className="bignum" style={{ marginTop: 8 }}>
                  —
                </div>
                {refusal !== null && <p className="refusal">{refusal}</p>}
              </>
            ) : (
              <>
                <div className="bignum" style={{ marginTop: 8 }}>
                  {gb(plan.entry_bytes)}
                  <small>GB</small>
                </div>
                <div className="gauge" style={{ marginTop: 16 }}>
                  <div className="bar">
                    <div
                      className="fill"
                      style={{
                        width:
                          chosen === null || chosen.bytes_on_disk === 0
                            ? '0%'
                            : `${Math.min(100, (plan.entry_bytes / chosen.bytes_on_disk) * 100).toFixed(0)}%`
                      }}
                    />
                  </div>
                  <div className="sub">
                    <span>{plan.bits_per_weight.toFixed(2)} bits / weight</span>
                    <span>{chosen === null ? '' : `source ${gb(chosen.bytes_on_disk)} GB`}</span>
                  </div>
                </div>
                <div className="kvline" style={{ marginTop: 14 }}>
                  <span className="eyebrow">If loaded now</span>
                  <b className="tn">
                    {held === null || system === null
                      ? '—'
                      : `${gb(held + plan.entry_bytes)} / ${Math.round(system.memory_bytes / GIB)} GB`}
                  </b>
                </div>
                <div className="kvline">
                  <span className="eyebrow">Leaves packed</span>
                  <b className="tn">
                    {packed} / {plan.leaves.length}
                  </b>
                </div>
              </>
            )}
            <p className="eyebrow" style={{ lineHeight: 1.6, margin: 'auto 0 0', paddingTop: 12 }}>
              {method === 'oq' && plan !== null
                ? 'The size and budget hold; which leaves the calibration promotes is measured by the job, not projected here.'
                : 'Priced by the daemon against the checkpoint’s own leaves — nothing written yet. Parity is measured after the job, never predicted.'}
            </p>
          </div>
        </div>

        <div className="card">
          <h3>
            Overrides <span>per leaf group, from the checkpoint itself</span>
          </h3>
          {groups.length === 0 ? (
            <p className="eyebrow">
              The leaves come with the price: pick a source the daemon can plan.
            </p>
          ) : (
            groups.map((entry) => (
              <Row
                key={entry.pattern}
                group={entry}
                bits={bits}
                override={overrides[entry.pattern]}
                overridden={entry.pattern in overrides}
                onPick={pick}
              />
            ))
          )}
        </div>
      </div>
    </>
  )
}

function Row({
  group: entry,
  bits,
  override,
  overridden,
  onPick
}: {
  group: LeafGroup
  bits: number
  override: number | null | undefined
  overridden: boolean
  onPick: (pattern: string, choice: string) => void
}): JSX.Element {
  const selected = !overridden ? '' : override === null ? 'dense' : String(override)
  return (
    <div className={entry.bits === null ? 'leaf frozen' : 'leaf'}>
      <code title={entry.pattern}>{entry.label}</code>
      {overridden && <span className="ovtag">override</span>}
      <span className="meta">
        {entry.leaves} {entry.leaves === 1 ? 'leaf' : 'leaves'} · {(entry.bytes / GIB).toFixed(2)} GB
      </span>
      <select
        className="pick"
        value={selected}
        onChange={(event) => onPick(entry.pattern, event.target.value)}
      >
        <option value="">{entry.mixed ? 'mixed' : `${bits}-bit`}</option>
        {BITS.filter((value) => value !== bits).map((value) => (
          <option key={value} value={value}>
            {value}-bit
          </option>
        ))}
        <option value="dense">dense</option>
      </select>
    </div>
  )
}

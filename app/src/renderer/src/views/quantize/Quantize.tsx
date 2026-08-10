import { type JSX, useCallback, useEffect, useRef, useState } from 'react'
import { cancelJob, isFinished, jobEvents } from '../../api/http'
import type { JobView } from '../../api/http'
import {
  BITS,
  GROUP_SIZES,
  METHODS,
  MODES,
  REPO,
  create,
  detail,
  listModels,
  machine,
  price,
  residency,
  selection
} from './api'
import type {
  CatalogEntry,
  Machine,
  Method,
  Mode,
  Override,
  PricedPlan,
  Residency
} from './api'
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
const suggest = (source: string, mode: Mode, bits: number): string =>
  source === ''
    ? ''
    : `local/${source.split('/').pop() ?? source}-${mode === 'affine' ? `${bits}bit` : mode}`

export function Quantize({
  initial,
  onClose
}: {
  /** The checkpoint Library asked to transform. */
  initial?: string
  onClose?: () => void
}): JSX.Element {
  const [entries, setEntries] = useState<CatalogEntry[] | null>(null)
  const [system, setSystem] = useState<Machine | null>(null)
  const [state, setState] = useState<Residency | null>(null)

  const [source, setSource] = useState(initial ?? '')
  const [method, setMethod] = useState<Method>('rtn')
  const [mode, setMode] = useState<Mode>('affine')
  const [bits, setBits] = useState(4)
  const [groupSize, setGroupSize] = useState(64)
  const [overrides, setOverrides] = useState<Record<string, Override | null>>({})
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
    if (!named) setRepo(suggest(source, mode, bits))
  }, [source, mode, bits, named])

  useEffect(() => () => stream.current?.abort(), [])

  /* Every change of the selection is a new price, and the one in flight is dropped:
     the panel never shows the answer to a question that is no longer on screen. */
  useEffect(() => {
    if (source === '') return
    const controller = new AbortController()
    const timer = setTimeout(() => {
      setPricing(true)
      price(
        { source, ...selection(mode, bits, groupSize), overrides, method },
        controller.signal
      )
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
  }, [source, mode, bits, groupSize, overrides, method])

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
        ...selection(mode, bits, groupSize),
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
      else if (choice === 'dense') next[pattern] = null
      /* The group size the row already carried survives a change of width: the two are
         picked apart, and re-picking one is not a way of forgetting the other. */
      else next[pattern] = { ...current[pattern], bits: Number(choice) }
      return next
    })

  /* Only where a width was already picked: the wire has no override that names a group
     size alone, and the plan's own is what a row without one means. */
  const regroup = (pattern: string, choice: string): void =>
    setOverrides((current) => {
      const found = current[pattern]
      if (found == null) return current
      const { group_size: _dropped, ...rest } = found
      return {
        ...current,
        [pattern]: choice === '' ? rest : { ...rest, group_size: Number(choice) }
      }
    })

  /* The mode's own shape when it has one — which is also what says it is not affine. */
  const exponent = MODES.find((entry) => entry.id === mode && entry.shape !== null) ?? null
  /* The two methods whose widths are the allocator's: what a group the caller left alone
     ends up at is decided by the calibration, so the screen has no number to name yet. */
  const allocated = method === 'oq' || method === 'oqe'
  /* Under the allocator the base width is a width like any other: it is not what a group
     left alone gets, so naming it is a choice and it stays on the list. */
  const widths =
    exponent !== null ? [] : allocated ? BITS : BITS.filter((value) => value !== bits)
  const chosen = entries?.find((entry) => entry.id === source) ?? null
  const groups = plan === null ? [] : group(plan.leaves)
  const running = job !== null && !isFinished(job)
  const ready = plan !== null && refusal === null && REPO.test(repo) && !running
  const held = state === null ? null : state.resident_bytes + state.kv_bytes
  const packed = plan === null ? 0 : plan.leaves.filter((leaf) => leaf.bits !== null).length

  return (
    <div className="sheet">
      <div className="sheetc qz">
      <div className="sheeth">
        <b>Quantize</b>
        <span className="arrow">{source === '' ? 'pick a checkpoint' : source}</span>
        <div className="trail">
          {job !== null && (
            <span className="chip" title={job.error ?? job.progress.message}>
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
            <button className="btn pri" disabled={!ready} onClick={() => void launch()}>
              Quantize
            </button>
          )}
          {onClose !== undefined && (
            <button className="btn quiet" onClick={onClose}>
              Close
            </button>
          )}
        </div>
      </div>

      <div className="qbody scroll">
        {failure !== null && <p className="refusal">{failure}</p>}

        <div className="qgrid">
          <div className="blk">
            <div className="field">
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

            <div className="field">
              <span className="eyebrow">Format</span>
              <div className="seg">
                {MODES.map((entry) => (
                  <button
                    key={entry.id}
                    className={entry.id === mode ? 'on' : undefined}
                    onClick={() => {
                      setMode(entry.id)
                      if (entry.shape === null) return
                      /* The exponent grid has no bias to search and no width to pick, so
                         the two controls it decides go with it. */
                      setMethod('rtn')
                      setOverrides((current) =>
                        Object.fromEntries(
                          Object.entries(current).filter(([, value]) => value === null)
                        )
                      )
                    }}
                  >
                    {entry.label}
                  </button>
                ))}
              </div>
              <span className="eyebrow" style={{ fontWeight: 400 }}>
                {exponent === null
                  ? 'A scale and a bias per group, at the width and group size below.'
                  : `An exponent per group and no bias: ${exponent.shape?.[1]} bits at group ${exponent.shape?.[0]}, packed by mx.quantize. Method, width and group size are the mode's.`}
              </span>
            </div>

            {exponent === null && (
              <>
                <div className="field">
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
                    AWQ, GPTQ, oQ and oQe run a calibration pass before writing.
                  </span>
                </div>

                <div className="field">
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

                <div className="field">
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
              </>
            )}

            <div className="field">
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
              {(method === 'oq' || method === 'oqe') && plan !== null
                ? 'The size and budget hold; which leaves the calibration promotes is measured by the job, not projected here.'
                : 'Priced by the daemon against the checkpoint’s own leaves — nothing written yet. Parity is measured after the job, never predicted.'}
            </p>
          </div>
        </div>

        <div className="blk">
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
                widths={widths}
                placeholder={
                  exponent?.label ??
                  (allocated ? 'auto' : entry.mixed ? 'mixed' : `${bits}-bit`)
                }
                groups={exponent === null ? GROUP_SIZES : []}
                planGroup={exponent === null ? groupSize : null}
                override={overrides[entry.pattern]}
                overridden={entry.pattern in overrides}
                onPick={pick}
                onRegroup={regroup}
              />
            ))
          )}
        </div>
      </div>
      </div>
    </div>
  )
}

function Row({
  group: entry,
  widths,
  groups,
  planGroup,
  placeholder,
  override,
  overridden,
  onPick,
  onRegroup
}: {
  group: LeafGroup
  /* The widths worth naming here — empty under an exponent-scaled mode, where the width is
     the mode and dense is the only override left. */
  widths: readonly number[]
  /* The group sizes, on the same rule. */
  groups: readonly number[]
  /* The group size this row falls back to, `null` where the mode decides it. */
  planGroup: number | null
  /* What the group is without an override of its own. */
  placeholder: string
  override: Override | null | undefined
  overridden: boolean
  onPick: (pattern: string, choice: string) => void
  onRegroup: (pattern: string, choice: string) => void
}): JSX.Element {
  const selected = !overridden ? '' : override == null ? 'dense' : String(override.bits)
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
        <option value="">{placeholder}</option>
        {widths.map((value) => (
          <option key={value} value={value}>
            {value}-bit
          </option>
        ))}
        <option value="dense">dense</option>
      </select>
      {override != null && groups.length > 0 && (
        <select
          className="pick"
          value={override.group_size === undefined ? '' : String(override.group_size)}
          onChange={(event) => onRegroup(entry.pattern, event.target.value)}
        >
          <option value="">group {planGroup}</option>
          {groups
            .filter((value) => value !== planGroup)
            .map((value) => (
              <option key={value} value={value}>
                group {value}
              </option>
            ))}
        </select>
      )}
    </div>
  )
}

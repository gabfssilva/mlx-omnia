/* One checkpoint: what it costs to read per token, what it reached when it was benched,
   what it is holding now, and the two things that can be done to it — residency, which is
   a job, and deletion from disk, which the daemon refuses while it is resident. */

import { type JSX, useEffect, useState } from 'react'
import * as api from './api'
import { CardBlock } from './Card'
import { ceilingRate, contextTokens, displayName, gib, homeless, idle, percent, perToken } from './format'
import type { Fleet } from './Models'
import { Profiles } from './Profiles'

interface Props {
  id: string
  fleet: Fleet
  onBack: () => void
}

function Copy({ value }: { value: string }): JSX.Element {
  const [copied, setCopied] = useState(false)
  return (
    <button
      className="iconbtn"
      title={copied ? 'Copied' : 'Copy'}
      aria-label="Copy"
      onClick={() => {
        void navigator.clipboard.writeText(value).then(
          () => {
            setCopied(true)
            setTimeout(() => setCopied(false), 1200)
          },
          () => setCopied(false)
        )
      }}
    >
      {copied ? (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  )
}

export function ModelDetail({ id, fleet, onBack }: Props): JSX.Element {
  const { say } = fleet
  const [entry, setEntry] = useState<api.CatalogEntry | null>(null)
  const [ttl, setTtl] = useState<number | null>(null)
  const [confirming, setConfirming] = useState(false)

  useEffect(() => {
    api
      .getModel(id)
      .then(setEntry)
      .catch((failure: unknown) => say(api.reason(failure)))
  }, [id, say])

  useEffect(() => {
    api
      .getConfig()
      .then((config) => {
        const value = config['idle_ttl_seconds']?.value
        setTtl(typeof value === 'number' ? value : null)
      })
      .catch(() => setTtl(null))
  }, [])

  const resident = fleet.residents.get(id)
  const bench = fleet.benches.get(id)
  const job = fleet.jobs[id]
  const system = fleet.system
  const memory = system?.memory_bytes ?? null

  const ceiling =
    entry === null || entry.bytes_per_token === null || system === null
      ? null
      : ceilingRate(entry.bytes_per_token, system.bandwidth_sustained_gbs)
  const fraction =
    bench === undefined
      ? null
      : (bench.ceiling_fraction ?? (ceiling === null ? null : bench.tokens_per_second / ceiling))
  const share =
    resident === undefined || memory === null
      ? null
      : percent((resident.weights_bytes + resident.kv_bytes) / memory)

  const remove = (): void => {
    say(null)
    void (async () => {
      try {
        await api.removeModel(id)
        fleet.rescan()
        onBack()
      } catch (failure) {
        setConfirming(false)
        say(api.reason(failure))
      }
    })()
  }

  return (
    <>
      <div className="vhead">
        <button className="iconbtn" title="Back" aria-label="Back" onClick={onBack}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
        <h1>{displayName(id)}</h1>
        {job !== undefined ? (
          <span className="chip hot">
            <span className="dot" />
            Loading
          </span>
        ) : fleet.live === id ? (
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
        <div className="trail">
          {job !== undefined ? (
            <button className="btn solid" onClick={() => fleet.cancel(id)}>
              Cancel load
            </button>
          ) : resident !== undefined ? (
            <button className="btn solid" onClick={() => fleet.unload(id)}>
              Unload
            </button>
          ) : (
            <button className="btn solid" onClick={() => fleet.load(id)}>
              Load
            </button>
          )}
          {confirming ? (
            <>
              <button className="btn danger" onClick={remove}>
                Delete from disk
              </button>
              <button className="btn" onClick={() => setConfirming(false)}>
                Keep
              </button>
            </>
          ) : (
            <button className="btn danger" onClick={() => setConfirming(true)}>
              Remove
            </button>
          )}
        </div>
      </div>

      <div className="vbody">
        {fleet.error !== null && (
          <div className="notice">
            {fleet.error}
            <button className="iconbtn" aria-label="Dismiss" onClick={() => say(null)}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        )}

        <div className="ceilblock">
          <div className="hero">
            <div className="bignum">
              {bench === undefined ? '—' : bench.tokens_per_second.toFixed(1)}
              <small>tok/s</small>
            </div>
            {fraction !== null && ceiling !== null && (
              <div className="heropct">
                {percent(fraction)}% <small>of the {Math.round(ceiling)} tok/s ceiling</small>
              </div>
            )}
            {bench === undefined && <div className="eyebrow">not benched on this machine</div>}
          </div>
          <div className="gauge ceil">
            <div className="bar">
              {fraction !== null && <div className="fill" style={{ width: `${percent(fraction)}%` }} />}
            </div>
            <div className="sub">
              <span>
                {entry === null || entry.bytes_per_token === null
                  ? 'bytes per token unknown'
                  : `${perToken(entry.bytes_per_token)} GB moved per token`}
              </span>
              <span>
                {system === null
                  ? 'ceiling'
                  : `ceiling · ${system.bandwidth_sustained_gbs} GB/s sustained`}
              </span>
            </div>
          </div>
        </div>

        <div className="stats">
          <div className="stat">
            <span className="eyebrow">Bytes / token</span>
            <b>
              {entry === null || entry.bytes_per_token === null ? '—' : perToken(entry.bytes_per_token)}
              <small>GB</small>
            </b>
          </div>
          <div className="stat">
            <span className="eyebrow">On disk</span>
            <b>
              {entry === null ? '—' : gib(entry.bytes_on_disk)}
              <small>GB</small>
            </b>
          </div>
          <div className="stat">
            <span className="eyebrow">Weights resident</span>
            <b>
              {resident === undefined ? '—' : gib(resident.weights_bytes)}
              {resident !== undefined && <small>GB</small>}
            </b>
          </div>
          <div className="stat">
            <span className="eyebrow">KV cache</span>
            <b>
              {resident === undefined ? '—' : gib(resident.kv_bytes)}
              {resident !== undefined && <small>GB</small>}
            </b>
          </div>
          <div className="stat">
            <span className="eyebrow">Memory share</span>
            <b>
              {share === null ? '—' : share}
              {share !== null && memory !== null && <small>% of {gib(memory, 0)} GB</small>}
            </b>
          </div>
          <div className="stat">
            <span className="eyebrow">Context</span>
            <b>
              {entry === null || entry.context === null ? '—' : contextTokens(entry.context)}
              {entry !== null && entry.context !== null && <small>tokens</small>}
            </b>
          </div>
        </div>

        <div className="cols2">
          <Profiles model={id} />
          <div className="card">
            <h3>Serving</h3>
            <div className="copyrow">
              <span className="eyebrow">Model id</span>
              <code>{id}</code>
              <Copy value={id} />
            </div>
            <div className="kvline">
              <span className="eyebrow">Idle unload</span>
              <b>{idle(ttl)}</b>
            </div>
            <div className="kvline">
              <span className="eyebrow">Architecture</span>
              <b className="mono arch">{entry?.architecture ?? '—'}</b>
            </div>
            <div className="kvline">
              <span className="eyebrow">Quantization</span>
              <b>
                {entry === null
                  ? '—'
                  : [entry.quantization ?? 'dense', entry.dtype].filter(Boolean).join(' · ')}
              </b>
            </div>
            <div className="copyrow">
              <span className="eyebrow">Checkpoint</span>
              <code>{entry?.directory ?? '—'}</code>
              {entry !== null && <Copy value={entry.directory} />}
            </div>
          </div>
        </div>

        {entry !== null && <CardBlock id={id} store={homeless(entry.directory)} />}
      </div>
    </>
  )
}

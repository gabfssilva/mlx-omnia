import { type JSX, useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'
import type { StoredMessage, TurnMetrics } from './api'

/* The turn being streamed, before it becomes a message. */
export interface Live {
  content: string
  reasoning: string
  error: string | null
}

const CHEVRON = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6" />
  </svg>
)

function Prose({ text, caret }: { text: string; caret: boolean }): JSX.Element {
  return (
    <div className="mdbody">
      <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
        {text}
      </Markdown>
      {caret ? <span className="caret" /> : null}
    </div>
  )
}

/* Seconds under a minute of them, milliseconds under a second: the load of a cold 30B and
   the ttft of a warm one are two orders of magnitude apart and share this line. */
const duration = (ms: number): string =>
  ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`

const rate = (value: number): string =>
  value >= 1000 ? `${(value / 1000).toFixed(1)}k tok/s` : `${value.toFixed(1)} tok/s`

/* What the turn cost, drawn the same while it is being written and once it is done — the
   numbers come from the daemon either way, so the line does not change shape at the end. */
function Metrics({ metrics, waiting }: { metrics?: TurnMetrics; waiting?: number }): JSX.Element {
  const parts: JSX.Element[] = []
  const say = (text: string): number => parts.push(<span key={parts.length}>{text}</span>)
  if (waiting != null) say(`waiting ${duration(waiting)}`)
  if (metrics?.load_ms != null) say(`load ${duration(metrics.load_ms)}`)
  if (metrics?.ttft_ms != null) say(`ttft ${duration(metrics.ttft_ms)}`)
  const prefill = metrics?.prefill_tokens_per_second
  if (prefill != null) say(`prefill ${rate(prefill)}`)
  const decode = metrics?.tokens_per_second
  if (decode != null) say(`decode ${rate(decode)}`)
  const fraction = metrics?.ceiling_fraction
  if (fraction != null) {
    parts.push(
      <span key={parts.length}>
        <b className="pct">{Math.round(fraction * 100)}%</b> of ceiling
      </span>
    )
  }
  /* The one thing on this line that is not a cost: whether a drafter wrote any of it. Only
     drawn when one did, because "no speculation" is what every turn on every other model
     looks like, and a chip saying so on all of them says nothing. */
  const speculation = metrics?.speculation
  if (speculation != null && speculation.rounds > 0) {
    /* Per round and not as a share of what was proposed: the share falls with the block
       length whatever the drafter does — a block of 16 proposes 15 for the same two or
       three that land — so two settings cannot be read against each other by it. What
       decides whether a round paid is how many ids it settled against how wide it was. */
    const landed = speculation.accepted / speculation.rounds
    const width = speculation.proposed / speculation.rounds
    parts.push(
      <span
        key={parts.length}
        title={`${speculation.accepted} of ${speculation.proposed} ids proposed over ${speculation.rounds} rounds`}
      >
        dflash <b className="pct">{landed.toFixed(1)}</b> of {width.toFixed(0)} a round
      </span>
    )
  }
  if (metrics?.finish === 'length') say('cut at the token budget')
  return <div className="metrics">{parts}</div>
}

function Answer({
  content,
  reasoning,
  metrics,
  waiting,
  error,
  streaming
}: {
  content: string
  reasoning: string
  metrics?: TurnMetrics
  waiting?: number
  error?: string
  streaming: boolean
}): JSX.Element {
  /* Closed until the reader opens it, streaming or not: the thinking is where the answer
     comes from and not the answer. */
  const [open, setOpen] = useState(false)
  return (
    <div className={open ? 'msg model open' : 'msg model'}>
      {reasoning === '' ? null : (
        <>
          <button className="reason" onClick={() => setOpen(!open)}>
            {CHEVRON} Reasoning
          </button>
          <div className="reasonbody">{reasoning}</div>
        </>
      )}
      {content === '' && streaming ? (
        <p>
          <span className="caret" />
        </p>
      ) : (
        <Prose text={content} caret={streaming} />
      )}
      {error === undefined ? null : <div className="msgerr">{error}</div>}
      {metrics === undefined && waiting === undefined ? null : (
        <Metrics metrics={metrics} waiting={waiting} />
      )}
    </div>
  )
}

export function Thread({
  messages,
  live,
  liveMetrics,
  waiting
}: {
  messages: StoredMessage[]
  live: Live | null
  /* The register's reading of the turn being written, and — before there is one — how long
     it has been waiting for the model and the queue. Both belong to `live` and to nothing
     already in `messages`. */
  liveMetrics: TurnMetrics | null
  waiting: number | null
}): JSX.Element {
  const box = useRef<HTMLDivElement>(null)
  /* Follows the stream while the reader is at the bottom, and stops the moment they
     scroll away from it. */
  const pinned = useRef(true)

  useEffect(() => {
    const element = box.current
    if (element !== null && pinned.current) element.scrollTop = element.scrollHeight
  }, [messages, live])

  const empty = messages.length === 0 && live === null

  return (
    <div
      className="thread"
      ref={box}
      onScroll={(event) => {
        const { scrollHeight, scrollTop, clientHeight } = event.currentTarget
        pinned.current = scrollHeight - scrollTop - clientHeight < 60
      }}
    >
      <div className="thread-inner">
        {empty ? <div className="threadempty">Nothing said yet.</div> : null}
        {messages.map((message, index) =>
          /* A system turn is somebody else's prompt — this window sends its own from the
             parameters pane, and never stores one. Nothing to draw for it. */
          message.role === 'system' ? null : message.role === 'user' ? (
            <div className="msg user" key={index}>
              <div className="bubble">{message.content}</div>
            </div>
          ) : (
            <Answer
              key={index}
              content={message.content}
              reasoning={message.reasoning_content ?? ''}
              metrics={message.metrics}
              error={message.error}
              streaming={false}
            />
          )
        )}
        {live === null ? null : (
          <Answer
            content={live.content}
            reasoning={live.reasoning}
            metrics={liveMetrics ?? undefined}
            waiting={waiting ?? undefined}
            error={live.error ?? undefined}
            streaming
          />
        )}
      </div>
    </div>
  )
}

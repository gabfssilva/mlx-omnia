import { type JSX, useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
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
      <Markdown remarkPlugins={[remarkGfm]}>{text}</Markdown>
      {caret ? <span className="caret" /> : null}
    </div>
  )
}

function Metrics({ metrics }: { metrics: TurnMetrics }): JSX.Element {
  const { tokens_per_second: rate, ceiling_fraction: fraction } = metrics
  return (
    <div className="metrics">
      <span>ttft {Math.round(metrics.ttft_ms)} ms</span>
      {rate === null ? null : (
        <>
          ·<span>decode {rate.toFixed(1)} tok/s</span>
        </>
      )}
      {fraction === null ? null : (
        <>
          ·<span className="pct">{Math.round(fraction * 100)}%</span>
          <span>of ceiling</span>
        </>
      )}
      {metrics.finish === 'length' ? (
        <>
          ·<span>cut at the token budget</span>
        </>
      ) : null}
    </div>
  )
}

function Answer({
  content,
  reasoning,
  metrics,
  error,
  streaming
}: {
  content: string
  reasoning: string
  metrics?: TurnMetrics
  error?: string
  streaming: boolean
}): JSX.Element {
  const [touched, setTouched] = useState<boolean | null>(null)
  /* Open while it is all there is to see, closed once the answer starts — and whatever
     the reader last chose from then on. */
  const open = touched ?? (streaming && reasoning !== '' && content === '')
  return (
    <div className={open ? 'msg model open' : 'msg model'}>
      {reasoning === '' ? null : (
        <>
          <button className="reason" onClick={() => setTouched(!open)}>
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
      {metrics === undefined ? null : <Metrics metrics={metrics} />}
    </div>
  )
}

export function Thread({
  messages,
  live
}: {
  messages: StoredMessage[]
  live: Live | null
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
            error={live.error ?? undefined}
            streaming
          />
        )}
      </div>
    </div>
  )
}

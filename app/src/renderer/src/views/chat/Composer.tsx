import { type JSX, useEffect, useRef } from 'react'
import { shortModel } from './format'

const ARROW = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="19" x2="12" y2="5" />
    <polyline points="5 12 12 5 19 12" />
  </svg>
)

const SQUARE = (
  <svg viewBox="0 0 24 24" fill="currentColor" stroke="none">
    <rect x="7" y="7" width="10" height="10" rx="1.5" />
  </svg>
)

export function Composer({
  draft,
  models,
  model,
  streaming,
  notice,
  onDraft,
  onModel,
  onSend,
  onStop
}: {
  draft: string
  models: string[]
  model: string
  streaming: boolean
  notice: string | null
  onDraft: (text: string) => void
  onModel: (model: string) => void
  onSend: () => void
  onStop: () => void
}): JSX.Element {
  const field = useRef<HTMLTextAreaElement>(null)

  /* Grows with the text up to the max-height the css sets, after which it scrolls. */
  useEffect(() => {
    const element = field.current
    if (element === null) return
    element.style.height = 'auto'
    /* Views stay mounted while hidden, and a hidden element measures 0 — writing that back
       would leave the field collapsed until the first keystroke. */
    if (element.scrollHeight > 0) element.style.height = `${element.scrollHeight}px`
  }, [draft])

  /* A model the catalog does not list — the one this chat was started on, gone since —
     is still what the chat says it uses, so it stays selectable. */
  const options = models.includes(model) || model === '' ? models : [model, ...models]

  return (
    <div className="chatfoot">
      <div className="composer">
        {notice === null ? null : <div className="notice">{notice}</div>}
        <textarea
          ref={field}
          className="draft"
          rows={1}
          placeholder="Ask anything…"
          value={draft}
          aria-label="Message"
          onChange={(event) => onDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              if (!streaming) onSend()
            }
          }}
        />
        <footer>
          <select
            className="modelpick"
            value={model}
            aria-label="Model"
            onChange={(event) => onModel(event.target.value)}
          >
            {options.length === 0 ? <option value="">no models</option> : null}
            {options.map((id) => (
              <option key={id} value={id}>
                {shortModel(id)}
              </option>
            ))}
          </select>
          <span className="hintk">{streaming ? 'generating…' : '⏎ send'}</span>
          <button
            className="send"
            aria-label={streaming ? 'Stop' : 'Send'}
            disabled={!streaming && (draft.trim() === '' || model === '')}
            onClick={streaming ? onStop : onSend}
          >
            {streaming ? SQUARE : ARROW}
          </button>
        </footer>
      </div>
    </div>
  )
}

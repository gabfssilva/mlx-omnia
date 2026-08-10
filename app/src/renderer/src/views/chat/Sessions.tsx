import { type JSX, useRef, useState } from 'react'
import type { Params, SessionSummary } from './api'
import { ago, shortModel } from './format'
import { ParamsPane } from './Params'

const PLUS = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
)

const PENCIL = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
  </svg>
)

const TRASH = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
)

function subtitle(entry: SessionSummary): string {
  const parts = [entry.model ? shortModel(entry.model) : 'no model']
  if (entry.message_count > 0) parts.push(`${entry.message_count} msg`)
  parts.push(ago(entry.updated_at))
  return parts.join(' · ')
}

function Rename({
  initial,
  onCommit,
  onCancel
}: {
  initial: string
  onCommit: (title: string) => void
  onCancel: () => void
}): JSX.Element {
  const [text, setText] = useState(initial)
  /* Escape closes the field, which blurs it; without this the blur would commit the
     edit the escape just abandoned. */
  const settled = useRef(false)
  const commit = (): void => {
    if (settled.current) return
    settled.current = true
    const title = text.trim()
    if (title && title !== initial) onCommit(title)
    else onCancel()
  }
  return (
    <input
      className="sessname"
      autoFocus
      value={text}
      aria-label="Chat title"
      onChange={(event) => setText(event.target.value)}
      onClick={(event) => event.stopPropagation()}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') {
          event.preventDefault()
          commit()
        }
        if (event.key === 'Escape') {
          settled.current = true
          onCancel()
        }
      }}
    />
  )
}

export function Sessions({
  summaries,
  current,
  params,
  contextLimit,
  onSelect,
  onNew,
  onRename,
  onDelete,
  onParams
}: {
  summaries: SessionSummary[]
  current: string | null
  params: Params
  contextLimit: number | null
  onSelect: (id: string) => void
  onNew: () => void
  onRename: (id: string, title: string) => void
  onDelete: (id: string) => void
  onParams: (params: Params) => void
}): JSX.Element {
  const [tab, setTab] = useState<'chats' | 'params'>('chats')
  const [renaming, setRenaming] = useState<string | null>(null)

  return (
    <div className="sessions">
      <div className="tabbar">
        <button className={tab === 'chats' ? 'on' : undefined} onClick={() => setTab('chats')}>
          Chats
        </button>
        <button className={tab === 'params' ? 'on' : undefined} onClick={() => setTab('params')}>
          Parameters
        </button>
      </div>

      <div className={tab === 'chats' ? 'sesspane on' : 'sesspane'}>
        <button className="newchat" onClick={onNew}>
          {PLUS}
          New chat
        </button>
        <div className="sesslist">
          {summaries.map((entry) => (
            <div
              key={entry.id}
              className={entry.id === current ? 'sess on' : 'sess'}
              role="button"
              tabIndex={0}
              onClick={() => onSelect(entry.id)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') onSelect(entry.id)
              }}
            >
              {renaming === entry.id ? (
                <Rename
                  initial={entry.title}
                  onCommit={(title) => {
                    setRenaming(null)
                    onRename(entry.id, title)
                  }}
                  onCancel={() => setRenaming(null)}
                />
              ) : (
                <b>{entry.title}</b>
              )}
              <span>{subtitle(entry)}</span>
              {renaming === entry.id ? null : (
                <div className="sessact">
                  <button
                    className="iconbtn"
                    title="Rename"
                    aria-label="Rename chat"
                    onClick={(event) => {
                      event.stopPropagation()
                      setRenaming(entry.id)
                    }}
                  >
                    {PENCIL}
                  </button>
                  <button
                    className="iconbtn"
                    title="Delete"
                    aria-label="Delete chat"
                    onClick={(event) => {
                      event.stopPropagation()
                      onDelete(entry.id)
                    }}
                  >
                    {TRASH}
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className={tab === 'params' ? 'sesspane on' : 'sesspane'}>
        <ParamsPane params={params} contextLimit={contextLimit} onChange={onParams} />
      </div>
    </div>
  )
}

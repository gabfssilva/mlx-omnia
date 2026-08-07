import { type JSX, useEffect, useRef, useState } from 'react'
import './chat.css'
import {
  type CatalogEntry,
  DEFAULT_PARAMS,
  type Params,
  type Session,
  type SessionSummary,
  type StoredMessage,
  type Turn,
  completion,
  createSession,
  deleteSession,
  getCatalogEntry,
  getSession,
  listCatalog,
  listModels,
  listSessions,
  patchSession,
  putMessages
} from './api'
import { Composer } from './Composer'
import { Sessions } from './Sessions'
import { Thread, type Live } from './Thread'
import { shortModel } from './format'

/* The sustained bandwidth every "% of the ceiling" in the house is measured against. */
const SUSTAINED = 490e9
const TITLE_MAX = 48

const reason = (error: unknown): string =>
  error instanceof Error ? error.message : String(error)

const summarize = (session: Session): SessionSummary => ({
  id: session.id,
  title: session.title,
  model: session.model,
  created_at: session.created_at,
  updated_at: session.updated_at,
  message_count: session.messages.length
})

/* The list is ordered by `updated_at` desc on the server; a write that bumps it has to
   move the row here too, or the order disagrees with the next reload. */
const merge = (all: SessionSummary[], session: Session): SessionSummary[] =>
  [summarize(session), ...all.filter((entry) => entry.id !== session.id)].sort(
    (a, b) => b.updated_at - a.updated_at
  )

function titleOf(text: string): string {
  const line = text.replace(/\s+/g, ' ').trim()
  return line.length > TITLE_MAX ? `${line.slice(0, TITLE_MAX).trimEnd()}…` : line
}

/* What goes to the model: the roles and the text, and the system prompt as the first turn
   when this chat carries one. The metrics and the thinking stay here. */
function turns(messages: StoredMessage[], params: Params): Turn[] {
  const history: Turn[] = messages.map(({ role, content }) => ({ role, content }))
  const prompt = params.system_prompt.trim()
  return prompt === '' ? history : [{ role: 'system', content: prompt }, ...history]
}

const contextLabel = (tokens: number): string =>
  tokens >= 1000 ? `${Math.round(tokens / 1000)}k` : String(tokens)

export function Chat(): JSX.Element {
  const [summaries, setSummaries] = useState<SessionSummary[]>([])
  const [current, setCurrent] = useState<string | null>(null)
  const [messages, setMessages] = useState<StoredMessage[]>([])
  /* Per chat, and only for as long as the window is open: the sessions table keeps the
     conversation, not how it was sampled. */
  const [knobs, setKnobs] = useState<Record<string, Params>>({})
  const [models, setModels] = useState<string[]>([])
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [model, setModel] = useState('')
  const [draft, setDraft] = useState('')
  const [live, setLive] = useState<Live | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [spent, setSpent] = useState<number | null>(null)
  const abort = useRef<AbortController | null>(null)
  /* Which session `messages` holds, so the fetch below does not undo a turn this window
     has already appended. */
  const loaded = useRef<string | null>(null)
  /* The same id the state holds, readable from a generation that is still running after
     the reader has moved to another chat: what it collected belongs to the chat it was
     asked in, and must not be drawn over the one now on screen. */
  const selected = useRef<string | null>(null)

  const open = (id: string | null): void => {
    selected.current = id
    setCurrent(id)
  }

  useEffect(() => {
    listSessions()
      .then((all) => {
        setSummaries(all)
        if (all.length > 0) open(all[0].id)
      })
      .catch((error: unknown) => setNotice(reason(error)))
    listModels().then(setModels).catch(() => setModels([]))
    listCatalog().then(setCatalog).catch(() => setCatalog([]))
  }, [])

  useEffect(() => {
    if (current === null || loaded.current === current) return
    let alive = true
    getSession(current)
      .then((session) => {
        if (!alive) return
        loaded.current = session.id
        setMessages(session.messages)
        setModel(session.model)
        setSpent(null)
      })
      .catch((error: unknown) => setNotice(reason(error)))
    return () => {
      alive = false
    }
  }, [current])

  /* A chat that names no model answers to whatever the daemon serves first. */
  useEffect(() => {
    if (model === '' && models.length > 0) setModel(models[0])
  }, [model, models])

  const params = knobs[current ?? ''] ?? DEFAULT_PARAMS
  const entry = catalog.find((candidate) => candidate.id === model) ?? null
  const streaming = live !== null
  const title = summaries.find((summary) => summary.id === current)?.title ?? 'New chat'
  /* What the last turn spent against what the checkpoint can hold — either half alone
     when the other is not known yet. */
  const tokens = [
    spent === null ? null : spent.toLocaleString('en-US'),
    entry?.context == null ? null : contextLabel(entry.context)
  ]
    .filter((part) => part !== null)
    .join(' / ')

  const setParams = (next: Params): void => {
    if (current !== null) setKnobs((all) => ({ ...all, [current]: next }))
  }

  const stop = (): void => abort.current?.abort()

  async function run(id: string, history: StoredMessage[]): Promise<void> {
    const controller = new AbortController()
    abort.current = controller
    const started = performance.now()
    let first: number | null = null
    let content = ''
    let reasoning = ''
    let failure: string | null = null
    let completed: number | null = null
    let finish: string | null = null
    let flushed = 0
    /* The entry read again, asked for on the first token and awaited after the last: by
       then the model is resident, and a resident entry is priced by the tree that is
       generating instead of by an estimate off the headers — a checkpoint carries blocks
       the loader drops. Asking here is what keeps the round trip off the end of the turn. */
    let priced: Promise<CatalogEntry | null> | null = null
    const paint = (force: boolean): void => {
      const at = performance.now()
      /* A token a frame is a render a frame; 30 a second is what the eye reads. */
      if (!force && at - flushed < 33) return
      flushed = at
      if (selected.current === id) setLive({ content, reasoning, error: failure })
    }
    paint(true)
    try {
      for await (const frame of completion(model, turns(history, params), params, controller.signal)) {
        switch (frame.kind) {
          case 'reasoning':
            first ??= performance.now()
            priced ??= getCatalogEntry(model).catch(() => null)
            reasoning += frame.text
            break
          case 'content':
            first ??= performance.now()
            priced ??= getCatalogEntry(model).catch(() => null)
            content += frame.text
            break
          case 'usage':
            completed = frame.completion_tokens
            if (selected.current === id) setSpent(frame.total_tokens)
            break
          case 'error':
            failure = frame.message
            break
          case 'finish':
            finish = frame.reason
            break
        }
        paint(false)
      }
    } catch (error) {
      if (!controller.signal.aborted) failure = reason(error)
    } finally {
      abort.current = null
    }

    const ended = performance.now()
    const rate =
      completed !== null && first !== null && ended > first
        ? completed / ((ended - first) / 1000)
        : null
    const perToken = (await priced)?.bytes_per_token ?? entry?.bytes_per_token ?? null
    const fraction = rate !== null && perToken ? rate / (SUSTAINED / perToken) : null
    const answer: StoredMessage = {
      role: 'assistant',
      content,
      metrics: {
        ttft_ms: (first ?? ended) - started,
        tokens_per_second: rate,
        ceiling_fraction: fraction,
        finish
      }
    }
    if (reasoning !== '') answer.reasoning_content = reasoning
    if (failure !== null) answer.error = failure
    /* A turn that was stopped before it said anything leaves nothing behind; everything
       else is persisted, partial or refused. */
    const whole =
      content === '' && reasoning === '' && failure === null ? history : [...history, answer]
    if (selected.current === id) setMessages(whole)
    setLive(null)
    try {
      const saved = await putMessages(id, whole)
      setSummaries((all) => merge(all, saved))
    } catch (error) {
      setNotice(reason(error))
    }
  }

  async function begin(): Promise<void> {
    const text = draft.trim()
    if (text === '' || streaming || model === '') return
    setNotice(null)
    setDraft('')
    try {
      let id = current
      if (id === null) {
        const created = await createSession({ title: titleOf(text), model })
        id = created.id
        loaded.current = id
        setSummaries((all) => merge(all, created))
        open(id)
      } else if (messages.length === 0) {
        /* The title is the first thing said, truncated — the row was created before
           anyone had typed. */
        const named = await patchSession(id, { title: titleOf(text), model })
        setSummaries((all) => merge(all, named))
      }
      const history: StoredMessage[] = [...messages, { role: 'user', content: text }]
      setMessages(history)
      await run(id, history)
    } catch (error) {
      /* Nothing was sent, so the text goes back where it was typed. */
      setDraft(text)
      setNotice(reason(error))
    }
  }

  async function newChat(): Promise<void> {
    stop()
    try {
      const created = await createSession({ model })
      loaded.current = created.id
      setSummaries((all) => merge(all, created))
      setMessages([])
      setSpent(null)
      open(created.id)
    } catch (error) {
      setNotice(reason(error))
    }
  }

  function select(id: string): void {
    if (id === current) return
    stop()
    setLive(null)
    open(id)
  }

  async function rename(id: string, next: string): Promise<void> {
    try {
      const named = await patchSession(id, { title: next })
      setSummaries((all) => merge(all, named))
    } catch (error) {
      setNotice(reason(error))
    }
  }

  async function remove(id: string): Promise<void> {
    try {
      await deleteSession(id)
    } catch (error) {
      setNotice(reason(error))
      return
    }
    const left = summaries.filter((summary) => summary.id !== id)
    setSummaries(left)
    if (id !== current) return
    stop()
    setLive(null)
    loaded.current = null
    setMessages([])
    setSpent(null)
    open(left.length > 0 ? left[0].id : null)
  }

  async function pick(next: string): Promise<void> {
    setModel(next)
    if (current === null) return
    try {
      const moved = await patchSession(current, { model: next })
      setSummaries((all) => merge(all, moved))
    } catch (error) {
      setNotice(reason(error))
    }
  }

  return (
    <div className="chatwrap">
      <Sessions
        summaries={summaries}
        current={current}
        params={params}
        contextLimit={entry?.context ?? null}
        onSelect={select}
        onNew={() => void newChat()}
        onRename={(id, next) => void rename(id, next)}
        onDelete={(id) => void remove(id)}
        onParams={setParams}
      />

      <div className="chatcol">
        <div className="chathead">
          <h1>{title}</h1>
          <span className="modeltag">
            {[shortModel(model) || 'no model', entry?.quantization, `t ${params.temperature}`]
              .filter((part) => part)
              .join(' · ')}
          </span>
          <div className="trail">
            {tokens === '' ? null : <span className="eyebrow tn">{tokens} tokens</span>}
          </div>
        </div>

        <Thread key={current ?? 'none'} messages={messages} live={live} />

        <Composer
          draft={draft}
          models={models}
          model={model}
          streaming={streaming}
          notice={notice}
          onDraft={setDraft}
          onModel={(next) => void pick(next)}
          onSend={() => void begin()}
          onStop={stop}
        />
      </div>
    </div>
  )
}

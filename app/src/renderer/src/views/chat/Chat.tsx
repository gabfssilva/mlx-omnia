import { type JSX, useEffect, useRef, useState } from 'react'
import './chat.css'
import {
  type CatalogEntry,
  type Params,
  type Session,
  type SessionSummary,
  type StoredMessage,
  type Timings,
  type Turn,
  type TurnMetrics,
  completion,
  createSession,
  deleteSession,
  getSession,
  listCatalog,
  listModels,
  listSessions,
  metricsOf,
  metricsOfSample,
  paramsOf,
  patchSession,
  putMessages
} from './api'
import { type Sample, metricEvents } from '../overview/api'
import { Composer } from './Composer'
import { Sessions } from './Sessions'
import { Thread, type Live } from './Thread'
import { shortModel } from './format'

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
  /* The daemon's register, as it stands for the turn being written. */
  const [pulse, setPulse] = useState<Sample | null>(null)
  /* How long this turn has been waiting for the register to know about it: the model being
     read off disk, and the queue in front of it. Both are before the meter exists. */
  const [waiting, setWaiting] = useState<number | null>(null)
  const abort = useRef<AbortController | null>(null)
  /* The same live entry, readable from inside a generation: what the turn is written with
     when the stream ends without the frame that carries its totals. */
  const pulsed = useRef<Sample | null>(null)
  const startedAt = useRef(0)
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

  const entry = catalog.find((candidate) => candidate.id === model) ?? null
  /* Untouched, a chat is sampled the way the checkpoint says it wants to be — the same
     values the daemon would fill in for a request that named none. `pick` drops the entry
     when the model changes, so the sliders follow the model rather than the chat. */
  const params = knobs[current ?? ''] ?? paramsOf(entry)
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

  /* Where the numbers under a turn come from while it is still being written. The register
     is the daemon's and not this window's, so only a request on this chat's model is drawn:
     another client's generation is not this one. It publishes at both ends of a request and
     resamples twice a second in between, which is what makes the rate move as it is read. */
  useEffect(() => {
    if (!streaming) return
    const controller = new AbortController()
    const follow = async (): Promise<void> => {
      for await (const snapshot of metricEvents(controller.signal)) {
        const running = snapshot.live
        const mine = running !== null && running.model === model ? running : null
        pulsed.current = mine
        setPulse(mine)
      }
    }
    /* A daemon that cannot be watched costs the live numbers and nothing else — the turn
       itself rides the completion stream, which has its own failure. */
    follow().catch(() => undefined)
    return () => {
      controller.abort()
      setPulse(null)
    }
  }, [streaming, model])

  /* The clock the reader is left with before the register knows anything: a cold 30B is tens
     of seconds with nothing to draw, and the load only becomes a number once the request has
     the model and reaches the gate. */
  useEffect(() => {
    if (!streaming || pulse !== null) {
      setWaiting(null)
      return
    }
    const tick = (): void => setWaiting(performance.now() - startedAt.current)
    tick()
    const timer = setInterval(tick, 250)
    return () => clearInterval(timer)
  }, [streaming, pulse])

  const setParams = (next: Params): void => {
    if (current !== null) setKnobs((all) => ({ ...all, [current]: next }))
  }

  const stop = (): void => abort.current?.abort()

  async function run(id: string, history: StoredMessage[]): Promise<void> {
    const controller = new AbortController()
    abort.current = controller
    startedAt.current = performance.now()
    pulsed.current = null
    let content = ''
    let reasoning = ''
    let failure: string | null = null
    let finish: string | null = null
    let timings: Timings | null = null
    let flushed = 0
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
            reasoning += frame.text
            break
          case 'content':
            content += frame.text
            break
          case 'usage':
            if (selected.current === id) setSpent(frame.total_tokens)
            break
          case 'timings':
            timings = frame.timings
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

    /* The turn's numbers are the daemon's, and the two ways it says them: the frame that
       closes a stream that reached its end, and the last live entry the register published
       when it did not — a turn the reader stopped is answered by no frame at all. */
    const metrics: TurnMetrics | null =
      timings !== null
        ? metricsOf(timings, finish)
        : pulsed.current === null
          ? null
          : metricsOfSample(pulsed.current)
    const answer: StoredMessage = { role: 'assistant', content }
    if (metrics !== null) answer.metrics = metrics
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
    /* The sampling knobs follow the model: another checkpoint is another set of defaults,
       and 0.6 carried over from the one before is a value nobody chose for this one. What
       does not follow are the three the checkpoint has no opinion on — the budget, the
       effort and the system prompt — which are the reader's and stay. */
    const chosen = catalog.find((candidate) => candidate.id === next) ?? null
    setKnobs((all) => ({ ...all, [current]: paramsOf(chosen, params) }))
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

        <Thread
          key={current ?? 'none'}
          messages={messages}
          live={live}
          liveMetrics={pulse === null ? null : metricsOfSample(pulse)}
          waiting={waiting}
        />

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

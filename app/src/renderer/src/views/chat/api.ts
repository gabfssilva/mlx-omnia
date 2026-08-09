/* Everything this view asks of the daemon: the session store under /admin/sessions and
   the OpenAI dialect under /api/openai/v1. The transport itself is api/http.ts. */

import { json, send, sse } from '../../api/http'
import type { Sample } from '../overview/api'

/* ── sessions ─────────────────────────────────────────────────────────── */

export type Role = 'system' | 'user' | 'assistant'

/* What a turn cost, kept next to the turn it belongs to. The server stores the array
   opaquely and its own ChatMessage drops fields it does not declare, so these ride along
   and are dropped on the way back into a completion.

   Every number is the daemon's own, off `x_sideros` on the usage frame or off the live
   register — never timed from out here, where the load, the queue and the prefill are one
   wait with no way to tell them apart. */
export interface TurnMetrics {
  /* The seconds this turn spent putting its model in memory, absent when it found it
     resident — which is every turn after the first on a model. */
  load_ms: number | null
  /* Prefill plus the step that draws the first token, with the weights already there. */
  ttft_ms: number | null
  prefill_tokens_per_second: number | null
  tokens_per_second: number | null
  ceiling_fraction: number | null
  /* Which of the three ended the turn, as the dialect spells it. `length` is the one a
     reader has to be told about: the answer stops mid-sentence and nothing is wrong. */
  finish: string | null
}

export interface StoredMessage {
  role: Role
  content: string
  reasoning_content?: string
  metrics?: TurnMetrics
  error?: string
}

export interface SessionSummary {
  id: string
  title: string
  model: string
  created_at: number
  updated_at: number
  message_count: number
}

export interface Session {
  id: string
  title: string
  model: string
  created_at: number
  updated_at: number
  messages: StoredMessage[]
}

interface RawSession {
  id: string
  title: string
  model: string
  created_at: number
  updated_at: number
  /* The column holds whatever the client wrote; it is read, not cast. */
  messages: unknown
}

const record = (value: unknown): Record<string, unknown> | null =>
  typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null

const ROLES: readonly string[] = ['system', 'user', 'assistant']

const numeric = (value: unknown): number | null => (typeof value === 'number' ? value : null)

function readMetrics(value: unknown): TurnMetrics | undefined {
  const row = record(value)
  if (row === null) return undefined
  return {
    load_ms: numeric(row.load_ms),
    ttft_ms: numeric(row.ttft_ms),
    prefill_tokens_per_second: numeric(row.prefill_tokens_per_second),
    tokens_per_second: numeric(row.tokens_per_second),
    ceiling_fraction: numeric(row.ceiling_fraction),
    finish: typeof row.finish === 'string' ? row.finish : null
  }
}

/* A turn this view cannot draw — a tool result, a message somebody hand-edited into the
   file — is dropped rather than rendered as a blank. The store is shared with whoever
   else talks to the daemon. */
function readMessage(value: unknown): StoredMessage | null {
  const row = record(value)
  if (row === null) return null
  const { role, content, reasoning_content: reasoning, error } = row
  if (typeof role !== 'string' || !ROLES.includes(role) || typeof content !== 'string') return null
  const message: StoredMessage = { role: role as Role, content }
  if (typeof reasoning === 'string' && reasoning) message.reasoning_content = reasoning
  if (typeof error === 'string' && error) message.error = error
  const metrics = readMetrics(row.metrics)
  if (metrics !== undefined) message.metrics = metrics
  return message
}

const readSession = (raw: RawSession): Session => ({
  id: raw.id,
  title: raw.title,
  model: raw.model,
  created_at: raw.created_at,
  updated_at: raw.updated_at,
  messages: Array.isArray(raw.messages)
    ? raw.messages.map(readMessage).filter((turn): turn is StoredMessage => turn !== null)
    : []
})

const at = (id: string): string => `/admin/sessions/${encodeURIComponent(id)}`

export const listSessions = (): Promise<SessionSummary[]> =>
  json<{ sessions: SessionSummary[] }>('/admin/sessions').then((body) => body.sessions)

export const createSession = (body: { title?: string; model?: string }): Promise<Session> =>
  send<RawSession>('POST', '/admin/sessions', body).then(readSession)

export const getSession = (id: string): Promise<Session> =>
  json<RawSession>(at(id)).then(readSession)

export const patchSession = (
  id: string,
  body: { title?: string; model?: string }
): Promise<Session> => send<RawSession>('PATCH', at(id), body).then(readSession)

export const putMessages = (id: string, messages: StoredMessage[]): Promise<Session> =>
  send<RawSession>('PUT', `${at(id)}/messages`, { messages }).then(readSession)

export const deleteSession = (id: string): Promise<void> => send<void>('DELETE', at(id))

/* ── the catalog, for what the header and the metrics line report ─────── */

/* What the checkpoint's own `generation_config.json` declares. A knob it does not declare
   is null, which is not the same as one it declares neutral. */
export interface SamplingDefaults {
  temperature: number | null
  top_p: number | null
  top_k: number | null
  min_p: number | null
  repetition_penalty: number | null
}

export interface CatalogEntry {
  id: string
  quantization: string | null
  context: number | null
  defaults: SamplingDefaults
  bytes_per_token: number | null
}

export const listModels = (): Promise<string[]> =>
  json<{ data: { id: string }[] }>('/api/openai/v1/models').then((body) =>
    body.data.map((entry) => entry.id)
  )

export const listCatalog = (): Promise<CatalogEntry[]> => json<CatalogEntry[]>('/admin/models')

/* ── completions ──────────────────────────────────────────────────────── */

/* `chat/completions`' own vocabulary, which is upstream's. There is no `reasoning_budget`
   beside it because the dialect has no field for one — how long the model may think is set
   on a profile, under Models. */
export type Effort = 'default' | 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'

export const EFFORTS: readonly Effort[] = [
  'default',
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max'
]

export interface Params {
  temperature: number
  top_p: number
  /* null is the dialect's "no cut": the field is left out of the body. */
  top_k: number | null
  min_p: number
  repetition_penalty: number
  max_tokens: number
  /* `default` leaves the field out, which is the checkpoint's own template deciding. */
  reasoning_effort: Effort
  system_prompt: string
}

/* The dialect's own defaults, which is what the daemon samples with when nothing else
   says otherwise — not any one checkpoint's. What a checkpoint does declare comes off its
   catalog entry and lands on top, in `paramsOf`. */
export const DEFAULT_PARAMS: Params = {
  temperature: 1,
  top_p: 1,
  top_k: null,
  min_p: 0,
  repetition_penalty: 1,
  max_tokens: 4096,
  reasoning_effort: 'default',
  system_prompt: ''
}

/* The knobs a chat opens on: the checkpoint's declared defaults over the dialect's, which
   is the same order the daemon resolves them in — so the sliders read what the request
   would have been sampled with had nobody touched them.

   `max_tokens`, the effort and the system prompt are not in that file and stay as they
   were: they are the reader's, not the checkpoint's. */
export const paramsOf = (entry: CatalogEntry | null, from = DEFAULT_PARAMS): Params => {
  const declared = entry?.defaults
  if (declared === undefined) return from
  return {
    ...from,
    temperature: declared.temperature ?? DEFAULT_PARAMS.temperature,
    top_p: declared.top_p ?? DEFAULT_PARAMS.top_p,
    top_k: declared.top_k,
    min_p: declared.min_p ?? DEFAULT_PARAMS.min_p,
    repetition_penalty: declared.repetition_penalty ?? DEFAULT_PARAMS.repetition_penalty
  }
}

export interface Turn {
  role: Role
  content: string
}

export type Frame =
  | { kind: 'content'; text: string }
  | { kind: 'reasoning'; text: string }
  | { kind: 'usage'; prompt_tokens: number; completion_tokens: number; total_tokens: number }
  | { kind: 'timings'; timings: Timings }
  | { kind: 'finish'; reason: string }
  | { kind: 'error'; message: string }

/* app.py `_timings`: what the turn cost in seconds, on the usage frame under a key the
   dialect does not define. Absent from any other server that speaks this route, which is
   what the `| undefined` on the chunk is for. */
export interface Timings {
  load_seconds: number | null
  ttft_seconds: number | null
  prefill_tokens_per_second: number | null
  tokens_per_second: number | null
  bytes_per_token: number | null
  ceiling_fraction: number | null
}

interface Delta {
  content?: string | null
  reasoning_content?: string | null
}

interface Choice {
  index: number
  delta: Delta
  finish_reason: string | null
}

interface Usage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

interface Chunk {
  choices?: Choice[]
  usage?: Usage | null
  x_sideros?: Timings
  /* The one frame that fails a request already answered 200. */
  error?: { message: string; type: string; code: string | null }
}

function body(model: string, messages: Turn[], params: Params): Record<string, unknown> {
  const asked: Record<string, unknown> = {
    model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: params.max_tokens,
    temperature: params.temperature,
    top_p: params.top_p
  }
  if (params.top_k !== null) asked.top_k = params.top_k
  if (params.min_p > 0) asked.min_p = params.min_p
  if (params.repetition_penalty > 1) asked.repetition_penalty = params.repetition_penalty
  if (params.reasoning_effort !== 'default') asked.reasoning_effort = params.reasoning_effort
  return asked
}

const ms = (seconds: number | null): number | null => (seconds === null ? null : seconds * 1000)

/* The turn's numbers as they are kept, from the frame that closes it. */
export const metricsOf = (timings: Timings, finish: string | null): TurnMetrics => ({
  load_ms: ms(timings.load_seconds),
  ttft_ms: ms(timings.ttft_seconds),
  prefill_tokens_per_second: timings.prefill_tokens_per_second,
  tokens_per_second: timings.tokens_per_second,
  ceiling_fraction: timings.ceiling_fraction,
  finish
})

/* The same numbers while the turn is still being written, off the register's live entry.
   `finish` is nobody's yet: the turn has not ended. */
export const metricsOfSample = (sample: Sample): TurnMetrics => ({
  load_ms: ms(sample.load_seconds),
  ttft_ms: ms(sample.ttft),
  prefill_tokens_per_second: sample.prefill_tokens_per_second,
  tokens_per_second: sample.tokens_per_second,
  ceiling_fraction: sample.ceiling_fraction,
  finish: null
})

/* The frames of one turn, in order. `[DONE]` ends it; an `error` frame is yielded and
   then ends it, because that is the whole failure — the stream will not say more. */
export async function* completion(
  model: string,
  messages: Turn[],
  params: Params,
  signal: AbortSignal
): AsyncGenerator<Frame> {
  const init: RequestInit = {
    method: 'POST',
    body: JSON.stringify(body(model, messages, params)),
    signal
  }
  for await (const data of sse('/api/openai/v1/chat/completions', init)) {
    if (data === '[DONE]') return
    const frame = JSON.parse(data) as Chunk
    if (frame.error) {
      yield { kind: 'error', message: frame.error.message }
      return
    }
    const choice = frame.choices?.[0]
    if (choice) {
      const { content, reasoning_content: reasoning } = choice.delta
      if (reasoning) yield { kind: 'reasoning', text: reasoning }
      if (content) yield { kind: 'content', text: content }
      if (choice.finish_reason) yield { kind: 'finish', reason: choice.finish_reason }
    }
    if (frame.usage) yield { kind: 'usage', ...frame.usage }
    if (frame.x_sideros) yield { kind: 'timings', timings: frame.x_sideros }
  }
}

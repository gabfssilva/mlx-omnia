/* Everything this view asks of the daemon: the session store under /admin/sessions and
   the OpenAI dialect under /api/openai/v1. The transport itself is api/http.ts. */

import { json, send, sse } from '../../api/http'

/* ── sessions ─────────────────────────────────────────────────────────── */

export type Role = 'system' | 'user' | 'assistant'

/* What a turn cost, kept next to the turn it belongs to. The server stores the array
   opaquely and its own ChatMessage drops fields it does not declare, so these ride along
   and are dropped on the way back into a completion. */
export interface TurnMetrics {
  ttft_ms: number
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

function readMetrics(value: unknown): TurnMetrics | undefined {
  const row = record(value)
  if (row === null || typeof row.ttft_ms !== 'number') return undefined
  return {
    ttft_ms: row.ttft_ms,
    tokens_per_second: typeof row.tokens_per_second === 'number' ? row.tokens_per_second : null,
    ceiling_fraction: typeof row.ceiling_fraction === 'number' ? row.ceiling_fraction : null,
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

export interface CatalogEntry {
  id: string
  quantization: string | null
  context: number | null
  bytes_per_token: number | null
}

export const listModels = (): Promise<string[]> =>
  json<{ data: { id: string }[] }>('/api/openai/v1/models').then((body) =>
    body.data.map((entry) => entry.id)
  )

export const listCatalog = (): Promise<CatalogEntry[]> => json<CatalogEntry[]>('/admin/models')

/* One entry, read again when a turn ends: by then the model is resident, and a resident
   entry is priced by the tree that answered instead of by an estimate off the headers. */
export const getCatalogEntry = (id: string): Promise<CatalogEntry> =>
  json<CatalogEntry>(`/admin/models/${encodeURIComponent(id)}`)

/* ── completions ──────────────────────────────────────────────────────── */

export interface Params {
  temperature: number
  top_p: number
  /* null is the dialect's "no cut": the field is left out of the body. */
  top_k: number | null
  min_p: number
  repetition_penalty: number
  max_tokens: number
  system_prompt: string
}

export const DEFAULT_PARAMS: Params = {
  temperature: 0.6,
  top_p: 0.95,
  top_k: 20,
  min_p: 0,
  repetition_penalty: 1,
  max_tokens: 4096,
  system_prompt: ''
}

export interface Turn {
  role: Role
  content: string
}

export type Frame =
  | { kind: 'content'; text: string }
  | { kind: 'reasoning'; text: string }
  | { kind: 'usage'; prompt_tokens: number; completion_tokens: number; total_tokens: number }
  | { kind: 'finish'; reason: string }
  | { kind: 'error'; message: string }

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
  return asked
}

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
  }
}

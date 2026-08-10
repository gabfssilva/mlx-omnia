/* The transport, and nothing about any screen. Every URL is relative: in dev Vite
   proxies /admin and /api to the daemon, in prod the desktop shell does. Domain calls
   belong next to the view that makes them. */

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string
  ) {
    super(`${status}: ${detail}`)
    this.name = 'HttpError'
  }
}

/* The house refusal is FastAPI's `{"detail": ...}`; the dialect routes answer
   `{"error": {"message": ...}}`. Both end up in `HttpError.detail`. */
async function refuse(answer: Response): Promise<never> {
  let detail = answer.statusText
  try {
    const body: unknown = await answer.json()
    if (body && typeof body === 'object') {
      const shape = body as { detail?: unknown; error?: { message?: unknown } }
      if (typeof shape.detail === 'string') detail = shape.detail
      else if (typeof shape.error?.message === 'string') detail = shape.error.message
      else detail = JSON.stringify(body)
    }
  } catch {
    /* a body that is not JSON leaves the status text standing */
  }
  throw new HttpError(answer.status, detail)
}

export async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const answer = await fetch(path, {
    ...init,
    headers:
      init?.body === undefined
        ? init?.headers
        : { 'content-type': 'application/json', ...init?.headers }
  })
  if (!answer.ok) await refuse(answer)
  if (answer.status === 204) return undefined as T
  return (await answer.json()) as T
}

export function send<T>(method: string, path: string, body?: unknown): Promise<T> {
  return json<T>(path, {
    method,
    ...(body === undefined ? {} : { body: JSON.stringify(body) })
  })
}

/* Every stream this daemon serves is `text/event-stream` carrying whole JSON states
   on `data:` lines, plus `: keep-alive` comments that mean nothing but the connection
   is warm. `fetch` rather than `EventSource` because half of them are POSTs and all of
   them need an abort that reaches the socket. */
export async function* sse(path: string, init?: RequestInit): AsyncGenerator<string> {
  const answer = await fetch(path, {
    ...init,
    headers:
      init?.body === undefined
        ? init?.headers
        : { 'content-type': 'application/json', ...init?.headers }
  })
  if (!answer.ok) await refuse(answer)
  if (answer.body === null) return
  const lines = answer.body.pipeThrough(new TextDecoderStream()).getReader()
  let held = ''
  try {
    for (;;) {
      const { value, done } = await lines.read()
      if (done) break
      held += value
      /* A frame ends at the blank line; a chunk may carry half of one. */
      let cut = held.indexOf('\n\n')
      while (cut !== -1) {
        const frame = held.slice(0, cut)
        held = held.slice(cut + 2)
        const data = frame
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n')
        if (data) yield data
        cut = held.indexOf('\n\n')
      }
    }
  } finally {
    await lines.cancel().catch(() => {})
  }
}

export async function* events<T>(path: string, init?: RequestInit): AsyncGenerator<T> {
  for await (const data of sse(path, init)) yield JSON.parse(data) as T
}

/* ── jobs ─────────────────────────────────────────────────────────────── */

export type JobState = 'pending' | 'running' | 'ok' | 'error' | 'cancelled'

export interface Progress {
  message: string
  completed: number
  /* Absent while unknown: a download learns its size from the first response. */
  total: number | null
}

interface JobBase {
  id: string
  state: JobState
  progress: Progress
  created_at: number
  updated_at: number
  error: string | null
}

/* `kind` is the tag `subject` is read back with, on this side too: a screen that draws a job
   beside the thing it acts on gets there by narrowing, not by parsing `progress.message`. */
export type JobView =
  | (JobBase & { kind: 'download'; subject: { model: string } })
  | (JobBase & { kind: 'load'; subject: { model: string } })
  | (JobBase & { kind: 'quantize'; subject: { model: string; target: string } })
  | (JobBase & { kind: 'bench'; subject: { model: string } })
  | (JobBase & { kind: 'benchmark'; subject: { kind: string; models: string[] } })

export const isFinished = (job: JobView): boolean =>
  job.state === 'ok' || job.state === 'error' || job.state === 'cancelled'

export const getJob = (id: string): Promise<JobView> => json<JobView>(`/admin/jobs/${id}`)

export const listJobs = (active = false): Promise<JobView[]> =>
  json<JobView[]>(`/admin/jobs${active ? '?active=true' : ''}`)

export const cancelJob = (id: string): Promise<JobView> => send<JobView>('DELETE', `/admin/jobs/${id}`)

/* Every frame is the whole state and the stream opens with the state as it is now, so
   a late subscriber loses nothing and a repeated frame costs nothing. It ends when the
   job reaches a terminal state. */
export const jobEvents = (id: string, signal?: AbortSignal): AsyncGenerator<JobView> =>
  events<JobView>(`/admin/jobs/${id}/events`, { signal })

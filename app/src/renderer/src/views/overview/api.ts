/* The routes this view reads, typed off the server's own models:
   app.py `/admin/health`, system.py, state.py, config.py, metrics.py, catalog.py.
   Nothing here is guessed — where a field is optional on the server it is `| null`
   here, because FastAPI serializes `None` rather than dropping the key. */

import { events, json, send } from '../../api/http'

/* ── system.py ─────────────────────────────────────────────────────────── */

export interface SystemInfo {
  chip: string
  gpu_cores: number
  memory_bytes: number
  bandwidth_theoretical_gbs: number
  /* Measured, not published: the denominator of every "% of ceiling". */
  bandwidth_sustained_gbs: number
  disk_free_bytes: number
  catalog: string
  version: string
  constants: Record<string, string>
}

/* ── app.py `/admin/health` ────────────────────────────────────────────── */

export interface Health {
  status: string
  models: string[]
  pid: number
  /* Seconds off `time.monotonic()`, not a wall clock. */
  uptime: number
  version: string
}

/* ── state.py ──────────────────────────────────────────────────────────── */

export interface Resident {
  id: string
  weights_bytes: number
  kv_bytes: number
  loaded_at: number
  last_used: number | null
}

export interface Queue {
  running: number
  waiting: number
}

export interface DaemonState {
  models: Resident[]
  queue: Queue
  /* max(MLX active, the process' RSS, the accumulator) — never one of them alone. */
  resident_bytes: number
  kv_bytes: number
  /* What the conversations spilled to disk weigh, over every model. A cache nobody can see
     is rubbish accumulating in silence, which is why it is reported at all. */
  prefix_disk_bytes: number
}

/* ── config.py ─────────────────────────────────────────────────────────── */

export type Effect = 'applied' | 'restart' | 'inert'
export type Policy = 'load' | 'fail'

/* The effect belongs to the field and not to the value, so it comes back on the GET
   too — the screen draws the warning beside the control before anyone edits it. */
export interface Setting {
  value: number | string | null
  effect: Effect
  note: string | null
}

export type ConfigKey =
  | 'memory_limit_bytes'
  | 'idle_ttl_seconds'
  | 'max_concurrent_requests'
  | 'prefix_cache_bytes'
  | 'prefix_disk_bytes'
  | 'port'
  | 'api_key'
  | 'catalog_directory'
  | 'not_resident'

export type ConfigView = Record<ConfigKey, Setting>

/* Only what this screen edits. `null` is a value the daemon means (no TTL), so an
   absent key and a null one are different requests: `model_fields_set` decides. */
export interface ConfigPatch {
  memory_limit_bytes?: number
  idle_ttl_seconds?: number | null
  port?: number
  not_resident?: Policy
}

export function whole(setting: Setting): number | null {
  return typeof setting.value === 'number' ? setting.value : null
}

export function policy(setting: Setting): Policy | null {
  return setting.value === 'load' || setting.value === 'fail' ? setting.value : null
}

/* ── catalog.py ────────────────────────────────────────────────────────── */

export interface CatalogEntry {
  id: string
  directory: string
  store: string
  architecture: string
  quantization: string | null
  dtype: string | null
  context: number | null
  bytes_on_disk: number
  bytes_per_token: number | null
  /* What one token adds to the cache, over the attending layers alone — `null` for a
     config that does not say enough to price it. It is what turns the prefix budget from
     a byte count into a length of conversation. */
  kv_bytes_per_token: number | null
  resident: boolean
}

/* ── metrics.py ────────────────────────────────────────────────────────── */

export type RequestState = 'queued' | 'running' | 'completed' | 'cancelled' | 'error'

export interface Sample {
  model: string
  state: RequestState
  prompt_tokens: number
  /* How much of `prompt_tokens` a stored prefix covered instead of a forward. Zero is a
     real answer — the trie was off, or the render did not reproduce what the last turn
     wrote — and telling those apart is why it sits beside the prompt. */
  reused_tokens: number
  /* Whether the run left the next turn anything to start from. It separates the third zero
     from the other two: a trunk whose layers neither rewind nor serialize keeps nothing at
     all, and no setting changes that. */
  kept_prefix: boolean
  completion_tokens: number
  started_at: number
  /* What this request paid to put its model in memory, absent when it found it there. It
     sits outside `ttft`, which starts at the prefill. */
  load_seconds: number | null
  ttft: number | null
  tokens_per_second: number | null
  prefill_tokens_per_second: number | null
  bytes_per_token: number | null
  ceiling_fraction: number | null
  /* What a drafter proposed and how much of it the target confirmed, absent for a request
     that did not speculate. The rate alone never says which: a model paired with a drafter
     decodes exactly like an unpaired one under a sampler. */
  speculation: { rounds: number; proposed: number; accepted: number } | null
}

export interface Aggregate {
  model: string
  requests: number
  prompt_tokens: number
  completion_tokens: number
  ttft: number | null
  tokens_per_second: number | null
  bytes_per_token: number | null
  ceiling_fraction: number | null
}

export interface Snapshot {
  live: Sample | null
  /* Newest first. */
  requests: Sample[]
  models: Aggregate[]
}

/* ── calls ─────────────────────────────────────────────────────────────── */

export const getSystem = (): Promise<SystemInfo> => json<SystemInfo>('/admin/system')
export const getHealth = (): Promise<Health> => json<Health>('/admin/health')
export const getState = (): Promise<DaemonState> => json<DaemonState>('/admin/state')
export const getConfig = (): Promise<ConfigView> => json<ConfigView>('/admin/config')
export const getCatalog = (): Promise<CatalogEntry[]> => json<CatalogEntry[]>('/admin/models')

/* The answer is the whole config as it stands after the write, effects included. */
export const patchConfig = (patch: ConfigPatch): Promise<ConfigView> =>
  send<ConfigView>('PATCH', '/admin/config', patch)

/* ── desktop shell (serve.ts /desktop/daemon) ──────────────────────────── */

/* These talk to the shell, not the daemon: only the shell holds the handle of the
   process it spawned. Absent shell (bare `npm run dev:web`) — the GET fails and
   the buttons stay locked. */
export const getOwnership = (): Promise<{ owned: boolean }> =>
  json<{ owned: boolean }>('/desktop/daemon')
export const daemonAction = (action: 'stop' | 'restart'): Promise<void> =>
  send<void>('POST', `/desktop/daemon/${action}`)

/* A native sheet off the title bar. No shell means no sheet to ask with, so the answer
   is yes — the same unguarded behaviour the action had before the sheet existed. */
export const confirmSheet = (message: string, action: string, detail?: string): Promise<boolean> =>
  send<{ ok: boolean }>('POST', '/desktop/confirm', { message, action, detail, danger: true })
    .then((answer) => answer.ok)
    .catch(() => true)

/* Every frame is a whole snapshot and the stream opens with the state as it is now,
   so a late subscriber loses nothing. It ends only when the client goes away. */
export const metricEvents = (signal: AbortSignal): AsyncGenerator<Snapshot> =>
  events<Snapshot>('/admin/metrics/events', { signal })

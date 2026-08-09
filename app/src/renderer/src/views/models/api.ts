/* The catalog's own routes, typed off the server source:
   sideros_server/{catalog,residency,state,system,bench,metrics,profiles,config}.py.
   Ids carry '/', so every one of them is a single encoded path segment — uvicorn
   unquotes before routing and `{model_id:path}` takes the slashes back. */

import { HttpError, json, send, type JobView } from '../../api/http'

/* What the screen prints when a call is refused: the daemon's own sentence, never a
   status code dressed up as one. */
export const reason = (error: unknown): string =>
  error instanceof HttpError ? error.detail : error instanceof Error ? error.message : String(error)

/* catalog.CatalogEntry */
export interface CatalogEntry {
  id: string
  directory: string
  /* What the id owns on disk, and what DELETE removes. */
  store: string
  architecture: string
  /* The label the config declares — null is dense. */
  quantization: string | null
  dtype: string | null
  context: number | null
  bytes_on_disk: number
  /* What one decode step reads. null when the shards' headers cannot be read. */
  bytes_per_token: number | null
  resident: boolean
}

/* state.Resident */
export interface Resident {
  id: string
  weights_bytes: number
  kv_bytes: number
  loaded_at: number
  last_used: number | null
}

/* state.State */
export interface EngineState {
  models: Resident[]
  queue: { running: number; waiting: number }
  resident_bytes: number
  kv_bytes: number
}

/* system.SystemInfo */
export interface SystemInfo {
  chip: string
  gpu_cores: number
  memory_bytes: number
  bandwidth_theoretical_gbs: number
  bandwidth_sustained_gbs: number
  disk_free_bytes: number
  catalog: string
  version: string
  constants: Record<string, string>
}

/* store.Bench */
export interface Bench {
  model: string
  tokens_per_second: number
  ttft_ms: number
  engine_version: string
  created_at: number
  ceiling_fraction: number | null
}

/* bench.History */
export interface BenchHistory {
  method: string
  benches: Bench[]
}

/* metrics.Sample */
export interface Sample {
  model: string
  state: 'queued' | 'running' | 'completed' | 'cancelled' | 'error'
  prompt_tokens: number
  completion_tokens: number
  started_at: number
  ttft: number | null
  tokens_per_second: number | null
  bytes_per_token: number | null
  ceiling_fraction: number | null
}

/* metrics.Snapshot — only `live` is read here; the rest is the dashboard's. */
export interface MetricsSnapshot {
  live: Sample | null
}

/* config.Setting */
export interface Setting {
  value: number | string | null
  effect: string
  note: string | null
}

/* sideros.chat.Effort — `auto` is the checkpoint's template deciding, `on` is thinking with
   no rung named, and the five rungs are what a template that reads `reasoning_effort` gets. */
export type Effort = 'auto' | 'off' | 'on' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'

export const EFFORTS: readonly Effort[] = [
  'auto',
  'off',
  'on',
  'low',
  'medium',
  'high',
  'xhigh',
  'max'
]

/* profiles.Sampling — every knob optional, and null is "the profile does not set it". */
export interface Sampling {
  temperature: number | null
  top_p: number | null
  top_k: number | null
  min_p: number | null
  repetition_penalty: number | null
  seed: number | null
  reasoning_effort: Effort | null
  /* The ids the reasoning block may spend. A profile is the only surface it has on the two
     OpenAI dialects: neither request body can spell one, and inventing a field there would
     be a field only this server answers. */
  reasoning_budget: number | null
}

/* profiles.ProfileView */
export interface ProfileView {
  model: string
  name: string
  sampling: Sampling
  system_prompt: string | null
}

/* profiles.ProfileBody — extra fields are refused, `template` among them. */
export interface ProfileBody {
  sampling: Partial<Sampling>
  system_prompt: string | null
}

/* catalog.CheckpointFile — size through the symlink, not the link. */
export interface CheckpointFile {
  name: string
  size: number
}

const at = (id: string): string => encodeURIComponent(id)

export const listModels = (): Promise<CatalogEntry[]> => json<CatalogEntry[]>('/admin/models')

export const listFiles = (id: string): Promise<CheckpointFile[]> =>
  json<CheckpointFile[]>(`/admin/models/${at(id)}/files`)

/* The README raw — null when the checkpoint ships none, which is not an error. */
export async function getCard(id: string): Promise<string | null> {
  const answer = await fetch(`/admin/models/${at(id)}/card`)
  if (answer.status === 404) return null
  if (!answer.ok) throw new HttpError(answer.status, answer.statusText)
  return await answer.text()
}

/* Where the card's relative images resolve to. */
export const assetBase = (id: string): string => `/admin/models/${at(id)}/assets`

export const getModel = (id: string): Promise<CatalogEntry> =>
  json<CatalogEntry>(`/admin/models/${at(id)}`)

/* 409 while the model is resident, and the daemon's own sentence comes back as the
   refusal's detail. */
export const removeModel = (id: string): Promise<void> =>
  send<void>('DELETE', `/admin/models/${at(id)}`)

/* 202 and the job's first frame: a cold 30B is seconds, so residency is a job. */
export const loadModel = (id: string): Promise<JobView> =>
  send<JobView>('PUT', `/admin/models/${at(id)}/residency`)

/* 204, not a job: what the caller wants back is whether the memory returned. */
export const unloadModel = (id: string): Promise<void> =>
  send<void>('DELETE', `/admin/models/${at(id)}/residency`)

export const getState = (): Promise<EngineState> => json<EngineState>('/admin/state')

export const getSystem = (): Promise<SystemInfo> => json<SystemInfo>('/admin/system')

export const getBenches = (): Promise<BenchHistory> => json<BenchHistory>('/admin/benches')

export const getMetrics = (): Promise<MetricsSnapshot> => json<MetricsSnapshot>('/admin/metrics')

export const getConfig = (): Promise<Record<string, Setting>> =>
  json<Record<string, Setting>>('/admin/config')

interface ServedModels {
  data: { id: string }[]
}

/* There is no route that lists a model's profiles. What does list them is the dialect's
   own catalog: a model with a profile `code` is served under `model` and `model:code`
   (profiles.served_ids), so the names are the suffixes of the ids under this one. */
export async function listProfileNames(model: string): Promise<string[]> {
  const served = await json<ServedModels>('/api/openai/v1/models')
  return served.data
    .filter((entry) => entry.id.startsWith(`${model}:`))
    .map((entry) => entry.id.slice(model.length + 1))
    .filter((name) => name.length > 0)
}

export const getProfile = (model: string, name: string): Promise<ProfileView> =>
  json<ProfileView>(`/admin/models/${at(model)}/profiles/${at(name)}`)

export const saveProfile = (
  model: string,
  name: string,
  body: ProfileBody
): Promise<ProfileView> =>
  send<ProfileView>('PUT', `/admin/models/${at(model)}/profiles/${at(name)}`, body)

export const removeProfile = (model: string, name: string): Promise<void> =>
  send<void>('DELETE', `/admin/models/${at(model)}/profiles/${at(name)}`)

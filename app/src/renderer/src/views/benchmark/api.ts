/* The benchmark routes, typed off the server's own models: `benchmarks.RunView`,
   `benchmarks.Plan`, `datasets.DatasetView` and `fidelity.References`.

   One rule runs through all of it: nothing here recomputes what the server decided. Whether
   a shape fits, what it needed, what the ceiling was — those come off the plan and off the
   row. Two copies of that arithmetic is two answers to "does 128k by 16 fit", and one of
   them goes stale the first time the cache dtype moves. */

import { json, send } from '../../api/http'
import type { JobView } from '../../api/http'

export type Kind = 'speed' | 'fidelity' | 'quality'

export const CONTEXTS: readonly number[] = [
  512, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576
]
export const GENERATES: readonly number[] = [128, 256, 512, 1024, 2048]
export const CONCURRENCIES: readonly number[] = [1, 2, 4, 8, 16, 32, 64]

/* 35 to 90 or none, recorded with the run. The gate decides when a round may start, not
   what it measures. */
export const GATES: readonly (number | null)[] = [null, 35, 40, 45, 50, 55, 60, 75, 90]

/* `benchmarks.SamplingBody`. Greedy is the default because a benchmark that says nothing
   about sampling is timing an argmax, and every filter under it is another pass over the
   same 150k logits. */
export interface Sampling {
  temperature: number
  top_p: number
  top_k: number | null
  min_p: number
  repetition_penalty: number
  seed: number | null
}

export const GREEDY: Sampling = {
  temperature: 0,
  top_p: 1,
  top_k: null,
  min_p: 0,
  repetition_penalty: 1,
  seed: null
}

/* `speed.Sampling.key`, spelled the same way on both sides — a key written two ways is two
   axes. Only what is on says its name. */
export function samplingKey(sampling: Sampling): string {
  const drawn = sampling.temperature !== 0
  if (!drawn && sampling.repetition_penalty === 1) return 'greedy'
  const parts = [drawn ? `t${sampling.temperature}` : 'greedy']
  if (drawn && sampling.top_k !== null) parts.push(`k${sampling.top_k}`)
  if (drawn && sampling.top_p < 1) parts.push(`p${sampling.top_p}`)
  if (drawn && sampling.min_p > 0) parts.push(`m${sampling.min_p}`)
  if (sampling.repetition_penalty !== 1) parts.push(`rp${sampling.repetition_penalty}`)
  return parts.join('·')
}

export interface SpeedBody {
  context: number
  generate: number
  concurrency: number
  rounds: number
  stream_source: 'queue' | 'engine'
  page_cache: 'warm' | 'cold'
  gate_c: number | null
  load_s: number | null
  prefill_tps: number | null
  ttft_p50_ms: number | null
  ttft_p95_ms: number | null
  decode_tps: number | null
  decode_per_stream_tps: number | null
  step_weight_bytes: number | null
  step_kv_bytes: number | null
  ceiling_tps: number | null
  ceiling_fraction: number | null
  /* JSON text: a list of kept rounds when the shape ran, and the refusal's own numbers
     when it did not. */
  per_round: string
}

export interface QualityBody {
  dataset: string
  items: number
  seed: number
  shots: number
  scoring: 'loglikelihood' | 'generate'
  accuracy: number | null
  correct: number | null
  wall_s: number | null
  breakdown: string
}

export interface FidelityBody {
  reference: string
  corpus: string
  tokens: number
  seed: number
  topk: number
  kl_mean: number | null
  kl_p95: number | null
  top1: number | null
  top5: number | null
  ppl_delta: number | null
  histogram: string
}

export interface Run {
  id: string
  kind: Kind
  model: string
  key: string
  /* `not_run` is a result and not an absence: the band draws it dashed and the list says
     why. */
  state: 'ok' | 'not_run' | 'error'
  reason: string | null
  engine_version: string
  mlx_version: string
  temp_c_start: number | null
  residents: string
  created_at: number
  speed: SpeedBody | null
  quality: QualityBody | null
  fidelity: FidelityBody | null
}

export interface PlannedShape {
  model: string
  key: string
}

export interface SkippedShape extends PlannedShape {
  reason: string
  detail: string | null
  needed_bytes: number | null
  budget_bytes: number | null
}

export interface Plan {
  shapes: PlannedShape[]
  skipped: SkippedShape[]
  already: PlannedShape[]
  /* Rows that will run and that the sheet marks — a fidelity pair whose architectures
     differ is the one case. A warning is not a refusal. */
  warnings: SkippedShape[]
  estimate_seconds: number | null
}

export interface DatasetDefinition {
  id: string
  use: 'multiple_choice' | 'generation' | 'corpus'
  repo: string
  split: string
  template: string
  config: string | null
  columns: Record<string, string>
  size: number | null
  builtin: boolean
}

export interface Preview {
  prompt: string
  continuations: string[]
  answer: string | null
  columns: string[]
  rows: number
}

export interface ReferenceEntry {
  id: string
  reference: string
  corpus: string
  tokens: number
  seed: number
  topk: number
  bytes: number
  created_at: number
  present: boolean
}

export interface References {
  entries: ReferenceEntry[]
  total_bytes: number
}

export type Request =
  | {
      kind: 'speed'
      models: string[]
      contexts: number[]
      generates: number[]
      concurrencies: number[]
      rounds: number
      sampling: Sampling
      page_cache: 'warm' | 'cold'
      thermal_gate_c: number | null
      skip_if_measured: boolean
    }
  | {
      kind: 'fidelity'
      pairs: { model: string; reference: string }[]
      corpus: string
      tokens: number
      seed: number
      skip_if_measured: boolean
    }
  | {
      kind: 'quality'
      models: string[]
      datasets: string[]
      items: number
      seed: number
      shots: number
      skip_if_measured: boolean
    }

const query = (params: Record<string, string | undefined>): string => {
  const pairs = Object.entries(params).filter((entry): entry is [string, string] => entry[1] !== undefined)
  return pairs.length === 0 ? '' : `?${new URLSearchParams(pairs).toString()}`
}

export function runs(kind: Kind, key?: string, model?: string): Promise<Run[]> {
  return json<Run[]>(`/admin/benchmarks/runs${query({ kind, key, model })}`)
}

export function run(id: string): Promise<Run> {
  return json<Run>(`/admin/benchmarks/runs/${id}`)
}

export function forget(id: string): Promise<void> {
  return send<void>('DELETE', `/admin/benchmarks/runs/${id}`)
}

export const exportUrl = (kind: Kind, key?: string): string =>
  `/admin/benchmarks/runs.csv${query({ kind, key })}`

export function plan(body: Request): Promise<Plan> {
  return send<Plan>('POST', '/admin/benchmarks/plan', body)
}

export function start(body: Request): Promise<JobView> {
  return send<JobView>('POST', '/admin/benchmarks', body)
}

export function datasets(): Promise<DatasetDefinition[]> {
  return json<DatasetDefinition[]>('/admin/benchmarks/datasets')
}

export function saveDataset(body: Omit<DatasetDefinition, 'builtin'>): Promise<DatasetDefinition> {
  return send<DatasetDefinition>('POST', '/admin/benchmarks/datasets', body)
}

export function preview(body: Omit<DatasetDefinition, 'builtin'>): Promise<Preview> {
  return send<Preview>('POST', '/admin/benchmarks/datasets/preview', body)
}

export function references(): Promise<References> {
  return json<References>('/admin/benchmarks/references')
}

export function forgetReference(id: string): Promise<void> {
  return send<void>('DELETE', `/admin/benchmarks/references/${id}`)
}

/* The shape's own spelling of a context, which is what the key uses: `4096` is `4k`
   everywhere or the same measurement sits on two axes. */
export const human = (context: number): string =>
  context >= 1024 * 1024
    ? `${context / (1024 * 1024)}M`
    : context >= 1024
      ? `${context / 1024}k`
      : String(context)

/* What every key of this view begins with: the three knobs the band carries. Everything
   after it — rounds, sampler, page cache, stream source — is the *variant*, and the view
   reads the variants that exist off the runs instead of assuming one. */
export const speedPrefix = (context: number, generate: number, concurrency: number): string =>
  `${human(context)}→${generate} · ${concurrency} streams · `

export const speedKey = (
  context: number,
  generate: number,
  concurrency: number,
  rounds: number,
  sampling: Sampling = GREEDY,
  page = 'warm',
  source = 'queue'
): string =>
  `${speedPrefix(context, generate, concurrency)}r${rounds}` +
  ` · ${samplingKey(sampling)} · ${page} · ${source}`

export const qualityKey = (dataset: string, items: number, seed: number, shots: number): string =>
  `${dataset} · n${items} · s${seed} · ${shots}shot`

export const fidelityKey = (
  corpus: string,
  tokens: number,
  seed: number,
  reference: string
): string => `${corpus} · n${tokens} · s${seed} · ref:${reference}`

/* The refusal the server wrote into `per_round` when a shape did not run. Parsed here
   rather than recomputed: `needs 206 GB` is the daemon's arithmetic, never the client's. */
export interface Refusal {
  reason: string
  detail: string | null
  needed_bytes: number | null
  budget_bytes: number | null
}

export function refusal(body: SpeedBody | null): Refusal | null {
  if (body === null) return null
  try {
    const parsed: unknown = JSON.parse(body.per_round)
    if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Refusal
    }
  } catch {
    /* a body that is not JSON is a body with no refusal in it */
  }
  return null
}

export interface RoundSample {
  prompt_tokens: number
  completion_tokens: number
  ttft_ms: number
  decode_tps: number
}

export function rounds(body: SpeedBody | null): RoundSample[] {
  if (body === null) return []
  try {
    const parsed: unknown = JSON.parse(body.per_round)
    return Array.isArray(parsed) ? (parsed as RoundSample[]) : []
  } catch {
    return []
  }
}

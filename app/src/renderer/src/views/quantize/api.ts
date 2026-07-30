/* The three routes this screen speaks to, typed off the server's own models:
   `catalog.CatalogEntry` (GET /admin/models), `quantize.PlanRequest` /
   `quantize.PricedPlan` (POST /admin/quantizations/plan) and the job every
   creator answers with (POST /admin/quantizations → 202). */

import { HttpError, json, send } from '../../api/http'
import type { JobView } from '../../api/http'

export interface CatalogEntry {
  id: string
  directory: string
  store: string
  architecture: string
  /* The label the config declares — `null` is dense, and dense is what can be
     quantized: the plan refuses anything else. */
  quantization: string | null
  dtype: string | null
  context: number | null
  bytes_on_disk: number
  bytes_per_token: number | null
  resident: boolean
}

/* The engine's four. AWQ and GPTQ price as RTN — neither moves a leaf's format. oQ is
   priced by the same allocator the job runs, handed no scores: the totals hold, and the
   calibration decides which free leaf the promotion lands on. */
export type Method = 'rtn' | 'awq' | 'gptq' | 'oq'

export const METHODS: readonly { id: Method; label: string }[] = [
  { id: 'rtn', label: 'RTN' },
  { id: 'awq', label: 'AWQ' },
  { id: 'gptq', label: 'GPTQ' },
  { id: 'oq', label: 'oQ' }
]

/* `Affine.__post_init__`: anything else raises out of the engine and comes back 400. */
export const BITS: readonly number[] = [2, 3, 4, 5, 6, 8]
export const GROUP_SIZES: readonly number[] = [32, 64, 128]

export interface PlanRequest {
  source: string
  bits: number
  group_size: number
  /* Width per group of leaves, keyed by the `fullmatch` pattern the engine's selection
     takes; `null` says dense. A pattern matching no leaf fails the request. */
  overrides: Record<string, number | null>
  method: Method
}

export interface QuantizeRequest extends PlanRequest {
  repo: string
}

export interface PlanLeaf {
  path: string
  kind: string
  shape: number[]
  /* `null` says the leaf stays dense, which is what the default or an override asked. */
  bits: number | null
  bytes: number
}

export interface PricedPlan {
  leaves: PlanLeaf[]
  total_bytes: number
  weights: number
  bits_per_weight: number
  /* What the produced `model.safetensors` will weigh: the leaves at their new width plus
     everything no plan touches. */
  entry_bytes: number
}

/* Two windows on the machine the projected panel needs and nothing else. */
export interface Residency {
  resident_bytes: number
  kv_bytes: number
}

export interface Machine {
  memory_bytes: number
}

export const listModels = (): Promise<CatalogEntry[]> => json<CatalogEntry[]>('/admin/models')

export const residency = (): Promise<Residency> => json<Residency>('/admin/state')

export const machine = (): Promise<Machine> => json<Machine>('/admin/system')

export const price = (request: PlanRequest, signal: AbortSignal): Promise<PricedPlan> =>
  json<PricedPlan>('/admin/quantizations/plan', {
    method: 'POST',
    body: JSON.stringify(request),
    signal
  })

export const create = (request: QuantizeRequest): Promise<JobView> =>
  send<JobView>('POST', '/admin/quantizations', request)

/* The daemon's refusals are the screen's copy: what a method lacks, why a source has no
   price, which repo id is taken. None of it is paraphrased here. */
export const detail = (error: unknown): string =>
  error instanceof HttpError ? error.detail : error instanceof Error ? error.message : String(error)

export const REPO = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/

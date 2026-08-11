/* What more than one screen needs to know about the machine: how much memory there is,
   what is resident, where the ceiling sits, and which material each resident model
   wears. The title bar's meter and Resident's band draw the same numbers, so they read
   them from here rather than each polling for itself. */

import { createContext, useContext, useEffect, useState } from 'react'
import { events, json, type JobView } from './http'
import type { Snapshot } from '../views/overview/api'

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

export interface Resident {
  id: string
  weights_bytes: number
  kv_bytes: number
  loaded_at: number
  last_used: number | null
}

export interface EngineState {
  models: Resident[]
  /* `reserved` is somebody holding the queue exclusively — a benchmark, today. */
  queue: { running: number; waiting: number; reserved: boolean }
  resident_bytes: number
  kv_bytes: number
  /* What the conversations spilled to disk weigh, over every model. */
  prefix_disk_bytes: number
}

export interface Health {
  status: string
  models: string[]
  pid: number
  uptime: number
  version: string
}

export interface Setting {
  value: number | string | null
  effect: 'applied' | 'restart' | 'inert'
  note: string | null
}

export type Config = Record<string, Setting>

/* catalog.CatalogEntry */
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
  /* What the cache grows by per token, over every attending layer, and where it stops
     growing — the two facts that decide whether a benchmark shape fits at all. `null` is a
     config that does not answer, never a guess. */
  kv_bytes_per_token: number | null
  attention_window: number | null
  vocab_size: number | null
  /* A print of the architecture. It validates a fidelity pair somebody declared; it never
     proposes one, since a fine-tune prints identically to its base. */
  shape: string | null
  resident: boolean
}

export interface Engine {
  system: SystemInfo | null
  state: EngineState | null
  health: Health | null
  config: Config | null
  catalog: CatalogEntry[] | null
  /* The live jobs and what the last requests cost — on the same stream as the rest, so a
     screen that draws progress beside a model is not a second connection. */
  jobs: JobView[] | null
  metrics: Snapshot | null
  /* Drops the stream and opens a new one, which re-sends every resource. Nothing needs it
     to see a change any more; what it is still for is a client that wants to be sure. */
  refresh: () => void
}

/* One stream for the whole window: four screens read the same machine. */
const EngineContext = createContext<Engine | null>(null)
export const EngineProvider = EngineContext.Provider

export function useEngineContext(): Engine {
  const engine = useContext(EngineContext)
  if (engine === null) throw new Error('useEngineContext outside the provider')
  return engine
}

export const GIB = 1024 ** 3

export const gb = (bytes: number, digits = 1): string => (bytes / GIB).toFixed(digits)

/* The six materials, handed out by load order and kept while a model stays resident:
   the colour is identity, so a block does not change colour because another one left. */
const MATERIALS = 6

export function materials(models: Resident[]): Map<string, string> {
  const byAge = [...models].sort((a, b) => a.loaded_at - b.loaded_at)
  return new Map(byAge.map((model, index) => [model.id, `m${(index % MATERIALS) + 1}`]))
}

export const whole = (setting: Setting | undefined): number | null =>
  typeof setting?.value === 'number' ? setting.value : null

/* How long to wait before dialling again when the stream drops. A daemon that is down is a
   daemon somebody is restarting, and a client reconnecting on a tight loop is the load the
   restart does not need. */
const REDIAL_MS = 1000

/* Each frame of `/admin/events` names what it carries. Anything else is a daemon newer than
   this window, and ignoring it is what lets one add a resource without breaking the other. */
interface Frame {
  resource: string
  value: unknown
}

/* The desktop shell keeps its window hidden until the first frame lands — or until the dial
   fails, since an engine that is down is also something to show. Once per window: a bare
   `dev:web` has no such route and the 404 is as ignorable as the shell's absence. */
let announced = false

function announce(): void {
  if (announced) return
  announced = true
  void fetch('/desktop/ready', { method: 'POST' }).catch(() => {})
}

export function useEngine(): Engine {
  const [system, setSystem] = useState<SystemInfo | null>(null)
  const [state, setState] = useState<EngineState | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [config, setConfig] = useState<Config | null>(null)
  const [catalog, setCatalog] = useState<CatalogEntry[] | null>(null)
  const [jobs, setJobs] = useState<JobView[] | null>(null)
  const [metrics, setMetrics] = useState<Snapshot | null>(null)
  const [tick, setTick] = useState(0)

  /* The machine does not change while the app runs, so it is the one thing still fetched. */
  useEffect(() => {
    json<SystemInfo>('/admin/system')
      .then(setSystem)
      .catch(() => setSystem(null))
  }, [])

  /* Pushed, not polled: the daemon opens the stream with every resource and then sends the
     one that moved. What used to be here was a beat every two seconds whose catalog leg read
     the safetensors headers of every checkpoint on disk — a quarter of a core, for ever, to
     watch a directory that changes twice a day. */
  useEffect(() => {
    let live = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const stop = new AbortController()

    const dial = async (): Promise<void> => {
      try {
        for await (const frame of events<Frame>('/admin/events', { signal: stop.signal })) {
          if (!live) return
          announce()
          switch (frame.resource) {
            case 'health':
              setHealth(frame.value as Health)
              break
            case 'config':
              setConfig(frame.value as Config)
              break
            case 'models':
              setCatalog(frame.value as CatalogEntry[])
              break
            case 'state':
              setState(frame.value as EngineState)
              break
            case 'jobs':
              setJobs(frame.value as JobView[])
              break
            case 'metrics':
              setMetrics(frame.value as Snapshot)
              break
          }
        }
      } catch {
        /* The daemon went away, or this window did. Which one it was is the next dial's
           to find out. */
      }
      if (!live) return
      announce()
      timer = setTimeout(() => void dial(), REDIAL_MS)
    }

    void dial()
    return () => {
      live = false
      clearTimeout(timer)
      stop.abort()
    }
  }, [tick])

  return {
    system,
    state,
    health,
    config,
    catalog,
    jobs,
    metrics,
    refresh: () => setTick((n) => n + 1)
  }
}

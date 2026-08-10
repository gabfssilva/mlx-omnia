/* The memory scale, twice: the band that is the subject of Resident, and the 126px
   meter that carries it into every other screen's title bar. Both read the same
   numbers and use the same materials, so the small one is the big one compressed and
   not a second opinion. */

import type { JSX } from 'react'
import { GIB, gb, materials, type EngineState, type Resident, type SystemInfo } from './api/engine'

interface Scale {
  total: number
  ceiling: number
  models: Resident[]
  material: Map<string, string>
}

export function scale(
  system: SystemInfo | null,
  state: EngineState | null,
  ceilingBytes: number | null
): Scale | null {
  if (system === null || state === null) return null
  return {
    total: system.memory_bytes,
    ceiling: ceilingBytes ?? system.memory_bytes,
    models: [...state.models].sort((a, b) => b.weights_bytes + b.kv_bytes - (a.weights_bytes + a.kv_bytes)),
    material: materials(state.models)
  }
}

const share = (bytes: number, total: number): string => `${((bytes / total) * 100).toFixed(2)}%`

/* A block narrower than this cannot hold two lines of label. */
const TIGHT_FRACTION = 0.055

export function Band({
  scale: s,
  loading,
  onCeiling
}: {
  scale: Scale
  loading?: { id: string; done: number; total: number | null } | null
  onCeiling?: (bytes: number) => void
}): JSX.Element {
  const used = s.models.reduce((sum, model) => sum + model.weights_bytes + model.kv_bytes, 0)
  const pending = loading?.done ?? 0
  const free = Math.max(0, s.ceiling - used - pending)
  const headroom = s.total - s.ceiling
  const over = used + pending > s.ceiling
  const ceilingAt = (s.ceiling / s.total) * 100

  return (
    <div className="band">
      <div className="bandh">
        <b>Memory</b>
        <span>
          {gb(s.total, 0)} GB · {s.models.length} resident
        </span>
        <span className="r">
          {gb(used + pending)} in use · ceiling {gb(s.ceiling, 0)} GB
        </span>
      </div>
      <div className="track">
        {s.models.map((model) => {
          const size = model.weights_bytes + model.kv_bytes
          const tight = size / s.total < TIGHT_FRACTION
          return (
            <div
              key={model.id}
              className={`slab ${s.material.get(model.id) ?? 'm1'}${tight ? ' tight' : ''}`}
              style={{ width: share(size, s.total) }}
              title={`${model.id} · ${gb(size)} GB`}
            >
              {model.kv_bytes > 0 && (
                <div className="kvp" style={{ width: share(model.kv_bytes, size) }} />
              )}
              <b>{tight ? short(model.id) : model.id.split('/').at(-1)}</b>
              <span>
                {gb(model.weights_bytes)}
                {model.kv_bytes > 0 && ` + ${gb(model.kv_bytes)} KV`}
              </span>
            </div>
          )
        })}
        {loading && (
          <div
            className="slab m6 loading tight"
            style={{ width: share(loading.done, s.total) }}
            title={`Loading ${loading.id}`}
          >
            <span>{gb(loading.done)}</span>
          </div>
        )}
        <div className="rest">
          <b>{gb(free)}</b>
          <span>GB free below the ceiling</span>
        </div>
        {headroom > 0 && (
          <div className="hroom" style={{ width: share(headroom, s.total) }}>
            {/* Under a tenth of the track there is no room for the label and the
                ceiling's own tag at the same time. */}
            {headroom / s.total > 0.1 && (
              <i>
                headroom
                <br />
                {gb(headroom, 0)} GB
              </i>
            )}
          </div>
        )}
        <div className={over ? 'ceilv over' : 'ceilv'} style={{ left: `${ceilingAt}%` }}>
          <span className="tag">ceiling {gb(s.ceiling, 0)} GB</span>
          <span
            className="grip"
            role={onCeiling ? 'slider' : undefined}
            aria-label={onCeiling ? 'Memory ceiling' : undefined}
            aria-valuenow={onCeiling ? Math.round(s.ceiling / GIB) : undefined}
            aria-valuemin={onCeiling ? 0 : undefined}
            aria-valuemax={onCeiling ? Math.round(s.total / GIB) : undefined}
            tabIndex={onCeiling ? 0 : undefined}
            onKeyDown={(event) => {
              if (!onCeiling) return
              const step = event.shiftKey ? 8 : 1
              if (event.key === 'ArrowLeft') onCeiling(Math.max(0, s.ceiling - step * GIB))
              if (event.key === 'ArrowRight') onCeiling(Math.min(s.total, s.ceiling + step * GIB))
            }}
          />
        </div>
      </div>
    </div>
  )
}

/* What fits in a block too narrow for the whole name. */
function short(id: string): string {
  const leaf = id.split('/').at(-1) ?? id
  return leaf.split('-').slice(0, 2).join('-')
}

export function Meter({
  scale: s,
  ghost
}: {
  scale: Scale
  /* The checkpoint under the cursor, drawn as what it would cost. */
  ghost?: number | null
}): JSX.Element {
  const used = s.models.reduce((sum, model) => sum + model.weights_bytes + model.kv_bytes, 0)
  const over = used + (ghost ?? 0) > s.ceiling
  return (
    <>
      <span
        className="meter"
        role="img"
        aria-label={`Memory: ${gb(used)} of ${gb(s.ceiling, 0)} GB in use`}
      >
        {s.models.map((model) => (
          <i
            key={model.id}
            className={s.material.get(model.id) ?? 'm1'}
            style={{ width: share(model.weights_bytes + model.kv_bytes, s.total) }}
          />
        ))}
        {ghost !== null && ghost !== undefined && ghost > 0 && (
          <i className="gh" style={{ width: share(ghost, s.total) }} />
        )}
        <u className={over ? 'over' : undefined} style={{ left: share(s.ceiling, s.total) }} />
      </span>
      <span className="chip mono">
        {gb(used)} / {gb(s.ceiling, 0)} GB
      </span>
    </>
  )
}

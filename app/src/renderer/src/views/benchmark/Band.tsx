/* Parallel coordinates: one axis per measurement, one line per checkpoint.

   Speed and quality never share a chart — a line crossing from tok/s into accuracy joins
   two things that don't trade against each other on the same run, and context and
   concurrency only exist on the speed side. The band follows the view switch instead.

   Load and TTFT are drawn upside down, because lower is better there and "up" has to mean
   the same thing on every axis. A dashed segment means that checkpoint has no run under
   this key — either it hasn't been measured, or the combination doesn't fit in memory.

   Six lines at most, one per material of the palette: past that the drawing is a mesh. */

import { type JSX } from 'react'

export interface Axis {
  id: string
  /* What it is, over the rail. */
  nm: string
  /* What scopes it, under the name — the part of the key this axis was read at. */
  key: string
  unit: string
  fix: number
  /* Lower is better: the top of the rail is the smallest value, and the name carries ↓. */
  down?: boolean
}

export interface Line {
  model: string
  values: Record<string, number | null>
  material: number
}

const W = 1112
const TOP = 56
const BOT = 118
const MIDDLE = (TOP + BOT) / 2

function bounds(lines: Line[], axis: Axis): [number, number] | null {
  const found = lines
    .map((line) => line.values[axis.id])
    .filter((value): value is number => value !== null && value !== undefined)
  if (found.length === 0) return null
  const low = Math.min(...found)
  const high = Math.max(...found)
  const pad = (high - low) * 0.22 || Math.abs(high) * 0.1 || 1
  return [low - pad, high + pad]
}

export function Band({
  axes,
  lines,
  selected,
  onSelect
}: {
  axes: Axis[]
  lines: Line[]
  selected: string | null
  onSelect: (model: string) => void
}): JSX.Element {
  const column = W / Math.max(1, axes.length)
  const ranges = new Map(axes.map((axis) => [axis.id, bounds(lines, axis)]))

  const at = (axis: Axis, value: number): number => {
    const range = ranges.get(axis.id)
    if (range === null || range === undefined) return MIDDLE
    const [low, high] = range
    const share = (value - low) / (high - low)
    return BOT - (axis.down === true ? 1 - share : share) * (BOT - TOP)
  }

  return (
    <div className="track2">
      <svg
        viewBox={`0 0 ${W} 152`}
        role="img"
        aria-label="Parallel coordinates: one axis per measurement, one line per checkpoint"
      >
        <g>
          {axes.map((axis, index) => {
            const x = column * (index + 0.5)
            const range = ranges.get(axis.id) ?? null
            return (
              <g key={axis.id}>
                <text x={x} y={22} className="axnm" textAnchor="middle">
                  {axis.nm}
                  {axis.down === true ? ' ↓' : ''}
                </text>
                <text x={x} y={35} className="axkey" textAnchor="middle">
                  {axis.key}
                </text>
                <line
                  x1={x}
                  y1={TOP - 8}
                  x2={x}
                  y2={BOT + 8}
                  className={range === null ? 'railghost' : 'rail'}
                />
                {range !== null && (
                  <>
                    <text x={x + 8} y={TOP - 2} className="axend">
                      {(axis.down === true ? range[0] : range[1]).toFixed(axis.fix)}
                      {axis.unit}
                    </text>
                    <text x={x + 8} y={BOT + 12} className="axend">
                      {(axis.down === true ? range[1] : range[0]).toFixed(axis.fix)}
                      {axis.unit}
                    </text>
                  </>
                )}
              </g>
            )
          })}
        </g>
        <g>
          {lines.map((line) => {
            const on = line.model === selected
            const points = axes.map((axis, index) => {
              const range = ranges.get(axis.id) ?? null
              const value = range === null ? null : (line.values[axis.id] ?? null)
              return {
                x: column * (index + 0.5),
                y: value === null ? null : at(axis, value),
                value,
                axis
              }
            })
            return (
              <g key={line.model} className={`m${line.material}`}>
                {points.slice(0, -1).map((from, index) => {
                  const to = points[index + 1]
                  if (from.y === null && to.y === null) return null
                  const broken = from.y === null || to.y === null
                  return (
                    <line
                      key={axes[index].id}
                      x1={from.x}
                      y1={from.y ?? MIDDLE}
                      x2={to.x}
                      y2={to.y ?? MIDDLE}
                      stroke="var(--m)"
                      className={`link${on ? ' on' : ''}${broken ? ' gap' : ''}`}
                    />
                  )
                })}
                {points.map((point) =>
                  point.y === null ? (
                    <circle key={point.axis.id} cx={point.x} cy={MIDDLE} className="pt miss" />
                  ) : (
                    <g key={point.axis.id}>
                      <circle cx={point.x} cy={point.y} fill="var(--m)" className={`pt${on ? ' on' : ''}`} />
                      {on && point.value !== null && (
                        <text x={point.x - 9} y={point.y + 3.5} className="val" textAnchor="end">
                          {point.value.toFixed(point.axis.fix)}
                        </text>
                      )}
                    </g>
                  )
                )}
                <polyline
                  points={points
                    .filter((point) => point.y !== null)
                    .map((point) => `${point.x},${point.y ?? 0}`)
                    .join(' ')}
                  fill="none"
                  stroke="transparent"
                  strokeWidth={16}
                  className="lane"
                  onClick={() => onSelect(line.model)}
                />
              </g>
            )
          })}
        </g>
      </svg>
    </div>
  )
}

/* Best, worst and four spread by rank between them — six, one per material. The automatic
   fill is a starting point and then it is the user's: `auto ∪ pinned − dropped`. */
export function autofill(models: string[], score: (model: string) => number | null): string[] {
  const ranked = models
    .map((model) => ({ model, value: score(model) }))
    .filter((entry): entry is { model: string; value: number } => entry.value !== null)
    .sort((left, right) => right.value - left.value)
    .map((entry) => entry.model)
  const ordered = [...ranked, ...models.filter((model) => !ranked.includes(model))]
  if (ordered.length <= 6) return ordered
  const picked = new Set([ordered[0], ordered[ordered.length - 1]])
  for (let step = 1; step <= 4; step += 1) picked.add(ordered[Math.round((step * (ordered.length - 1)) / 5)])
  return ordered.filter((model) => picked.has(model))
}

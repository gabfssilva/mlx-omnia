/* The Speed view: four measurements under one shape, and the shape is three knobs in the
   band header.

   Context, generate and concurrency together *are* the key, so they scope the whole view —
   band and list move in one action. They are absent from the other two views, and that
   absence is the most direct way of saying that concurrency does not exist on the quality
   side.

   A shape that did not fit shows the reason with the server's own numbers. What fits is the
   daemon's arithmetic and never recomputed here: two copies of it is two answers, and one
   of them goes stale the first time the cache dtype moves. */

import { type JSX } from 'react'
import { gb } from '../../api/engine'
import type { Axis } from './Band'
import { CONCURRENCIES, human, refusal, rounds, speedPrefix, type Run } from './api'

export function speedAxes(context: number, generate: number, concurrency: number): Axis[] {
  const streams = `${concurrency} stream${concurrency > 1 ? 's' : ''}`
  return [
    { id: 'load', nm: 'Load', key: 'warm cache', unit: ' s', fix: 1, down: true },
    { id: 'pre', nm: 'Prefill', key: `${streams} · ${human(context)}`, unit: '', fix: 0 },
    { id: 'ttft', nm: 'TTFT', key: `${streams} · p50`, unit: ' ms', fix: 0, down: true },
    { id: 'dec', nm: 'Decode', key: `${streams} · ${human(context)}→${generate}`, unit: '', fix: 1 }
  ]
}

export function speedValues(run: Run | undefined): Record<string, number | null> {
  const body = run?.speed ?? null
  return {
    load: body?.load_s ?? null,
    pre: body?.prefill_tps ?? null,
    ttft: body?.ttft_p50_ms ?? null,
    dec: body?.decode_tps ?? null
  }
}

const nb = (value: number | null): string => (value === null ? '—' : Math.round(value).toLocaleString('en'))

function spark(samples: number[]): JSX.Element | null {
  if (samples.length < 2) return null
  const low = Math.min(...samples)
  const high = Math.max(...samples)
  const span = high - low || 1
  const points = samples
    .map((value, index) => `${((index / (samples.length - 1)) * 84).toFixed(1)},${(22 - ((value - low) / span) * 20).toFixed(1)}`)
    .join(' ')
  return (
    <svg viewBox="0 0 84 24" fill="none">
      <polyline points={points} stroke="currentColor" strokeWidth="1.2" />
    </svg>
  )
}

export function SpeedRows({
  runs,
  context,
  generate,
  concurrency,
  selected,
  materials,
  onSelect
}: {
  runs: Run[]
  context: number
  generate: number
  concurrency: number
  selected: string | null
  materials: Map<string, number>
  onSelect: (model: string) => void
}): JSX.Element {
  const ordered = [...runs].sort(
    (left, right) => (right.speed?.decode_tps ?? -1) - (left.speed?.decode_tps ?? -1)
  )
  /* The shape is the group's own heading, so the row says only what varies under it. */
  const prefix = speedPrefix(context, generate, concurrency)
  return (
    <>
      <div className="qhead">
        <b>
          {human(context)} → {generate} tok · {concurrency} stream{concurrency > 1 ? 's' : ''}
        </b>
        <span>{ordered.length} measured</span>
      </div>
      <ul className="runs">
        {ordered.map((run) => {
          const material = materials.get(run.model)
          const refused = refusal(run.speed)
          const ok = run.state === 'ok'
          return (
            <li
              key={run.id}
              className={`run s ${material === undefined ? '' : `m${material}`}${run.model === selected ? ' on' : ''}${ok ? '' : ' nofit'}`}
              onClick={() => onSelect(run.model)}
            >
              <div className="id">
                <span className="top">
                  {material !== undefined && <i className="inband" />}
                  <b>{run.model}</b>
                </span>
                <span title={run.key}>
                  {run.key.startsWith(prefix) ? run.key.slice(prefix.length) : run.key} · load{' '}
                  {run.speed?.load_s === null || run.speed === null ? '—' : `${run.speed.load_s.toFixed(1)} s`}
                </span>
              </div>
              <div className={`num${ok ? '' : ' dim'}`}>
                <b>{ok ? (run.speed?.decode_tps ?? 0).toFixed(1) : '—'}</b>
                <span>decode</span>
              </div>
              <div className="num dim">
                <b>{ok ? nb(run.speed?.ttft_p50_ms ?? null) : '—'}</b>
                <span>ms ttft</span>
              </div>
              <div className="num dim">
                <b>{ok ? nb(run.speed?.prefill_tps ?? null) : '—'}</b>
                <span>prefill</span>
              </div>
              <div className="spark">{ok ? spark(rounds(run.speed).map((entry) => entry.decode_tps)) : null}</div>
              <div className="state">
                <div className="l1">
                  {ok ? (
                    <>
                      <i className="dot ok" />
                      {run.speed?.ceiling_fraction === null || run.speed === null
                        ? 'measured'
                        : `${(run.speed.ceiling_fraction * 100).toFixed(1)}% of ceiling`}
                    </>
                  ) : (
                    <>
                      <i className="dot bad" />
                      {refused?.reason === 'kv_over_budget' && refused.needed_bytes !== null
                        ? `needs ${gb(refused.needed_bytes, 0)} GB`
                        : (run.reason ?? 'not run').replace(/_/g, ' ')}
                    </>
                  )}
                </div>
                <div className="l2">
                  {ok
                    ? `${(run.speed?.decode_per_stream_tps ?? 0).toFixed(1)} per stream`
                    : refused?.reason === 'kv_over_budget'
                      ? 'kv over budget'
                      : '—'}
                </div>
              </div>
            </li>
          )
        })}
      </ul>
      {ordered.length === 0 && (
        <p className="note" style={{ padding: '12px 3px' }}>
          Nothing measured under this shape. New benchmark runs it — or pick another context,
          generate and concurrency above.
        </p>
      )}
    </>
  )
}

export function SpeedResult({
  run,
  context,
  concurrency,
  series,
  onConcurrency
}: {
  run: Run
  context: number
  concurrency: number
  series: { concurrency: number; decode: number | null }[]
  onConcurrency: (value: number) => void
}): JSX.Element {
  const body = run.speed
  const refused = refusal(body)
  const samples = rounds(body)
  const spread =
    samples.length < 2
      ? null
      : ((Math.max(...samples.map((entry) => entry.decode_tps)) -
          Math.min(...samples.map((entry) => entry.decode_tps))) /
          (body?.decode_tps ?? 1)) *
        100
  const high = Math.max(1, ...series.map((point) => point.decode ?? 0))
  const tall = Math.max(1, ...samples.map((entry) => entry.decode_tps))

  return (
    <>
      <div className="grouplab">
        Load <em style={{ fontStyle: 'normal', textTransform: 'none', letterSpacing: 0 }}>· no context, no concurrency</em>
      </div>
      <div className="kvs">
        <div className="kvr">
          <span className="k">time to ready</span>
          <span className="v">{body?.load_s === null || body === null ? '—' : `${body.load_s.toFixed(2)} s`}</span>
        </div>
        <div className="kvr">
          <span className="k">page cache</span>
          <span className="v">{body?.page_cache ?? '—'}</span>
        </div>
      </div>

      {refused !== null && (
        <p className="note bad" style={{ marginTop: 9 }}>
          {refused.detail ??
            (refused.needed_bytes !== null && refused.budget_bytes !== null
              ? `${refused.reason.replace(/_/g, ' ')}: needed ${gb(refused.needed_bytes, 1)} GB against a budget of ${gb(refused.budget_bytes, 1)} GB`
              : refused.reason.replace(/_/g, ' '))}
        </p>
      )}

      <div className="grouplab">
        At {concurrency} stream{concurrency > 1 ? 's' : ''} · {human(context)} context
      </div>
      <div className="kvs">
        <div className="kvr">
          <span className="k">prefill</span>
          <span className="v">{nb(body?.prefill_tps ?? null)} tok/s</span>
        </div>
        <div className="kvr">
          <span className="k">ttft p50</span>
          <span className="v">{nb(body?.ttft_p50_ms ?? null)} ms</span>
        </div>
        <div className="kvr">
          <span className="k">ttft p95</span>
          <span className="v">{nb(body?.ttft_p95_ms ?? null)} ms</span>
        </div>
        <div className="kvr">
          <span className="k">decode total</span>
          <span className="v">{(body?.decode_tps ?? 0).toFixed(1)} tok/s</span>
        </div>
        <div className="kvr">
          <span className="k">decode per stream</span>
          <span className="v">{(body?.decode_per_stream_tps ?? 0).toFixed(1)} tok/s</span>
        </div>
        <div className="kvr">
          <span className="k">spread across rounds</span>
          <span className="v">{spread === null ? '—' : `±${spread.toFixed(1)}%`}</span>
        </div>
      </div>

      <div className="grouplab">
        Scaling with concurrency{' '}
        <em style={{ fontStyle: 'normal', textTransform: 'none', letterSpacing: 0 }}>· at {human(context)}</em>
      </div>
      <div className="curve">
        {series.map((point) => (
          <button
            key={point.concurrency}
            type="button"
            className={`${point.concurrency === concurrency ? 'on' : ''}${point.decode === null ? ' dead' : ''}`}
            onClick={() => onConcurrency(point.concurrency)}
          >
            <b style={{ height: `${((point.decode ?? high * 0.12) / high) * 100}%` }} />
            <span>{point.concurrency}</span>
          </button>
        ))}
      </div>
      <div className="curvelab">
        <span>decode total</span>
        <span>{high.toFixed(0)} tok/s</span>
      </div>

      {/* The ceiling opened: weights amortise between streams and the cache does not, which
          is why decode collapses with context long before it collapses with concurrency. */}
      <div className="grouplab">Bandwidth ceiling</div>
      <div className="kvs">
        <div className="kvr">
          <span className="k">weights read / step</span>
          <span className="v">
            {body?.step_weight_bytes === null || body === null ? '—' : `${gb(body.step_weight_bytes, 2)} GB`}
          </span>
        </div>
        <div className="kvr">
          <span className="k">kv read / step</span>
          <span className="v">
            {body?.step_kv_bytes === null || body === null ? '—' : `${gb(body.step_kv_bytes, 2)} GB`}
          </span>
        </div>
        <div className="kvr">
          <span className="k">ceiling here</span>
          <span className="v">{nb(body?.ceiling_tps ?? null)} tok/s</span>
        </div>
        <div className="kvr">
          <span className="k">% of ceiling</span>
          <span className="v">
            {body?.ceiling_fraction === null || body === null ? '—' : `${(body.ceiling_fraction * 100).toFixed(1)}%`}
          </span>
        </div>
      </div>
      <p className="note" style={{ marginTop: 8 }}>
        The ceiling is recomputed for this shape. Change the weight format and it moves — a
        percentage that rose because its denominator shrank is not a gain.
      </p>

      {samples.length > 0 && (
        <>
          <div className="grouplab">Rounds</div>
          <div className="rounds">
            {samples.map((sample, index) => (
              <i key={index} style={{ height: `${(sample.decode_tps / tall) * 100}%` }} />
            ))}
          </div>
          <p className="note" style={{ marginTop: 8 }}>
            One warm-up round ran before the first shape of this checkpoint and stayed out of
            every median.
          </p>
        </>
      )}
    </>
  )
}

export function SpeedSetup({ run }: { run: Run }): JSX.Element {
  const body = run.speed
  const residents = JSON.parse(run.residents) as string[]
  return (
    <>
      <div className="grouplab">Run</div>
      <div className="kvs">
        <div className="kvr">
          <span className="k">key</span>
          <span className="v">{run.key}</span>
        </div>
        <div className="kvr">
          <span className="k">context → generate</span>
          <span className="v">{body === null ? '—' : `${human(body.context)} → ${body.generate}`}</span>
        </div>
        <div className="kvr">
          <span className="k">rounds</span>
          <span className="v">{body?.rounds ?? '—'}</span>
        </div>
        <div className="kvr">
          <span className="k">streams from</span>
          <span className="v">{body?.stream_source ?? '—'}</span>
        </div>
      </div>
      <div className="grouplab">Provenance</div>
      <div className="kvs">
        <div className="kvr">
          <span className="k">engine</span>
          <span className="v">{run.engine_version}</span>
        </div>
        <div className="kvr">
          <span className="k">mlx</span>
          <span className="v">{run.mlx_version}</span>
        </div>
        <div className="kvr">
          <span className="k">gate</span>
          <span className="v">{body?.gate_c === null || body === null ? 'none' : `${body.gate_c} °C`}</span>
        </div>
        <div className="kvr">
          <span className="k">temp at start</span>
          <span className="v">{run.temp_c_start === null ? '—' : `${run.temp_c_start.toFixed(1)} °C`}</span>
        </div>
        <div className="kvr">
          <span className="k">other residents</span>
          <span className="v">{residents.length <= 1 ? 'none' : residents.length - 1}</span>
        </div>
        <div className="kvr">
          <span className="k">queue</span>
          <span className="v">exclusive</span>
        </div>
      </div>
      <p className="note" style={{ marginTop: 9 }}>
        Every line here is stored with the number. Without it, next week&rsquo;s comparison
        isn&rsquo;t one.
      </p>
    </>
  )
}

export const CONCURRENCY_LEVELS = CONCURRENCIES

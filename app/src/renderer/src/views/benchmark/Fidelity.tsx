/* The Fidelity view: how far a checkpoint drifted from the one it was made out of.

   Logits, not answers. A 3-bit can lose 0.3 points of MMLU — inside the noise of 1400 items
   — and change its first choice on one token in ten, and only this view sees that.

   The list teaches the constraint in three blocks instead of hiding what cannot be done:
   measured against this reference, could be but has not been, and a different vocabulary —
   which has no number and no click, because logits of two vocabularies do not subtract.
   Changing the reference changes the whole set of candidates, because the reference is
   inside the key.

   The calculation is task 59.9's `# TODO`. Until it lands the rows are `not_run` and this
   view says so. */

import { type JSX } from 'react'
import type { CatalogEntry } from '../../api/engine'
import type { Axis } from './Band'
import type { Run } from './api'

export function fidelityAxes(corpus: string): Axis[] {
  return [
    { id: 'kl', nm: 'KL', key: corpus, unit: '', fix: 3, down: true },
    { id: 'kl95', nm: 'KL p95', key: 'tail', unit: '', fix: 3, down: true },
    { id: 'top1', nm: 'Top-1', key: 'agreement', unit: '%', fix: 1 },
    { id: 'ppl', nm: 'Perplexity', key: 'vs reference', unit: '%', fix: 1, down: true }
  ]
}

export function fidelityValues(run: Run | undefined): Record<string, number | null> {
  const body = run?.fidelity ?? null
  return {
    kl: body?.kl_mean ?? null,
    kl95: body?.kl_p95 ?? null,
    top1: body?.top1 === null || body === null ? null : body.top1 * 100,
    ppl: body?.ppl_delta ?? null
  }
}

export interface Candidate {
  entry: CatalogEntry
  run: Run | undefined
  /* Same vocabulary, other architecture. It computes and it does not mean what this view
     means — a warning, never a refusal: a fine-tune prints identically to its base, so
     blocking on a different print would block only the pairs somebody declared. */
  otherShape: boolean
}

export function split(
  catalog: CatalogEntry[],
  reference: string,
  runs: Run[]
): { measured: Candidate[]; comparable: Candidate[]; incompatible: CatalogEntry[] } {
  const referred = catalog.find((entry) => entry.id === reference)
  const byModel = new Map(runs.map((run) => [run.model, run]))
  const measured: Candidate[] = []
  const comparable: Candidate[] = []
  const incompatible: CatalogEntry[] = []
  for (const entry of catalog) {
    if (entry.id === reference) continue
    const sameVocab =
      referred === undefined ||
      entry.vocab_size === null ||
      referred.vocab_size === null ||
      entry.vocab_size === referred.vocab_size
    if (!sameVocab) {
      incompatible.push(entry)
      continue
    }
    const candidate: Candidate = {
      entry,
      run: byModel.get(entry.id),
      otherShape:
        referred !== undefined &&
        entry.shape !== null &&
        referred.shape !== null &&
        entry.shape !== referred.shape
    }
    if (candidate.run !== undefined && candidate.run.state === 'ok') measured.push(candidate)
    else comparable.push(candidate)
  }
  measured.sort(
    (left, right) => (left.run?.fidelity?.kl_mean ?? 0) - (right.run?.fidelity?.kl_mean ?? 0)
  )
  return { measured, comparable, incompatible }
}

const four = (value: number | null): string => (value === null ? '—' : value.toFixed(4))

export function FidelityRows({
  reference,
  corpus,
  groups,
  selected,
  materials,
  onSelect
}: {
  reference: string
  corpus: string
  groups: ReturnType<typeof split>
  selected: string | null
  materials: Map<string, number>
  onSelect: (model: string) => void
}): JSX.Element {
  return (
    <>
      <div className="qhead">
        <b>Against {reference}</b>
        <span>{corpus}</span>
      </div>
      <ul className="runs">
        {groups.measured.map((candidate) => {
          const material = materials.get(candidate.entry.id)
          const body = candidate.run?.fidelity ?? null
          const kl = body?.kl_mean ?? null
          return (
            <li
              key={candidate.entry.id}
              className={`run q ${material === undefined ? '' : `m${material}`}${candidate.entry.id === selected ? ' on' : ''}`}
              onClick={() => onSelect(candidate.entry.id)}
            >
              <div className="id">
                <span className="top">
                  {material !== undefined && <i className="inband" />}
                  <b>{candidate.entry.id}</b>
                </span>
                <span>{candidate.run?.key ?? corpus}</span>
              </div>
              <div className="num">
                <b>{kl === null ? '—' : kl.toFixed(3)}</b>
                <span>kl nats</span>
              </div>
              <div className="num dim">
                <b>{body?.top1 === null || body === null ? '—' : (body.top1 * 100).toFixed(1)}</b>
                <span>% top-1</span>
              </div>
              <div className="spark" />
              <div className="state">
                <div className="l1">
                  <i className={`dot ${kl === null ? '' : kl < 0.02 ? 'ok' : kl < 0.08 ? 'warn' : 'bad'}`} />
                  {body?.ppl_delta === null || body === null ? '—' : `ppl ${body.ppl_delta > 0 ? '+' : ''}${body.ppl_delta.toFixed(1)}%`}
                </div>
                <div className="l2">
                  {body?.top5 === null || body === null ? '' : `top-5 ${(body.top5 * 100).toFixed(1)}%`}
                </div>
              </div>
            </li>
          )
        })}
      </ul>

      {groups.comparable.length > 0 && (
        <>
          <div className="qhead">
            <b>Same vocabulary, not measured against this reference</b>
            <span>can run</span>
          </div>
          <ul className="runs">
            {groups.comparable.map((candidate) => (
              <li
                key={candidate.entry.id}
                className={`run q${candidate.otherShape ? ' nofit' : ''}`}
                style={{ opacity: 0.6 }}
                onClick={() => onSelect(candidate.entry.id)}
              >
                <div className="id">
                  <span className="top">
                    <b>{candidate.entry.id}</b>
                  </span>
                  <span>shape {candidate.entry.shape ?? '—'}</span>
                </div>
                <div className="num dim">
                  <b>—</b>
                  <span>kl nats</span>
                </div>
                <div className="num dim">
                  <b>—</b>
                  <span>% top-1</span>
                </div>
                <div className="spark" />
                <div className="state">
                  <div className="l1">
                    <i className="dot" />
                    {candidate.otherShape ? 'other architecture' : 'not run'}
                  </div>
                  <div className="l2">
                    {candidate.otherShape ? 'kl defined, not meaningful' : 'same shape'}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </>
      )}

      {/* No number and no click: logits of two vocabularies do not subtract. */}
      {groups.incompatible.length > 0 && (
        <>
          <div className="qhead">
            <b>Different vocabulary</b>
            <span>{groups.incompatible.length} checkpoints</span>
          </div>
          <ul className="runs">
            {groups.incompatible.map((entry) => (
              <li key={entry.id} className="run q nofit" style={{ opacity: 0.45, cursor: 'default' }}>
                <div className="id">
                  <span className="top">
                    <b>{entry.id}</b>
                  </span>
                  <span>vocab {entry.vocab_size ?? '—'}</span>
                </div>
                <div className="num dim">
                  <b>—</b>
                  <span>kl nats</span>
                </div>
                <div className="num dim">
                  <b>—</b>
                  <span>% top-1</span>
                </div>
                <div className="spark" />
                <div className="state">
                  <div className="l1">
                    <i className="dot" />
                    not comparable
                  </div>
                  <div className="l2">logits don&rsquo;t subtract</div>
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
    </>
  )
}

function histogram(text: string): number[] {
  try {
    const parsed: unknown = JSON.parse(text)
    return Array.isArray(parsed) ? (parsed as number[]) : []
  } catch {
    return []
  }
}

export function FidelityResult({ run, reference }: { run: Run; reference: string }): JSX.Element {
  const body = run.fidelity
  const bars = histogram(body?.histogram ?? '[]')
  const high = Math.max(1, ...bars)
  const flipped = body?.top1 === null || body === undefined || body === null ? null : (1 - body.top1) * 100
  return (
    <>
      <div className="grouplab">Against {reference}</div>
      <div className="kvs">
        <div className="kvr">
          <span className="k">KL mean</span>
          <span className="v">{four(body?.kl_mean ?? null)}</span>
        </div>
        <div className="kvr">
          <span className="k">KL p95</span>
          <span className="v">{four(body?.kl_p95 ?? null)}</span>
        </div>
        <div className="kvr">
          <span className="k">Top-1 agreement</span>
          <span className="v">
            {body?.top1 === null || body === null ? '—' : `${(body.top1 * 100).toFixed(2)}%`}
          </span>
        </div>
        <div className="kvr">
          <span className="k">Top-5 overlap</span>
          <span className="v">
            {body?.top5 === null || body === null ? '—' : `${(body.top5 * 100).toFixed(2)}%`}
          </span>
        </div>
        <div className="kvr">
          <span className="k">Δ perplexity</span>
          <span className="v">{four(body?.ppl_delta ?? null)}</span>
        </div>
      </div>

      <div className="grouplab">Comparison key</div>
      <div className="keyline">
        key <b>{run.key}</b>
      </div>
      <p className="note" style={{ marginTop: 6 }}>
        The reference is part of the key. Measured against a different checkpoint, these are
        different numbers and do not compare.
      </p>

      {run.state !== 'ok' && (
        <>
          <div className="grouplab">Did not run</div>
          <div className="err">
            {run.reason === 'vocabulary_mismatch'
              ? 'The two vocabularies differ, so there is no vector to subtract. This pair has no comparison, not a missing one.'
              : run.reason === 'not_implemented'
                ? 'The teacher-forced pass is task 59.9. The pair is recorded so nothing about it has to be typed twice.'
                : (run.reason ?? 'unknown')}
          </div>
        </>
      )}

      {bars.length > 0 && (
        <>
          <div className="grouplab">KL per token</div>
          {/* The tail is what matters: a low mean with a fat tail is a model that agrees
              almost always and diverges hard when it does not. */}
          <div className="rounds">
            {bars.map((count, index) => (
              <i key={index} style={{ height: `${(count / high) * 100}%` }} />
            ))}
          </div>
          <div className="curvelab">
            <span>kl per token, bucketed</span>
            <span>tail on the right</span>
          </div>
        </>
      )}

      {flipped !== null && (
        <p className="note" style={{ marginTop: 8 }}>
          {flipped.toFixed(1)} tokens in every hundred come out with a different first choice,
          and a 1400-item benchmark does not see that in the aggregate.
        </p>
      )}
    </>
  )
}

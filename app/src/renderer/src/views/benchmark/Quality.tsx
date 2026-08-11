/* The Quality view: accuracy on a dataset, one axis per dataset.

   No context and no concurrency button. Accuracy does not move with either, and the absence
   of the control is the most direct way of saying so. What authorises the comparison is the
   key on the group header — `n1400 · s42 · 5shot` — because a sample of a different size,
   or drawn under a different seed, is a different question.

   The execution is task 59.8's `# TODO`. Until it lands the daemon records the rows as
   `not_run` and finishes the job in error, and this view shows that as an error rather than
   as a zero. */

import { type JSX, useState } from 'react'
import { useEscape } from '../../keys'
import type { Axis } from './Band'
import { preview, saveDataset, type DatasetDefinition, type Preview, type Run } from './api'

export function qualityAxes(datasets: { id: string; key: string }[]): Axis[] {
  return datasets.map((dataset) => ({
    id: dataset.id,
    nm: dataset.id,
    key: dataset.key,
    unit: '%',
    fix: 1
  }))
}

export function qualityValues(runs: Run[]): Record<string, number | null> {
  const values: Record<string, number | null> = {}
  for (const run of runs) {
    if (run.quality !== null) {
      values[run.quality.dataset] =
        run.quality.accuracy === null ? null : run.quality.accuracy * 100
    }
  }
  return values
}

const percent = (value: number | null): string =>
  value === null ? '—' : `${(value * 100).toFixed(1)}`

function breakdown(text: string): [string, number][] {
  try {
    const parsed: unknown = JSON.parse(text)
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return []
    return Object.entries(parsed as Record<string, number>)
  } catch {
    return []
  }
}

export function QualityRows({
  runs,
  selected,
  materials,
  onSelect
}: {
  runs: Run[]
  selected: string | null
  materials: Map<string, number>
  onSelect: (model: string, dataset: string) => void
}): JSX.Element {
  const groups = new Map<string, Run[]>()
  for (const run of runs) {
    const dataset = run.quality?.dataset ?? 'unknown'
    groups.set(dataset, [...(groups.get(dataset) ?? []), run])
  }
  if (groups.size === 0) {
    return (
      <p className="note" style={{ padding: '12px 3px' }}>
        Nothing scored yet. Declaring a dataset and running it is New benchmark → Quality.
      </p>
    )
  }
  return (
    <>
      {[...groups.entries()].map(([dataset, entries]) => {
        const ordered = [...entries].sort(
          (left, right) => (right.quality?.accuracy ?? -1) - (left.quality?.accuracy ?? -1)
        )
        const best = ordered[0]?.quality?.accuracy ?? null
        return (
          <div key={dataset}>
            {/* The key on the header is what authorises the comparison: change n or the seed
                and it becomes a different axis. */}
            <div className="qhead">
              <b>{dataset}</b>
              <span>{ordered[0]?.key ?? ''}</span>
            </div>
            <ul className="runs">
              {ordered.map((run) => {
                const material = materials.get(run.model)
                const accuracy = run.quality?.accuracy ?? null
                return (
                  <li
                    key={run.id}
                    className={`run q ${material === undefined ? '' : `m${material}`}${run.model === selected ? ' on' : ''}${run.state === 'ok' ? '' : ' nofit'}`}
                    onClick={() => onSelect(run.model, dataset)}
                  >
                    <div className="id">
                      <span className="top">
                        {material !== undefined && <i className="inband" />}
                        <b>{run.model}</b>
                      </span>
                      <span>{run.key}</span>
                    </div>
                    <div className={`num${accuracy === null ? ' dim' : ''}`}>
                      <b>
                        {accuracy === null ? '—' : (accuracy * 100).toFixed(1)}
                        {accuracy !== null && (
                          <em style={{ fontStyle: 'normal', fontSize: 11 }}>%</em>
                        )}
                      </b>
                      <span>accuracy</span>
                    </div>
                    <div className="num dim">
                      <b>{run.quality?.items ?? '—'}</b>
                      <span>items</span>
                    </div>
                    <div className="spark" />
                    <div className="state">
                      <div className="l1">
                        {accuracy === null ? (
                          <>
                            <i className="dot" />
                            {(run.reason ?? 'not run').replace(/_/g, ' ')}
                          </>
                        ) : accuracy === best ? (
                          <>
                            <i className="dot ok" />
                            best here
                          </>
                        ) : (
                          <>
                            <i className="dot ok" />
                            {((accuracy - (best ?? 0)) * 100).toFixed(1)} pt off best
                          </>
                        )}
                      </div>
                      <div className="l2">{run.quality?.scoring ?? '—'}</div>
                    </div>
                  </li>
                )
              })}
            </ul>
          </div>
        )
      })}
    </>
  )
}


export function QualityResult({
  run,
  others,
  onDataset
}: {
  run: Run
  others: Run[]
  onDataset: (dataset: string) => void
}): JSX.Element {
  const body = run.quality
  const areas = breakdown(body?.breakdown ?? '{}')
  return (
    <>
      <div className="grouplab">Score</div>
      <div className="kvs">
        <div className="kvr">
          <span className="k">Accuracy</span>
          <span className="v">{percent(body?.accuracy ?? null)}%</span>
        </div>
        <div className="kvr">
          <span className="k">Correct</span>
          <span className="v">
            {body?.correct ?? '—'} / {body?.items ?? '—'}
          </span>
        </div>
        <div className="kvr">
          <span className="k">Scoring</span>
          <span className="v">{body?.scoring ?? '—'}</span>
        </div>
        <div className="kvr">
          <span className="k">Wall</span>
          <span className="v">{body?.wall_s === null || body === null ? '—' : `${body.wall_s?.toFixed(0)} s`}</span>
        </div>
      </div>

      <div className="grouplab">Comparison key</div>
      <div className="keyline">
        key <b>{run.key}</b>
      </div>
      <p className="note" style={{ marginTop: 6 }}>
        Only comparable to runs carrying exactly this key. Change n or the seed and it becomes
        a different axis.
      </p>

      {run.state !== 'ok' && (
        <>
          <div className="grouplab">Did not run</div>
          <div className="err">
            {run.reason === 'not_implemented'
              ? 'Quality scoring is not implemented yet: the log-likelihood pass over a dataset’s continuations is task 59.8. The shape is recorded so nothing about it has to be typed twice.'
              : (run.reason ?? 'unknown')}
          </div>
        </>
      )}

      {areas.length > 0 && (
        <>
          <div className="grouplab">By area</div>
          <div className="kvs">
            {areas.map(([area, value]) => (
              <div className="kvr" key={area}>
                <span className="k">{area}</span>
                <span className="v">{(value * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </>
      )}

      {others.length > 0 && (
        <>
          <div className="grouplab">This checkpoint elsewhere</div>
          <div className="kvs">
            {others.map((other) => (
              <div
                className="mini"
                key={other.id}
                onClick={() => onDataset(other.quality?.dataset ?? '')}
              >
                <span className="t">{other.quality?.dataset}</span>
                <span className="w">{percent(other.quality?.accuracy ?? null)}%</span>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  )
}

const EMPTY: Omit<DatasetDefinition, 'builtin'> = {
  id: '',
  use: 'multiple_choice',
  repo: '',
  config: null,
  split: 'test',
  template: '{question}\nAnswer:',
  columns: { question: 'question', choices: 'choices', answer: 'answer' },
  size: null
}

/* Declaring a dataset, with the first item rendered beside the form.

   The preview is the point of the form. A column mapped to the wrong name is a silent
   error: the run completes, twenty minutes later, at chance accuracy, and nothing says
   why. Here it fails in a second, by name, with the columns the repository actually
   ships. */
export function DatasetForm({
  onClose,
  onSaved
}: {
  onClose: () => void
  onSaved: () => void
}): JSX.Element {
  const [draft, setDraft] = useState(EMPTY)
  const [shown, setShown] = useState<Preview | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEscape(onClose)

  const field = (key: keyof typeof EMPTY, value: string): void =>
    setDraft({ ...draft, [key]: value === '' && key === 'config' ? null : value })

  const column = (role: string, value: string): void =>
    setDraft({ ...draft, columns: { ...draft.columns, [role]: value } })

  const look = async (): Promise<void> => {
    setBusy(true)
    setFailure(null)
    try {
      setShown(await preview(draft))
    } catch (error) {
      setShown(null)
      setFailure(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  const keep = async (): Promise<void> => {
    setBusy(true)
    setFailure(null)
    try {
      await saveDataset(draft)
      onSaved()
      onClose()
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="sheet" onClick={onClose}>
      <div className="sheetc" onClick={(event) => event.stopPropagation()}>
        <div className="sheeth">
          <b>Add a dataset from the Hub</b>
          <span className="arrow">the preview is what catches a wrong mapping</span>
        </div>
        <div className="sheetb fixed">
          <div className="sheetl">
            <label className="fld">
              <span>Id</span>
              <input value={draft.id} onChange={(event) => field('id', event.target.value)} />
            </label>
            <label className="fld">
              <span>Repository</span>
              <input value={draft.repo} onChange={(event) => field('repo', event.target.value)} />
            </label>
            <label className="fld">
              <span>Config</span>
              <input
                value={draft.config ?? ''}
                onChange={(event) => field('config', event.target.value)}
              />
            </label>
            <label className="fld">
              <span>Split</span>
              <input value={draft.split} onChange={(event) => field('split', event.target.value)} />
            </label>
            <div className="seg">
              {(['multiple_choice', 'generation', 'corpus'] as const).map((use) => (
                <button
                  key={use}
                  className={draft.use === use ? 'on' : undefined}
                  onClick={() => setDraft({ ...draft, use })}
                >
                  {use === 'multiple_choice' ? 'Choice' : use === 'generation' ? 'Generate' : 'Corpus'}
                </button>
              ))}
            </div>
          </div>
          <div className="sheetr">
            <div className="grouplab">Columns</div>
            {['question', 'choices', 'answer', 'group'].map((role) => (
              <label className="fld" key={role}>
                <span>{role}</span>
                <input
                  value={draft.columns[role] ?? ''}
                  onChange={(event) => column(role, event.target.value)}
                  placeholder="a column, or a.b to reach into a struct"
                />
              </label>
            ))}
            <label className="fld">
              <span>Template</span>
              <textarea
                value={draft.template}
                rows={2}
                onChange={(event) => field('template', event.target.value)}
              />
            </label>
            {failure !== null && <div className="err">{failure}</div>}
            {shown !== null && (
              <div className="prev">
                <div className="grouplab">First item, as the model reads it</div>
                <pre>{shown.prompt}</pre>
                <div className="grouplab">Continuations scored</div>
                <ul>
                  {shown.continuations.map((option, index) => (
                    <li key={index} className={String(index) === shown.answer ? 'right' : undefined}>
                      {option}
                    </li>
                  ))}
                </ul>
                <p className="empty quiet">
                  {shown.rows} rows · columns: {shown.columns.join(', ')}
                </p>
              </div>
            )}
          </div>
        </div>
        <div className="sheetf">
          <button className="btn" onClick={() => void look()} disabled={busy}>
            Preview
          </button>
          <button
            className="btn pri"
            onClick={() => void keep()}
            disabled={busy || draft.id === '' || draft.repo === ''}
          >
            Save definition
          </button>
          <button className="btn quiet" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}

import { type JSX, useEffect, useRef, useState } from 'react'
import { cancelJob, isFinished, jobEvents, type JobView, type Progress } from '../../api/http'
import { size, sizePair } from './format'

interface Props {
  id: string
  /* Known when this window started the pull; a job adopted from `/admin/jobs` has only
     what its progress message says. */
  repo: string | null
  /* `null` when the stream ended without a terminal frame — the daemon went away. */
  onDone: (view: JobView | null) => void
}

const START: Progress = { message: '', completed: 0, total: null }

export function DownloadJob({ id, repo, onDone }: Props): JSX.Element {
  const [view, setView] = useState<JobView | null>(null)
  const [stopping, setStopping] = useState(false)
  const done = useRef(onDone)

  useEffect(() => {
    done.current = onDone
  })

  /* One stream per job id, and the id is what this component is keyed by: a job
     arriving or leaving never reopens anybody else's. */
  useEffect(() => {
    const abort = new AbortController()
    const watch = async (): Promise<void> => {
      try {
        for await (const frame of jobEvents(id, abort.signal)) {
          setView(frame)
          if (isFinished(frame)) {
            done.current(frame)
            return
          }
        }
        if (!abort.signal.aborted) done.current(null)
      } catch {
        if (!abort.signal.aborted) done.current(null)
      }
    }
    void watch()
    return () => abort.abort()
  }, [id])

  /* The DELETE only sets the flag; `cancelled` arrives as the stream's last frame. */
  const cancel = (): void => {
    setStopping(true)
    void cancelJob(id).catch(() => setStopping(false))
  }

  const progress = view?.progress ?? START
  const total = progress.total
  const percent = total ? Math.round(Math.min(1, progress.completed / total) * 100) : null

  return (
    <div className="card dljob" title={repo ?? undefined}>
      <h3>
        Downloading
        <span className="tn">
          {total ? sizePair(progress.completed, total) : size(progress.completed)}
        </span>
        <button className="btn danger" onClick={cancel} disabled={stopping}>
          {stopping ? 'Cancelling…' : 'Cancel'}
        </button>
      </h3>
      <div className="gauge">
        <div className="bar">
          {percent !== null && <div className="fill" style={{ width: `${percent}%` }} />}
        </div>
        <div className="sub">
          <span className="mono">{progress.message || repo || 'starting…'}</span>
          <span>{percent === null ? '—' : `${percent}%`}</span>
        </div>
      </div>
    </div>
  )
}

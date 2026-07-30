import { type FormEvent, type JSX, useCallback, useEffect, useState } from 'react'
import { HttpError, type JobView, listJobs } from '../../api/http'
import { dropStaged, type HubModel, listStaged, pull, searchHub, type Staged } from './api'
import { DownloadJob } from './DownloadJob'
import { size } from './format'
import { HubDetail } from './HubDetail'
import './download.css'

/* A download this window is watching. `repo` is null for a job that was already running
   when the view opened — `/admin/jobs` says the kind, not what is being fetched. */
interface Tracked {
  id: string
  repo: string | null
}

interface Note {
  text: string
  bad: boolean
}

const reason = (error: unknown): string =>
  error instanceof HttpError ? error.detail : error instanceof Error ? error.message : String(error)

const name = (id: string): string => id.slice(id.lastIndexOf('/') + 1)

const origin = (hit: HubModel): string[] => {
  const cut = hit.id.lastIndexOf('/')
  return cut === -1 ? [] : [hit.id.slice(0, cut)]
}

const describe = (hit: HubModel): string =>
  [
    ...origin(hit),
    ...(hit.downloads === null ? [] : [`${hit.downloads.toLocaleString()} downloads`])
  ].join(' · ')

export function Download(): JSX.Element {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HubModel[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [note, setNote] = useState<Note | null>(null)
  const [jobs, setJobs] = useState<Tracked[]>([])
  const [staged, setStaged] = useState<Staged[]>([])
  const [selected, setSelected] = useState<HubModel | null>(null)

  const refreshStaged = useCallback((): void => {
    void listStaged()
      .then(setStaged)
      .catch(() => setStaged([]))
  }, [])

  /* Whatever was already downloading when the view opened; each row then follows its
     own SSE. */
  useEffect(() => {
    let live = true
    void listJobs(true)
      .then((all) => {
        if (live) {
          setJobs(
            all.filter((job) => job.kind === 'download').map((job) => ({ id: job.id, repo: null }))
          )
        }
      })
      .catch(() => {})
    refreshStaged()
    return () => {
      live = false
    }
  }, [refreshStaged])

  const search = async (event: FormEvent): Promise<void> => {
    event.preventDefault()
    const wanted = query.trim()
    if (wanted === '') return
    setSearching(true)
    try {
      setResults(await searchHub(wanted))
      setNote(null)
    } catch (error) {
      setResults(null)
      setNote({ text: reason(error), bad: true })
    } finally {
      setSearching(false)
    }
  }

  const start = async (repo: string): Promise<void> => {
    try {
      const job = await pull(repo)
      setJobs((current) => [...current, { id: job.id, repo }])
      setNote(null)
    } catch (error) {
      setNote({ text: reason(error), bad: true })
    }
  }

  const finished = useCallback(
    (job: Tracked, view: JobView | null): void => {
      setJobs((current) => current.filter((entry) => entry.id !== job.id))
      const what = job.repo ?? 'the download'
      if (view === null) setNote({ text: `lost the stream watching ${what}`, bad: true })
      else if (view.state === 'ok') setNote({ text: `${what} is on disk`, bad: false })
      else if (view.state === 'cancelled')
        setNote({ text: `${what} was cancelled — its staged bytes went with it`, bad: false })
      else setNote({ text: view.error ?? `${what} failed`, bad: true })
      refreshStaged()
    },
    [refreshStaged]
  )

  const discard = async (repo: string): Promise<void> => {
    try {
      await dropStaged(repo)
      setNote({ text: `dropped what was staged for ${repo}`, bad: false })
    } catch (error) {
      setNote({ text: reason(error), bad: true })
    }
    refreshStaged()
  }

  const stagedBytes = staged.reduce((total, entry) => total + entry.bytes_on_disk, 0)

  /* The detail replaces the list, as in Models. The job rows unmount with it, which is
     safe: each stream aborts on unmount and reopens with the job's current state on the
     way back. */
  if (selected !== null) {
    return (
      <HubDetail
        hit={selected}
        busy={jobs.some((job) => job.repo === selected.id)}
        onPull={() => {
          setSelected(null)
          void start(selected.id)
        }}
        onBack={() => setSelected(null)}
      />
    )
  }

  return (
    <>
      <div className="vhead">
        <h1>Download</h1>
        <form className="trail dlform" onSubmit={(event) => void search(event)}>
          <input
            className="input dlsearch"
            placeholder="Search the Hub…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button className="btn solid" type="submit" disabled={searching || query.trim() === ''}>
            {searching ? 'Searching…' : 'Search'}
          </button>
        </form>
      </div>

      <div className="vbody dlbody">
        {note !== null && <p className={note.bad ? 'dlnote bad' : 'dlnote'}>{note.text}</p>}

        {jobs.map((job) => (
          <DownloadJob
            key={job.id}
            id={job.id}
            repo={job.repo}
            onDone={(view) => finished(job, view)}
          />
        ))}

        {/* Results come before the partials: they are the answer to the click that just
            happened, and a long partials card was burying them below the fold. */}
        {results !== null &&
          (results.length === 0 ? (
            <div className="listfoot">nothing on the Hub matches that</div>
          ) : (
            <div className="list">
              {results.map((hit) => {
                const busy = jobs.some((job) => job.repo === hit.id)
                return (
                  <div
                    className="row"
                    role="button"
                    tabIndex={0}
                    key={hit.id}
                    onClick={() => setSelected(hit)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        setSelected(hit)
                      }
                    }}
                  >
                    <div className="id">
                      <b>{name(hit.id)}</b>
                      <span>{describe(hit)}</span>
                    </div>
                    <button
                      className="btn solid dlpull"
                      disabled={busy}
                      onClick={(event) => {
                        event.stopPropagation()
                        void start(hit.id)
                      }}
                    >
                      {busy ? 'Pulling…' : 'Pull'}
                    </button>
                  </div>
                )
              })}
            </div>
          ))}

        {staged.length > 0 && (
          <div className="card">
            <h3>
              Partial downloads<span className="tn">{size(stagedBytes)} on disk</span>
            </h3>
            {staged.map((entry) => (
              <div className="staged" key={entry.repo}>
                <code>{entry.repo}</code>
                <span className="meta tn">{size(entry.bytes_on_disk)}</span>
                <button className="btn danger" onClick={() => void discard(entry.repo)}>
                  Discard
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  )
}

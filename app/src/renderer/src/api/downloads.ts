/* Every download the daemon is running, keyed by the repository it is fetching.

   It lives above the views because a download outlives the screen that started it: the tile
   in Library, the ring in the title bar and a reload three minutes later all read the same
   frames. The boot pass is what makes the reload work — `listJobs(true)` answers which
   downloads are running, and each one's `subject.model` says which tile it belongs under. */

import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { cancelJob, jobEvents, listJobs, send, type JobState, type JobView } from './http'

/* Dock badge and Notification Center live on the shell; without one (bare dev:web,
   CLI) both calls 404 and the download is none the worse. */
const shell = (path: string, body: unknown): void => {
  void send('POST', path, body).catch(() => {})
}

export interface Pull {
  job: string
  repo: string
  state: JobState
  message: string
  completed: number
  /* Absent until the first response: a bar drawn from a guessed total is worse than none. */
  total: number | null
  error: string | null
  /* Bytes per second, null until a second frame gives it something to divide. */
  rate: number | null
  at: number
}

export interface Downloads {
  /* By repository, which is what a tile has and a job id is not. */
  active: Map<string, Pull>
  /* What a `pull` hands back, so the tile starts moving before the next poll. */
  follow: (job: JobView) => void
  cancel: (job: string) => Promise<void>
  /* A terminal failure stays on screen until it is acted on; this is the acting on it. */
  forget: (repo: string) => void
}

const DownloadsContext = createContext<Downloads | null>(null)
export const DownloadsProvider = DownloadsContext.Provider

export function useDownloadsContext(): Downloads {
  const downloads = useContext(DownloadsContext)
  if (downloads === null) throw new Error('useDownloadsContext outside the provider')
  return downloads
}

export const eta = (pull: Pull): number | null =>
  pull.total === null || pull.rate === null || pull.rate <= 0
    ? null
    : (pull.total - pull.completed) / pull.rate

/* Smoothed, because a shard already in the staging cache completes in a single frame and the
   instantaneous reading there is gigabytes per second. */
const SMOOTHING = 0.35

function rate(previous: Pull | undefined, frame: JobView): number | null {
  if (previous === undefined) return null
  const seconds = frame.updated_at - previous.at
  if (seconds <= 0) return previous.rate
  const instant = (frame.progress.completed - previous.completed) / seconds
  if (previous.rate === null) return instant
  return previous.rate + SMOOTHING * (instant - previous.rate)
}

function fold(pulls: Map<string, Pull>, repo: string, frame: JobView): Map<string, Pull> {
  const next = new Map(pulls)
  const previous = next.get(repo)
  /* A finished download is the catalog's now, and a cancelled one took its bytes with it;
     only a failure has something left to offer, so only a failure stays. */
  if (frame.state === 'ok' || frame.state === 'cancelled') {
    next.delete(repo)
    return next
  }
  next.set(repo, {
    job: frame.id,
    repo,
    state: frame.state,
    message: frame.progress.message,
    completed: frame.progress.completed,
    total: frame.progress.total,
    error: frame.error,
    rate: rate(previous, frame),
    at: frame.updated_at
  })
  return next
}

export function useDownloads(onSettled: () => void): Downloads {
  const [active, setActive] = useState<Map<string, Pull>>(new Map())
  const followed = useRef<Set<string>>(new Set())
  const settled = useRef(onSettled)
  settled.current = onSettled

  const follow = useCallback((job: JobView) => {
    if (job.kind !== 'download' || followed.current.has(job.id)) return
    followed.current.add(job.id)
    const repo = job.subject.model
    setActive((pulls) => fold(pulls, repo, job))
    void (async () => {
      try {
        for await (const frame of jobEvents(job.id)) {
          if (frame.state === 'ok') shell('/desktop/notify', { title: 'Download complete', body: repo })
          setActive((pulls) => fold(pulls, repo, frame))
        }
      } catch {
        /* The stream dying is not itself news: the job's own state is what the next poll
           reads, and a daemon that went away is already drawn by the engine chip. */
      } finally {
        followed.current.delete(job.id)
        settled.current()
      }
    })()
  }, [])

  /* What was already running before this window opened, or before this reload. */
  useEffect(() => {
    void listJobs(true)
      .then((jobs) => {
        for (const job of jobs) follow(job)
      })
      .catch(() => {})
  }, [follow])

  /* The badge counts what is still moving; a failure waiting to be acted on is not. */
  useEffect(() => {
    const moving = [...active.values()].filter((pull) => pull.state !== 'error').length
    shell('/desktop/badge', { count: moving })
  }, [active])

  const forget = useCallback((repo: string) => {
    setActive((pulls) => {
      const next = new Map(pulls)
      next.delete(repo)
      return next
    })
  }, [])

  return {
    active,
    follow,
    cancel: (job) => cancelJob(job).then(() => {}),
    forget
  }
}

/* The download screen's routes, typed off `sideros_server/download.py`.
   The job side (202 body, SSE frames, cancel) is the shared one in api/http.ts. */

import { HttpError, json, send, type JobView } from '../../api/http'
import type { CheckpointFile } from '../models/api'

/* GET /admin/hub/models — `downloads`/`likes` are the Hub's, and the Hub omits them. */
export interface HubModel {
  id: string
  downloads: number | null
  likes: number | null
}

/* GET /admin/hub/staging — bytes with no model to show for them. */
export interface Staged {
  repo: string
  bytes_on_disk: number
}

export const searchHub = (query: string, limit = 20): Promise<HubModel[]> =>
  json<HubModel[]>(`/admin/hub/models?query=${encodeURIComponent(query)}&limit=${limit}`)

/* POST /admin/models — 202 with the job's first frame; 409 when the repository is
   already downloading, already on disk, or being collected. */
export const pull = (repo: string, quant?: string): Promise<JobView> =>
  send<JobView>('POST', '/admin/models', quant === undefined ? { repo } : { repo, quant })

export const listStaged = (): Promise<Staged[]> => json<Staged[]>('/admin/hub/staging')

/* The path parameter is a `:path`, so the slash in `org/name` stays a slash. */
const at = (repo: string): string => repo.split('/').map(encodeURIComponent).join('/')

export const dropStaged = (repo: string): Promise<void> =>
  send<void>('DELETE', `/admin/hub/staging/${at(repo)}`)

/* What a pull would fetch, priced by the Hub before any byte moves. */
export const hubFiles = (repo: string): Promise<CheckpointFile[]> =>
  json<CheckpointFile[]>(`/admin/hub/models/${at(repo)}/files`)

/* The repository's README raw — null when it ships none, which is not an error. */
export async function hubCard(repo: string): Promise<string | null> {
  const answer = await fetch(`/admin/hub/models/${at(repo)}/card`)
  if (answer.status === 404) return null
  if (!answer.ok) throw new HttpError(answer.status, answer.statusText)
  return await answer.text()
}

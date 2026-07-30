/* A repository before it costs anything: the card, the files a pull would fetch and what
   they weigh — the same block the local detail shows, read through the hub routes. */

import { type JSX, useState } from 'react'
import type { CheckpointFile } from '../models/api'
import { bytes, CardBlock } from '../models/Card'
import type { HubModel } from './api'

interface Props {
  hit: HubModel
  busy: boolean
  onPull: () => void
  onBack: () => void
}

export function HubDetail({ hit, busy, onPull, onBack }: Props): JSX.Element {
  const [total, setTotal] = useState<number | null>(null)
  const cut = hit.id.lastIndexOf('/')
  const meta = [
    ...(cut === -1 ? [] : [hit.id.slice(0, cut)]),
    ...(hit.downloads === null ? [] : [`${hit.downloads.toLocaleString()} downloads`]),
    ...(hit.likes === null ? [] : [`${hit.likes.toLocaleString()} likes`]),
    ...(total === null ? [] : [`${bytes(total)} to download`])
  ].join(' · ')

  return (
    <>
      <div className="vhead">
        <button className="iconbtn" title="Back" aria-label="Back" onClick={onBack}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
        <h1>{hit.id.slice(cut + 1)}</h1>
        <span className="hubmeta tn">{meta}</span>
        <div className="trail">
          <button className="btn solid" disabled={busy} onClick={onPull}>
            {busy ? 'Pulling…' : 'Pull'}
          </button>
        </div>
      </div>
      <div className="vbody">
        <CardBlock
          id={hit.id}
          hub
          onFiles={(files: CheckpointFile[]) =>
            setTotal(files.reduce((sum, file) => sum + file.size, 0))
          }
        />
      </div>
    </>
  )
}

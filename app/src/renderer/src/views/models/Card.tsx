/* The checkpoint's own README, rendered where the checkpoint lives: a hub card is
   arbitrary markdown with raw HTML, so it renders through the sanitizer — it runs in the
   same webview that can reach the local API, and scripts never survive the pipeline.
   Relative images resolve against the daemon's asset route; links leave through the
   system browser, because navigating the webview would replace the app. */

import { Fragment, type JSX, useEffect, useState } from 'react'
import Markdown, { defaultUrlTransform } from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import { hubCard, hubFiles } from '../download/api'
import * as api from './api'

const openExternal = (url: string): void => {
  void fetch('/desktop/open', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ url })
  }).catch(() => {})
}

export function bytes(size: number): string {
  if (size >= 1024 ** 3) return `${(size / 1024 ** 3).toFixed(2)} GB`
  if (size >= 1024 ** 2) return `${(size / 1024 ** 2).toFixed(1)} MB`
  if (size >= 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${size} B`
}

/* A YAML subset: cards use scalars and string lists, and a card that uses more keeps
   its extra shapes in the Source tab rather than breaking the chips. */
interface Frontmatter {
  scalars: Record<string, string>
  tags: string[]
}

function split(raw: string): { front: Frontmatter | null; body: string } {
  const lines = raw.split(/\r?\n/)
  if (lines[0]?.trim() !== '---') return { front: null, body: raw }
  const close = lines.findIndex((line, i) => i > 0 && line.trim() === '---')
  if (close < 0) return { front: null, body: raw }
  const scalars: Record<string, string> = {}
  const tags: string[] = []
  let list: string | null = null
  for (const line of lines.slice(1, close)) {
    const key = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/)
    if (key) {
      list = key[2] === '' ? key[1] : null
      if (key[2] !== '') scalars[key[1]] = key[2]
      continue
    }
    const item = line.match(/^\s*-\s+(.+)$/)
    if (item && list === 'tags') tags.push(item[1])
  }
  return { front: { scalars, tags }, body: lines.slice(close + 1).join('\n') }
}

function Chips({ front }: { front: Frontmatter }): JSX.Element {
  const { scalars, tags } = front
  const base = scalars['base_model']
  const relation = scalars['base_model_relation']
  return (
    <div className="fm">
      {scalars['license'] && (
        <span className="lic">
          <i>license</i>
          {scalars['license']}
        </span>
      )}
      {base && (
        <span className="base">
          <i>base</i>
          {relation ? `${base} · ${relation}` : base}
        </span>
      )}
      {scalars['library_name'] && (
        <span>
          <i>library</i>
          {scalars['library_name']}
        </span>
      )}
      {scalars['pipeline_tag'] && <span>{scalars['pipeline_tag']}</span>}
      {tags.map((tag) => (
        <span key={tag}>{tag}</span>
      ))}
    </div>
  )
}

function Rendered({ body, assets }: { body: string; assets: string }): JSX.Element {
  const transform = (url: string): string => {
    const absolute = /^[a-z][a-z0-9+.-]*:|^\/\/|^#/i.test(url)
    const resolved = absolute ? url : `${assets}/${url.replace(/^\.?\//, '')}`
    return defaultUrlTransform(resolved)
  }
  return (
    <div className="mdcard">
      <Markdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeSanitize]}
        urlTransform={transform}
        components={{
          a: ({ href, children }) => (
            <a
              href={href}
              onClick={(event) => {
                event.preventDefault()
                if (href?.startsWith('http')) openExternal(href)
              }}
            >
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="tablewrap">
              <table>{children}</table>
            </div>
          )
        }}
      >
        {body}
      </Markdown>
    </div>
  )
}

/* The source highlights exactly what the reader would check: where the frontmatter
   ends, and which keys it declares. */
function Source({ raw }: { raw: string }): JSX.Element {
  const lines = raw.split(/\r?\n/)
  const opens = lines[0]?.trim() === '---'
  const closes = opens ? lines.findIndex((line, i) => i > 0 && line.trim() === '---') : -1
  return (
    <pre className="srcview">
      {lines.map((line, i) => {
        const fence = opens && (i === 0 || i === closes)
        const key = closes > 0 && i > 0 && i < closes ? line.match(/^([A-Za-z0-9_-]+:)(.*)$/) : null
        return (
          <Fragment key={i}>
            {fence ? (
              <span className="fence">{line}</span>
            ) : key ? (
              <>
                <span className="k">{key[1]}</span>
                {key[2]}
              </>
            ) : /^#{1,6}\s/.test(line) ? (
              <span className="h">{line}</span>
            ) : (
              line
            )}
            {'\n'}
          </Fragment>
        )
      })}
    </pre>
  )
}

const SHARD = /^model-\d+-of-\d+\.safetensors$/

function Files({ files, tail }: { files: api.CheckpointFile[]; tail: string }): JSX.Element {
  const [open, setOpen] = useState(false)
  const shards = files.filter((file) => SHARD.test(file.name))
  const total = files.reduce((sum, file) => sum + file.size, 0)
  const row = (file: api.CheckpointFile, shard: boolean): JSX.Element => (
    <div className={shard ? 'frow shard' : 'frow'} key={file.name}>
      <span>{file.name}</span>
      {file.name === 'README.md' && <span className="tag">card</span>}
      <span className="size">{bytes(file.size)}</span>
    </div>
  )
  return (
    <div className="files">
      {files.map((file) => {
        /* The shard run collapses into one row at its first member; the rest ride it. */
        if (!SHARD.test(file.name)) return row(file, false)
        if (file.name !== shards[0].name) return null
        return (
          <Fragment key="shards">
            <button className="frow group" onClick={() => setOpen((was) => !was)}>
              <svg
                className={open ? 'chev open' : 'chev'}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="9 18 15 12 9 6" />
              </svg>
              <span>model-*-of-{shards[0].name.split('-of-')[1]}</span>
              <span className="tag">{shards.length} shards</span>
              <span className="size">{bytes(shards.reduce((sum, file) => sum + file.size, 0))}</span>
            </button>
            {open && shards.map((shard) => row(shard, true))}
          </Fragment>
        )
      })}
      <div className="filefoot">
        <span>{files.length} files</span>
        <span>
          {bytes(total)} {tail}
        </span>
      </div>
    </div>
  )
}

type Tab = 'card' | 'files' | 'source'

/* `hub` reads the repository through the daemon's hub routes instead of the catalog's:
   same card, same listing, before any byte is on disk. */
export function CardBlock({
  id,
  store,
  hub = false,
  shown: asked,
  onFiles,
  onReady
}: {
  id: string
  store?: string
  hub?: boolean
  /** Which tab to show. Given one, the block drops its own tab bar: the caller's
      inspector already carries it. */
  shown?: Tab
  /** The one listing this block fetches, shared out so a caller can price it. */
  onFiles?: (files: api.CheckpointFile[]) => void
  /** Fired once card and listing have both settled, however they went. A caller that
      reveals several blocks together needs to know this one has stopped being blank. */
  onReady?: () => void
}): JSX.Element | null {
  const [raw, setRaw] = useState<string | null | undefined>(undefined)
  const [files, setFiles] = useState<api.CheckpointFile[]>([])
  const [tab, setTab] = useState<Tab>('card')

  useEffect(() => {
    let live = true
    setRaw(undefined)
    setFiles([])
    setTab('card')
    const card = (hub ? hubCard(id) : api.getCard(id)).then(setRaw, () => setRaw(null))
    const listing = (hub ? hubFiles(id) : api.listFiles(id)).then(
      (listed) => {
        setFiles(listed)
        onFiles?.(listed)
      },
      () => setFiles([])
    )
    void Promise.all([card, listing]).then(() => live && onReady?.())
    // `onFiles` and `onReady` stay out of the deps: they are notifications, not sources, and
    // refetching because the caller re-rendered would ask the Hub again for the same listing.
    return () => {
      live = false
    }
  }, [id, hub])

  if (raw === undefined) return null
  if (raw === null && files.length === 0) return null

  const { front, body } = split(raw ?? '')
  const tabs: Tab[] = raw === null ? ['files'] : ['card', 'files', 'source']
  const shown = raw === null ? 'files' : (asked ?? tab)
  const linked = /^[\w.-]+\/[\w.-]+$/.test(id)

  return (
    <div className="cardblock">
      {asked === undefined && (
      <div className="mdhead">
        {tabs.length > 1 && (
          <div className="seg">
            {tabs.map((name) => (
              <button
                className={shown === name ? 'on' : undefined}
                key={name}
                onClick={() => setTab(name)}
              >
                {name === 'card' ? 'Card' : name === 'files' ? 'Files' : 'Source'}
              </button>
            ))}
          </div>
        )}
        {tabs.length === 1 && <h3>Files</h3>}
        <div className="grow" />
        {linked && (
          <button className="hublink" onClick={() => openExternal(`https://huggingface.co/${id}`)}>
            hf.co/{id} ↗
          </button>
        )}
      </div>
      )}
      {shown === 'card' && raw !== null && (
        <>
          {front !== null && <Chips front={front} />}
          <Rendered
            body={body}
            assets={hub ? `https://huggingface.co/${id}/resolve/main` : api.assetBase(id)}
          />
        </>
      )}
      {shown === 'files' && (
        <Files files={files} tail={hub ? 'to download' : `on disk · ${store ?? ''}`} />
      )}
      {shown === 'source' && raw !== null && <Source raw={raw} />}
    </div>
  )
}

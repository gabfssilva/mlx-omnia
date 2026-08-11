/* The window's origin in prod: `sideros://app/` handled here, serving the built
   renderer plus a reverse proxy for /admin and /api. Nothing here knows about the
   daemon beyond its base URL — and nothing buffers, because a token stream and a job's
   SSE both have to arrive frame by frame.

   It is a module of its own so the two rules it encodes (which prefixes are the
   daemon's, and that every other path is the app) can be exercised without opening a
   window. */

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { readFile } from 'node:fs/promises'
import { join, normalize } from 'node:path'
import { app, BrowserWindow, dialog, nativeTheme, Notification, shell } from 'electron'

export const SCHEME = 'sideros'
export const ORIGIN = `${SCHEME}://app`

export interface Origin {
  daemon: string
  dist: string
  control: Shell
}

/* What the window can ask of the process hosting it — main.ts implements it, this module
   only routes to it. The daemon half is authority over the one *it* spawned: a daemon
   that was already up when the app arrived is not ours to kill, and `stop`/`restart`
   refuse with the reason instead. `ready` is the renderer saying its first poll landed,
   which is what puts the window on screen. */
export interface Shell {
  owned(): boolean
  stop(): Promise<void>
  restart(): Promise<void>
  ready(): void
}

const TYPES: Record<string, string> = {
  html: 'text/html; charset=utf-8',
  js: 'text/javascript; charset=utf-8',
  css: 'text/css; charset=utf-8',
  json: 'application/json; charset=utf-8',
  svg: 'image/svg+xml',
  png: 'image/png',
  jpg: 'image/jpeg',
  woff2: 'font/woff2',
  ico: 'image/x-icon'
}

export const proxied = (pathname: string): boolean =>
  pathname === '/admin' ||
  pathname === '/api' ||
  pathname.startsWith('/admin/') ||
  pathname.startsWith('/api/')

export function handler(origin: Origin): (request: Request) => Promise<Response> {
  return async (request) => {
    const url = new URL(request.url)
    if (proxied(url.pathname)) return await forward(origin.daemon, request, url)
    return (await desktop(request, url, origin.control)) ?? (await asset(origin.dist, url.pathname))
  }
}

/* The routes that are the shell's own rather than the daemon's. In prod they arrive
   through the scheme handler above; in dev the window is on Vite, which proxies
   /desktop to the listener below. */
async function desktop(
  request: Request,
  url: URL,
  control: Shell
): Promise<Response | undefined> {
  if (url.pathname === '/desktop/open' && request.method === 'POST') return await open(request)
  if (url.pathname === '/desktop/ready' && request.method === 'POST') {
    control.ready()
    return Response.json({ ok: true })
  }
  if (url.pathname === '/desktop/daemon' && request.method === 'GET') {
    return Response.json({ owned: control.owned() })
  }
  if (url.pathname === '/desktop/daemon/stop' && request.method === 'POST') {
    return await act(() => control.stop())
  }
  if (url.pathname === '/desktop/daemon/restart' && request.method === 'POST') {
    return await act(() => control.restart())
  }
  if (url.pathname === '/desktop/theme' && request.method === 'POST') return await theme(request)
  if (url.pathname === '/desktop/confirm' && request.method === 'POST') return await confirm(request)
  if (url.pathname === '/desktop/notify' && request.method === 'POST') return await notify(request)
  if (url.pathname === '/desktop/badge' && request.method === 'POST') return await badge(request)
  return undefined
}

/* The renderer's Appearance choice, mirrored into the process: native menus, sheets and
   dialogs open on the side the app is drawn on, not the side the OS happens to be on. */
async function theme(request: Request): Promise<Response> {
  const { theme } = (await request.json()) as { theme?: unknown }
  if (theme !== 'light' && theme !== 'dark' && theme !== 'system') {
    return Response.json({ detail: 'theme is light, dark or system' }, { status: 400 })
  }
  nativeTheme.themeSource = theme
  return Response.json({ ok: true })
}

/* A destructive action asks as a sheet off the title bar — the parent window is what
   makes showMessageBox a sheet rather than a floating alert. */
async function confirm(request: Request): Promise<Response> {
  const { message, detail, action, danger } = (await request.json()) as {
    message?: unknown
    detail?: unknown
    action?: unknown
    danger?: unknown
  }
  if (typeof message !== 'string' || typeof action !== 'string') {
    return Response.json({ detail: 'message and action are required' }, { status: 400 })
  }
  const [win] = BrowserWindow.getAllWindows()
  const options = {
    type: danger === true ? ('warning' as const) : ('question' as const),
    buttons: [action, 'Cancel'],
    defaultId: 0,
    cancelId: 1,
    message,
    ...(typeof detail === 'string' ? { detail } : {})
  }
  const answer = win === undefined
    ? await dialog.showMessageBox(options)
    : await dialog.showMessageBox(win, options)
  return Response.json({ ok: answer.response === 0 })
}

async function notify(request: Request): Promise<Response> {
  const { title, body } = (await request.json()) as { title?: unknown; body?: unknown }
  if (typeof title !== 'string') {
    return Response.json({ detail: 'title is required' }, { status: 400 })
  }
  if (Notification.isSupported()) {
    new Notification({ title, ...(typeof body === 'string' ? { body } : {}) }).show()
  }
  return Response.json({ ok: true })
}

/* How many downloads are running, worn on the dock icon — the app keeps working behind
   other windows, and this is how it says so. */
async function badge(request: Request): Promise<Response> {
  const { count } = (await request.json()) as { count?: unknown }
  if (typeof count !== 'number' || !Number.isInteger(count) || count < 0) {
    return Response.json({ detail: 'count is a non-negative integer' }, { status: 400 })
  }
  app.dock?.setBadge(count === 0 ? '' : String(count))
  return Response.json({ ok: true })
}

/* A refusal (foreign daemon, one that would not die) arrives as the house
   `{"detail": ...}` shape, which is what the renderer's transport already reads. */
async function act(run: () => Promise<void>): Promise<Response> {
  try {
    await run()
    return Response.json({ ok: true })
  } catch (error) {
    return Response.json({ detail: String(error instanceof Error ? error.message : error) }, {
      status: 409
    })
  }
}

/* Card links leave through the system browser; navigating the webview would replace
   the app with huggingface.co. */
async function open(request: Request): Promise<Response> {
  const { url } = (await request.json()) as { url?: unknown }
  if (typeof url !== 'string' || !/^https?:\/\//.test(url)) {
    return Response.json({ detail: 'only http(s) URLs leave the app' }, { status: 400 })
  }
  void shell.openExternal(url)
  return Response.json({ ok: true })
}

/* Hop-by-hop, plus the two the proxy itself invalidates: undici hands the body back
   already decoded and re-chunked, so passing the upstream's encoding and length on
   would have Chromium decode it a second time. */
const DROPPED = new Set([
  'connection',
  'keep-alive',
  'transfer-encoding',
  'content-encoding',
  'content-length'
])

/* Plain pass-through, both bodies streamed. `duplex` is what fetch demands the moment a
   request carries a body that is not already buffered.

   The response rides through a stream this side owns, which is the only way an abort
   reaches the daemon: `request.signal` never fires for a scheme handler (Chromium
   cancels the response instead), so a cancelled chat would otherwise keep generating
   with nobody reading. */
async function forward(daemon: string, request: Request, url: URL): Promise<Response> {
  const headers = new Headers(request.headers)
  headers.delete('host')
  const control = new AbortController()
  try {
    const answer = await fetch(daemon + url.pathname + url.search, {
      method: request.method,
      headers,
      body: request.body,
      signal: control.signal,
      redirect: 'manual',
      ...(request.body ? { duplex: 'half' } : {})
    } as RequestInit)
    const out = new Headers()
    for (const [name, value] of answer.headers) {
      if (!DROPPED.has(name.toLowerCase())) out.append(name, value)
    }
    return new Response(answer.body === null ? null : relay(answer.body, control), {
      status: answer.status,
      statusText: answer.statusText,
      headers: out
    })
  } catch (error) {
    return Response.json(
      { detail: `the daemon at ${daemon} did not answer: ${error}` },
      { status: 502 }
    )
  }
}

function relay(body: ReadableStream<Uint8Array>, control: AbortController): ReadableStream {
  const source = body.getReader()
  return new ReadableStream({
    async pull(target) {
      const { value, done } = await source.read()
      if (done) target.close()
      else target.enqueue(value)
    },
    cancel() {
      control.abort()
    }
  })
}

async function asset(dist: string, pathname: string): Promise<Response> {
  const relative = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '')
  const file = normalize(join(dist, relative))
  if (!file.startsWith(dist)) return new Response('not found', { status: 404 })
  try {
    const body = await readFile(file)
    const extension = relative.split('.').pop() ?? ''
    return new Response(body, {
      headers: { 'content-type': TYPES[extension] ?? 'application/octet-stream' }
    })
  } catch {
    /* Every unknown path is the app itself: view switching is client-side. */
    if (relative === 'index.html') return new Response('renderer not built', { status: 500 })
    return await asset(dist, '/')
  }
}

/* Dev only. The window is on Vite there, so the scheme handler above is never reached;
   Vite proxies /admin and /api straight to the daemon and /desktop to this listener,
   which is the one thing neither of them owns. Bodies here are small JSON, so the
   Request is built whole rather than streamed. */
export async function serveDesktop(control: Shell): Promise<{ url: string; close(): void }> {
  const server = createServer((incoming, outgoing) => {
    void answer(incoming, outgoing, control)
  })
  await new Promise<void>((ready) => server.listen(0, '127.0.0.1', ready))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('the shell listener has no port')
  return { url: `http://127.0.0.1:${address.port}`, close: () => server.close() }
}

async function answer(
  incoming: IncomingMessage,
  outgoing: ServerResponse,
  control: Shell
): Promise<void> {
  const chunks: Buffer[] = []
  for await (const chunk of incoming) chunks.push(chunk as Buffer)
  const url = new URL(incoming.url ?? '/', 'http://shell')
  const request = new Request(url, {
    method: incoming.method,
    ...(chunks.length === 0 ? {} : { body: Buffer.concat(chunks) })
  })
  const response =
    (await desktop(request, url, control)) ?? Response.json({ detail: 'no such route' }, { status: 404 })
  outgoing.writeHead(response.status, Object.fromEntries(response.headers))
  outgoing.end(Buffer.from(await response.arrayBuffer()))
}

/* The window's origin in prod: the built renderer, plus a reverse proxy for /admin
   and /api. Nothing here knows about the daemon beyond its base URL — and nothing
   buffers, because a token stream and a job's SSE both have to arrive frame by frame.

   It is a module of its own so the two rules it encodes (which prefixes are the
   daemon's, and that every other path is the app) can be exercised without opening a
   window. */

export interface Origin {
  /* In dev the window points at Vite; anything that still reaches this listener is
     sent there rather than served from a `dist/` that may not exist. */
  dev: boolean
  devUi: string
  daemon: string
  dist: URL
  control: DaemonControl
}

/* The shell's authority over the daemon *it* spawned — main.ts implements it, this
   module only routes to it. A daemon that was already up when the app arrived is not
   ours to kill; `stop`/`restart` refuse with the reason instead. */
export interface DaemonControl {
  owned(): boolean
  stop(): Promise<void>
  restart(): Promise<void>
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
    if (url.pathname === '/desktop/open' && request.method === 'POST') return await open(request)
    if (url.pathname === '/desktop/daemon' && request.method === 'GET') {
      return Response.json({ owned: origin.control.owned() })
    }
    if (url.pathname === '/desktop/daemon/stop' && request.method === 'POST') {
      return await control(() => origin.control.stop())
    }
    if (url.pathname === '/desktop/daemon/restart' && request.method === 'POST') {
      return await control(() => origin.control.restart())
    }
    if (origin.dev) return Response.redirect(origin.devUi + url.pathname + url.search, 302)
    return await asset(origin.dist, url.pathname)
  }
}

/* A refusal (foreign daemon, one that would not die) arrives as the house
   `{"detail": ...}` shape, which is what the renderer's transport already reads. */
async function control(act: () => Promise<void>): Promise<Response> {
  try {
    await act()
    return Response.json({ ok: true })
  } catch (error) {
    return Response.json({ detail: String(error instanceof Error ? error.message : error) }, {
      status: 409
    })
  }
}

/* Card links leave through the system browser; navigating the webview would replace
   the app with huggingface.co. In dev Vite proxies /desktop here (vite.config.ts). */
async function open(request: Request): Promise<Response> {
  const { url } = (await request.json()) as { url?: unknown }
  if (typeof url !== 'string' || !/^https?:\/\//.test(url)) {
    return Response.json({ detail: 'only http(s) URLs leave the app' }, { status: 400 })
  }
  new Deno.Command('open', { args: [url] }).spawn()
  return Response.json({ ok: true })
}

/* Plain pass-through, both bodies streamed. `duplex` is what fetch demands the moment
   a request carries a body that is not already buffered. */
async function forward(daemon: string, request: Request, url: URL): Promise<Response> {
  const headers = new Headers(request.headers)
  headers.delete('host')
  try {
    const answer = await fetch(daemon + url.pathname + url.search, {
      method: request.method,
      headers,
      body: request.body,
      redirect: 'manual',
      ...(request.body ? { duplex: 'half' } : {})
    } as RequestInit)
    return new Response(answer.body, {
      status: answer.status,
      statusText: answer.statusText,
      headers: answer.headers
    })
  } catch (error) {
    return Response.json(
      { detail: `the daemon at ${daemon} did not answer: ${error}` },
      { status: 502 }
    )
  }
}

async function asset(dist: URL, pathname: string): Promise<Response> {
  const relative = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '')
  if (relative.includes('..')) return new Response('not found', { status: 404 })
  try {
    const body = await Deno.readFile(new URL(relative, dist))
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

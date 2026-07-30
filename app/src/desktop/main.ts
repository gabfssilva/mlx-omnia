import { enableWindowDrag, hideTitleBar } from './macos.ts'
import { handler } from './serve.ts'

/* Desktop shell. Unlike a shell that hosts its own server, the backend here is an
   external Python daemon (`uv run sideros-server`): this process discovers it on
   /admin/health, starts it when nobody answers, and owns the one it started —
   quitting the app takes that daemon with it, and the Overview's Stop/Restart act
   on it through /desktop/daemon/*. A daemon that was already up (CLI, another
   client) is not adopted: closing the window never touches it.

   Under `deno desktop` the runtime picks a port (DENO_SERVE_ADDRESS) and redirects
   the process's FIRST listener to it, which makes the server below the window's
   origin: it serves the built renderer and reverse-proxies /admin and /api to the
   daemon. In dev (SIDEROS_DEV=1) the window points at Vite instead, and Vite proxies
   the same two prefixes. Either way the renderer uses relative URLs only. */

const dev = Deno.env.get('SIDEROS_DEV') === '1'
const mock = Deno.env.get('SIDEROS_MOCK') === '1'
const DEV_UI = 'http://localhost:5173'

const app = new URL('../../', import.meta.url)

/* Under `deno desktop` this script runs out of the compiled bundle: `app` above is a
   virtual path the embedded VFS reads (which is how `dist` below is served) but the
   kernel has never heard of — using it as a spawn's cwd, or inheriting it, fails with
   "No such cwd". Spawns need the real project dir, told apart by vite.config.ts,
   which is not among the embedded files. */
const appDir = [
  Deno.env.get('SIDEROS_APP'),
  Deno.env.get('INIT_CWD'),
  app.pathname
].find((dir) => {
  try {
    return dir !== undefined && Deno.statSync(`${dir}/vite.config.ts`).isFile
  } catch {
    return false
  }
}) ?? Deno.env.get('HOME') ?? '/'
Deno.chdir(appDir)
const project = Deno.env.get('SIDEROS_PROJECT') ?? `${appDir}/..`
const dist = new URL('dist/renderer/', app)

const daemon = (Deno.env.get('SIDEROS_DAEMON_URL') ?? 'http://127.0.0.1:8642').replace(/\/$/, '')

/* Spawned children survive an --hmr restart of this script, so both of them are
   tracked on globalThis, which outlives it. `claimed` outlives a failed start too,
   so it is not retried on every --hmr restart. */
const g = globalThis as {
  __sideros?: {
    vite?: Deno.ChildProcess
    daemon?: Deno.ChildProcess
    claimed?: boolean
    hooked?: boolean
  }
}
g.__sideros ??= {}

if (!g.__sideros.claimed && !(await isUp(`${daemon}/admin/health`))) {
  /* Claimed before the spawn — a daemon that never comes up leaves the window open
     on the rail's "down" dot rather than taking the app with it. */
  g.__sideros.claimed = true
  g.__sideros.daemon = startDaemon()
  await waitFor(`${daemon}/admin/health`, 300).catch((error) =>
    console.error(`the daemon at ${daemon} did not come up:`, error)
  )
}

/* Quit is a stop. SIGTERM reaches `uv run`, which forwards it to the server; the
   window's own close stays a hide (dock reopen), so only ending the process kills
   the daemon — and only the one this app spawned. */
if (!g.__sideros.hooked) {
  g.__sideros.hooked = true
  globalThis.addEventListener('unload', () => {
    try {
      g.__sideros?.daemon?.kill('SIGTERM')
    } catch {
      /* already gone */
    }
  })
  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    Deno.addSignalListener(signal, () => {
      try {
        g.__sideros?.daemon?.kill('SIGTERM')
      } catch {
        /* already gone */
      }
      g.__sideros?.vite?.kill()
      Deno.exit(0)
    })
  }
}

/* The process's first listener, which is the one `deno desktop` redirects to the port
   it picked (DENO_SERVE_ADDRESS) — so its address, read back here, is the window's
   origin. */
const server = Deno.serve(
  { onListen: () => {} },
  handler({ dev, devUi: DEV_UI, daemon, dist, control: { owned, stop: stopDaemon, restart } })
)
const origin = `http://${server.addr.hostname === '0.0.0.0' ? '127.0.0.1' : server.addr.hostname}:${server.addr.port}`

if (dev && !g.__sideros.vite) {
  if (await isUp(DEV_UI)) {
    /* A Vite left by a previous process froze SIDEROS_DAEMON_URL on that process's
       view of the world; kill it and respawn rather than proxy through it. */
    await new Deno.Command('pkill', { args: ['-f', 'npm:vite'] }).output()
    for (let i = 0; i < 50 && (await isUp(DEV_UI)); i++) {
      await new Promise((r) => setTimeout(r, 100))
    }
  }
  /* The child must not inherit DENO_SERVE_ADDRESS or its own listen gets hijacked. */
  g.__sideros.vite = new Deno.Command('/usr/bin/env', {
    args: ['-u', 'DENO_SERVE_ADDRESS', 'deno', 'run', '-A', 'npm:vite'],
    cwd: appDir,
    /* The shell's own origin rides along so Vite can proxy /desktop back here —
       the one route (opening links externally) the daemon does not own. */
    env: { SIDEROS_DAEMON_URL: daemon, SIDEROS_SHELL_URL: origin },
    stdout: 'inherit',
    stderr: 'inherit'
  }).spawn()
  await waitFor(DEV_UI, 100)
}

let win = createWindow()

if (Deno.build.os === 'darwin') {
  Deno.dock.onreopen = () => {
    if (win.isClosed()) win = createWindow()
    else win.show()
  }
} else {
  win.onclose = () => {
    g.__sideros?.vite?.kill()
    Deno.exit(0)
  }
}

function createWindow(): Deno.BrowserWindow {
  const w = new Deno.BrowserWindow({
    title: '',
    width: 1300,
    height: 860,
    transparentTitlebar: true
  })
  w.navigate(dev ? DEV_UI : origin)
  void hideTitleBar(w)
  if (Deno.build.os === 'darwin') enableWindowDrag(w)
  return w
}

function startDaemon(): Deno.ChildProcess {
  const command = mock
    ? new Deno.Command('/usr/bin/env', {
        args: ['-u', 'DENO_SERVE_ADDRESS', 'deno', 'run', '-A', 'src/mock/daemon.ts'],
        cwd: appDir,
        stdout: 'inherit',
        stderr: 'inherit'
      })
    : new Deno.Command('/usr/bin/env', {
        args: ['-u', 'DENO_SERVE_ADDRESS', 'uv', 'run', '--project', project, 'sideros-server'],
        stdout: 'inherit',
        stderr: 'inherit'
      })
  /* Unreffed so the child never holds this event loop open; the handle stays on
     globalThis, which is what quit and /desktop/daemon/* act through. */
  const child = command.spawn()
  child.unref()
  return child
}

function owned(): boolean {
  return g.__sideros?.daemon !== undefined
}

/* SIGTERM and wait: uv forwards it, uvicorn shuts down, `status` resolves and the
   port is free for a respawn. No SIGKILL fallback — killing `uv` alone would orphan
   the very python process this is about; a daemon that ignores SIGTERM is reported,
   not escalated. */
async function stopDaemon(): Promise<void> {
  const child = g.__sideros?.daemon
  if (child === undefined) {
    throw new Error('the daemon was not started by this app; stop it where it was started')
  }
  g.__sideros!.daemon = undefined
  g.__sideros!.claimed = false
  try {
    child.kill('SIGTERM')
  } catch {
    /* already gone */
  }
  const gone = await Promise.race([
    child.status.then(() => true),
    new Promise<boolean>((answer) => setTimeout(() => answer(false), 10_000))
  ])
  if (!gone) {
    g.__sideros!.daemon = child
    g.__sideros!.claimed = true
    throw new Error('the daemon did not exit within 10s of SIGTERM')
  }
}

async function restart(): Promise<void> {
  if (owned()) await stopDaemon()
  else if (await isUp(`${daemon}/admin/health`)) {
    throw new Error('the daemon was not started by this app; restart it where it was started')
  }
  g.__sideros!.claimed = true
  g.__sideros!.daemon = startDaemon()
  await waitFor(`${daemon}/admin/health`, 300)
}

async function isUp(target: string): Promise<boolean> {
  try {
    const answer = await fetch(target)
    await answer.body?.cancel()
    return true
  } catch {
    return false
  }
}

async function waitFor(target: string, tries: number): Promise<void> {
  for (let i = 0; i < tries; i++) {
    if (await isUp(target)) return
    await new Promise((r) => setTimeout(r, 200))
  }
  throw new Error(`timed out waiting for ${target}`)
}

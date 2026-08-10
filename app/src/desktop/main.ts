import { app, BrowserWindow, protocol } from 'electron'
import { spawn, type ChildProcess } from 'node:child_process'
import { join } from 'node:path'
import { handler, ORIGIN, SCHEME, serveDesktop } from './serve.ts'

/* Desktop shell. Unlike a shell that hosts its own server, the backend here is an
   external Python daemon (`uv run sideros-server`): this process discovers it on
   /admin/health, starts it when nobody answers, and owns the one it started —
   quitting the app takes that daemon with it, and the Overview's Stop/Restart act
   on it through /desktop/daemon/*. A daemon that was already up (CLI, another
   client) is not adopted: closing the window never touches it.

   In prod the window's origin is the `sideros:` scheme, whose handler serves the built
   renderer and reverse-proxies /admin and /api to the daemon. In dev (SIDEROS_DEV=1)
   the window points at Vite instead, and Vite proxies the same two prefixes plus
   /desktop back to the small listener this process opens for it. Either way the
   renderer uses relative URLs only. */

const dev = process.env.SIDEROS_DEV === '1'
const mock = process.env.SIDEROS_MOCK === '1'
const DEV_UI = 'http://localhost:5173'

/* Packaged, this file runs from inside app.asar and the renderer sits beside it; from
   a checkout it runs out of dist/desktop/ and the project root is two levels up. */
const appDir = app.isPackaged ? app.getAppPath() : join(import.meta.dirname, '../..')
const dist = join(appDir, 'dist/renderer')
/* Where the daemon is run from, which is never `appDir` when packaged: that path is
   inside app.asar, which the asset reads above go through but no spawn can use as a
   cwd. A packaged app has to be told where the checkout is. */
const project = process.env.SIDEROS_PROJECT ?? (app.isPackaged ? app.getPath('home') : join(appDir, '..'))

const daemonUrl = (process.env.SIDEROS_DAEMON_URL ?? 'http://127.0.0.1:8642').replace(/\/$/, '')

let daemon: ChildProcess | undefined
let vite: ChildProcess | undefined
/* Outlives a failed start, so it is not retried on every activation. */
let claimed = false

/* Same-origin fetch, streamed bodies, and a secure context (the renderer's `zoom` and
   fonts do not care, but Chromium refuses parts of the platform without it). */
protocol.registerSchemesAsPrivileged([
  {
    scheme: SCHEME,
    privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true }
  }
])

const control = { owned, stop: stopDaemon, restart, ready: reveal }

/* `then`, not a top-level await: this module is ESM, and Electron only emits `ready`
   once it finishes evaluating — awaiting it up here deadlocks the boot. */
void app.whenReady().then(boot)

async function boot(): Promise<void> {
  /* Packaged, the dock reads the icon out of the bundle; from a checkout it would show
     Electron's own until told otherwise. */
  if (!app.isPackaged) app.dock?.setIcon(join(appDir, 'build/icon.png'))

  if (!claimed && !(await isUp(`${daemonUrl}/admin/health`))) {
    /* Claimed before the spawn — a daemon that never comes up leaves the window open
       on the rail's "down" dot rather than taking the app with it. */
    claimed = true
    daemon = startDaemon()
    await waitFor(`${daemonUrl}/admin/health`, 300).catch((error: unknown) =>
      console.error(`the daemon at ${daemonUrl} did not come up:`, error)
    )
  }

  if (dev) {
    const shell = await serveDesktop(control)
    if (await isUp(DEV_UI)) {
      /* A Vite left by a previous process froze SIDEROS_DAEMON_URL on that process's
         view of the world; kill it and respawn rather than proxy through it. */
      spawn('pkill', ['-f', 'vite']).unref()
      for (let i = 0; i < 50 && (await isUp(DEV_UI)); i++) {
        await new Promise((r) => setTimeout(r, 100))
      }
    }
    /* The shell's own origin rides along so Vite can proxy /desktop back here — the one
       route (opening links externally, stopping the daemon) the daemon does not own. */
    vite = spawn('npm', ['exec', '--', 'vite'], {
      cwd: appDir,
      env: { ...process.env, SIDEROS_DAEMON_URL: daemonUrl, SIDEROS_SHELL_URL: shell.url },
      stdio: 'inherit'
    })
    await waitFor(DEV_UI, 100)
  } else {
    protocol.handle(SCHEME, handler({ daemon: daemonUrl, dist, control }))
  }

  createWindow()
}

/* Quit is a stop. SIGTERM reaches `uv run`, which forwards it to the server; the
   window's own close stays a hide (dock reopen), so only ending the process kills the
   daemon — and only the one this app spawned. A kill this process cannot answer
   (SIGKILL, a crash) is caught on the other side: the daemon is spawned watching this
   pid (`--parent-pid`) and stops itself within a second of it disappearing. */
app.on('before-quit', () => {
  daemon?.kill('SIGTERM')
  vite?.kill()
})
for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => app.quit())
}

app.on('activate', () => {
  const [win] = BrowserWindow.getAllWindows()
  if (win === undefined) createWindow()
  else {
    win.show()
    win.focus()
  }
})

/* macOS keeps the app alive with no window (dock reopen); elsewhere the last window
   closing is the quit. */
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

function createWindow(): BrowserWindow {
  const win = new BrowserWindow({
    title: '',
    /* One size, both axes. The layout is three measured columns and a memory band drawn
       to scale — neither gains anything from stretching, and a fixed canvas is what makes
       the band a real ruler: the same model is always the same width. 760 + the 37pt menu
       bar clears the 800pt of the smallest current Mac display. */
    width: 1160,
    height: 760,
    resizable: false,
    maximizable: false,
    fullscreenable: false,
    center: true,
    /* Shown by `reveal`, not on creation: the app arrives already drawn rather than as
       an empty dark rectangle waiting for React to mount and the first poll to land. */
    show: false,
    titleBarStyle: 'hiddenInset',
    /* Where theme.css leaves room for them: `.lights` reserves 72px at the head of the
       40px title bar. There is no page zoom any more, so these native points are the
       same points the stylesheet counts. */
    trafficLightPosition: { x: 16, y: 14 },
    backgroundColor: '#1B1E20',
    webPreferences: { contextIsolation: true, nodeIntegration: false }
  })
  void win.loadURL(dev ? DEV_UI : `${ORIGIN}/`)
  /* The renderer reveals itself over /desktop/ready once its first poll lands. These are
     the two ways that signal never arrives — a renderer that failed to load, and one too
     old or too broken to send it — and neither may leave the app with no window at all. */
  win.webContents.once('did-fail-load', reveal)
  win.webContents.once('did-finish-load', () => setTimeout(reveal, 10_000))
  return win
}

/* Idempotent: whichever of the ready signal and the fallbacks arrives first wins. */
function reveal(): void {
  const [win] = BrowserWindow.getAllWindows()
  if (win !== undefined && !win.isVisible()) {
    win.show()
    win.focus()
  }
}

function startDaemon(): ChildProcess {
  const child = mock
    ? spawn('node', [join(appDir, 'src/mock/daemon.ts')], { cwd: project, stdio: 'inherit' })
    : spawn(
        'uv',
        [
          'run',
          '--project',
          project,
          'sideros-server',
          /* The daemon's half of "quit is a stop": it watches this pid and shuts itself
             down when it goes, which is the only path that survives a force quit or a
             crash — the handlers above never run then. */
          '--parent-pid',
          String(process.pid)
        ],
        { cwd: project, stdio: 'inherit' }
      )
  child.unref()
  return child
}

function owned(): boolean {
  return daemon !== undefined
}

/* SIGTERM and wait: uv forwards it, uvicorn shuts down, the child exits and the port is
   free for a respawn. No SIGKILL fallback — killing `uv` alone would orphan the very
   python process this is about; a daemon that ignores SIGTERM is reported, not
   escalated. */
async function stopDaemon(): Promise<void> {
  const child = daemon
  if (child === undefined) {
    throw new Error('the daemon was not started by this app; stop it where it was started')
  }
  daemon = undefined
  claimed = false
  const exited = new Promise<boolean>((answer) => {
    child.once('exit', () => answer(true))
    setTimeout(() => answer(false), 10_000)
  })
  child.kill('SIGTERM')
  if (!(await exited)) {
    daemon = child
    claimed = true
    throw new Error('the daemon did not exit within 10s of SIGTERM')
  }
}

async function restart(): Promise<void> {
  if (owned()) await stopDaemon()
  else if (await isUp(`${daemonUrl}/admin/health`)) {
    throw new Error('the daemon was not started by this app; restart it where it was started')
  }
  claimed = true
  daemon = startDaemon()
  await waitFor(`${daemonUrl}/admin/health`, 300)
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

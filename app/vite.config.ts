import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/* Where /admin and /api go while the window points at this dev server. The desktop
   shell exports it after discovering (or starting) the daemon; a bare `npm run
   dev:web` talks to whatever is on the default port. Either way the renderer only
   ever uses relative URLs, so there is no CORS on any path. */
const daemon = process.env.SIDEROS_DAEMON_URL ?? 'http://127.0.0.1:8642'
/* The desktop shell's own server, for the one route that is neither the daemon's nor an
   asset: /desktop/open. Absent under a bare `npm run dev:web`, where the proxy entry
   simply is not added and card links stay in-page no-ops. */
const shell = process.env.SIDEROS_SHELL_URL

const at = (path: string) => new URL(path, import.meta.url).pathname

export default defineConfig({
  root: at('./src/renderer/'),
  /* The terminal is shared with the daemon's uvicorn output; Vite's startup clear
     would wipe it. */
  clearScreen: false,
  plugins: [react()],
  build: {
    outDir: at('./dist/renderer/'),
    emptyOutDir: true
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/admin': daemon,
      '/api': daemon,
      ...(shell === undefined ? {} : { '/desktop': shell })
    }
  }
})

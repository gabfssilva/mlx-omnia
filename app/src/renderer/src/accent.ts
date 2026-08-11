/* The system accent reaches the stylesheet as --accent, which the shell writes onto the
   document and rewrites whenever the person changes it in System Settings. One thing the
   stylesheet cannot then do with it is put it inside an image: `url()` is a single token
   and no `var()` substitutes into a data URI. So the picture that needs the accent — the
   chevron well of a pop-up button — is drawn here and stamped as --well.

   Derived in the renderer rather than in the shell so there is one copy of the drawing,
   and so a window with no shell under it (`dev:web`) still follows the fallback accent. */

const well = (accent: string): string =>
  `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='17' height='18' viewBox='0 0 17 18'%3E%3Crect width='17' height='18' rx='5' fill='${encodeURIComponent(accent)}'/%3E%3Cg fill='none' stroke='%23fff' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M5.6 7.4 8.5 4.6 11.4 7.4'/%3E%3Cpath d='M5.6 10.6 8.5 13.4 11.4 10.6'/%3E%3C/g%3E%3C/svg%3E")`

/* The palette's own --accent, and macOS's blue behind it for the one case that reads
   empty: a stylesheet that has not been applied yet. */
const current = (): string =>
  getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#0A84FF'

function redraw(): void {
  document.documentElement.style.setProperty('--well', well(current()))
}

/* The shell announces the accent by writing the property, so the style attribute changing
   is the only signal there is. Guarded against the write this observer's own callback
   makes: --well is set, --accent is not, and reading it back yields the same string. */
export function followAccent(): void {
  redraw()
  let last = current()
  new MutationObserver(() => {
    const accent = current()
    if (accent === last) return
    last = accent
    redraw()
  }).observe(document.documentElement, { attributes: true, attributeFilter: ['style'] })
}

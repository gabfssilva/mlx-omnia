/* The two keyboard behaviours a native window has and a page does not: Escape dismisses
   the thing on top, and typing letters into a list jumps to a row. Both are here because
   both are about the whole window rather than any one view. */

import { useCallback, useEffect, useRef, type RefCallback } from 'react'

/* Only the innermost sheet answers Escape — Library can open Quantize over itself, and
   Benchmark's sheet opens the dataset form over its own. A stack rather than one listener
   per sheet is what makes "the one on top" a fact instead of a race between handlers. */
const stack: (() => void)[] = []

function dismiss(event: KeyboardEvent): void {
  if (event.key !== 'Escape') return
  const top = stack[stack.length - 1]
  if (top === undefined) return
  event.preventDefault()
  top()
}

/* Undefined is "this one is not dismissable" — Quantize is a sheet over Library and a view
   of its own, and the view has nothing to close. It must not take Escape off the stack. */
export function useEscape(onEscape: (() => void) | undefined): void {
  const held = useRef(onEscape)
  held.current = onEscape
  const dismissable = onEscape !== undefined

  useEffect(() => {
    if (!dismissable) return
    const entry = (): void => held.current?.()
    if (stack.length === 0) window.addEventListener('keydown', dismiss)
    stack.push(entry)
    return () => {
      stack.splice(stack.indexOf(entry), 1)
      if (stack.length === 0) window.removeEventListener('keydown', dismiss)
    }
  }, [dismissable])
}

/* Letters typed within this of each other are one prefix; a pause starts a new one. The
   figure is AppKit's. */
const RUN_MS = 900

/* Type-ahead over the rows of a list, the way an NSTableView answers it: the rows carry
   what they are matched on in `data-key`, so the hook needs no copy of the model, and the
   match takes focus. Whether focus is also the selection is the list's own business — a
   roster selects on focus, a list whose rows open something does not.

   A ref callback rather than a ref object, because these lists are not mounted until they
   have rows: an effect reading `.current` on mount would find nothing and never look
   again. */
export function useTypeAhead(): RefCallback<HTMLElement> {
  const typed = useRef({ prefix: '', at: 0 })

  return useCallback((node: HTMLElement | null) => {
    if (node === null) return

    const onKey = (event: KeyboardEvent): void => {
      /* A single printable character and nothing modified: Escape, the arrows and ⌘F are
         somebody else's, and Space activates the focused row. */
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (event.key.length !== 1 || event.key === ' ') return
      /* A field inside a row — a chat being renamed — is typing, not jumping. */
      const from = event.target
      if (from instanceof HTMLInputElement || from instanceof HTMLTextAreaElement) return

      const now = Date.now()
      const prefix =
        (now - typed.current.at < RUN_MS ? typed.current.prefix : '') + event.key.toLowerCase()
      typed.current = { prefix, at: now }

      const rows = [...node.querySelectorAll<HTMLElement>('[data-key]')]
      const found = rows.find((row) => (row.dataset['key'] ?? '').toLowerCase().startsWith(prefix))
      if (found === undefined) return
      event.preventDefault()
      found.focus()
    }

    node.addEventListener('keydown', onKey)
    return () => node.removeEventListener('keydown', onKey)
  }, [])
}

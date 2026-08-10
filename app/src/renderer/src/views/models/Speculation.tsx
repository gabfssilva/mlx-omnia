/* The model's own DFlash setting. A profile can override it (`features` on the profile),
   but this is what every request gets when none does.

   There is no on/off here: what turns speculation on is naming the checkpoint to draft
   with, since that is the thing the daemon loads beside the model. The daemon suggests the
   drafters it knows pair with this architecture, and accepts any id in the catalog — a
   drafter quantized here is a checkpoint like any other. Empty is off.

   It never changes what the model writes: verification accepts only the target's own
   argmax, so the setting buys speed or it buys nothing. */

import { type JSX, useCallback, useEffect, useState } from 'react'
import * as api from './api'

/* The value the picker uses for "an id the list does not offer". Not a valid model id —
   a `/` is mandatory in one — so it can never collide with something selectable. */
const OTHER = '?other'

export function Speculation({ model }: { model: string }): JSX.Element | null {
  const [view, setView] = useState<api.SettingsView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [drafter, setDrafter] = useState('')
  const [typing, setTyping] = useState(false)
  const [block, setBlock] = useState('')

  const take = useCallback((next: api.SettingsView): void => {
    setView(next)
    setDrafter(next.features.dflash?.drafter ?? '')
    setTyping(false)
    setBlock(String(next.features.dflash?.block_size ?? ''))
  }, [])

  const read = useCallback(async (): Promise<void> => {
    try {
      take(await api.getSettings(model))
    } catch (failure) {
      setError(api.reason(failure))
    }
  }, [model, take])

  useEffect(() => {
    setView(null)
    setError(null)
    void read()
  }, [read])

  if (view === null) return null

  const write = async (next: api.DFlash): Promise<void> => {
    setSaving(true)
    setError(null)
    try {
      take(await api.saveSettings(model, { dflash: next }))
    } catch (failure) {
      setError(api.reason(failure))
      /* The daemon refused, so what is on screen is not what is stored. */
      void read()
    } finally {
      setSaving(false)
    }
  }

  const stored = view.features.dflash
  const commit = (named: string): void => {
    const length = block.trim() === '' ? null : Number(block)
    if (named === (stored?.drafter ?? '') && length === (stored?.block_size ?? null)) return
    void write({ drafter: named === '' ? null : named, block_size: named === '' ? null : length })
  }

  /* What the picker offers: off, everything installed, and — when the stored id is not
     among them — the stored one, so opening the list never proposes to forget it. */
  const offered = view.available.includes(drafter) || drafter === ''
    ? view.available
    : [drafter, ...view.available]

  return (
    <>
      <div className="grouplab">Speculation</div>
      <div className="field">
        <span className="eyebrow">DFlash drafter</span>
        {typing || offered.length === 0 ? (
          <input
            className="input mono"
            value={drafter}
            placeholder="off"
            disabled={saving}
            aria-label="DFlash drafter checkpoint"
            autoFocus={typing}
            onChange={(event) => setDrafter(event.target.value)}
            onBlur={(event) => commit(event.target.value.trim())}
            onKeyDown={(event) => event.key === 'Enter' && event.currentTarget.blur()}
          />
        ) : (
          <select
            className="input mono"
            value={drafter}
            disabled={saving}
            aria-label="DFlash drafter checkpoint"
            onChange={(event) => {
              if (event.target.value === OTHER) {
                setDrafter('')
                setTyping(true)
                return
              }
              setDrafter(event.target.value)
              commit(event.target.value)
            }}
          >
            <option value="">off</option>
            {offered.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
            <option value={OTHER}>Another checkpoint…</option>
          </select>
        )}
      </div>
      {stored?.drafter != null && (
        <div className="field">
          <span className="eyebrow">Ids proposed a round</span>
          <input
            className="input tn"
            type="number"
            min={1}
            value={block}
            placeholder="the engine's default"
            disabled={saving}
            aria-label="Ids proposed a round"
            onChange={(event) => setBlock(event.target.value)}
            onBlur={() => commit(drafter.trim())}
            onKeyDown={(event) => event.key === 'Enter' && event.currentTarget.blur()}
          />
        </div>
      )}
      {view.unavailable_reason !== null && <p className="note">{view.unavailable_reason}.</p>}
      {error !== null && <div className="err">{error}</div>}
    </>
  )
}

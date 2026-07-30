/* A profile is sampling plus a system prompt, saved on the model and served as
   `model:name`. There is no route that lists them, so the names come out of the dialect's
   own catalog (api.listProfileNames) and each one is read back by name.

   A knob left blank is not part of the profile: the request's own value — or the dialect's
   default — stands. That is what `null` means on the way out, and what '—' means here. */

import { type JSX, useCallback, useEffect, useState } from 'react'
import * as api from './api'

interface Draft {
  name: string
  temperature: string
  top_p: string
  top_k: string
  min_p: string
  repetition_penalty: string
  seed: string
  system_prompt: string
}

const text = (value: number | null): string => (value === null ? '' : String(value))

const num = (raw: string): number | null => {
  const trimmed = raw.trim()
  if (trimmed === '') return null
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : null
}

const EMPTY: Draft = {
  name: '',
  temperature: '',
  top_p: '',
  top_k: '',
  min_p: '',
  repetition_penalty: '',
  seed: '',
  system_prompt: ''
}

const draftOf = (view: api.ProfileView): Draft => ({
  name: view.name,
  temperature: text(view.sampling.temperature),
  top_p: text(view.sampling.top_p),
  top_k: text(view.sampling.top_k),
  min_p: text(view.sampling.min_p),
  repetition_penalty: text(view.sampling.repetition_penalty),
  seed: text(view.sampling.seed),
  system_prompt: view.system_prompt ?? ''
})

function Line({ label, value }: { label: string; value: number | null }): JSX.Element {
  return (
    <div className="kvline">
      <span className="eyebrow">{label}</span>
      <b className={value === null ? 'unset' : undefined}>{value === null ? '—' : value}</b>
    </div>
  )
}

interface KnobProps {
  label: string
  value: string
  step: string
  min: string
  onChange: (next: string) => void
}

function Knob({ label, value, step, min, onChange }: KnobProps): JSX.Element {
  return (
    <div className="fieldcol">
      <span className="eyebrow">{label}</span>
      <input
        className="input tn"
        type="number"
        step={step}
        min={min}
        value={value}
        placeholder="—"
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  )
}

export function Profiles({ model }: { model: string }): JSX.Element {
  const [names, setNames] = useState<string[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [profile, setProfile] = useState<api.ProfileView | null>(null)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [error, setError] = useState<string | null>(null)

  const list = useCallback(async (): Promise<string[]> => {
    const found = await api.listProfileNames(model)
    setNames(found)
    return found
  }, [model])

  useEffect(() => {
    setSelected(null)
    setProfile(null)
    setDraft(null)
    setError(null)
    void list()
      .then((found) => setSelected(found[0] ?? null))
      .catch((failure: unknown) => setError(api.reason(failure)))
  }, [list])

  useEffect(() => {
    if (selected === null) {
      setProfile(null)
      return
    }
    api
      .getProfile(model, selected)
      .then(setProfile)
      .catch((failure: unknown) => setError(api.reason(failure)))
  }, [model, selected])

  const edit = (next: Partial<Draft>): void =>
    setDraft((current) => ({ ...(current ?? EMPTY), ...next }))

  const save = (): void => {
    if (draft === null) return
    const name = draft.name.trim()
    if (name === '') {
      setError('a profile needs a name — it is what a request selects it by')
      return
    }
    if (name.includes(':')) {
      setError("profile name may not contain ':' — resolution splits the model at the last one")
      return
    }
    setError(null)
    void (async () => {
      try {
        const saved = await api.saveProfile(model, name, {
          sampling: {
            temperature: num(draft.temperature),
            top_p: num(draft.top_p),
            top_k: num(draft.top_k),
            min_p: num(draft.min_p),
            repetition_penalty: num(draft.repetition_penalty),
            seed: num(draft.seed)
          },
          system_prompt: draft.system_prompt.trim() === '' ? null : draft.system_prompt
        })
        await list()
        setProfile(saved)
        setSelected(saved.name)
        setDraft(null)
      } catch (failure) {
        setError(api.reason(failure))
      }
    })()
  }

  const remove = (name: string): void => {
    setError(null)
    void (async () => {
      try {
        await api.removeProfile(model, name)
        const left = await list()
        setSelected(left[0] ?? null)
        setDraft(null)
      } catch (failure) {
        setError(api.reason(failure))
      }
    })()
  }

  return (
    <div className="card profiles">
      <h3>
        Profiles <span>{names.length === 0 ? 'none saved' : `${names.length} saved`}</span>
      </h3>

      {names.length > 0 && draft === null && (
        <div className="seg profpick">
          {names.map((name) => (
            <button
              key={name}
              className={name === selected ? 'on' : undefined}
              onClick={() => setSelected(name)}
            >
              {name}
            </button>
          ))}
        </div>
      )}

      {error !== null && <div className="notice">{error}</div>}

      {draft !== null ? (
        <>
          <div className="fieldcol">
            <span className="eyebrow">Name</span>
            <input
              className="input mono"
              value={draft.name}
              placeholder="code"
              onChange={(event) => edit({ name: event.target.value })}
            />
            <span className="eyebrow hint">
              Served as <code className="mono">{`${model}:${draft.name.trim() || 'name'}`}</code>
            </span>
          </div>
          <div className="knobs">
            <Knob label="Temperature" value={draft.temperature} step="0.05" min="0" onChange={(next) => edit({ temperature: next })} />
            <Knob label="Top-p" value={draft.top_p} step="0.01" min="0" onChange={(next) => edit({ top_p: next })} />
            <Knob label="Top-k" value={draft.top_k} step="1" min="1" onChange={(next) => edit({ top_k: next })} />
            <Knob label="Min-p" value={draft.min_p} step="0.01" min="0" onChange={(next) => edit({ min_p: next })} />
            <Knob label="Repetition penalty" value={draft.repetition_penalty} step="0.01" min="0" onChange={(next) => edit({ repetition_penalty: next })} />
            <Knob label="Seed" value={draft.seed} step="1" min="0" onChange={(next) => edit({ seed: next })} />
          </div>
          <div className="fieldcol">
            <span className="eyebrow">System prompt</span>
            <textarea
              className="input"
              value={draft.system_prompt}
              placeholder="None — the checkpoint's template stands."
              onChange={(event) => edit({ system_prompt: event.target.value })}
            />
          </div>
          <footer>
            <button className="btn solid" onClick={save}>
              Save
            </button>
            <button className="btn" onClick={() => { setDraft(null); setError(null) }}>
              Cancel
            </button>
          </footer>
        </>
      ) : profile === null ? (
        <>
          <p className="eyebrow hint">
            No profile saved. A profile fills the knobs a request leaves out, and is selected by
            asking for <code className="mono">{`${model}:name`}</code>.
          </p>
          <footer>
            <button className="btn solid" onClick={() => setDraft(EMPTY)}>
              New profile
            </button>
          </footer>
        </>
      ) : (
        <>
          <Line label="Temperature" value={profile.sampling.temperature} />
          <Line label="Top-p" value={profile.sampling.top_p} />
          <Line label="Top-k" value={profile.sampling.top_k} />
          {profile.sampling.min_p !== null && <Line label="Min-p" value={profile.sampling.min_p} />}
          {profile.sampling.repetition_penalty !== null && (
            <Line label="Repetition penalty" value={profile.sampling.repetition_penalty} />
          )}
          {profile.sampling.seed !== null && <Line label="Seed" value={profile.sampling.seed} />}
          <div className="kvline">
            <span className="eyebrow">System prompt</span>
            <b className={profile.system_prompt === null ? 'unset' : 'prompt'}>
              {profile.system_prompt ?? 'none'}
            </b>
          </div>
          <footer>
            <button className="btn solid" onClick={() => setDraft(draftOf(profile))}>
              Edit
            </button>
            <button className="btn" onClick={() => setDraft(EMPTY)}>
              New
            </button>
            <button className="btn danger" onClick={() => remove(profile.name)}>
              Delete
            </button>
          </footer>
        </>
      )}
    </div>
  )
}

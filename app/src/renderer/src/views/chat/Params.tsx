import type { JSX } from 'react'
import type { Params } from './api'

/* One decimal place when one is what the number has: 0.6 and not 0.60, which is what
   the mockup reads. */
const decimal = (value: number): string => String(Number(value.toFixed(2)))

function Slider({
  label,
  shown,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string
  shown: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}): JSX.Element {
  return (
    <div className="param">
      <div className="prow">
        <label>{label}</label>
        <b>{shown}</b>
      </div>
      <input
        type="range"
        className="mini"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  )
}

/* The knobs this chat's next request carries. They are held per chat and not stored:
   the sessions table keeps the conversation, not how it was sampled. */
export function ParamsPane({
  params,
  contextLimit,
  onChange
}: {
  params: Params
  /* The selected model's own window (the config's max_position_embeddings), null when
     the catalog does not know it. The daemon caps the generation by it either way; here
     it is the slider's ceiling, so the dial cannot ask past it. */
  contextLimit: number | null
  onChange: (params: Params) => void
}): JSX.Element {
  const set = <K extends keyof Params>(key: K, value: Params[K]): void =>
    onChange({ ...params, [key]: value })
  const budgetCap = contextLimit ?? 32768
  const budget = Math.min(params.max_tokens, budgetCap)

  return (
    <div className="params">
      <div className="phint">Applied to this chat from the next message on.</div>

      <Slider
        label="Temperature"
        shown={decimal(params.temperature)}
        value={Math.round(params.temperature * 100)}
        min={0}
        max={200}
        step={1}
        onChange={(value) => set('temperature', value / 100)}
      />
      {/* From 1 and not 0: the dialect refuses a top_p of zero, and a slider that can
          reach a value the daemon will not take is a 400 nobody asked for. */}
      <Slider
        label="Top-p"
        shown={decimal(params.top_p)}
        value={Math.round(params.top_p * 100)}
        min={1}
        max={100}
        step={1}
        onChange={(value) => set('top_p', value / 100)}
      />
      <Slider
        label="Top-k"
        shown={params.top_k === null ? '—' : String(params.top_k)}
        value={params.top_k ?? 0}
        min={0}
        max={100}
        step={1}
        onChange={(value) => set('top_k', value === 0 ? null : value)}
      />
      {/* To 99 and not 100: min_p is a fraction below 1, and 1.0 keeps nothing. */}
      <Slider
        label="Min-p"
        shown={params.min_p === 0 ? '—' : decimal(params.min_p)}
        value={Math.round(params.min_p * 100)}
        min={0}
        max={99}
        step={1}
        onChange={(value) => set('min_p', value / 100)}
      />
      <Slider
        label="Repetition penalty"
        shown={params.repetition_penalty <= 1 ? '—' : decimal(params.repetition_penalty)}
        value={Math.round(params.repetition_penalty * 100)}
        min={100}
        max={200}
        step={1}
        onChange={(value) => set('repetition_penalty', value / 100)}
      />
      <Slider
        label="Max tokens"
        shown={budget.toLocaleString('en-US')}
        value={budget}
        min={256}
        max={budgetCap}
        step={256}
        onChange={(value) => set('max_tokens', value)}
      />

      <div className="param">
        <div className="prow">
          <label htmlFor="chat-system">System prompt</label>
        </div>
        <textarea
          id="chat-system"
          placeholder="None — the checkpoint's template stands."
          value={params.system_prompt}
          onChange={(event) => set('system_prompt', event.target.value)}
        />
      </div>
    </div>
  )
}

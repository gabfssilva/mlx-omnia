/* The units this screen prints, and the one arithmetic it does.

   Storage and memory are GiB under the label GB, which is how the catalog's sizes read
   everywhere else. Bytes per token is decimal GB, because it is the numerator of the
   ceiling — 490 GB/s ÷ bytes/token — and the two have to divide out. */

const GIB = 1024 ** 3

export const gib = (bytes: number, digits = 1): string => (bytes / GIB).toFixed(digits)

export const perToken = (bytes: number): string => (bytes / 1e9).toFixed(2)

/** tok/s a decode step can reach at the sustained bandwidth: measured, never published. */
export const ceilingRate = (bytesPerToken: number, sustainedGbs: number): number =>
  (sustainedGbs * 1e9) / bytesPerToken

export const percent = (fraction: number): number => Math.round(fraction * 100)

export const contextTokens = (tokens: number): string =>
  tokens >= 1000 ? `${Math.round(tokens / 1000)}k` : String(tokens)

/* A quantization suffix in a repository name is not part of the model's name — the meta
   line under it already says what it is packed as. */
const PACKED = /-(?:\d+bit|mxfp4|nf4|bf16|fp16|fp32|int[48])$/i

export function displayName(id: string): string {
  const last = id.split('/').filter(Boolean).at(-1) ?? id
  return last.replace(PACKED, '')
}

/** The hub organisation, or the directory a quantized entry was written into. */
export function owner(id: string): string {
  const parts = id.split('/').filter(Boolean)
  return parts.length > 1 ? parts[parts.length - 2] : 'local'
}

export const homeless = (path: string): string => path.replace(/^\/(?:Users|home)\/[^/]+/, '~')

export function idle(seconds: number | null): string {
  if (seconds === null) return 'never'
  if (seconds < 60) return `after ${Math.round(seconds)} s`
  if (seconds < 3600) return `after ${Math.round(seconds / 60)} min`
  const hours = seconds / 3600
  return `after ${hours % 1 === 0 ? hours : hours.toFixed(1)} h`
}

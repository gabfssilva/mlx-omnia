/* Decimal GB, which is the unit the Hub prices a repository in. */

const scale = (amount: number): [number, string] => (amount >= 1e9 ? [1e9, 'GB'] : [1e6, 'MB'])

export function size(amount: number): string {
  const [div, unit] = scale(amount)
  return `${(amount / div).toFixed(2)} ${unit}`
}

/* Both sides in the total's unit: "2.31 / 4.48 GB". */
export function sizePair(done: number, total: number): string {
  const [div, unit] = scale(total)
  return `${(done / div).toFixed(2)} / ${(total / div).toFixed(2)} ${unit}`
}

export const rate = (bytesPerSecond: number): string =>
  `${Math.round(bytesPerSecond / 1e6)} MB/s`

/* Rounded to the unit above the one it is about to leave: a countdown that reads "119s"
   is a number nobody subtracts from. */
export function left(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)}s left`
  const minutes = Math.round(seconds / 60)
  return minutes < 90 ? `${minutes}m left` : `${Math.round(minutes / 60)}h left`
}

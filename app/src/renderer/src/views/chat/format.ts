/* The two things both the sidebar and the header spell out. */

/* The name without the repository that owns it: the sidebar has 218px and the org is
   the same for every row. */
export const shortModel = (id: string): string => id.slice(id.lastIndexOf('/') + 1)

const DAY = 86400

export function ago(seconds: number): string {
  const delta = Date.now() / 1000 - seconds
  if (delta < 90) return 'now'
  if (delta < 3600) return `${Math.round(delta / 60)} min`
  if (delta < DAY) return `${Math.round(delta / 3600)} h`
  if (delta < 2 * DAY) return 'yesterday'
  return new Date(seconds * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

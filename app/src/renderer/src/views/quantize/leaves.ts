/* The plan is priced leaf by leaf — a 32B has some six hundred of them — and the
   override it takes back is a `fullmatch` pattern. Both ends meet here: the leaves
   collapse into the groups the numbered segments make, and the group's key is the
   regex that matches exactly the leaves it was built from.

   Two rules of the engine's `claims` decide the shape of what is generated: a pattern
   matching no leaf raises, and two patterns over the same leaf raise. A group whose key
   is a function of the path satisfies both — every leaf belongs to exactly one. */

import type { PlanLeaf } from './api'

export interface LeafGroup {
  /* The override key: `model\.layers\.\d+\.mlp\.down_proj`. */
  pattern: string
  /* The same thing to read: `model.layers.*.mlp.down_proj`. */
  label: string
  leaves: number
  bytes: number
  /* What the plan gave them, `null` when they stay dense; `mixed` when they disagree,
     which only a hand-written override could produce. */
  bits: number | null
  groupSize: number | null
  mixed: boolean
}

const INDEX = /^\d+$/

const escape = (segment: string): string => segment.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

export function group(leaves: readonly PlanLeaf[]): LeafGroup[] {
  const groups = new Map<string, LeafGroup>()
  for (const leaf of leaves) {
    const segments = leaf.path.split('.')
    const pattern = segments.map((part) => (INDEX.test(part) ? '\\d+' : escape(part))).join('\\.')
    const found = groups.get(pattern)
    if (found === undefined) {
      groups.set(pattern, {
        pattern,
        label: segments.map((part) => (INDEX.test(part) ? '*' : part)).join('.'),
        leaves: 1,
        bytes: leaf.bytes,
        bits: leaf.bits,
        groupSize: leaf.group_size,
        mixed: false
      })
      continue
    }
    found.leaves += 1
    found.bytes += leaf.bytes
    if (found.bits !== leaf.bits || found.groupSize !== leaf.group_size) found.mixed = true
  }
  return [...groups.values()]
}

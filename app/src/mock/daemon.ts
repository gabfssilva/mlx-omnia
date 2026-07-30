/* A daemon-shaped fixture: the same routes `sideros-server` serves, answering the
   numbers the design mockup shows, so the app can be developed and screenshotted
   without an M5 and 63 GB of checkpoints. Shapes come from
   packages/sideros-server/src/sideros_server/*.py — where those disagree with this
   file, they are right.

   `deno task mock`, or `deno task dev:mock` to have the desktop shell start it. */

const PORT = Number(Deno.env.get('SIDEROS_MOCK_PORT') ?? 8642)
const STARTED = performance.now()
const VERSION = '0.1.0'

const GIB = 1024 ** 3
const gib = (n: number) => Math.round(n * GIB)

/* ── the machine ──────────────────────────────────────────────────────── */

const SYSTEM = {
  chip: 'Apple M5 Max',
  gpu_cores: 40,
  memory_bytes: 128 * GIB,
  bandwidth_theoretical_gbs: 614.0,
  bandwidth_sustained_gbs: 490.0,
  disk_free_bytes: gib(412),
  catalog: '/Users/you/.cache/huggingface/hub',
  version: VERSION,
  constants: {
    bandwidth_theoretical_gbs:
      "Apple's published figure for this chip. Nothing on the machine reports it.",
    bandwidth_sustained_gbs:
      'Measured, not published: what a serial chain of dependent kernels holds on this ' +
      'machine (~80% of theoretical), calibrated with bench/interleaved.py. Every ' +
      '% of ceiling divides by it.'
  }
}

/* ── the catalog ──────────────────────────────────────────────────────── */

interface Entry {
  id: string
  directory: string
  store: string
  architecture: string
  quantization: string | null
  dtype: string | null
  context: number | null
  bytes_on_disk: number
  bytes_per_token: number | null
  resident: boolean
}

/* Ceilings the mockup prints come out of these: 490e9 / bytes_per_token. */
const CATALOG: Entry[] = [
  entry('mlx-community/Qwen3-30B-A3B-4bit', 'qwen3_moe', '4-bit', 'bfloat16', 262144, 17.2, 1_711_000_000),
  entry('openai/gpt-oss-20b', 'gpt_oss', 'mxfp4', 'bfloat16', 131072, 12.9, 940_500_000),
  entry('mlx-community/gemma-3-12b-it-8bit', 'gemma3', '8-bit', 'bfloat16', 131072, 12.8, 12_250_000_000),
  entry('mlx-community/LFM2-8B-A1B-4bit', 'lfm2_moe', '4-bit', 'bfloat16', 32768, 4.5, 830_500_000),
  entry('local/Qwen3.5-VL-27B-4bit', 'qwen3_5_vision', '4-bit', 'bfloat16', 131072, 15.6, 14_410_000_000)
]

function entry(
  id: string,
  architecture: string,
  quantization: string | null,
  dtype: string,
  context: number,
  gigabytes: number,
  bytesPerToken: number
): Entry {
  const directory = `${SYSTEM.catalog}/models--${id.replace(/\//g, '--')}/snapshots/a1b2c3d`
  return {
    id,
    directory,
    store: directory.replace(/\/snapshots\/.*/, ''),
    architecture,
    quantization,
    dtype,
    context,
    bytes_on_disk: gib(gigabytes),
    bytes_per_token: bytesPerToken,
    resident: false
  }
}

const RESIDENT = new Map<string, { weights: number; kv: number; loaded_at: number; last_used: number }>([
  [CATALOG[0].id, { weights: gib(17.2), kv: gib(2.1), loaded_at: now() - 8040, last_used: now() - 12 }],
  [CATALOG[1].id, { weights: gib(12.9), kv: gib(1.2), loaded_at: now() - 3600, last_used: now() - 2 }]
])

/* ── benches, metrics ─────────────────────────────────────────────────── */

const BENCHES = [
  bench(CATALOG[0].id, 161.9, 142.0),
  bench(CATALOG[1].id, 248.7, 96.0),
  bench(CATALOG[2].id, 33.1, 310.0),
  bench(CATALOG[3].id, 302.4, 61.0)
]

function bench(model: string, rate: number, ttft: number) {
  const perToken = CATALOG.find((e) => e.id === model)?.bytes_per_token ?? null
  return {
    model,
    tokens_per_second: rate,
    ttft_ms: ttft,
    engine_version: VERSION,
    created_at: now() - 7200,
    ceiling_fraction: perToken === null ? null : rate / (490e9 / perToken)
  }
}

/* ── jobs ─────────────────────────────────────────────────────────────── */

interface JobView {
  id: string
  kind: string
  state: 'pending' | 'running' | 'ok' | 'error' | 'cancelled'
  progress: { message: string; completed: number; total: number | null }
  created_at: number
  updated_at: number
  error: string | null
}

/* One running download, the one the mockup draws at 2.31 / 4.48 GB. */
const JOBS = new Map<string, JobView>()
JOBS.set('9f2c41ab7d5e4c0f8b1a', {
  id: '9f2c41ab7d5e4c0f8b1a',
  kind: 'download',
  state: 'running',
  progress: {
    message: 'model-00002-of-00002.safetensors',
    completed: 2.31e9,
    total: 4.48e9
  },
  created_at: now() - 180,
  updated_at: now(),
  error: null
})
JOBS.set('1c7e0d93a4b2489f6a35', {
  id: '1c7e0d93a4b2489f6a35',
  kind: 'quantize',
  state: 'ok',
  progress: { message: 'wrote local/Qwen3-32B-4bit', completed: 1, total: 1 },
  created_at: now() - 5400,
  updated_at: now() - 4980,
  error: null
})

/* ── sessions (the contract the real Store grows in migration v2) ─────── */

interface Session {
  id: string
  title: string
  model: string
  created_at: number
  updated_at: number
  messages: { role: string; content: string }[]
}

const SESSIONS = new Map<string, Session>()
for (const [title, model, ago, messages] of [
  ['bf16 rounding floors', CATALOG[0].id, 30, THREAD()],
  ['Kernel fusion ideas', CATALOG[0].id, 7200, []],
  ['Router tie-breaking', CATALOG[1].id, 93600, []],
  ['LFM2 conv state', CATALOG[3].id, 345600, []],
  ['Sink attention notes', CATALOG[1].id, 518400, []]
] as [string, string, number, Session['messages']][]) {
  const id = crypto.randomUUID().replace(/-/g, '')
  SESSIONS.set(id, {
    id,
    title,
    model,
    created_at: now() - ago - 600,
    updated_at: now() - ago,
    messages
  })
}

function THREAD(): Session['messages'] {
  return [
    {
      role: 'user',
      content: "Why doesn't an N-row matmul round like N single-row matmuls in bf16?"
    },
    {
      role: 'assistant',
      content:
        "Because floating-point addition isn't associative, and the two shapes accumulate in " +
        'different orders. A GEMV reduces K serially per output; a batched GEMM tiles K and ' +
        'combines partials, so the same mathematical sum associates differently.\n\n' +
        'In bf16 each reorder can flip the last bit, and over a 40-layer trunk those flips ' +
        'compound into a measurable floor — which is why prefill vs. stepwise is never `== 0`, ' +
        'even when both paths are correct.'
    },
    { role: 'user', content: 'So what floor should a prefill-vs-stepwise test use?' }
  ]
}

/* ── the canned answer the fake stream types out ──────────────────────── */

const REASONING =
  'The question is about accumulation order. A batched GEMM tiles the K dimension ' +
  'differently than a GEMV, so partial sums associate differently — and bf16 addition is ' +
  'not associative.'

const ANSWER =
  'Measure it, never invent it: run prefill against step-by-step decode in the reference ' +
  'over the same checkpoint and record the observed difference as `noise.batching` in the ' +
  'fixture. The test then asserts against that floor rather than against zero, and a ' +
  'regression is a number that moved rather than a tolerance somebody relaxed.'

/* ── routing ──────────────────────────────────────────────────────────── */

function now(): number {
  return Date.now() / 1000
}

const ok = (body: unknown, status = 200) => Response.json(body, { status })
const refused = (status: number, detail: string) => Response.json({ detail }, { status })

function stream(frames: () => AsyncGenerator<string>): Response {
  const body = new ReadableStream<Uint8Array>({
    async start(controller) {
      const encoder = new TextEncoder()
      try {
        for await (const frame of frames()) controller.enqueue(encoder.encode(frame))
      } catch {
        /* the client walked away */
      }
      controller.close()
    }
  })
  return new Response(body, {
    headers: { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' }
  })
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

async function handle(request: Request): Promise<Response> {
  const url = new URL(request.url)
  const path = url.pathname
  const method = request.method

  if (path === '/admin/health') {
    return ok({
      status: 'ok',
      models: [...RESIDENT.keys()],
      pid: Deno.pid,
      uptime: (performance.now() - STARTED) / 1000,
      version: VERSION
    })
  }

  if (path === '/admin/system') return ok(SYSTEM)

  if (path === '/admin/state') {
    const models = [...RESIDENT].map(([id, live]) => ({
      id,
      weights_bytes: live.weights,
      kv_bytes: live.kv,
      loaded_at: live.loaded_at,
      last_used: live.last_used
    }))
    return ok({
      models,
      queue: { running: 1, waiting: 0 },
      resident_bytes: models.reduce((sum, m) => sum + m.weights_bytes, 0),
      kv_bytes: models.reduce((sum, m) => sum + m.kv_bytes, 0)
    })
  }

  if (path === '/admin/config') {
    if (method === 'PATCH') {
      const patch = (await request.json()) as Record<string, unknown>
      const known = Object.keys(CONFIG())
      for (const [key, value] of Object.entries(patch)) {
        if (known.includes(key)) PATCHED[key] = value
      }
    }
    return ok(CONFIG())
  }

  if (path === '/admin/models' && method === 'GET') {
    const listed = CATALOG.map((e) => ({ ...e, resident: RESIDENT.has(e.id) }))
    return ok(url.searchParams.get('resident') === 'true' ? listed.filter((e) => e.resident) : listed)
  }

  if (path === '/admin/models' && method === 'POST') {
    const body = (await request.json()) as { repo: string }
    return started('download', `fetching ${body.repo}`)
  }

  const residency = path.match(/^\/admin\/models\/(.+)\/residency$/)
  if (residency) {
    const id = decodeURIComponent(residency[1])
    const found = CATALOG.find((e) => e.id === id)
    if (!found) return refused(404, `${id} is not in the catalog`)
    if (method === 'PUT') {
      RESIDENT.set(id, {
        weights: found.bytes_on_disk,
        kv: gib(1),
        loaded_at: now(),
        last_used: now()
      })
      return started('load', `loading ${id}`)
    }
    if (method === 'DELETE') {
      if (!RESIDENT.delete(id)) return refused(404, `${id} is not resident`)
      return new Response(null, { status: 204 })
    }
  }

  const model = path.match(/^\/admin\/models\/(.+)$/)
  if (model && (method === 'GET' || method === 'DELETE')) {
    const id = decodeURIComponent(model[1])
    const found = CATALOG.find((e) => e.id === id)
    if (!found) return refused(404, `'${id}' is not in the catalog`)
    if (method === 'DELETE') return new Response(null, { status: 204 })
    return ok({ ...found, resident: RESIDENT.has(id) })
  }

  if (path === '/admin/benches') {
    const wanted = url.searchParams.get('model')
    return ok({
      method:
        'One measurement of this machine against itself, over the daemon queue. It is not a ' +
        'comparison against mlx-lm: that is bench/interleaved.py.',
      benches: wanted === null ? BENCHES : BENCHES.filter((b) => b.model === wanted)
    })
  }

  if (path === '/admin/metrics') return ok(METRICS())
  if (path === '/admin/metrics/events') {
    return stream(async function* () {
      for (;;) {
        yield `data: ${JSON.stringify(METRICS())}\n\n`
        await sleep(1000)
      }
    })
  }

  if (path === '/admin/jobs' && method === 'GET') {
    const listed = [...JOBS.values()].sort((a, b) => b.created_at - a.created_at)
    return ok(
      url.searchParams.get('active') === 'true'
        ? listed.filter((j) => j.state === 'pending' || j.state === 'running')
        : listed
    )
  }

  const jobEvents = path.match(/^\/admin\/jobs\/([^/]+)\/events$/)
  if (jobEvents) {
    const job = JOBS.get(jobEvents[1])
    if (!job) return refused(404, `no job '${jobEvents[1]}'`)
    return stream(async function* () {
      yield `data: ${JSON.stringify(job)}\n\n`
      while (job.state === 'running' || job.state === 'pending') {
        await sleep(400)
        const total = job.progress.total
        job.state = 'running'
        job.progress = {
          ...job.progress,
          completed: total === null ? job.progress.completed + 1e8 : Math.min(total, job.progress.completed + 6e7)
        }
        job.updated_at = now()
        if (total !== null && job.progress.completed >= total) job.state = 'ok'
        yield `data: ${JSON.stringify(job)}\n\n`
      }
    })
  }

  const job = path.match(/^\/admin\/jobs\/([^/]+)$/)
  if (job) {
    const found = JOBS.get(job[1])
    if (!found) return refused(404, `no job '${job[1]}'`)
    if (method === 'DELETE') {
      if (found.state !== 'running' && found.state !== 'pending') {
        return refused(409, `job '${found.id}' already finished as ${found.state}`)
      }
      return ok(found, 202)
    }
    return ok(found)
  }

  if (path === '/admin/hub/models') {
    const query = (url.searchParams.get('query') ?? '').toLowerCase()
    return ok(
      [
        { id: 'mlx-community/LFM2-8B-A1B-4bit', downloads: 184203, likes: 231 },
        { id: 'LiquidAI/LFM2-8B-A1B', downloads: 57410, likes: 604 },
        { id: 'mlx-community/Qwen3-30B-A3B-4bit', downloads: 96124, likes: 412 },
        { id: 'openai/gpt-oss-20b', downloads: 1204553, likes: 3102 }
      ].filter((hit) => hit.id.toLowerCase().includes(query))
    )
  }

  if (path === '/admin/hub/staging') return ok([])

  if (path === '/admin/quantizations/plan' && method === 'POST') {
    const body = (await request.json()) as { bits?: number; group_size?: number }
    return ok(PLAN(body.bits ?? 4, body.group_size ?? 64))
  }

  if (path === '/admin/quantizations' && method === 'POST') {
    const body = (await request.json()) as { repo: string }
    return started('quantize', `quantizing into ${body.repo}`)
  }

  if (path === '/admin/sessions' && method === 'GET') {
    return ok({
      sessions: [...SESSIONS.values()]
        .sort((a, b) => b.updated_at - a.updated_at)
        .map(({ messages, ...rest }) => ({ ...rest, message_count: messages.length }))
    })
  }

  if (path === '/admin/sessions' && method === 'POST') {
    const body = (await request.json().catch(() => ({}))) as { title?: string; model?: string }
    const created: Session = {
      id: crypto.randomUUID().replace(/-/g, ''),
      title: body.title ?? 'New chat',
      model: body.model ?? '',
      created_at: now(),
      updated_at: now(),
      messages: []
    }
    SESSIONS.set(created.id, created)
    return ok(created, 201)
  }

  const sessionMessages = path.match(/^\/admin\/sessions\/([^/]+)\/messages$/)
  if (sessionMessages && method === 'PUT') {
    const found = SESSIONS.get(sessionMessages[1])
    if (!found) return refused(404, `no session '${sessionMessages[1]}'`)
    const body = (await request.json()) as { messages: Session['messages'] }
    found.messages = body.messages
    found.updated_at = now()
    return ok(found)
  }

  const session = path.match(/^\/admin\/sessions\/([^/]+)$/)
  if (session) {
    const found = SESSIONS.get(session[1])
    if (!found) return refused(404, `no session '${session[1]}'`)
    if (method === 'DELETE') {
      SESSIONS.delete(found.id)
      return new Response(null, { status: 204 })
    }
    if (method === 'PATCH') {
      const body = (await request.json()) as { title?: string; model?: string }
      if (body.title !== undefined) found.title = body.title
      if (body.model !== undefined) found.model = body.model
      found.updated_at = now()
    }
    return ok(found)
  }

  if (path === '/api/openai/v1/models') {
    return ok({
      object: 'list',
      data: CATALOG.map((e) => ({ id: e.id, object: 'model', created: 0, owned_by: 'sideros' }))
    })
  }

  if (path === '/api/openai/v1/chat/completions' && method === 'POST') {
    const body = (await request.json()) as { model: string; stream?: boolean }
    return body.stream ? completionStream(body.model) : ok(completion(body.model))
  }

  return refused(404, `nothing is mounted at ${path}`)
}

function started(kind: string, message: string): Response {
  const created: JobView = {
    id: crypto.randomUUID().replace(/-/g, '').slice(0, 20),
    kind,
    state: 'running',
    progress: { message, completed: 0, total: 1e9 },
    created_at: now(),
    updated_at: now(),
    error: null
  }
  JOBS.set(created.id, created)
  return new Response(JSON.stringify(created), {
    status: 202,
    headers: { 'content-type': 'application/json', location: `/admin/jobs/${created.id}` }
  })
}

/* What PATCH /admin/config has accepted so far. The real daemon answers with the config
   it now holds, and the app adopts that answer — a mock that echoed the defaults would
   make every control snap back. */
const PATCHED: Record<string, unknown> = {}

function CONFIG() {
  const setting = (key: string, value: unknown, effect: string, note?: string) => ({
    value: key in PATCHED ? PATCHED[key] : value,
    effect,
    note: note ?? null
  })
  return {
    memory_limit_bytes: setting('memory_limit_bytes', gib(120), 'applied'),
    idle_ttl_seconds: setting('idle_ttl_seconds', 1800, 'applied'),
    max_concurrent_requests: setting('max_concurrent_requests', 1, 'inert', 'Kept, not honoured: the FCFS queue serializes generation at effective depth 1.'),
    prefix_cache_bytes: setting('prefix_cache_bytes', GIB, 'applied'),
    port: setting('port', PORT, 'restart'),
    api_key: setting('api_key', null, 'applied'),
    catalog_directory: setting('catalog_directory', SYSTEM.catalog, 'inert', 'Kept, not honoured, and a restart does not honour it either.'),
    not_resident: setting('not_resident', 'load', 'applied')
  }
}

function METRICS() {
  const perToken = CATALOG[1].bytes_per_token
  return {
    live: {
      model: CATALOG[1].id,
      state: 'running',
      prompt_tokens: 1284,
      completion_tokens: 210,
      started_at: now() - 4,
      ttft: 0.096,
      tokens_per_second: 248.7,
      bytes_per_token: perToken,
      ceiling_fraction: 248.7 / (490e9 / (perToken ?? 1))
    },
    requests: [],
    models: BENCHES.map((b) => ({
      model: b.model,
      requests: 12,
      prompt_tokens: 15400,
      completion_tokens: 8210,
      ttft: b.ttft_ms / 1000,
      tokens_per_second: b.tokens_per_second,
      bytes_per_token: CATALOG.find((e) => e.id === b.model)?.bytes_per_token ?? null,
      ceiling_fraction: b.ceiling_fraction
    }))
  }
}

/* The mockup's quantize screen: Qwen3-32B dense bf16 65.5 GB → 19.0 GB at 4.75
   bits/weight, with lm_head overridden to dense. */
function PLAN(bits: number, groupSize: number) {
  const leaves = [
    { path: 'model.embed_tokens', kind: 'Embedding', shape: [151936, 5120] as number[], bits, bytes: gib(1.1) },
    { path: 'model.layers.*.self_attn.{q,k,v,o}_proj', kind: 'Linear', shape: [5120, 5120] as number[], bits, bytes: gib(10.5) },
    { path: 'model.layers.*.mlp.{gate,up,down}_proj', kind: 'Linear', shape: [25600, 5120] as number[], bits, bytes: gib(52.8) },
    { path: 'lm_head', kind: 'Linear', shape: [151936, 5120] as number[], bits: null, bytes: gib(1.1) }
  ]
  const weights = 32_500_000_000
  const total = gib(19.0)
  return {
    leaves,
    total_bytes: total,
    weights,
    bits_per_weight: (total * 8) / weights,
    entry_bytes: total,
    group_size: groupSize
  }
}

function completion(model: string) {
  return {
    id: `chatcmpl-${crypto.randomUUID().replace(/-/g, '')}`,
    object: 'chat.completion',
    created: Math.floor(now()),
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: ANSWER, reasoning_content: REASONING },
        finish_reason: 'stop'
      }
    ],
    usage: { prompt_tokens: 128, completion_tokens: 96, total_tokens: 224 }
  }
}

/* Token by token, reasoning first, in the frames the dialect defines: the chat view's
   accumulator is exercised for real. */
function completionStream(model: string): Response {
  const id = `chatcmpl-${crypto.randomUUID().replace(/-/g, '')}`
  const created = Math.floor(now())
  const chunk = (delta: Record<string, unknown>, finish: string | null) =>
    `data: ${JSON.stringify({
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }]
    })}\n\n`

  return stream(async function* () {
    yield chunk({ role: 'assistant', content: '' }, null)
    await sleep(140)
    for (const token of tokens(REASONING)) {
      yield chunk({ reasoning_content: token }, null)
      await sleep(12)
    }
    for (const token of tokens(ANSWER)) {
      yield chunk({ content: token }, null)
      await sleep(22)
    }
    yield chunk({}, 'stop')
    yield `data: ${JSON.stringify({
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [],
      usage: { prompt_tokens: 128, completion_tokens: 96, total_tokens: 224 }
    })}\n\n`
    yield 'data: [DONE]\n\n'
  })
}

const tokens = (text: string): string[] => text.match(/\s*\S+/g) ?? []

Deno.serve({ port: PORT, hostname: '127.0.0.1' }, handle)

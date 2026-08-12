"""Where what is neither a block nor a checkpoint lives: the three dialects, the port, the
ceiling, idle unloading, the engine process and the theme.

Controlling the daemon is this process's own business (see daemon.py): Stop and Restart
are offered only on the daemon this window started, and any other one says where to stop
it instead.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Coroutine

import flet as ft

from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.daemon import FOREIGN, Daemon
from mlx_omnia.app.api.engine import GIB, CatalogEntry, Engine
from mlx_omnia.app.ui import parts, theme
from mlx_omnia.app.ui.chrome import Chrome
from mlx_omnia.app.ui.format import display_name, gb
from mlx_omnia.app.ui.hooks import act, use_tick
from mlx_omnia.app.ui.theme import t

NO_ENGINE = "No engine is answering, and this window did not start one."

DIALECTS = [
    ("OpenAI", "/api/openai/v1"),
    ("Anthropic", "/api/anthropic"),
    ("Gemini", "/api/gemini"),
]

TTLS: list[tuple[str, int | None]] = [
    ("10 min", 600),
    ("30 min", 1800),
    ("2 h", 7200),
    ("Never", None),
]

CLIENTS = ["Claude Code", "Zed", "aider"]

# Claude Code picks a model per alias: ANTHROPIC_MODEL is what it starts on, and each
# ANTHROPIC_DEFAULT_<TIER>_MODEL is what that alias resolves to when /model switches.
TIERS = ["opus", "sonnet", "haiku", "fable"]

HOME = re.compile(r"^/(?:Users|home)/[^/]+")


def homeless(path: str) -> str:
    return HOME.sub("~", path)


def slots(client: str) -> list[str]:
    """Zed's snippet names no model; aider's line names one; Claude Code names one per
    alias."""
    if client == "Claude Code":
        return ["default", *TIERS]
    if client == "aider":
        return ["default"]
    return []


def line(client: str, base: str, key: str, picks: dict[str, str], fallback: str) -> str:
    model = picks.get("default") or fallback
    if client == "Zed":
        return f'"language_models": {{ "openai": {{ "api_url": "{base}/api/openai/v1" }} }}'
    if client == "aider":
        return f"aider --openai-api-base {base}/api/openai/v1 --model {model}"
    env = [f"ANTHROPIC_BASE_URL={base}/api/anthropic"]
    if key:
        env.append(f"ANTHROPIC_AUTH_TOKEN={key}")
    if picks.get("default"):
        env.append(f"ANTHROPIC_MODEL={picks['default']}")
    for tier in TIERS:
        if picks.get(tier):
            env.append(f"ANTHROPIC_DEFAULT_{tier.upper()}_MODEL={picks[tier]}")
    # A local model answers slower than the API and has nothing to report home about: the
    # default timeout would cut a long generation short.
    env += ["API_TIMEOUT_MS=3000000", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"]
    return " ".join(env) + " claude"


def holds(budget: int | None, catalog: list[CatalogEntry]) -> tuple[str, int] | None:
    """What the prefix budget buys, read on a model that is actually loaded.

    A byte count on its own says nothing here: the trie holds conversations, and how many
    tokens fit in a gigabyte is the checkpoint's `kv_bytes_per_token` away. The costliest
    resident cache is the one quoted — it is the model that runs out first, and a screen
    that quoted the cheapest would promise a length the expensive one never reaches.
    """
    if budget is None:
        return None
    worst: tuple[str, int] | None = None
    for entry in catalog:
        per_token = entry.get("kv_bytes_per_token")
        if not entry["resident"] or not per_token or per_token <= 0:
            continue
        tokens = int(budget // per_token)
        if worst is None or tokens < worst[1]:
            worst = (entry["id"], tokens)
    return worst


def thousands(tokens: int) -> str:
    return f"{round(tokens / 1000)}k" if tokens >= 1000 else str(tokens)


def _process(daemon: Daemon, up: bool) -> str:
    if daemon.owned:
        return "started by this app"
    return "started elsewhere" if up else "unknown"


def elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, minutes = total // 3600, (total % 3600) // 60
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min"
    return f"{total} s"


@ft.component
def Settings(engine: Engine, chrome: Chrome, daemon: Daemon) -> ft.Control:
    desk, _ = ft.use_state(Desk)
    # A copied endpoint says so for 1.4 s, and nothing else redraws it back.
    use_tick(0.5)
    return desk.tree(engine, chrome, daemon)


@ft.observable
class Desk:
    """What this screen holds that the daemon does not: which client is being wired, what
    was just copied, and the ceiling under the thumb."""

    def __init__(self) -> None:
        self.client = "Claude Code"
        self.picks: dict[str, str] = {}
        self.copied: tuple[str, float] | None = None
        self.failure: str | None = None
        self.ceiling_gb: int | None = None
        # The ceiling rail's measured width, which has to outlive the render that the
        # next frame does. Private, so measuring it is not a state change.
        self._rail = [1.0]

    def tree(self, engine: Engine, chrome: Chrome, daemon: Daemon) -> ft.Control:
        head: list[ft.Control] = [parts.head("Settings", self._machine(engine))]
        blocks: list[ft.Control] = []
        if self.failure is not None:
            blocks.append(
                ft.Container(content=parts.err(self.failure), margin=ft.Margin.only(bottom=9))
            )
        blocks.append(
            parts.cols(
                [self._endpoints(engine), self._clients(engine), *self._storage(engine)],
                [self._memory(engine), self._engine(engine, daemon), self._appearance(chrome)],
            )
        )
        return ft.Container(
            expand=True,
            bgcolor=t().win,
            padding=12,
            content=parts.pane([*head, parts.body(blocks)]),
        )

    def _machine(self, engine: Engine) -> str | None:
        system = engine.system
        if system is None:
            return None
        return (
            f"{system['chip']} · {system['gpu_cores']} GPU cores · "
            f"{gb(system['memory_bytes'], 0)} GB · {system['bandwidth_sustained_gbs']} GB/s"
        )

    # ── left column ───────────────────────────────────────────────────────

    def _base(self, engine: Engine) -> str:
        port = engine.config.get("port")
        value = 8642 if port is None else port["value"]
        return f"http://127.0.0.1:{value}"

    def _endpoints(self, engine: Engine) -> ft.Control:
        base = self._base(engine)
        port = engine.config.get("port")
        rows: list[ft.Control] = [
            parts.urlrow(
                f"{base}{path}",
                parts.btn(
                    "Copied" if self._is_copied(path) else "Copy",
                    lambda path=path, base=base: self._copy(path, f"{base}{path}"),
                    "quiet",
                ),
            )
            for _, path in DIALECTS
        ]
        rows.append(
            ft.Row(
                [
                    parts.note("Port"),
                    parts.field(
                        str(8642 if port is None else port["value"]),
                        self._port,
                        width=82,
                        mono=True,
                    ),
                    parts.note("Changing the port restarts the engine."),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
        return parts.blk("Endpoints", rows)

    def _clients(self, engine: Engine) -> ft.Control:
        base = self._base(engine)
        key = engine.config.get("api_key")
        api_key = key["value"] if key is not None and isinstance(key.get("value"), str) else ""
        first = engine.catalog[0]["id"] if engine.catalog else "model-id"
        command = line(self.client, base, str(api_key), self.picks, first)

        options = [("", "—"), *((e["id"], e["id"]) for e in engine.catalog)]
        rows: list[ft.Control] = [
            parts.urlrow(
                command,
                parts.btn(
                    "Copied" if self._is_copied(self.client) else "Copy",
                    lambda: self._copy(self.client, command),
                    "quiet",
                ),
            ),
            ft.Row(
                [
                    parts.fchip(name, name == self.client, lambda name=name: self._client(name))
                    for name in CLIENTS
                ],
                spacing=6,
            ),
        ]
        names = slots(self.client)
        for index, slot in enumerate(names):
            rows.append(
                ft.Container(
                    padding=ft.Padding(left=0, right=0, top=5.5, bottom=5.5),
                    border=None
                    if index == len(names) - 1
                    else ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
                    content=ft.Row(
                        [
                            ft.Text(slot.capitalize(), style=theme.sans(11, t().fg3), no_wrap=True),
                            ft.Container(expand=True),
                            parts.pick(
                                [
                                    ("", "—" if slot == "default" else "follows the default"),
                                    *options[1:],
                                ],
                                self.picks.get(slot, ""),
                                lambda value, slot=slot: self._pick(slot, value),
                                width=260,
                            ),
                        ],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                )
            )
        rows.append(
            parts.note(
                "Default is what Claude Code starts on; each alias is what /model resolves to."
                if self.client == "Claude Code"
                else "Copy writes the whole line, ready to paste."
            )
        )
        return parts.blk("Clients", rows)

    def _storage(self, engine: Engine) -> list[ft.Control]:
        system = engine.system
        if system is None:
            return []
        on_disk = sum(entry["bytes_on_disk"] for entry in engine.catalog)
        return [
            parts.blk(
                "Storage",
                [
                    ft.Column(
                        [
                            parts.kvr(
                                "Checkpoints", f"{gb(on_disk)} GB", f"· {len(engine.catalog)}"
                            ),
                            parts.kvr("Folder", homeless(system["catalog"])),
                            parts.kvr(
                                "Disk free", f"{gb(system['disk_free_bytes'], 0)} GB", last=True
                            ),
                        ],
                        spacing=0,
                        tight=True,
                    )
                ],
            )
        ]

    # ── right column ──────────────────────────────────────────────────────

    def _memory(self, engine: Engine) -> ft.Control:
        system = engine.system
        configured = engine.ceiling
        total = None if system is None else system["memory_bytes"]
        ceiling = self.ceiling_gb if self.ceiling_gb is not None else (
            None if configured is None else round(configured / GIB)
        )

        rows: list[ft.Control] = []
        if total is not None and ceiling is not None:
            in_use = 0 if engine.state is None else (
                engine.state["resident_bytes"] + engine.state["kv_bytes"]
            )
            rows += [
                parts.sublab(f"ceiling {ceiling} GB", f"{gb(total, 0)} GB"),
                parts.slider(
                    ceiling,
                    4,
                    round(total / GIB),
                    self._drag_ceiling,
                    self._commit_ceiling,
                    self._rail,
                ),
                parts.note(
                    "Above the ceiling the engine refuses to load rather than letting macOS "
                    f"swap. {gb(in_use)} GB in use right now."
                ),
            ]
        else:
            rows.append(parts.note("Waiting for the engine."))

        prefix = engine.config.get("prefix_cache_bytes")
        budget = None if prefix is None or not isinstance(prefix["value"], int | float) else int(
            prefix["value"]
        )
        if budget is not None:
            conversation = holds(budget, engine.catalog)
            prose = (
                f"Conversation cache: {gb(budget)} GB per loaded model, inside the ceiling above."
            )
            if conversation is not None:
                prose += (
                    f" About {thousands(conversation[1])} tokens on "
                    f"{display_name(conversation[0])} — a turn that continues one of those is "
                    "answered without reading the whole conversation again."
                )
            rows.append(parts.note(prose))

        ttl = engine.config.get("idle_ttl_seconds")
        current = "Never" if ttl is not None and ttl["value"] is None else next(
            (label for label, secs in TTLS if ttl is not None and ttl["value"] == secs), ""
        )
        rows.append(
            ft.Row(
                [
                    ft.Container(content=parts.note("Unload a model after"), expand=True),
                    parts.seg([(label, label) for label, _ in TTLS], current, self._ttl),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
        policy = engine.config.get("not_resident")
        rows.append(
            ft.Row(
                [
                    ft.Container(
                        content=parts.note("A request for a model that is not loaded"), expand=True
                    ),
                    parts.seg(
                        [("load", "Loads it"), ("fail", "Fails")],
                        "" if policy is None else str(policy["value"]),
                        lambda value: self._commit({"not_resident": value}),
                    ),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
        return parts.blk("Memory", rows)

    def _engine(self, engine: Engine, daemon: Daemon) -> ft.Control:
        health = engine.health
        system = engine.system
        mlx = None if system is None else system["constants"].get("mlx")
        return parts.blk(
            "Engine",
            [
                ft.Column(
                    [
                        parts.kvr(
                            "Status",
                            "not answering" if health is None else "running",
                            None if health is None else f"· {elapsed(health['uptime'])}",
                        ),
                        parts.kvr(
                            "Process",
                            _process(daemon, health is not None),
                            None if health is None else f"· pid {health['pid']}",
                        ),
                        parts.kvr(
                            "Version",
                            f"mlx_omnia {'—' if system is None else system['version']}",
                            None if mlx is None else f"· mlx {mlx}",
                            last=True,
                        ),
                    ],
                    spacing=0,
                    tight=True,
                ),
                ft.Row(
                    [
                        parts.btn("Restart", lambda: act(self._control(daemon.restart))),
                        parts.btn("Stop", lambda: act(self._control(daemon.stop)), "danger"),
                    ],
                    spacing=6,
                )
                if daemon.owned
                else parts.note(FOREIGN if health is not None else NO_ENGINE),
            ],
        )

    def _appearance(self, chrome: Chrome) -> ft.Control:
        return parts.blk(
            "Appearance",
            [
                parts.seg(
                    [("light", "Light"), ("dark", "Dark"), ("system", "System")],
                    chrome.mode,
                    chrome.choose,
                )
            ],
        )

    # ── what this screen can do ───────────────────────────────────────────

    def _is_copied(self, key: str) -> bool:
        if self.copied is None or self.copied[0] != key:
            return False
        return time.time() - self.copied[1] < 1.4

    def _copy(self, key: str, text: str) -> None:
        parts.CLIPBOARD.set(text)
        self.copied = (key, time.time())

    def _client(self, name: str) -> None:
        self.client = name

    def _pick(self, slot: str, value: str) -> None:
        self.picks[slot] = value

    def _ttl(self, label: str) -> None:
        """"Never" is a null the daemon means, so the key is sent carrying it rather than
        left out."""
        self._commit({"idle_ttl_seconds": next(s for text, s in TTLS if text == label)})

    def _port(self, value: str) -> None:
        if value.isdigit():
            self._commit({"port": int(value)})

    def _drag_ceiling(self, value: float) -> None:
        self.ceiling_gb = round(value)

    def _commit_ceiling(self, _: float) -> None:
        if self.ceiling_gb is not None:
            self._commit({"memory_limit_bytes": self.ceiling_gb * GIB})

    def _commit(self, patch: dict[str, object]) -> None:
        self.failure = None
        act(self._patch(patch))

    async def _patch(self, patch: dict[str, object]) -> None:
        try:
            await engine_api.patch_config(patch)
        except Exception as error:  # noqa: BLE001 — a refusal is shown, not raised
            self.failure = str(error)

    async def _control(self, action: Callable[[], Coroutine[None, None, None]]) -> None:
        self.failure = None
        try:
            await action()
        except Exception as error:  # noqa: BLE001 — a refusal is shown, not raised
            self.failure = str(error)

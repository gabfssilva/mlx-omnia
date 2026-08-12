"""What the window will one day read off the daemon, invented in one place.

Every number is plausible for the M5 Max the app runs on, so the mockup's proportions
are real ones: the band is still a ruler, just of invented bytes. Wiring a screen later
means replacing its import of this module, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

TOTAL_GB = 128.0
CEILING_GB = 96.0
PORT = "8642"


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    arch: str
    kind: str
    quant: str
    size_gb: float
    active_gb: float
    context: str
    state: str
    """resident | disk | hub | downloading"""
    material: int | None = None
    kv_gb: float = 0.0
    progress: float | None = None
    decode: str | None = None
    ceiling_share: str | None = None
    blurb: str = ""

    @property
    def resident_gb(self) -> float:
        return self.size_gb + self.kv_gb


MODELS: tuple[Model, ...] = (
    Model(
        id="mlx-community/Qwen3-30B-A3B-Instruct",
        name="Qwen3-30B-A3B-Instruct",
        arch="qwen3_moe",
        kind="MoE",
        quant="4-bit · mixed",
        size_gb=17.2,
        active_gb=2.1,
        context="262,144",
        state="resident",
        material=0,
        kv_gb=3.1,
        decode="31.4 tok/s",
        ceiling_share="89% of ceiling",
        blurb="MoE from the Qwen3 family: 30.5 B parameters with 3.3 B active per "
        "token, 128 experts with 8 routed. Native context is 262k.",
    ),
    Model(
        id="mlx-community/Nemotron-H-47B",
        name="Nemotron-H-47B",
        arch="nemotron_h",
        kind="Hybrid",
        quant="4-bit · mixed",
        size_gb=26.4,
        active_gb=26.4,
        context="131,072",
        state="resident",
        material=1,
        kv_gb=2.1,
        decode="21.3 tok/s",
        ceiling_share="92% of ceiling",
        blurb="Mamba-Transformer hybrid from the Nemotron-H family: attention layers "
        "interleaved with state blocks, which shrinks the cache at long context.",
    ),
    Model(
        id="mlx-community/Llama-3.3-70B-Instruct",
        name="Llama-3.3-70B-Instruct",
        arch="llama",
        kind="Dense",
        quant="4-bit",
        size_gb=39.7,
        active_gb=39.7,
        context="131,072",
        state="downloading",
        progress=0.62,
        blurb="Dense 70 B from the Llama 3.3 family, instruction-tuned.",
    ),
    Model(
        id="mlx-community/gpt-oss-20b",
        name="GPT-OSS-20B",
        arch="gpt_oss",
        kind="MoE",
        quant="mxfp4",
        size_gb=12.1,
        active_gb=3.6,
        context="131,072",
        state="disk",
        decode="58.7 tok/s",
        ceiling_share="84% of ceiling",
        blurb="OpenAI's open MoE: 21 B parameters with 3.6 B active, published "
        "already quantized in mxfp4.",
    ),
    Model(
        id="zai-org/GLM-4.5-Air",
        name="GLM-4.5-Air",
        arch="glm4_moe",
        kind="MoE",
        quant="bf16",
        size_gb=61.0,
        active_gb=12.0,
        context="131,072",
        state="hub",
        blurb="MoE from the GLM-4.5 family in full precision — 106 B parameters "
        "with 12 B active per token.",
    ),
)


def residents() -> list[Model]:
    return [model for model in MODELS if model.state == "resident"]


def used_gb() -> float:
    return sum(model.resident_gb for model in residents())


@dataclass(frozen=True)
class Turn:
    role: str
    text: str
    meta: str | None = None


THREAD: tuple[Turn, ...] = (
    Turn("user", "why is decode below the 610 GB/s ceiling?"),
    Turn(
        "assistant",
        "With 17.2 GB active per token, the physical ceiling gives ~35 tok/s — you "
        "measured 31.4, which is 89% of it. The remaining 11% is the cost of MoE "
        "routing and the KV step, not a bug. To close the gap, the next target is "
        "the router's gather kernel.",
        meta="31.4 tok/s · 42 ms TTFT · 1,204 tokens",
    ),
)

CANNED_REPLY = Turn(
    "assistant",
    "This window is still a mockup: nothing actually answered. Once Chat is wired "
    "to the daemon, the reply arrives here, with the turn's numbers underneath.",
    meta="0.0 tok/s · mock",
)

CHATS: tuple[str, ...] = (
    "RoPE scaling on Nemotron",
    "Sampling profiles",
    "Release draft",
    "Decode ceiling on M5",
)

TRACE: tuple[tuple[str, str, str], ...] = (
    ("14:32", "Qwen3-30B-A3B loaded", "11.4 s"),
    ("14:35", "Chat · 1,204 tokens", "31.4 tok/s"),
    ("14:41", "Nemotron-H-47B loaded", "18.2 s"),
)

FILES: tuple[tuple[str, str], ...] = (
    ("config.json", "1.8 kB"),
    ("model-00001-of-00006.safetensors", "4.9 GB"),
    ("tokenizer.json", "11.4 MB"),
)

PROFILES: tuple[tuple[str, str], ...] = (
    ("default", "temp 0.7 · top-p 0.95"),
    ("precise", "temp 0.2 · no prompt"),
)

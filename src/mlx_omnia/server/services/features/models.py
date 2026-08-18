from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mlx_omnia.engine.quant.quantization import MXFP, NVFP, Affine, Quantization


class Speculation(BaseModel):
    """Speculative decoding, by whichever technique this model has one for.

    - `dflash` is a **second checkpoint**, so `drafter` names it and is required.
    - `mtp` is a head **inside this model's own checkpoint**, so `drafter` must stay empty.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["dflash", "mtp"] | None = None
    drafter: str | None = None
    block_size: int | None = Field(default=None, ge=1)
    """How many ids to propose a round, `None` for the pair's own default."""

    @model_validator(mode="after")
    def _named(self) -> "Speculation":
        if self.kind == "dflash" and not self.drafter:
            raise ValueError("dflash speculation needs a drafter checkpoint to load")
        if self.kind == "mtp" and self.drafter:
            raise ValueError(
                f"mtp speculation drafts with the model's own head, not with {self.drafter!r}"
            )
        if self.kind is None and self.drafter:
            raise ValueError(f"{self.drafter!r} is named with no kind: speculation is off")
        return self


def format_of(spelled: str) -> Quantization:
    """The format a row names, as the object a cache stores its rows with.

    `name/bits/group_size` — the spelling `QuantizedKVCache.signature` already puts in the
    prefix key, so a policy and the files written under it read alike."""
    name, _, rest = spelled.partition("/")
    bits, _, group = rest.partition("/")
    if not bits.isdigit() or not group.isdigit():
        raise ValueError(f"{spelled!r} is not a format: expected name/bits/group_size")
    match name:
        case "affine":
            return Affine(group_size=int(group), bits=int(bits))
        case "mxfp4" | "mxfp8":
            return MXFP(name, group_size=int(group), bits=int(bits))
        case "nvfp4":
            return NVFP(group_size=int(group), bits=int(bits))
        case _:
            raise ValueError(f"unknown quantization format {name!r}")


class KvCache(BaseModel):
    """The KV cache of this model, stored compressed. Naming the formats is what turns the
    feature on; both sides or neither, and `save` refuses half a policy."""

    model_config = ConfigDict(extra="forbid")

    k: str | None = None
    v: str | None = None
    start_tokens: int = Field(default=0, ge=0)
    """How many tokens at the head of the context stay dense."""

    @field_validator("k", "v")
    @classmethod
    def _spellable(cls, value: str | None) -> str | None:
        if value is not None:
            format_of(value)
        return value


class Features(BaseModel):
    """Every switch, each one optional. Adding one is a field here and a row nowhere: the
    column is JSON text on both levels."""

    model_config = ConfigDict(extra="forbid")

    speculation: Speculation | None = None
    kv_cache: KvCache | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_dflash(cls, data: object) -> object:
        """`speculation` used to be `dflash`, and the rows already on users' disks say so.
        Migrated on read rather than by a pass over the store; the next write puts the new
        spelling down."""
        if not isinstance(data, dict) or "dflash" not in data:
            return data
        moved = dict(data)
        old = moved.pop("dflash")
        if "speculation" in moved:
            raise ValueError("features carry both `dflash` and `speculation`")
        if old is None:
            moved["speculation"] = None
        elif isinstance(old, dict):
            named = old.get("drafter")
            moved["speculation"] = {**old, "kind": "dflash" if named else None}
        else:
            raise ValueError(f"`dflash` is not an object: {old!r}")
        return moved


def resolve(model: Features, profile: Features | None) -> Features:
    """The profile over the model, field by field. A field the profile leaves `None` is one
    it does not opine on, so the model's stands."""
    if profile is None:
        return model
    filled = {
        name: value
        for name, value in vars(model).items()
        if value is not None and getattr(profile, name) is None
    }
    return profile.model_copy(update=filled)


def parse(text: str) -> Features:
    """The column is JSON text, so the way back is a parse: a row written by an older
    version fails here, named, instead of reaching the engine as a switch nobody set."""
    return Features.model_validate_json(text)

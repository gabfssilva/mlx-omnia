"""The compressed-KV gate: whether a policy can hold on a trunk, and putting it there."""

import asyncio
from functools import partial

import mlx.core as mx

from mlx_omnia import LanguageModel, ModelInput
from mlx_omnia.engine.core import api
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.quantizing import Quantizing, admits
from mlx_omnia.server.runtime.environment import Compression, Environment, KvCompression
from mlx_omnia.server.runtime.grammars import Compiling
from mlx_omnia.server.runtime.residency import Residency, refuse
from mlx_omnia.server.runtime.walks import trunk_of


def _probe(trunk: api.LanguageModel[LayerCache], policy: Compression) -> str | None:
    """One short prefill and one decode step through the wrapped trunk, `None` when it ran.

    The half of the gate arithmetic cannot answer. A family that reaches its cache with
    `update_and_fetch` never enters `QuantizedKVCache.attend`, and the first thing that says so is
    the attribute that is not there. Two ids and then one, because a policy that survives the
    prefill and dies on the step appending to what the prefill wrote is the failure a single
    forward misses.

    Built with `start_tokens=0` whatever the policy says: what has to be reached is the compressed
    region, and a dense head longer than the probe's three ids would let every policy pass by
    never entering it.
    """
    wrapped = Quantizing(trunk, policy.k_format, policy.v_format)
    try:
        cache = wrapped.make_cache()
        mx.eval(wrapped(mx.array([[0, 1]]), cache))
        mx.eval(wrapped(mx.array([[1]]), cache))
    except Exception as failure:
        return f"{type(failure).__name__}: {failure}"
    return None


def _verdict(
    environment: Environment,
    model_id: str,
    trunk: api.LanguageModel[LayerCache],
    policy: Compression,
) -> str | None:
    """Why this policy cannot hold on this trunk, `None` when it can. Run off the loop, for the
    reason the loader is: the probe is a forward, and the catalog scan behind the width stats
    every file of every checkpoint."""
    width = environment.kv_head_width(model_id)
    if width is None:
        return (
            "the catalog cannot read this checkpoint's KV head width, and a format whose "
            "groups nobody checked would be quantizing a shape it may not describe"
        )
    if not admits(width, policy.k_format, policy.v_format):
        return f"a KV head of {width} does not close the groups both formats pack in"
    return _probe(trunk, policy)


class Compressing(Compiling):
    async def _compress(
        self, model_id: str, model: LanguageModel[ModelInput], entry: Residency
    ) -> LanguageModel[ModelInput]:
        """The model this request generates through, under whatever KV policy its settings ask for
        — with the verdict on that policy written to the record either way.

        The gate has two halves because they cost different things. `admits` is arithmetic over
        the head's width against the groups both formats pack in, and it answers with no forward
        at all. The probe runs the trunk, and only exists for what no config predicts. Both
        answers are memoized on `(model, policy)`.

        Nothing is cached about the *wrapping*: `Quantizing` holds a reference and no weights, so
        building one per request costs a `list` comprehension the first time the request makes a
        cache. What that buys is that a patched policy is in force for the next request rather
        than for the next load.
        """
        environment = self._environment
        policy = None if environment is None else environment.compression(model_id)
        found = trunk_of(model)
        if policy is None:
            entry.kv_cache = None
            if found is not None:
                # A policy withdrawn while the model is resident. Put back rather than left
                # standing: the row is what the daemon answers with, and a trunk still wrapped
                # would go on compressing for a switch the screen shows as off.
                found[0].model = found[1]
            return model
        assert environment is not None
        if found is None:
            refuse(
                model_id,
                entry,
                "nothing under this model's facades answers make_cache, so there is no trunk "
                "for a cache policy to be about",
            )
        holder, trunk = found
        key = (model_id, policy)
        if key not in self._quantizable:
            self._quantizable[key] = await asyncio.get_running_loop().run_in_executor(
                self._model_thread, partial(_verdict, environment, model_id, trunk, policy)
            )
        refusal = self._quantizable[key]
        if refusal is not None:
            refuse(model_id, entry, refusal)
        holder.model = Quantizing(
            trunk, policy.k_format, policy.v_format, start_tokens=policy.start_tokens
        )
        entry.kv_cache = KvCompression(applied=True)
        return model

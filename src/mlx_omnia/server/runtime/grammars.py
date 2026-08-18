"""The token table a resident model's grammars compile against, and the walks off it."""

import asyncio
import json
from collections.abc import Mapping, Sequence

from mlx_omnia import LanguageModel, ModelInput
from mlx_omnia.engine.generate import Constraint
from mlx_omnia.engine.grammar import Vocabulary
from mlx_omnia.engine.language import tokenizer_of
from mlx_omnia.engine.parsers import ToolFamily
from mlx_omnia.server.runtime.admission import Admitting
from mlx_omnia.server.runtime.errors import NotConstrainable
from mlx_omnia.server.runtime.residency import Residency
from mlx_omnia.server.runtime.walks import stop_ids


def _vocabulary(model_id: str, model: LanguageModel[ModelInput], size: int | None) -> Vocabulary:
    """The token table this model's grammars compile against. Off the loop: it decodes every id
    the head can draw and hands the table to Rust, which is 0.27 s over 150k of them."""
    tokenizer = tokenizer_of(model)
    stop = stop_ids(model)
    if tokenizer is None or not size or not stop:
        missing = [
            name
            for name, found in (("tokenizer", tokenizer), ("vocab_size", size), ("stop id", stop))
            if not found
        ]
        raise NotConstrainable(
            f"{model_id!r} has no {' and no '.join(missing)}: a strict schema is compiled "
            "against the checkpoint's own token table, and there is none to compile against. "
            "The same schema without strict is checked after the answer and needs neither."
        )
    return Vocabulary(tokenizer, size=size, stop=stop)


class Compiling(Admitting):
    async def constrain(self, model_id: str, schema: Mapping[str, object]) -> Constraint:
        """One request's walk over `schema`, compiled against this model's own token table.

        Three lifetimes, and the whole of what this method is: the table is per resident model and
        dies with its record, the compiled grammar is per (model, schema) and shared, and the walk
        is per request and shared with nobody.

        No lease is taken. Between here and `submit` an eviction can take the model and the next
        request load it again — what comes back is the same table, so the walk stays valid and
        what the race costs is a reload, not a wrong mask.
        """
        return await self._walk(model_id, json.dumps(schema, sort_keys=True), schema)

    async def constrain_envelope(
        self, model_id: str, family: ToolFamily, tools: Sequence[Mapping[str, object]]
    ) -> Constraint:
        """The walk that makes a forced `tool_choice` mean what it says: the decode constrained to
        this checkpoint's own call envelope over the tools offered.

        The family builds the grammar and the vocabulary tells it how to spell a marker, so this
        method is where the two meet — the family cannot know whether `<tool_call>` is an added id
        here, and the vocabulary lives inside this class.
        """
        assert family.grammar is not None, "a family with no grammar is refused by the route"
        model = await self._reachable(model_id)
        entry = self._residency[model_id]
        async with self._compiling:
            vocabulary = await self._table(model_id, model, entry)
            source = family.grammar(tools, vocabulary.literal)
            grammar = entry.grammars.get(source)
            if grammar is None:
                grammar = vocabulary.written(source)
                entry.grammars[source] = grammar
        return grammar.constrain()

    async def _walk(self, model_id: str, key: str, schema: Mapping[str, object]) -> Constraint:
        model = await self._reachable(model_id)
        entry = self._residency[model_id]
        async with self._compiling:
            grammar = entry.grammars.get(key)
            if grammar is None:
                vocabulary = await self._table(model_id, model, entry)
                grammar = vocabulary.compile(schema)
                entry.grammars[key] = grammar
        return grammar.constrain()

    async def _table(
        self, model_id: str, model: LanguageModel[ModelInput], entry: Residency
    ) -> Vocabulary:
        vocabulary = entry.vocabulary
        if vocabulary is None:
            width = None if self._environment is None else self._environment.head_width(model_id)
            vocabulary = await asyncio.to_thread(_vocabulary, model_id, model, width)
            entry.vocabulary = vocabulary
        return vocabulary

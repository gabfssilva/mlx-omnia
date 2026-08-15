"""The mutation tests' one move: break an attribute, assert the floor catches it, put it
back."""

from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def mutated(owner: object, attribute: str, value: object) -> Generator[None]:
    """`attribute` set to `value` for the block, restored on the way out however it ends.

    Written as a contextmanager and not as save/try/finally at each site because the
    restoration is the part that must not be forgotten: a mutation left in place is a model
    that stays broken for every test after it in the module, which reads as an unrelated
    failure somewhere else.
    """
    original = getattr(owner, attribute)
    setattr(owner, attribute, value)
    try:
        yield
    finally:
        setattr(owner, attribute, original)

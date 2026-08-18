"""What residency and the compression gate refuse, named."""


class ModelTooLarge(Exception):
    """A checkpoint that does not fit the memory limit with nothing else resident. Raised before
    anything is evicted: no sequence of evictions makes room for it, so a loop chasing that space
    would empty the daemon of every other model and still fail."""


class NotResident(Exception):
    """A model a request named while the config says `not_resident: "fail"`. A cold load is
    seconds of the queue for whoever is behind it, and a daemon told to fail fast says so instead
    of paying it. Loading on purpose is still an order and not a request."""


class NotConstrainable(Exception):
    """A resident model no grammar can be compiled against: nothing under the facades holds a
    tokenizer, the catalog has no config to read the head's width out of, or the load resolved no
    stop id for a document to end on.

    Named rather than left as whatever fails first, because the way out belongs to the client:
    the same schema without `strict` is checked after the answer and needs none of the three.
    """


class NotQuantizable(Exception):
    """A model whose settings ask for a compressed KV cache it cannot hold.

    Raised in `submit`, before the request becomes a job, because the alternative is the one
    thing a compression switch must never do — generate densely while the screen says the cache
    is compressed.
    """

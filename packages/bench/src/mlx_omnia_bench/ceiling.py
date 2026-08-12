"""Decode is memory-bandwidth-bound, so a tok/s alone is not a result: beating a baseline
says nothing about how much of the machine is left.

`gbs` is required and there is no table of chips behind it. Apple publishes a memory
bandwidth per chip, and what a serial chain of dependent kernels actually sustains is a
fraction of it measured per machine — about 95% on the M5 Max this was written on. Shipping a
guessed sustained figure for hardware nobody here has measured would put an invented
denominator under every percentage computed from it.

Prefill has no equivalent. It is compute-bound, so the denominator would be sustained TFLOPs,
which is not cheap to establish; that axis stays relative to a baseline and says nothing about
what is left.
"""


def ceiling(active_bytes_per_token: int, gbs: float) -> float:
    """The decode tok/s the bandwidth allows. The numerator is the *active* bytes — an MoE
    reads the experts a token routes to, not the whole stack — and it has to be recomputed
    whenever the weight format changes, or the percentage silently means something else."""
    return gbs * 1e9 / active_bytes_per_token

# A/B switches for the bench: falsify one and the ops chain takes over — it is also the parity
# reference. They gate the sparse trunk only; the dense one takes the rope epilogue whenever its
# predicate holds.
#
# Only `ADD_RMS_NORM_KERNEL` is read on every call. `ROPE_EPILOGUE_KERNEL` is read once, when
# the attention module is built (it becomes `fused_decode`), because the model declares the
# operation and no longer names the kernel — so patching this name after a model is loaded
# changes nothing, and what a test or a bench flips on a live model is `_fused_decode` on the
# attention modules themselves (`tests/conftest.py: with_the_fused_step`).
#
# Both default off: measured on the M5 Max (Qwen3-30B-A3B-4bit, 435-token prompt,
# interleaved A/B, median of 5) each kernel LOSES decode here — 150.6 (both on) vs
# 159.9 tok/s (both off), reproduced in a second battery (142.7 vs 152.4). The same
# rope_epilogue wins on the dense 0.6B (310.0 vs 303.3), so the loss is specific to
# this step, not to the kernel. Kept wired for re-measurement.
ROPE_EPILOGUE_KERNEL = False
ADD_RMS_NORM_KERNEL = False

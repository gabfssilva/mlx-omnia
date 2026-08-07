# A/B switches for the bench, read as module attributes on every call: falsify one and the
# ops chain takes over — it is also the parity reference. They gate the sparse trunk only;
# the dense one takes the rope epilogue whenever its predicate holds.
#
# Both default off: measured on the M5 Max (Qwen3-30B-A3B-4bit, 435-token prompt,
# interleaved A/B, median of 5) each kernel LOSES decode here — 150.6 (both on) vs
# 159.9 tok/s (both off), reproduced in a second battery (142.7 vs 152.4). The same
# rope_epilogue wins on the dense 0.6B (310.0 vs 303.3), so the loss is specific to
# this step, not to the kernel. Kept wired for re-measurement.
ROPE_EPILOGUE_KERNEL = False
ADD_RMS_NORM_KERNEL = False

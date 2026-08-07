# A/B switches for the bench: set either to False to fall back to the op path, which
# stays the parity reference. The predicates read them on every call.
CONV_MIX_FUSED = True
MOE_GEMV_DENSE_FUSED = True
# The routing kernel scores from a float32 dot where the op chain rounds the gemv to T.
# In bfloat16 that flips the selection on genuine near-ties (measured: one layer of 22 on
# a synthetic prompt, the 4th and 5th experts 6e-5 apart in a selector quantized to 3e-4).
# Set False to keep the fused gemvs while the op path decides the experts.
MOE_ROUTE_FUSED = True

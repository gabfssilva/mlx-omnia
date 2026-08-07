# A/B switches for the bench stage: set either to False (module attribute, no code edit)
# and the corresponding op path — which is also the parity reference — runs instead.
USE_MXFP4_MOE_GEMV = True
# Measured on the M5 Max (gpt-oss-20b, 435-token prompt, interleaved A/B, median of 5)
# the sink kernel costs ~1.4% of decode against mx.fast.scaled_dot_product_attention
# (107.7 vs 109.3 and 107.9 vs 109.4 in two batteries) — but it stays ON: with it off,
# `test_greedy_matches_mlxlm` diverges from the fixture at index 209. 1.4% does not buy
# a broken parity gate. Re-measure at long context, where docs/models/gpt_oss.md expects
# it to win.
USE_SINK_ATTENTION = True

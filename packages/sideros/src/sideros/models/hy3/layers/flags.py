# Default off (see Qwen3-MoE): rope_epilogue and add_rms_norm each lose decode on the
# 30B step. Kept wired for re-measurement on Hy3 once a local checkpoint exists.
ROPE_EPILOGUE_KERNEL = False
ADD_RMS_NORM_KERNEL = False

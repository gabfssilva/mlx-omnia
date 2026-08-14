# A/B switches for the bench: set either to False (module attribute, read on every call)
# and the ops chain takes over — it is also the parity reference. The DeltaNet one
# changes the recurrent state's layout, so flip it before a cache is created.
GATED_DELTA_KERNEL = True
ADD_RMS_NORM_KERNEL = True

# The compiled one-token DeltaNet step (the Swift port's `linearStep`): the ~20
# elementwise ops around the recurrence collapse into a handful of kernels. Rides on
# the gated_delta kernel, so it only engages when that one does.
COMPILED_STEP = True

# The speculative verify's DeltaNet checkpoints: True is the trace kernel (one dispatch
# per layer, the state stored after every token), False walks the rows as single-token
# dispatches of the plain kernel. Bit-identical — the state round-trips through the same
# f32 buffers either way — so the flag is a measurement, not a semantic.
VERIFY_TRACE_KERNEL = True

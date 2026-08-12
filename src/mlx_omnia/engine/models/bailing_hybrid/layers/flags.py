# A/B switches for the bench: set either to False (module attribute, read on every call)
# and the ops chain takes over, which is also the parity reference. The recurrence one
# changes the state's layout, so flip it before a cache is created.
GATED_DELTA_KERNEL = True

# The compiled one-token step: the ~40 elementwise ops around the recurrence and the
# residual join collapse into a handful of kernels. Rides on the gated_delta kernel for
# the KDA layers, so it only engages there when that one does.
COMPILED_STEP = True

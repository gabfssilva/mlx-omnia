# A/B switches for the bench: set either to False (module attribute, read on every call)
# and the ops chain takes over — it is also the parity reference. The DeltaNet one
# changes the recurrent state's layout, so flip it before a cache is created.
GATED_DELTA_KERNEL = True
ADD_RMS_NORM_KERNEL = True

# The compiled one-token DeltaNet step (the Swift port's `linearStep`): the ~20
# elementwise ops around the recurrence collapse into a handful of kernels. Rides on
# the gated_delta kernel, so it only engages when that one does.
COMPILED_STEP = True

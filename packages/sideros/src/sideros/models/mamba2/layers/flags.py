# A/B switches for the bench: set to False (module attribute, read on every call)
# and the ops chain takes over — it is also the parity reference.
SSM_KERNEL = True

# The compiled one-token step: the ~20 elementwise ops around the recurrence
# collapse into the kernel + a handful of ops. Rides on ssm_step, so it only
# engages when that one does.
COMPILED_STEP = True

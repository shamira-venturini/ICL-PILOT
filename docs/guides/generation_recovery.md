# Interrupted generation recovery

Fresh model runs initialize Python, NumPy and PyTorch CPU/CUDA once with master seed 42 before model loading. Jobs, stages and replicate IDs do not reseed the streams.

Completed-call checkpoints retain the exact finished output, all RNG states, the next scheduled call, journal boundaries and configuration/input hashes. Stage 1 parents remain immutable.

To resume, select the existing run with the notebook's RESUME_RUN control, use its original model/mode/smoke selection, load the model, and only then restore the checkpoint's RNG states. The recovery helpers validate identities, hashes, completed records and the schedule boundary before continuing.

Do not manually skip calls, regenerate completed Stage 1 parents, truncate journals or append a changed design to an existing run. Configuration, template, roster, input, model or protocol changes require a fresh run directory.

Checkpoint and exported-copy verification establish recovery integrity. They do not guarantee identical GPU sampling across different hardware/software environments. Missing RNG states or parent records cannot be reconstructed by assigning a new seed.

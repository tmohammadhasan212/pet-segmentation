# Project continuation

When the user asks to continue this project, read `NEXT_SESSION.md` and
`PROGRESS.md` before restarting work. They record the paused accuracy study,
checkpoint paths/hashes, missing resume support, and exact remaining steps.

Respect an explicit pause: do not start training until the user resumes it.
Preserve existing checkpoints and locked data manifests. Use validation for
model/threshold selection; test results are reporting evidence. Keep datasets,
large generated predictions, environments and weights out of Git.

The user expects meaningful local Git commits for completed implementation and
measured results. Inspect the working tree first and preserve unrelated changes.

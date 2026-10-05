# Evidence from the T4 run

The final call is **DON'T SHIP**. The adapter passed the independent learning
checks, but the measured Soup shipping gate rejected a task tie (4/8 correct
for both models). The data is translated general Russian preference text;
it does not establish support-ticket quality.

Read [the two-page report](../report/report.pdf) first. The files below retain
the failures as well as the completed measurements. Raw logs are byte copies
of subprocess output; adjacent timestamped logs add UTC timestamps for review.
Command JSON records contain the exact argv, start/end times and return code.

| Attempt | Outcome and useful evidence |
|---|---|
| [01](attempt-01) | TorchAO 0.10 conflicted with PEFT's optional-package probe. Training was skipped. The legacy collector's outer zero exit was misleading. |
| [02](attempt-02) | Tiny streamed/resident parity passed. Full setup failed because Soup did not accept symlinked Hub weights. |
| [03](attempt-03) | Training reached step 33. The logger rejected an infinite AMP-scaled gradient and left malformed telemetry; it is preserved. |
| [04](attempt-04) | Full DPO run: 100 attempted steps, 97 successful updates, 3 AMP skips. Parity, gradients, reference stability and memory passed. First resident verification failed on the tokenizer API return type; the collection remains marked incomplete. FP16 generation and Soup ship completed, yielding DON'T SHIP. |
| [05](attempt-05) | Verification-only retry using the unchanged saved adapter from 04. Explicit tokenizer return types fixed the verifier; all nine checks passed and the unchanged-adapter control was rejected. This collection is complete. |
| [Bootstrap](bootstrap) | Installation, GPU/environment, optional TorchAO removal, source checkout and test logs. The missing-Jedi `pip check` advisory remains recorded. |

Training used commit `55bf7f4c500157a4d921cf6d9818cb3947a8f542`; retry 05 used
`b21a3499290c6317baeaf41da6fb0406fe85fab4`. Each phase records source/input hashes.
The retry's provenance includes the original training provenance and saved
adapter SHA-256. Its internal `SHIP` result means the Part 2 adapter gates
passed; it does not override the measured or final deployment verdict.

The main measurements are:

- `attempt-04/memory-before.json`: calculations saved before training.
- `attempt-04/nvidia-smi-before.raw.log`, `nvidia-smi-after.raw.log` and
  `nvidia-smi-sampled.raw.csv`: raw GPU evidence; the sample stream covers training.
- `attempt-04/streaming-parity.json`: four-layer resident/streamed forward,
  DPO loss, gradients and AdamW update comparison, plus a rejected zero-gradient
  control. It does not establish full 0.5B backward equivalence.
- `attempt-04/dpo-training.raw.log` and `.timestamped.log`: complete training
  output, including the initial/final loss and memory events.
- `attempt-05/verification.json`, `trained.json`, `baseline.json` and
  `negative-control.json`: independent saved-adapter evidence. Mean margin gain
  is 0.369583; the paired bootstrap 95% lower bound is 0.207135.
- `attempt-04/ship-input-fp16.*.responses.jsonl`: all 278 generated responses
  per model, with timestamps, durations and errors. The eight task fixtures
  and 270 mini-suite items are functional checks.
- `attempt-04/ship-verdict.json`, `ship-evidence.json` and `ship.raw.log`:
  Soup's offline gate over measured fp16 scores, with no remote judge.

The original run's failed verifier output is not replaced by the retry. Check
both `attempt-04/run-summary.json` and `attempt-05/verification-summary.json`.
The separate [reproduction notebook](../notebooks/run_colab.ipynb) uses the
corrected collector and verifier for a new attempt; it refuses existing output
directories.

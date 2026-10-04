# Russian DPO with layer streaming on a T4

This repository runs a small DPO experiment through Soup 0.75.2 on a Colab T4.
It uses `Qwen/Qwen2.5-0.5B-Instruct`, plain LoRA, and RAM layer streaming. The
runner records the commands, raw output, timestamps, memory measurements, and
verification evidence needed to review the result.

The production verdict is **DON'T SHIP**. The 500 preference pairs contain
translated general Russian dialogue, not support tickets or company policies.
A passing training run or a small public-data evaluation cannot establish
support quality. Deployment needs reviewed domain data and independent evidence
that the saved adapter changes behavior and improves held-out outcomes.

## Review the submission

| Deliverable | Path |
|---|---|
| Report | [PDF](report/report.pdf) · [Markdown](report/report.md) |
| Reproducible Colab notebook | [run_colab.ipynb](notebooks/run_colab.ipynb) |
| Executed T4 notebook | [completed_t4.ipynb](notebooks/completed_t4.ipynb) |
| Exact training settings | [configs/soup.yaml](configs/soup.yaml) |
| Initial setup logs | [artifacts/bootstrap](artifacts/bootstrap) |
| Preserved first attempt | [artifacts/attempt-01](artifacts/attempt-01) |
| Second attempt | [artifacts/attempt-02](artifacts/attempt-02) |
| Third attempt, interrupted at step 33 | [artifacts/attempt-03](artifacts/attempt-03) |
| Full-run retry, results pending | [artifacts/attempt-04](artifacts/attempt-04) |
| Static source audit | [docs/silent_failures.md](docs/silent_failures.md) |
| Verification rules and memory plan | [docs/verification.md](docs/verification.md) |
| Dataset provenance and split checks | [data/README.md](data/README.md) · [manifest.json](data/manifest.json) |

The report and completed-run links above are reserved for the measured results.
Review the files that exist before making a runtime claim; the template notebook
has empty outputs. The report will distinguish completed checks, failed checks,
and evidence that the run did not produce.

Attempts 01–03 used an older collector that returned zero after failures. Their
raw logs remain part of the evidence; that outer exit code does not establish
completed training or evaluation. Attempt 03 stopped at step 33 when telemetry
tried to serialize an infinite scaled AMP gradient magnitude into JSON. The
full-run retry will preserve its own results in attempt 04.

## Run on Colab

Open [the notebook on Colab](https://colab.research.google.com/github/Huzyefahwasim/Soup/blob/main/notebooks/run_colab.ipynb),
select **T4 GPU** under **Runtime → Change runtime type**, and run the cells in
order. The notebook clones the public repository into `/content/soup-assignment`.
Set `REPO_REF` to the recorded commit SHA for an exact source replay. The default
`main` resolves the latest submission and records the commit it used.

The setup installs [requirements-colab.txt](requirements-colab.txt), verifies
the four core package pins, records `pip freeze`, `pip check`, CUDA/GPU details,
and host memory. It retains Colab's CUDA-enabled PyTorch. Soup requires Python
3.10 through 3.12. If Colab has TorchAO older than 0.16.0, the notebook logs its
removal for this unquantized, non-QAT configuration: PEFT 0.20 otherwise rejects
the unused optional package while creating LoRA layers.

The notebook launches this command with streamed stdout:

```sh
python -u scripts/run_experiment.py --config configs/soup.yaml --run-dir artifacts/<fresh-attempt>
```

Each attempt uses a new directory. The runner refuses an existing path so a
retry cannot overwrite a failed run. The current collector exits **1** when a
required command or its evidence is missing, invalid, or incomplete. It exits
**0** when it collected and validated the required artifacts, including a valid
measured `DON'T SHIP` verdict. Read `collection_status` and `failed_commands` in
`run-summary.json`, alongside `commands.json` and the individual reports.

The collector validates expected nonzero inner exits against their artifacts:
the DPO doctor refusal, a rejected unchanged-adapter control, adapter
verification reporting `DON'T SHIP`, and Soup's measured `DON'T SHIP` verdict.
Soup uses exit 2 for that verdict. Collection completeness, the
`measured_ship_verdict`, and the support-domain `domain_verdict` are separate
fields. A complete collection does not imply that either verdict is `SHIP`.

The final notebook cell downloads a ZIP containing setup logs, run artifacts,
the executed source/config/data snapshot, and SHA-256 checksums. It also works
after a failed experiment. Download the executed `.ipynb` from Colab's **File →
Download** menu to retain cell outputs.

## Configuration and checks

The config uses one epoch, learning rate `2e-5`, one preference pair per
microbatch, accumulation 4, seed 42, beta 0.1, and a 512-token sequence limit.
LoRA uses rank 8, alpha 16, zero dropout, and `q_proj`/`v_proj` targets. Two RAM
streaming buffers serve the frozen decoder. Soup selects FP16 on the T4 and
keeps trainable adapters in FP32. Its streamed decoder uses its own checkpoint
path; the runtime disables nested Hugging Face gradient checkpointing.

The model revision is fixed to
`7ae557604adf67be50417f59c2c2f167def9a775`. The package pins are Soup `0.75.2`,
Transformers `5.16.1`, TRL `0.29.0`, and PEFT `0.20.0`. The exact PyTorch and
resolved dependency versions belong to each run's environment evidence.

The experiment runs doctor, DPO lint, a chosen-only chat doctor projection, and
a dry run before the GPU work. The data doctor refuses native DPO format, so
the runner preserves that refusal and keeps its chat projection separate from
the DPO tokenization audit.

The streamed/resident probe uses a random four-layer Qwen2 fixture. It compares
forward values, DPO loss, policy/reference behavior, adapter gradients, and an
optimizer update. Passing this fixture does not prove full-checkpoint backward
equivalence, larger-model fit, or NF4 correctness. The main training path uses
the real 0.5B checkpoint through Soup's DPO wrapper and records trainable tensors,
per-layer gradients, reference stability, tokenization, and synchronized memory.

Fresh resident loads then compare the saved initial/final adapter tensors,
adapter-on/off inference, and 50 held-out preference pairs. A no-op control must
fail the same gate. The separate FP16 task/general evaluation records responses
and passes its measured scores to `soup ship --evidence`; this avoids the pinned
release's live shipping path selecting BF16 on a T4. See the
[verification contract](docs/verification.md) for the thresholds and limitations.

## Data and local tests

The dataset is
[`d0rj/rlhf-reward-datasets-ru`](https://huggingface.co/datasets/d0rj/rlhf-reward-datasets-ru/tree/ec1b66da5885fb4af7cc90096036c3cd4ee0dab5),
pinned at `ec1b66da5885fb4af7cc90096036c3cd4ee0dab5`. Its publisher declares MIT
and identifies a translated upstream preference corpus. The card provides
little annotation or translation provenance. The official source test contains
English examples, so the project draws 400 train, 50 validation, and 50 test
pairs from unique Russian source-train prompts. These project holdouts have
zero overlap after Unicode/case/whitespace normalization; that check does not
establish semantic novelty. No preference scores or support context are
fabricated.

The checked-in JSONL files allow a run without downloading the full corpus.
Rebuild them and run the offline tests with:

```sh
python scripts/prepare_data.py
python -m pytest tests -q
```

Preparation streams checksum-verified Parquet batches, filters invalid pairs,
and selects seeded hash ranks. [The data notes](data/README.md) describe the
filters, license declaration, source-card mismatch, and label limitations.

# Soup 0.75.2 source audit

This is a static source audit, performed on 2026-10-05. It does not record a GPU run or prove numerical parity. Runtime results belong in the generated evidence artifacts.

## Reproducible source

- [PyPI soup-cli 0.75.2](https://pypi.org/project/soup-cli/0.75.2/) was published on 2026-10-01. PyPI provenance identifies commit `3966ef95cead56f500a67b1f68cbc84d22b11cfb`.
- [Pinned upstream source](https://github.com/MakazhanAlpamys/Soup/tree/3966ef95cead56f500a67b1f68cbc84d22b11cfb) is the authority for this audit. The website and `main` can describe options unavailable in this release.
- [Official dependency-floor constraints](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/.github/constraints/transformers-floor.txt) pin torch 2.6.0, transformers 5.16.1, TRL 0.29.0, PEFT 0.20.0 and plotext 6.0.0. Python must be 3.10 through 3.12. Record the actual Colab package versions and run `pip check`; do not infer compatibility from installation succeeding.

PEFT 0.20 probes optional TorchAO while wrapping ordinary linear layers. If TorchAO is absent the dispatcher continues; if an installed version is older than 0.16.0, the availability probe raises an import error. For this run's `quantization: none` and default `quantization_aware: false`, TorchAO is unused and can be removed from the ephemeral Colab environment before launching fresh subprocesses. Soup declares it only in the optional `qat` extra. [PEFT availability probe](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/import_utils.py#L121-L140), [dispatcher](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/torchao.py#L128-L149), [Soup extras](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/pyproject.toml#L140).

## Supported entry-test configuration

`Qwen/Qwen2.5-0.5B-Instruct` uses the admitted `qwen2` family. A small model allows a resident control on the same T4 and avoids a large RAM or disk requirement. It demonstrates streaming mechanics; it does not demonstrate that a model larger than the T4 fits.

The configuration uses `task: dpo`, `backend: transformers`, `modality: text`, `data.format: dpo`, `data.max_length: 512`, explicit `training.batch_size: 1`, accumulation 4, `training.quantization: none`, and plain LoRA on `q_proj` and `v_proj`. Streaming keys are `training.stream_layers`, `stream_source`, `stream_buffers`, and optional `stream_pin`. `ram` selects the RAM tier explicitly; `auto` can fall back to the disk tier. A concrete batch size is required. Accumulation is supported despite stale schema descriptions suggesting otherwise.

Do not add `bf16`, `fp16`, `dtype`, `max_prompt_length` or `stream_read_ahead` to this release's YAML. They are not declared fields. Unknown keys fail at configuration load. Prompt length is derived internally as `data.max_length // 2`, giving 256 here. `training.stream_vram_probe: true` is refused for DPO because the measured probe implements an SFT step. [Schema](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/config/schema.py#L5199), [DPO setup](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/trainer/dpo.py#L163).

The T4 has no native bf16 hardware. Soup asks `torch.cuda.is_bf16_supported(including_emulation=False)`, selects fp16 for compute and streamed storage, and maintains trainable LoRA parameters in fp32. The bare PyTorch capability call can return true through software emulation. Leave FlashAttention disabled and record the actual model attention implementation; YAML has no arbitrary `attn_implementation` field. [GPU precision helpers](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/gpu.py#L376), [stream dtype](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/layer_stream.py#L216).

## What the built-in checks cover

| Check | Actual scope | Remaining check |
|---|---|---|
| `python -m soup_cli doctor --config configs/soup.yaml` | Environment, dependencies, resources and declared settings not consumed by a backend | Missing or incompatible training extras are advisory and do not make the command fail. Explicitly import DPOConfig/DPOTrainer and verify CUDA/device/versions. |
| `python -m soup_cli train --config configs/soup.yaml --dry-run` | Schema and data loading | Exits before streaming setup. Does not establish GPU memory fit, reference correctness, gradients, or optimizer updates. |
| Streamer setup | Supported architecture, RAM/disk costs, predicted GPU memory, materialized adapters and compatible modes | A predicted fit is not a measured DPO peak. Capture synchronized allocated/reserved GPU peaks and process RAM. |
| `python -m soup_cli data lint data/train.jsonl --format dpo --sample 400 --output "$RUN_DIR/data-lint.json"` | Word-count length bias, identical pairs, near duplicates and prompt echoes | The runner does not pass the optional tokenizer flag. It does not verify preference truth, train/test separation, completion masks, or EOS after truncation. Near-duplicate detection is advisory if `datasketch` is absent. |
| `soup data doctor` on a chat projection | Chat-template rendering, roles, BOS/EOS, truncation risk and SFT loss-mask preview | The command refuses DPO input. Its mask preview is not the DPO completion mask. |
| `soup ship` | Strict task improvement plus no general-suite regression exceeding a threshold | A tie is DON'T SHIP. Tiny suites and synthetic tasks are limited evidence. Training/parity success does not imply SHIP. |

Sources: [doctor](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/commands/doctor.py#L135), [dry run](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/commands/train.py#L1255), [data commands](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/commands/data_doctor.py#L197), [linter](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/data_lint.py#L279), [memory budget](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/trainer/stream_setup.py#L919).

## Silent correctness risks to test

1. **Adapters with no storage.** A meta skeleton can create LoRA tensors on meta. An optimizer may accept them while changing nothing. Current source materializes them and asserts that trainable LoRA tensors are real. Independently record each trainable tensor's name, shape, device and dtype, and require a positive trainable parameter count. [Guard](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/layer_stream_runtime.py#L1896).

2. **Severed lower-layer gradients.** Frozen base weights still participate in the derivative with respect to hidden states. A `detach` or `no_grad` around the base can let top adapters learn while lower adapters remain dead. Measure finite, nonzero gradients in layer 0 and a high layer before an optimizer step, then verify actual policy adapter changes. Zero gradient in LoRA A on the first step can be expected because LoRA B initializes at zero. [Streaming forward](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/layer_stream_runtime.py#L1091), [upstream gradient test](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/tests/test_v07204.py#L738).

3. **Reference equals the evolving policy.** TRL 0.29 creates a frozen initial-policy `ref` LoRA adapter on the same base, rather than loading a second full model. Soup materializes that snapshot after DPOTrainer construction. Initially policy/reference equality is expected with zero LoRA B. After a policy change, their log probabilities should differ while the reference tensors/log probabilities remain unchanged. A loss near `log(2)` alone cannot distinguish legitimate initialization from a broken reference. [Snapshot materialization](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/trainer/dpo.py#L235), [snapshot helper](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/layer_stream_runtime.py#L1956), [reference control](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/tests/test_v07204.py#L496).

4. **Double checkpointing.** The streamed decoder already uses non-reentrant checkpointing. HF checkpointing of the inner layer can recompute after temporary base-weight substitution has ended. Soup explicitly disables HF checkpointing when streaming, even if the YAML requests it. Verify `wrapper.trainer.args.gradient_checkpointing is False` and run a real backward step. [Guard and CUDA test](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/tests/test_v07204.py#L607).

5. **Prompt and inference formatting differ.** Soup's DPO converter passes fields through. Conversational message lists let TRL render the model chat template; string fields remain string fields. Soup's live evaluator templates a single user message. Use conversational preference rows and inspect the prepared token IDs. Do not assume a chat doctor projection proves the DPO path. [Converter](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/data/formats.py#L245), [inference renderer](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/live_eval.py#L126).

6. **Truncation erases the useful preference or EOS.** Soup restores the prompt cap after TRL tokenization and slices completions to the remaining budget. This can remove answer differences or terminating tokens. Check chosen/rejected token IDs remain distinct, have nonempty completion masks and retain a terminating token at the actual training shape. [Length enforcement](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/trainer/_trl_compat.py#L134).

7. **Data leakage and trivial preferences.** The linter does not establish correct labels or independently held-out tasks. Separate train/test prompts and template families where feasible, retain known-invalid fixtures, and inspect representative pairs. Report the limitations of synthetic labels and small tests.

8. **Live shipping uses different precision.** In 0.75.2, `soup ship` with an unquantized base hardcodes bf16 for CUDA, including a T4. Training selects fp16 there. Record the live evaluator precision explicitly, or produce honest fp16 evidence using the shared evaluation tools and replay it through `soup ship --evidence ...`. Do not label live ship scores as fp16 training-parity evidence. [Ship dtype selection](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/commands/ship.py#L512).

9. **Saved adapter does not reload faithfully.** Check `adapter_config.json` names the intended base, adapter tensors are finite and changed, and a fresh resident load reproduces the trained policy at matching precision. Current source stamps the base reference on the meta skeleton; older streaming artifacts could lose it. [Base provenance](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/layer_stream_runtime.py#L1780).

10. **A green configuration hides an unconsumed option.** Restrict the demonstration to consumed standard DPO/LoRA options. For example, current docs acknowledge that `gdpo_variant` was accepted while not applied on supported TRL. Static field validity is weaker than verifying the effective trainer configuration. Record beta, loss types, optimizer, precision, shapes and actual adapter targets.

11. **Scaled AMP overflow is mistaken for a broken update.** Autograd hooks see scaled microbatch gradients before unscale. An overflow may be handled by GradScaler skipping that optimizer update and reducing its scale. Preserve those raw nonfinite events, the actual skipped-update flag and scale before/after. Measure accumulated gradients in `on_pre_optimizer_step` after Trainer's clipping/unscale, then accept them into gradient gates only after `on_optimizer_step` confirms a successful update. An infinite scaled value also needs an explicit JSON string or count; strict JSON refuses a numeric infinity. The current telemetry writes raw counts and finite-only maxima, encodes nonfinite log/history numbers as strings, and keeps successful-update gradient gates strict. Skipping every update still fails. [Trainer callback order](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py#L1623-L1638), [unscale through clipping](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py#L2311-L2324).

## Streamed versus resident parity protocol

Hold the frozen base checkpoint/revision, fp16 storage/compute, attention implementation, tokenizer/template, batch token IDs/masks, adapter configuration and adapter weights fixed. Match both policy and frozen reference adapters. Disable dropout. Reset gradients and random state before each arm; compare each metric independently and record absolute and relative error.

Run these controls before long training:

1. Zero-B control: policy and reference agree.
2. Nonzero-B control: policy and reference differ, and the frozen reference remains stable after a policy update.
3. Streamed/resident logits and sequence log probabilities agree at the same numerics.
4. Standard DPO loss agrees.
5. LoRA gradients agree, including layer 0.
6. One identical optimizer update produces matching policy adapter changes.
7. Repeat with pinned RAM and, if practical, pageable RAM. Record which path was actually used.
8. Save/reload and compare outputs at the same precision.

Upstream uses bit equality for its matching forward/loss controls. Record exact equality where obtained; if using tolerances, report the actual maxima and justification rather than calling the result bit-exact. Distinguish a measured forward result from backward or optimizer parity.

TRL 0.29 removed `concatenated_forward` and `get_batch_loss_metrics`. Upstream's compatibility test reconstructs policy sequence log probabilities from `_truncate_inputs`, shifted logits and `selective_log_softmax`. Its private DPO loss path can read `self.model` even when a separate model argument is supplied; the reference control must use the correct model. Reuse the pinned source's [test helper](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/tests/test_v07204.py#L187) as a specification and inspect the installed trainer API.

## Shipping evidence

The runner uses the fp16 score producer, then Soup's offline gate. Here `RUN_DIR` is the unique existing attempt directory; `LOCAL_MODEL` and `MODEL_REVISION` come from that attempt's `model.json`:

```bash
python scripts/evaluate_ship.py --model "$LOCAL_MODEL" --revision "$MODEL_REVISION" --adapter "$RUN_DIR/adapter" --out "$RUN_DIR/ship-input-fp16.json"
python -m soup_cli ship --evidence "$RUN_DIR/ship-input-fp16.json" --config "$RUN_DIR/resolved-soup.yaml" --output "$RUN_DIR/ship-verdict.json" --emit-evidence "$RUN_DIR/ship-evidence.json"
```

The task file uses `prompt`, `expected`, optional `category` and `scoring` (`exact` by default). The producer uses greedy inference and the default bundled general suite, loads the arms sequentially, and preserves per-arm response journals. Its exit 0 means measurement completed, not SHIP. Soup ship exit codes are 0 for SHIP, 2 for DON'T SHIP, 3 for usage errors and 1 for runtime errors. Preserve the actual verdict. `--emit-evidence` stamps the resolved configuration hash onto the supplied scores; it does not prove that they were measured, and it omits the producer's additional measurement fields. Retain the raw input, journals and command logs. A later replay without `--emit-evidence` checks the stamped configuration hash. [CLI](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/commands/ship.py#L1154), [verdict rule](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/ship_verdict.py#L375).

The current collector records `source-provenance.json` before commands, including script and input SHA-256 hashes. Its summary separates `collection_status`, `measured_ship_verdict` and `domain_verdict`. Incomplete collection exits 1; completed collection exits 0 even when a valid gate rejects shipping. Expected nonzero command results require the matching verdict/control artifacts. An older archived collector may return 0 after failed child commands; inspect its `commands.json`, raw logs and actual artifacts rather than interpreting that top-level exit as a completed experiment.

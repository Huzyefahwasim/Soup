# Verification contract

A completed training process is a smoke test. This submission requires independently saved tensor updates, a portable adapter reload, an observable inference effect, and held-out preference evidence. The repository verdict must also require the streaming probe and training run evidence. No run is marked successful until its output exists.

## Reproduce the checks

Use the pinned Soup source and immutable model revision recorded by the runner. The main run saves the initial adapter before the first optimizer update.

```bash
python scripts/check_streaming.py --device cuda --dtype float16 --out artifacts/streaming.json
python scripts/verify_adapter.py snapshot --adapter artifacts/initial_adapter --out artifacts/before.npz
python scripts/verify_adapter.py snapshot --adapter artifacts/adapter --out artifacts/after.npz
python scripts/verify_adapter.py evaluate --model "$MODEL" --revision "$REV" --data data/test.jsonl --out artifacts/base.json
python scripts/verify_adapter.py evaluate --model "$MODEL" --revision "$REV" --adapter artifacts/adapter --data data/test.jsonl --out artifacts/trained.json
python scripts/verify_adapter.py compare --before artifacts/before.npz --after artifacts/after.npz --metrics-before artifacts/base.json --metrics-after artifacts/trained.json --out artifacts/verification.json
```

The `evaluate` commands run in fresh processes and load a normal resident model with PEFT. They apply the Qwen user/assistant chat template, verify an unchanged prompt token prefix, preserve at most the first 256 prompt tokens (`keep_start`, matching pinned TRL 0.29 and Soup's prepared-data cap), and truncate the total sequence at 512. Position `t` predicts token `t+1`; only assistant completion targets contribute to summed sequence log probabilities. Template end tokens count as part of the completion. Prompt and padding tokens receive no score.

For pair `i`, the reported gain is `g_i = (log p_after(chosen) - log p_after(rejected)) - (log p_base(chosen) - log p_base(rejected))`. DPO's implicit reward margin is `beta * g_i`; it is separate from the raw chosen-versus-rejected probability margin. See the [original DPO objective](https://arxiv.org/abs/2305.18290).

The comparator exits zero only when every check passes:

- Initial/final snapshots have the same canonical tensor layout and adapter configuration. Every tensor is finite, and every saved `lora_B` changes and ends nonzero. Random initial `lora_A` weights alone cannot pass.
- Reloaded tensors exactly match every saved tensor, with no missing or extra adapter keys. This verifies the portable artifact through an ordinary PEFT loader.
- On the first eight fixed test pairs, the maximum adapter-on/off change in the first 32 completion-token log probabilities exceeds `max(1e-5, 10 * repeat_noise)`. Repeating the first pair measures numerical noise. Adapter-off sequence scores reproduce the independently loaded base within `1e-4` absolute tolerance.
- Both score files identify the same immutable model revision, dtype, truncation settings, data bytes and ordered unique test IDs, with at least 30 pairs.
- Mean held-out `g_i` is at least `0.05` unscaled log-probability units and the lower endpoint of a paired 95% percentile bootstrap interval is positive (5,000 row resamples, seed 2026). This is a predeclared engineering threshold, not a literature-derived guarantee. Fifty test pairs can be inconclusive; an inconclusive result produces `DON'T SHIP`.

The gate can be tested against the same checkpoint and base scores on both sides. It must exit nonzero:

```bash
python scripts/verify_adapter.py compare --before artifacts/before.npz --after artifacts/before.npz --metrics-before artifacts/base.json --metrics-after artifacts/base.json --out artifacts/noop_control.json
```

## Streaming probe and its limits

`check_streaming.py` constructs a random, four-layer Qwen2 fixture, shards the same weights with Soup's `shard_checkpoint`, and builds its streamed counterpart with the official `build_streamed_model` API. It synchronizes adapters and their dtypes, matches the resident model to the streamed model's effective attention implementation, records that implementation, and sets dropout zero. The probe computes the same completion-masked DPO loss with adapters disabled for its reference. Nonzero B matrices exercise both A and B gradients. The disabled streamed reference must reproduce the resident reference, and its sequence log probabilities must differ from the active policy by more than `1e-5`. It checks every expected adapter gradient, a real finite AdamW parameter update, frozen base parameters and decoder weights remaining on the meta device. A zero-gradient control must be rejected even when forward logits agree.

The full training run uses TRL 0.29's frozen copy of the initial LoRA under the `ref` adapter name, sharing the same streamed base. At initialization its B matrices are zero, so this reference represents the base policy. The training callback verifies the reference tensors remain frozen and unchanged. Fresh held-out verification independently compares the trained adapter enabled and disabled on a resident base; it does not substitute for full-model backward equivalence.

Predeclared float32 tolerances are `atol=1e-6, rtol=1e-5`; float16 uses `atol=2e-4, rtol=2e-3`. Gradient relative L2 error must also be at most `rtol`, with cosine similarity at least `0.9999`. The report preserves exact equality and maximum error independently. These tolerances are engineering choices and must not be loosened after observing a failure.

The fixture checks the same architecture and runtime, but it does **not** prove full 0.5B backward equivalence, convergence, large-model or NF4 correctness. Saved-adapter checks and held-out inference use the real 0.5B checkpoint. Broader claims require separate, matching resident comparisons. [PyTorch checkpointing](https://docs.pytorch.org/docs/stable/checkpoint) trades activation storage for recomputation; changing the function's behavior between forward and recompute can silently produce incorrect gradients. Soup's [streaming history](https://trysoup.dev/docs/layer-streaming) documents both bad saved adapter keys and NF4 backward errors despite matching forward/loss values. The pinned version must include the repairs, including the [pre-Ampere dtype correction](https://trysoup.dev/docs/free-gpu-tier).

## T4 memory plan

These are analytical estimates for `Qwen/Qwen2.5-0.5B-Instruct`, fp16 frozen weights, r=8 targeting q/v, one preference pair per microbatch, 512 total tokens, and two streaming buffers. They are **not measured peaks**. Dimensions come from the [model's official config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json): 24 layers, hidden width 896, MLP width 4,864, 14 attention heads, two KV heads, vocabulary 151,936, tied embedding/head.

| Component | Analytical storage |
| --- | ---: |
| One fp16 decoder layer (14,912,384 parameters) | 28.44 MiB |
| Two streaming decoder buffers | 56.89 MiB |
| Resident tied embedding/LM head; counted once | 259.66 MiB |
| q/v LoRA, 540,672 parameters, FP32 weight + grad + Adam m/v | 8.25 MiB |
| Frozen initial-reference LoRA, FP32 weights only | 2.06 MiB |
| Checkpoint boundary allowance, two rows × 512 × 896 × 25 × 2 bytes | 43.75 MiB |
| One eager fp32 attention-score matrix, two rows × 14 × 512² | 28 MiB |
| Full chosen/rejected logits, two rows × 512 × 151,936 | 296.75 MiB fp16; 593.50 MiB fp32 |
| Conservative logits/loss planning charge, 14 bytes per element | 2.03 GiB |
| Within-layer activation planning allowance | 76 MiB |
| CUDA context and allocator planning allowance | 1 GiB |
| CPU fp16 decoder weight store | 682.63 MiB |
| Full fp16 model weights, including tied vocabulary matrix | 942.29 MiB |

DPO's chosen and rejected sequences double the physical row count. Four accumulation microsteps do not multiply the live activation count when each microstep backpropagates immediately; gradients remain between them. The reference reuses the frozen base with a frozen initial LoRA, but its inference logits and temporaries can overlap with the policy graph. Budget those separately, including shifted-logit copies, per-layer MLP/RMSNorm temporaries, CUDA kernels, optimizer first-step allocations and allocator fragmentation. SDPA can change attention storage; the vocabulary logits term remains. The 14-byte charge is a conservative planning coefficient from Soup, not a universal DPO bound. The current `scripts/memory_budget.py` component sum is approximately **3.465 GiB** for these settings. Its activation and CUDA allowances are planning choices; only an actual run establishes the fit.

Measure synchronized CUDA peak **allocated and reserved** bytes over setup, policy/reference forwards, backward and the first optimizer step (after accumulation), then separately over saved-adapter reload/inference. Also record the GPU's total/free bytes and host process RSS/available RAM, including sharding/loading peaks, Python, tokenized data, pinned/pageable store and disk cache. Inspect steady-state microsteps for continuing growth. The CPU store fits a normal Colab host easily at this size, but read actual RAM rather than assuming a notebook quota. [PyTorch's CUDA memory documentation](https://docs.pytorch.org/docs/stable/notes/cuda) distinguishes tensor allocations from allocator reservations; `empty_cache()` does not release live tensors.

Use fp16 on a T4, and record adapter dtypes independently: PEFT commonly promotes adapters to float32. The [NVIDIA T4 specification](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf) lists 16 GB GDDR6. This deliberately small model makes the resident reload and controls affordable; it demonstrates layer streaming without claiming that streaming is necessary for this model.

The artifact gate establishes changed, loadable, active adapters and preference movement on this held-out fixture. It does not establish general instruction quality, safety, production readiness, or statistically reliable improvement across tasks. Report measured failures and missing GPU evidence explicitly.

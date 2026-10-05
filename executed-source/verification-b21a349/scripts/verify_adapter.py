"""Compare saved LoRA tensors and independently reloaded held-out token scores."""
import argparse
import contextlib
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_snapshot(path, tensors, metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **tensors, __metadata__=json.dumps(metadata))


def read_snapshot(path):
    with np.load(path, allow_pickle=False) as archive:
        return ({key: archive[key].copy() for key in archive.files if key != "__metadata__"},
                json.loads(str(archive["__metadata__"])))


def snapshot(adapter, out):
    from safetensors.numpy import load_file
    root = Path(adapter)
    tensors = load_file(str(root / "adapter_model.safetensors"))
    config = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    write_snapshot(out, tensors, config)


def compare_evidence(before, after, base, trained, min_margin=0.05):
    same_keys = bool(before) and set(before) == set(after)
    common = set(before) & set(after)
    same_shapes = same_keys and all(before[k].shape == after[k].shape for k in common)
    finite = all(np.isfinite(v).all() for v in list(before.values()) + list(after.values()))
    canonical = same_keys and all(re.search(r"\.lora_[AB]\.weight$", k) and
                                 ".inner." not in k for k in after)
    b_keys = [k for k in common if ".lora_B." in k]
    changed = same_shapes and bool(b_keys) and all(
        np.any(before[k] != after[k]) and np.any(after[k] != 0) for k in b_keys)
    rows, old = trained.get("samples", []), base.get("samples", [])
    identity_keys = {"model", "revision", "data_sha256", "dtype", "max_length", "max_prompt_length", "truncation_mode"}
    provenance = identity_keys <= set(base.get("identity", {}))
    matched = (provenance and base.get("identity") == trained.get("identity") and len(rows) >= 30 and
               [r["id"] for r in rows] == [r["id"] for r in old] and
               len({r["id"] for r in rows}) == len(rows) and base.get("adapter") is None)
    effect = trained.get("effect", {})
    delta, noise = effect.get("probe_delta", 0.0), effect.get("repeat_noise", 0.0)
    active = bool(np.isfinite([delta, noise]).all() and
                  delta > max(1e-5, 10 * noise) and noise >= 0)
    control, improved, mean, lower = False, False, None, None
    if matched:
        scores = np.array([[r["chosen"], r["rejected"], o["chosen"], o["rejected"],
                            r.get("base_chosen", np.nan), r.get("base_rejected", np.nan)]
                           for r, o in zip(rows, old)], dtype=np.float64)
        if np.isfinite(scores).all():
            control = bool(np.allclose(scores[:, 2:4], scores[:, 4:6], atol=1e-4, rtol=0))
            gains = scores[:, 0] - scores[:, 1] - scores[:, 2] + scores[:, 3]
            mean = float(gains.mean())
            rng = np.random.default_rng(2026)
            boot = gains[rng.integers(0, len(gains), (5000, len(gains)))].mean(axis=1)
            lower = float(np.quantile(boot, 0.025))
            improved = mean >= min_margin and lower > 0
    checks = {"matching_tensor_layout": bool(same_shapes and canonical),
              "finite_tensors": bool(finite), "updated_lora_b": bool(changed),
              "matched_data": bool(matched), "portable_reload": trained.get("reload_exact") is True,
              "active_adapter": active, "base_control": control,
              "heldout_improvement": improved}
    return {"verdict": "SHIP" if all(checks.values()) else "DON'T SHIP", "checks": checks,
            "n_test": len(rows), "mean_margin_gain": mean, "paired_bootstrap_95_lower": lower,
            "min_margin_gain": min_margin, "adapter_probe_delta": float(delta),
            "repeat_noise": float(noise), "scope": "adapter verification; combine with streaming parity and run evidence"}


def response_tokens(tokenizer, prompt, response, max_prompt, max_length):
    """Use the same Qwen user/assistant template as conversational DPO training."""
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    answer = response if isinstance(response, list) else [{"role": "assistant", "content": response}]
    # Match TRL 0.29's explicit return types; Transformers 5 defaults to a mapping.
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=False)
    full = tokenizer.apply_chat_template(messages + answer, tokenize=True,
                                         return_dict=True)["input_ids"]
    prefix = prefix[0] if prefix and isinstance(prefix[0], list) else prefix
    full = full[0] if full and isinstance(full[0], list) else full
    if full[:len(prefix)] != prefix:
        raise ValueError("chat tokenization does not preserve the prompt prefix")
    completion = full[len(prefix):]
    prefix = prefix[:max_prompt]
    completion = completion[:max_length - len(prefix)]
    if not prefix or not completion:
        raise ValueError("truncation removed the prompt or all completion tokens")
    return prefix + completion, len(prefix)


def evaluate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("--revision must be an immutable 40-character Hub commit")
    if args.device == "cpu" and args.dtype != "float32":
        raise ValueError("CPU verification uses float32")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager").to(args.device).eval()
    exact = False
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        saved = load_file(str(Path(args.adapter) / "adapter_model.safetensors"))
        loaded = get_peft_model_state_dict(model)
        exact = set(saved) == set(loaded) and all(
            torch.equal(saved[k].float(), loaded[k].detach().cpu().float()) for k in saved)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    data_bytes = Path(args.data).read_bytes()
    dataset = [json.loads(line) for line in data_bytes.decode("utf-8").splitlines() if line.strip()]

    def score(prompt, response, disabled=False):
        ids, boundary = response_tokens(tokenizer, prompt, response,
                                        args.max_prompt_length, args.max_length)
        inputs = torch.tensor([ids], device=args.device)
        context = model.disable_adapter() if disabled and args.adapter else contextlib.nullcontext()
        with context, torch.inference_mode():
            # Position t predicts token t+1; only completion targets are scored.
            logits = model(input_ids=inputs, use_cache=False).logits[:, boundary - 1:-1, :]
            labels = inputs[:, boundary:]
            token_logps = logits.float().log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
            probe = token_logps[0, :32].cpu().numpy().copy()
            return float(token_logps.sum().item()), probe

    rows, probe_delta, repeat_noise = [], 0.0, 0.0
    for index, row in enumerate(dataset):
        record = {"id": str(row.get("id", hashlib.sha256(
            json.dumps(row, sort_keys=True).encode()).hexdigest()))}
        for response in ("chosen", "rejected"):
            total, probe = score(row["prompt"], row[response])
            record[response] = total
            if args.adapter:
                base_total, base_probe = score(row["prompt"], row[response], disabled=True)
                record["base_" + response] = base_total
                if index < 8:
                    probe_delta = max(probe_delta, float(np.abs(probe - base_probe).max()))
            if index == 0:
                _, repeated = score(row["prompt"], row[response])
                repeat_noise = max(repeat_noise, float(np.abs(probe - repeated).max()))
        rows.append(record)
    result = {"identity": {"model": args.model, "revision": args.revision,
                           "data_sha256": hashlib.sha256(data_bytes).hexdigest(), "dtype": args.dtype,
                           "max_length": args.max_length, "max_prompt_length": args.max_prompt_length,
                           "truncation_mode": "keep_start"},
              "adapter": args.adapter, "reload_exact": bool(exact), "samples": rows,
              "effect": {"probe_delta": probe_delta, "repeat_noise": repeat_noise},
              "runtime": {"torch": torch.__version__, "device": args.device}}
    if args.device == "cuda":
        torch.cuda.synchronize()
        result["runtime"].update(peak_allocated=torch.cuda.max_memory_allocated(),
                                  peak_reserved=torch.cuda.max_memory_reserved(),
                                  gpu=torch.cuda.get_device_name())
    write_json(args.out, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--adapter", required=True)
    snap.add_argument("--out", required=True)
    ev = sub.add_parser("evaluate")
    for name in ("model", "revision", "data", "out"):
        ev.add_argument("--" + name, required=True)
    ev.add_argument("--adapter")
    ev.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ev.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ev.add_argument("--max-length", type=int, default=512)
    ev.add_argument("--max-prompt-length", type=int, default=256)
    comp = sub.add_parser("compare")
    for name in ("before", "after", "metrics-before", "metrics-after", "out"):
        comp.add_argument("--" + name, required=True)
    comp.add_argument("--min-margin", type=float, default=0.05)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.adapter, args.out)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        before, config_before = read_snapshot(args.before)
        after, config_after = read_snapshot(args.after)
        result = compare_evidence(before, after,
                                  json.loads(Path(args.metrics_before).read_text()),
                                  json.loads(Path(args.metrics_after).read_text()), args.min_margin)
        result["checks"]["matching_adapter_config"] = config_before == config_after
        if not all(result["checks"].values()):
            result["verdict"] = "DON'T SHIP"
        write_json(args.out, result)
        return 0 if result["verdict"] == "SHIP" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

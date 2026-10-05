"""Independent Soup forward/backward DPO parity probe; no training success is inferred."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def parity_report(stream_logits, resident_logits, stream_grads, resident_grads, dtype):
    atol, rtol = ((2e-4, 2e-3) if dtype == "float16" else (1e-6, 1e-5))
    forward = np.allclose(stream_logits, resident_logits, atol=atol, rtol=rtol)
    layout = bool(stream_grads) and set(stream_grads) == set(resident_grads)
    finite, close, relative, cosine = False, False, None, None
    if layout:
        left = np.concatenate([stream_grads[k].ravel() for k in sorted(stream_grads)]).astype(float)
        right = np.concatenate([resident_grads[k].ravel() for k in sorted(resident_grads)]).astype(float)
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        norm = np.linalg.norm(right)
        if finite and norm > 0 and np.linalg.norm(left) > 0:
            relative = float(np.linalg.norm(left - right) / norm)
            cosine = float(np.dot(left, right) / (np.linalg.norm(left) * norm))
            close = bool(np.allclose(left, right, atol=atol, rtol=rtol) and
                         relative <= rtol and cosine >= 0.9999)
    return {"forward_pass": bool(forward), "backward_pass": bool(layout and finite and close),
            "forward_bit_exact": bool(np.array_equal(stream_logits, resident_logits)),
            "forward_max_abs": float(np.max(np.abs(stream_logits - resident_logits))),
            "gradient_relative_l2": relative, "gradient_cosine": cosine,
            "gradient_tensors": len(stream_grads), "atol": atol, "rtol": rtol}


def reference_report(policy, streamed_ref, resident_ref, dtype):
    atol, rtol = ((2e-4, 2e-3) if dtype == "float16" else (1e-6, 1e-5))
    gap = float(np.max(np.abs(policy - streamed_ref)))
    valid = np.isfinite([policy, streamed_ref, resident_ref]).all()
    return {"reference_pass": bool(valid and gap > 1e-5 and np.allclose(
                streamed_ref, resident_ref, atol=atol, rtol=rtol)),
            "policy_reference_logprob_gap": gap}


def loss_report(streamed_loss, resident_loss, dtype):
    atol, rtol = ((2e-4, 2e-3) if dtype == "float16" else (1e-6, 1e-5))
    finite = bool(np.isfinite([streamed_loss, resident_loss]).all())
    return {"loss_pass": finite and bool(np.isclose(streamed_loss, resident_loss, atol=atol, rtol=rtol)),
            "loss_bit_exact": finite and streamed_loss == resident_loss,
            "loss_difference": abs(streamed_loss - resident_loss) if finite else None}


def optimizer_report(streamed, resident, dtype):
    atol, rtol = ((2e-4, 2e-3) if dtype == "float16" else (1e-6, 1e-5))
    layout = bool(streamed) and set(streamed) == set(resident) and all(
        streamed[k].shape == resident[k].shape for k in streamed)
    exact, passed, max_abs, relative, cosine = False, False, None, None, None
    if layout:
        left = np.concatenate([streamed[k].ravel() for k in sorted(streamed)]).astype(float)
        right = np.concatenate([resident[k].ravel() for k in sorted(resident)]).astype(float)
        if np.isfinite(left).all() and np.isfinite(right).all():
            max_abs = float(np.abs(left - right).max())
            exact = bool(np.array_equal(left, right))
            norm_left, norm_right = np.linalg.norm(left), np.linalg.norm(right)
            relative = float(np.linalg.norm(left - right) / norm_right) if norm_right else (0.0 if exact else None)
            cosine = float(np.dot(left, right) / (norm_left * norm_right)) if norm_left and norm_right else (1.0 if exact else None)
            passed = all(np.allclose(streamed[k], resident[k], atol=atol, rtol=rtol) for k in streamed)
            passed = passed and relative is not None and relative <= rtol and cosine is not None and cosine >= 0.9999
    return {"optimizer_state_pass": bool(passed), "optimizer_state_bit_exact": exact,
            "optimizer_state_max_abs": max_abs, "optimizer_state_relative_l2": relative,
            "optimizer_state_cosine": cosine, "optimizer_state_tensors": len(streamed)}


def probe_pass(result):
    gates = ["forward_pass", "backward_pass", "loss_pass", "reference_pass",
             "all_expected_gradients", "base_frozen", "decoder_base_on_meta",
             "optimizer_changed_parameters", "negative_control_rejected", "updated_parameters_finite",
             "resident_optimizer_changed_parameters", "optimizer_state_pass", "optimizer_state_expected"]
    return all(result.get(key) is True for key in gates)


def run_check(device="cuda", dtype="float16"):
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model, canonical_named_parameters

    if device == "cpu" and dtype != "float32":
        raise ValueError("CPU probe uses float32")
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    with tempfile.TemporaryDirectory() as work:
        weights, shards = Path(work) / "weights", Path(work) / "shards"
        config = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                             tie_word_embeddings=True, attention_dropout=0.0)
        config._attn_implementation = "eager"
        Qwen2ForCausalLM(config).to(getattr(torch, dtype)).save_pretrained(weights)
        index = shard_checkpoint(str(weights), str(shards), dtype=dtype)
        lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                          lora_dropout=0.0, task_type="CAUSAL_LM")
        streamed, runtime = build_streamed_model(
            model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
            device=device, dtype=dtype, buffers=2, pin=device == "cuda", seed=2026)
        try:
            resident = get_peft_model(AutoModelForCausalLM.from_pretrained(
                weights, torch_dtype=getattr(torch, dtype), attn_implementation=streamed.config._attn_implementation).to(device), lora)
            stream_params = dict(canonical_named_parameters(streamed))
            for name, parameter in canonical_named_parameters(resident):
                if "lora_" in name:
                    parameter.data = parameter.data.to(stream_params[name].dtype)
            # Nonzero B exercises both A and B gradients rather than only initial B.
            with torch.no_grad():
                for name, parameter in stream_params.items():
                    if "lora_B" in name:
                        parameter.fill_(0.01)
            set_peft_model_state_dict(resident, get_peft_model_state_dict(streamed))
            streamed.eval()
            resident.eval()
            prompt = torch.tensor([7, 11, 13, 17], device=device)
            ids = torch.stack([torch.cat([prompt, torch.tensor([19, 23, 29, 31], device=device)]),
                               torch.cat([prompt, torch.tensor([37, 41, 43, 47], device=device)])])
            mask = torch.zeros_like(ids[:, 1:])
            mask[:, 3:] = 1

            def loss_of(model):
                def logps():
                    logits = model(input_ids=ids, use_cache=False).logits
                    selected = logits[:, :-1].float().log_softmax(-1).gather(
                        -1, ids[:, 1:, None]).squeeze(-1)
                    return (selected * mask).sum(1), logits
                policy, logits = logps()
                with torch.no_grad(), model.disable_adapter():
                    reference, reference_logits = logps()
                margin = (policy[0] - policy[1]) - (reference[0] - reference[1])
                return -torch.nn.functional.logsigmoid(0.1 * margin), logits, policy, reference, reference_logits

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            streamed_loss, stream_logits, policy, reference, reference_logits = loss_of(streamed)
            resident_loss, resident_logits, _, resident_ref, resident_ref_logits = loss_of(resident)
            streamed_loss.backward()
            resident_loss.backward()
            grads = lambda model: {n: p.grad.detach().cpu().float().numpy().copy()
                                    for n, p in canonical_named_parameters(model)
                                    if "lora_" in n and p.grad is not None}
            stream_grads, resident_grads = grads(streamed), grads(resident)
            result = parity_report(stream_logits.detach().cpu().float().numpy(),
                                   resident_logits.detach().cpu().float().numpy(),
                                   stream_grads, resident_grads, dtype)
            result.update(reference_report(policy.detach().cpu().numpy(), reference.cpu().numpy(),
                                           resident_ref.cpu().numpy(), dtype))
            result["reference_pass"] &= bool(np.allclose(
                reference_logits.cpu().float().numpy(), resident_ref_logits.cpu().float().numpy(),
                atol=result["atol"], rtol=result["rtol"]))
            expected = [n for n, p in stream_params.items() if p.requires_grad]
            result["all_expected_gradients"] = set(expected) == set(stream_grads)
            result.update(loss_report(float(streamed_loss), float(resident_loss), dtype))
            result["base_frozen"] = all("lora_" in n for n in expected)
            result["decoder_base_on_meta"] = any(p.is_meta and ".layers." in n
                                                    for n, p in stream_params.items())
            initial = {n: p.detach().clone() for n, p in stream_params.items() if p.requires_grad}
            resident_params = dict(canonical_named_parameters(resident))
            resident_initial = {n: p.detach().clone() for n, p in resident_params.items() if p.requires_grad}
            optimizer_kwargs = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
            torch.optim.AdamW([p for p in stream_params.values() if p.requires_grad], **optimizer_kwargs).step()
            torch.optim.AdamW([p for p in resident_params.values() if p.requires_grad], **optimizer_kwargs).step()
            result["optimizer_changed_parameters"] = any(
                not torch.equal(initial[n], stream_params[n]) for n in initial)
            result["resident_optimizer_changed_parameters"] = any(
                not torch.equal(resident_initial[n], resident_params[n]) for n in resident_initial)
            result["updated_parameters_finite"] = all(
                bool(torch.isfinite(p).all()) for params in (stream_params, resident_params)
                for p in params.values() if p.requires_grad)
            states = lambda params: {n: p.detach().cpu().float().numpy().copy()
                                     for n, p in params.items() if p.requires_grad}
            stream_state, resident_state = states(stream_params), states(resident_params)
            result.update(optimizer_report(stream_state, resident_state, dtype))
            result["optimizer_state_expected"] = set(stream_state) == set(resident_state) == set(expected)
            result["optimizer_settings"] = optimizer_kwargs
            corrupt = {k: np.zeros_like(v) for k, v in stream_grads.items()}
            result["negative_control_rejected"] = not parity_report(
                stream_logits.detach().cpu().float().numpy(),
                resident_logits.detach().cpu().float().numpy(), corrupt, resident_grads, dtype)["backward_pass"]
            result.update(scope="random four-layer Qwen2; this does not prove checkpoint-scale or NF4 parity",
                          attention_implementation=streamed.config._attn_implementation,
                          dtype=dtype, device=device, runtime=runtime.stats(), torch=torch.__version__,
                          adapter_dtypes=sorted({str(p.dtype) for p in stream_params.values() if p.requires_grad}))
            if device == "cuda":
                torch.cuda.synchronize()
                result.update(gpu=torch.cuda.get_device_name(), peak_allocated=torch.cuda.max_memory_allocated(),
                              peak_reserved=torch.cuda.max_memory_reserved())
            result["pass"] = probe_pass(result)
            return result
        finally:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_check(args.device, args.dtype)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

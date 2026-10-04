"""Use Soup's DPO wrapper with evidence callbacks; no replacement training loop."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'


def json_safe(value):
    """Preserve nonfinite telemetry explicitly without invalid JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dumps(value, **kwargs):
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, default=str, **kwargs)


def write(path, value):
    Path(path).write_text(dumps(value, indent=2)+'\n', encoding='utf-8')


def merge_gradient_stats(destination, summaries):
    """Aggregate only gradients belonging to a successful optimizer update."""
    for name, summary in summaries.items():
        item = destination.setdefault(name, {'calls': 0, 'finite': True, 'max_abs': 0.0})
        item['calls'] += 1
        item['finite'] = item['finite'] and summary['finite']
        item['max_abs'] = max(item['max_abs'], summary['max_abs'])


def conversational(rows):
    return [{'prompt': [{'role': 'user', 'content': row['prompt']}],
             'chosen': [{'role': 'assistant', 'content': row['chosen']}],
             'rejected': [{'role': 'assistant', 'content': row['rejected']}]} for row in rows]


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def materialize_checkpoint(snapshot, destination):
    """Resolve Hub links into regular files and verify every copied byte."""
    snapshot, destination = Path(snapshot), Path(destination)
    sources = {p.name: p for p in snapshot.iterdir() if p.is_file()}
    destination.mkdir(parents=True, exist_ok=True)
    extras = {p.name for p in destination.iterdir()} - set(sources)
    if extras:
        raise ValueError(f'Unexpected files in checkpoint cache: {sorted(extras)}')
    hashes = {}
    for name, source in sources.items():
        target = destination / name
        if target.is_symlink():
            raise ValueError(f'Checkpoint cache target is a symlink: {name}')
        source_sha = file_sha256(source)
        if not target.exists() or file_sha256(target) != source_sha:
            shutil.copy2(source, target)
        if file_sha256(target) != source_sha:
            raise ValueError(f'Checkpoint copy failed checksum: {name}')
        hashes[name] = source_sha
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    import torch
    import yaml
    from huggingface_hub import snapshot_download
    from transformers import TrainerCallback
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.dpo import DPOTrainerWrapper
    from scripts.memory_budget import estimate

    assert torch.cuda.is_available(), 'This experiment requires a real CUDA T4.'
    properties = torch.cuda.get_device_properties(0)
    assert 'T4' in properties.name, f'Required T4, got {properties.name}'
    assert (properties.major, properties.minor) == (7, 5)
    from soup_cli.utils.gpu import cuda_supports_bf16
    assert not cuda_supports_bf16(), 'Recheck native fp16 assumptions on this hardware.'
    snapshot = Path(snapshot_download(MODEL, revision=REVISION,
                                   allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    )
    # Soup's sharder rejects symlinked weights. HF snapshots use links to blobs.
    # Keep the pinned snapshot intact and materialize a regular-file checkpoint.
    materialized = Path('model-cache') / REVISION
    source_hashes = materialize_checkpoint(snapshot, materialized)
    local_model = str(materialized.resolve())
    write(run_dir/'model.json', {'model': MODEL, 'revision': REVISION, 'local_path': local_model,
                               'materialized_source_sha256': source_hashes})
    model_config = json.loads((Path(local_model)/'config.json').read_text())
    write(run_dir/'model-config.json', model_config)
    budget = estimate(model_config)
    budget['gpu_total_gib'] = properties.total_memory/1024**3
    write(run_dir/'memory-before.json', budget)
    print('PRE-TRAINING MEMORY BUDGET', dumps(budget, indent=2), flush=True)

    raw = yaml.safe_load(Path(args.config).read_text())
    raw['base'], raw['output'] = local_model, str(run_dir/'adapter')
    config = load_config_from_string(yaml.safe_dump(raw))
    (run_dir/'resolved-soup.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    dataset = {key: conversational([json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()])
               for key, path in [('train', 'data/train.jsonl'), ('val', 'data/validation.jsonl')]}
    telemetry = (run_dir/'nvidia-smi-sampled.raw.csv').open('wb')
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu', '--format=csv', '-l', '1'], stdout=telemetry, stderr=subprocess.STDOUT)
    torch.cuda.reset_peak_memory_stats()
    wrapper = DPOTrainerWrapper(config, device='cuda', report_to='none')
    events, grads, raw_grads, raw_nonfinite_events, optimizer_steps = [], {}, {}, [], []
    pending_gradients, scale_before = {}, None

    def gradient_summary(gradient):
        gradient = gradient.detach()
        finite = torch.isfinite(gradient)
        all_finite = bool(finite.all().item())
        max_abs = float(gradient.abs().max().item()) if all_finite else float(
            torch.where(finite, gradient.abs(), 0).max().item())
        return {'finite': all_finite, 'max_abs': max_abs,
                'nonfinite_values': 0 if all_finite else int((~finite).sum().item()),
                'nan_values': 0 if all_finite else int(torch.isnan(gradient).sum().item()),
                'positive_infinity_values': 0 if all_finite else int(torch.isposinf(gradient).sum().item()),
                'negative_infinity_values': 0 if all_finite else int(torch.isneginf(gradient).sum().item())}

    def scaler_value():
        scaler = getattr(trainer.accelerator, 'scaler', None)
        return float(scaler.get_scale()) if scaler is not None else None

    def save_gradient_telemetry():
        write(run_dir/'gradients.json', grads)
        write(run_dir/'raw-scaled-gradients.json', {'parameters': raw_grads,
              'nonfinite_events': raw_nonfinite_events,
              'scope': 'Autograd hooks observe AMP-scaled microbatch gradients before unscale.'})
        write(run_dir/'optimizer-steps.json', {'steps': optimizer_steps,
              'successful_updates': sum(not step['skipped'] for step in optimizer_steps),
              'skipped_updates': sum(step['skipped'] for step in optimizer_steps),
              'gradient_scope': 'Unscaled accumulated gradients after Trainer clipping; only successful updates enter gradients.json.'})
    def memory(phase):
        torch.cuda.synchronize()
        value = {'utc': datetime.now(timezone.utc).isoformat(), 'phase': phase,
                 'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        import resource
        value['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        events.append(value)
        write(run_dir/'memory-events.json', events)
        print('MEMORY', dumps(value), flush=True)
    try:
        wrapper.setup(dataset)
        memory('setup_and_streaming_preflight')
        trainer, model = wrapper.trainer, wrapper.model
        write(run_dir/'effective-training.json', {
            'fp16': trainer.args.fp16, 'bf16': trainer.args.bf16,
            'beta': trainer.args.beta, 'loss_type': trainer.args.loss_type,
            'truncation_mode': trainer.args.truncation_mode,
            'max_length': trainer.args.max_length,
            'max_prompt_length': config.data.max_length // 2,
            'hf_gradient_checkpointing': trainer.args.gradient_checkpointing,
            'attention_implementation': model.config._attn_implementation,
            'stream_runtime': wrapper._stream_runtime.stats(),
            'nonfinite_numeric_encoding': 'NaN, Infinity and -Infinity are explicit JSON strings in telemetry.',
        })
        assert trainer.args.fp16 and not trainer.args.bf16
        assert config.training.stream_layers and wrapper._stream_runtime is not None
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        assert trainable and all('lora_' in name and '.ref.' not in name for name, _ in trainable)
        write(run_dir/'trainable-parameters.json', [{'name': n, 'shape': list(p.shape), 'dtype': str(p.dtype), 'device': str(p.device)} for n,p in trainable])
        for name, parameter in trainable:
            def hook(gradient, name=name):
                summary = gradient_summary(gradient)
                item = raw_grads.setdefault(name, {'calls': 0, 'nonfinite_calls': 0,
                    'nonfinite_values': 0, 'max_finite_abs': 0.0})
                item['calls'] += 1
                item['nonfinite_calls'] += int(not summary['finite'])
                item['nonfinite_values'] += summary['nonfinite_values']
                item['max_finite_abs'] = max(item['max_finite_abs'], summary['max_abs'])
                if not summary['finite']:
                    raw_nonfinite_events.append({'parameter': name,
                        'optimizer_step_attempt': trainer.state.global_step + 1, **summary})
            parameter.register_hook(hook)
        reference = {n: p.detach().cpu().clone() for n,p in model.named_parameters() if '.ref.' in n and 'lora_' in n}
        assert reference, 'Expected frozen initial-policy reference adapter under TRL 0.29.'
        assert all(not p.requires_grad and p.device.type != 'meta' for n,p in model.named_parameters() if n in reference)
        prepared = trainer.train_dataset
        audit = {'features': str(prepared.features), 'rows': len(prepared), 'identical_after_tokenization': 0,
                 'empty_completions': 0, 'missing_eos': 0, 'max_prepared_sequence': 0,
                 'validation_note': '50 rows are prepared; Soup wrapper does not schedule periodic validation evaluation.'}
        # TRL 0.29 uses chosen/rejected IDs; record actual prepared fields rather than guess a mask.
        for row in prepared:
            prompt = row.get('prompt_ids', [])
            chosen, rejected = row.get('chosen_ids', []), row.get('rejected_ids', [])
            if not chosen or not rejected:
                audit['empty_completions'] += 1
            if chosen == rejected:
                audit['identical_after_tokenization'] += 1
            for response in (chosen, rejected):
                audit['max_prepared_sequence'] = max(audit['max_prepared_sequence'], len(prompt)+len(response))
                audit['missing_eos'] += int(bool(response) and wrapper.tokenizer.eos_token_id not in response)
        batch = trainer.data_collator([prepared[0]])
        audit['batch_shapes'] = {k: list(v.shape) for k,v in batch.items() if torch.is_tensor(v)}
        if 'completion_mask' in batch:
            audit['completion_tokens_first_pair'] = batch['completion_mask'].sum(-1).tolist()
        write(run_dir/'tokenization-audit.json', audit)
        print('TOKENIZATION AUDIT', dumps(audit, indent=2), flush=True)
        assert audit['empty_completions'] == 0 and audit['identical_after_tokenization'] == 0
        assert audit['missing_eos'] == 0, 'Completion truncation removed the response terminator.'
        assert audit['max_prepared_sequence'] <= config.data.max_length

        class EvidenceCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                model.save_pretrained(run_dir/'initial_adapter', selected_adapters=['default'])
                wrapper.tokenizer.save_pretrained(run_dir/'initial_adapter')
                memory('before_first_training_step')
            def on_log(self, args, state, control, logs=None, **kwargs):
                print('TRAIN LOG', dumps({'step': state.global_step, **(logs or {})}), flush=True)
            def on_pre_optimizer_step(self, args, state, control, **kwargs):
                nonlocal pending_gradients, scale_before
                # Transformers 5.16.1 clips through Accelerate (and unscales)
                # before this callback. Calling unscale again would be invalid.
                pending_gradients = {name: gradient_summary(parameter.grad)
                                     for name, parameter in trainable if parameter.grad is not None}
                scale_before = scaler_value()
            def on_optimizer_step(self, args, state, control, **kwargs):
                skipped = bool(trainer.accelerator.optimizer_step_was_skipped)
                step = {'optimizer_step_attempt': state.global_step + 1, 'skipped': skipped,
                        'scale_before': scale_before, 'scale_after': scaler_value(),
                        'gradients': pending_gradients,
                        'missing_gradient_parameters': [name for name, _ in trainable
                                                        if name not in pending_gradients]}
                optimizer_steps.append(step)
                if not skipped:
                    merge_gradient_stats(grads, pending_gradients)
                print('OPTIMIZER STEP', dumps({key: value for key, value in step.items()
                      if key != 'gradients'}), flush=True)
            def on_step_end(self, args, state, control, **kwargs):
                memory(f'optimizer_step_attempt_{state.global_step}')
                save_gradient_telemetry()
            def on_train_end(self, args, state, control, **kwargs):
                drift = {n: float((dict(model.named_parameters())[n].detach().cpu()-value).abs().max()) for n,value in reference.items()}
                write(run_dir/'reference-stability.json', {'tensor_count': len(reference), 'max_abs_change': max(drift.values()), 'all_frozen': True})
                assert max(drift.values()) == 0, 'DPO reference drifted.'
                memory('train_end_before_stream_release')

        trainer.add_callback(EvidenceCallback())
        def reference_memory(module, inputs, outputs):
            if 'ref' in module.active_adapters and len(events) < 8:
                memory('reference_forward_completed')
        model.register_forward_hook(reference_memory)
        result = wrapper.train()
        write(run_dir/'training-result.json', result)
        write(run_dir/'trainer-log-history.json', trainer.state.log_history)
        save_gradient_telemetry()
        assert any(not step['skipped'] for step in optimizer_steps), 'AMP skipped every optimizer update.'
        assert all(g['finite'] for g in grads.values())
        assert len(grads) == len(trainable), 'A trainable adapter never received a gradient.'
        for layer in range(model_config['num_hidden_layers']):
            layer_grads = [g for n,g in grads.items() if f'.layers.{layer}.' in n]
            assert layer_grads and any(g['max_abs'] > 0 for g in layer_grads), f'Layer {layer} never received a nonzero adapter gradient.'
        assert all(bool(torch.isfinite(parameter).all()) for _, parameter in trainable), 'An updated policy adapter contains nonfinite values.'
        write(run_dir/'memory-actual.json', {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'estimate_gib': budget['conservative_total_gib'],
              'gap_allocated_gib': torch.cuda.max_memory_allocated()/1024**3-budget['conservative_total_gib']})
        print('TRAINING RESULT', dumps(result, indent=2), flush=True)
    finally:
        runtime = getattr(wrapper, '_stream_runtime', None)
        if runtime is not None:
            runtime.close()
        monitor.terminate()
        monitor.wait(timeout=10)
        telemetry.close()
        save_gradient_telemetry()
        memory('final_or_failure')


if __name__ == '__main__':
    main()

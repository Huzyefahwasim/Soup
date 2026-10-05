"""Use Soup's DPO wrapper with evidence callbacks; no replacement training loop."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def conversational(rows):
    return [{'prompt': [{'role': 'user', 'content': row['prompt']}],
             'chosen': [{'role': 'assistant', 'content': row['chosen']}],
             'rejected': [{'role': 'assistant', 'content': row['rejected']}]} for row in rows]


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
    materialized.mkdir(parents=True, exist_ok=True)
    for source in snapshot.iterdir():
        if source.is_file():
            target = materialized / source.name
            if not target.exists() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
            assert not target.is_symlink()
    local_model = str(materialized.resolve())
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in materialized.iterdir() if p.is_file()}
    write(run_dir/'model.json', {'model': MODEL, 'revision': REVISION, 'local_path': local_model,
                               'materialized_source_sha256': source_hashes})
    model_config = json.loads((Path(local_model)/'config.json').read_text())
    write(run_dir/'model-config.json', model_config)
    budget = estimate(model_config)
    budget['gpu_total_gib'] = properties.total_memory/1024**3
    write(run_dir/'memory-before.json', budget)
    print('PRE-TRAINING MEMORY BUDGET', json.dumps(budget, indent=2), flush=True)

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
    events, grads = [], {}
    def memory(phase):
        torch.cuda.synchronize()
        value = {'utc': datetime.now(timezone.utc).isoformat(), 'phase': phase,
                 'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        import resource
        value['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        events.append(value)
        write(run_dir/'memory-events.json', events)
        print('MEMORY', json.dumps(value), flush=True)
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
        })
        assert trainer.args.fp16 and not trainer.args.bf16
        assert config.training.stream_layers and wrapper._stream_runtime is not None
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        assert trainable and all('lora_' in name and '.ref.' not in name for name, _ in trainable)
        write(run_dir/'trainable-parameters.json', [{'name': n, 'shape': list(p.shape), 'dtype': str(p.dtype), 'device': str(p.device)} for n,p in trainable])
        for name, parameter in trainable:
            def hook(gradient, name=name):
                item = grads.setdefault(name, {'calls': 0, 'finite': True, 'max_abs': 0.0})
                item['calls'] += 1
                item['finite'] = item['finite'] and bool(torch.isfinite(gradient).all().item())
                item['max_abs'] = max(item['max_abs'], float(gradient.detach().abs().max().item()))
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
        print('TOKENIZATION AUDIT', json.dumps(audit, indent=2), flush=True)
        assert audit['empty_completions'] == 0 and audit['identical_after_tokenization'] == 0
        assert audit['missing_eos'] == 0, 'Completion truncation removed the response terminator.'
        assert audit['max_prepared_sequence'] <= config.data.max_length

        class EvidenceCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                model.save_pretrained(run_dir/'initial_adapter', selected_adapters=['default'])
                wrapper.tokenizer.save_pretrained(run_dir/'initial_adapter')
                memory('before_first_training_step')
            def on_log(self, args, state, control, logs=None, **kwargs):
                print('TRAIN LOG', json.dumps({'step': state.global_step, **(logs or {})}, default=str), flush=True)
            def on_step_end(self, args, state, control, **kwargs):
                memory(f'optimizer_step_{state.global_step}')
                write(run_dir/'gradients.json', grads)
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
        write(run_dir/'gradients.json', grads)
        assert all(g['finite'] for g in grads.values())
        assert len(grads) == len(trainable), 'A trainable adapter never received a gradient.'
        for layer in range(model_config['num_hidden_layers']):
            layer_grads = [g for n,g in grads.items() if f'.layers.{layer}.' in n]
            assert layer_grads and any(g['max_abs'] > 0 for g in layer_grads), f'Layer {layer} never received a nonzero adapter gradient.'
        write(run_dir/'memory-actual.json', {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'estimate_gib': budget['conservative_total_gib'],
              'gap_allocated_gib': torch.cuda.max_memory_allocated()/1024**3-budget['conservative_total_gib']})
        print('TRAINING RESULT', json.dumps(result, indent=2), flush=True)
    finally:
        runtime = getattr(wrapper, '_stream_runtime', None)
        if runtime is not None:
            runtime.close()
        monitor.terminate()
        monitor.wait(timeout=10)
        telemetry.close()
        memory('final_or_failure')


if __name__ == '__main__':
    main()

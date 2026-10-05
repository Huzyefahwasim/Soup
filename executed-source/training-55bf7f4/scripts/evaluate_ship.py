"""Measure fp16 task/general scores for Soup's offline ship gate.

Uses Soup 0.75.2's own greedy generator and scorers. The base is released before
the tuned model loads. A completed score file is evidence, not a SHIP verdict.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def measure_side(generator, tasks, suite_names, *, model_id, record=None):
    """Measure with real Soup scorers, preserving responses and inference errors."""
    from soup_cli.eval.custom import run_eval
    from soup_cli.eval.gate_suites import score_bundled_suite

    scope = 'task'
    errors = []
    generated = 0

    def observed(prompt):
        nonlocal generated
        started = time.perf_counter()
        item = {'scope': scope, 'prompt': prompt, 'started_utc': utc()}
        try:
            output = generator(prompt)
            if not isinstance(output, str):
                raise TypeError(f'generator returned {type(output).__name__}, expected str')
            item['output'] = output
            return output
        except Exception as exc:
            item['error'] = f'{type(exc).__name__}: {exc}'
            errors.append(item['error'])
            raise
        finally:
            generated += 1
            item['duration_seconds'] = time.perf_counter() - started
            if record:
                record(item)

    task_results = run_eval(model_id, tasks, generate_fn=observed)
    benchmarks = {}
    for name in suite_names:
        scope = name
        benchmarks[name] = score_bundled_suite(name, observed)
        # Some bundled scorers convert generator exceptions to a failed item.
        # A model-load/inference failure is not a measured benchmark regression.
        if errors:
            raise RuntimeError(f'generation failed while measuring {name}: {errors[0]}')
    return {
        'task_accuracy': task_results.accuracy,
        'task_total': task_results.total,
        'task_correct': task_results.correct,
        'task_category_scores': task_results.category_scores,
        'task_results': [
            {'prompt': result.task.prompt, 'expected': result.task.expected,
             'category': result.task.category, 'scoring': result.task.scoring,
             'output': result.output, 'score': result.score, 'matched': result.matched}
            for result in task_results.results
        ],
        'benchmarks': benchmarks,
        'generation_calls': generated,
        'generation_errors': len(errors),
    }


def build_evidence(base, tuned, measurement):
    if set(base['benchmarks']) != set(tuned['benchmarks']) or not base['benchmarks']:
        raise ValueError('base and tuned must have the same nonempty general suite')
    return {
        'task': {'mode': 'metric', 'base': base['task_accuracy'], 'tuned': tuned['task_accuracy']},
        'benchmarks': {name: {'base': base['benchmarks'][name], 'tuned': tuned['benchmarks'][name]}
                       for name in base['benchmarks']},
        'measurement': measurement,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Already downloaded immutable model directory')
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--out', required=True, help='Measured input JSON for soup ship --evidence')
    parser.add_argument('--task-eval', default='data/task_eval.jsonl')
    parser.add_argument('--revision', default=None, help='Recorded immutable model revision')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--max-new-tokens', type=int, default=None)
    args = parser.parse_args()

    import torch
    from soup_cli import __version__ as soup_version
    from soup_cli.eval.custom import load_eval_tasks
    from soup_cli.eval.gate import current_baseline_stamp
    from soup_cli.eval.gate_suites import BEHAVIOURAL_MAX_NEW_TOKENS, DEFAULT_GENERAL_SUITE
    from soup_cli.utils import live_eval

    if soup_version != '0.75.2':
        raise ValueError(f'This producer is source-audited for Soup 0.75.2, found {soup_version}')
    if not Path(args.model).is_dir():
        raise ValueError('--model must be an existing downloaded model directory')
    if not (Path(args.adapter) / 'adapter_model.safetensors').is_file():
        raise ValueError('--adapter must contain adapter_model.safetensors')
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable')
        if 'T4' not in torch.cuda.get_device_name(0) or args.dtype != 'float16':
            raise ValueError('This Colab demonstration requires a T4 and float16')
    elif args.dtype != 'float32':
        raise ValueError('CPU evaluation requires --dtype float32')
    tokens = BEHAVIOURAL_MAX_NEW_TOKENS if args.max_new_tokens is None else args.max_new_tokens
    if not 1 <= tokens <= BEHAVIOURAL_MAX_NEW_TOKENS:
        raise ValueError(f'--max-new-tokens must be 1..{BEHAVIOURAL_MAX_NEW_TOKENS}')
    tasks = load_eval_tasks(args.task_eval)
    if not tasks:
        raise ValueError('task evaluation fixture is empty')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError('--out already exists; preserve the prior measurement')
    source_revision = args.revision
    if source_revision is None and re.fullmatch(r'[0-9a-f]{40}', Path(args.model).name):
        source_revision = Path(args.model).name
    metadata = {
        'started_utc': utc(), 'dtype': args.dtype, 'device': args.device,
        'base_model_directory': args.model, 'base_model_revision': source_revision,
        'adapter': args.adapter, 'decode': 'greedy', 'max_new_tokens': tokens,
        'general_suite': list(DEFAULT_GENERAL_SUITE), 'task_count': len(tasks),
        'task_sha256': hashlib.sha256(Path(args.task_eval).read_bytes()).hexdigest(),
        'adapter_sha256': hashlib.sha256(
            (Path(args.adapter) / 'adapter_model.safetensors').read_bytes()).hexdigest(),
        'versions': {name: importlib.metadata.version(name)
                     for name in ['soup-cli', 'torch', 'transformers', 'peft', 'trl']},
        'scorer': current_baseline_stamp(),
        'limitations': ['The eight task cases and bundled mini suites are small functional checks.',
                        'No remote judge is used; this is not a support-ticket quality assessment.',
                        'An offline verdict trusts this producer; keep response journals and raw logs.'],
    }
    if args.device == 'cuda':
        metadata['gpu'] = torch.cuda.get_device_name(0)
    write_json(out.with_suffix('.measurement.json'), metadata)

    def side(label, adapter):
        side_path = out.with_name(f'{out.stem}.{label}.json')
        responses_path = out.with_name(f'{out.stem}.{label}.responses.jsonl')
        report = {'status': 'started', 'started_utc': utc(), 'adapter': adapter}
        write_json(side_path, report)
        loaded = generator = None
        started = time.perf_counter()
        try:
            loaded = live_eval.load_model_and_tokenizer(
                args.model, adapter=adapter, device=args.device, dtype=args.dtype)
            report['model_attention_implementation'] = loaded[0].config._attn_implementation
            report['parameter_dtypes'] = sorted({str(p.dtype) for p in loaded[0].parameters()})
            generator = live_eval.make_generator(
                args.model, loaded=loaded, max_new_tokens=tokens)
            if args.device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            with responses_path.open('w', encoding='utf-8') as journal:
                def record(item):
                    journal.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
                    journal.flush()
                report.update(measure_side(generator, tasks, DEFAULT_GENERAL_SUITE,
                                           model_id=args.model, record=record))
            if args.device == 'cuda':
                torch.cuda.synchronize()
                report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
            report.update(status='complete', finished_utc=utc(),
                          duration_seconds=time.perf_counter() - started)
            write_json(side_path, report)
            return report
        except Exception as exc:
            report.update(status='failed', finished_utc=utc(), error=f'{type(exc).__name__}: {exc}')
            write_json(side_path, report)
            raise
        finally:
            # Both the generator closure and loaded tuple own the resident model.
            generator = loaded = None
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(json.dumps({'event': 'resident_released', 'side': label,
                                  'allocated_bytes': torch.cuda.memory_allocated(),
                                  'reserved_bytes': torch.cuda.memory_reserved()}), flush=True)

    base = side('base', None)
    tuned = side('tuned', args.adapter)
    metadata['finished_utc'] = utc()
    evidence = build_evidence(base, tuned, metadata)
    evidence['provenance'] = current_baseline_stamp()
    write_json(out, evidence)
    write_json(out.with_suffix('.measurement.json'), metadata)
    print(json.dumps({'event': 'fp16_ship_scores_measured', 'output': str(out),
                      'task': evidence['task'], 'benchmarks': evidence['benchmarks']},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

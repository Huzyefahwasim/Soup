"""Run checks in separate processes and preserve failed command output."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import re


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


REQUIRED_COMMANDS = ['nvidia-smi-before', 'environment', 'soup-version', 'doctor',
    'dpo-doctor-refusal', 'data-lint', 'data-doctor-chat', 'preflight-dry-run',
    'streaming-parity', 'dpo-training', 'baseline-heldout', 'trained-heldout',
    'snapshot-before', 'snapshot-after', 'verify-adapter', 'negative-control',
    'ship-score-fp16', 'ship', 'nvidia-smi-after']


def source_provenance(root, config):
    """Fingerprint the submitted inputs before any subprocess can run."""
    root = Path(root).resolve()
    def hashes(paths):
        return {str(path.relative_to(root)).replace('\\', '/'):
                hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    inputs = [config_path, *sorted((root/'data').glob('*.jsonl'))]
    return {'started_utc': utc(), 'python_executable': sys.executable,
            'script_sha256': hashes((root/'scripts').rglob('*.py')),
            'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in inputs}}


def command_problem(record, directory):
    """Return an evidence failure, allowing only substantiated expected exits."""
    name, code = record['name'], record.get('returncode')
    directory = Path(directory)
    def payload(filename):
        value = json.loads((directory/filename).read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError(f'{filename} is not a JSON object')
        return value
    try:
        if name == 'dpo-doctor-refusal':
            log = (directory/'dpo-doctor-refusal.raw.log').read_text(encoding='utf-8', errors='replace')
            log = ' '.join(re.sub(r'\x1b\[[0-9;]*m', '', log).split())
            if code != 2 or 'preference data' not in log or 'soup data lint' not in log:
                return 'expected DPO doctor refusal was not observed'
            return None
        if name in ('verify-adapter', 'negative-control'):
            report = payload('verification.json' if name == 'verify-adapter' else 'negative-control.json')
            verdict, checks = report.get('verdict'), report.get('checks')
            if verdict not in ('SHIP', "DON'T SHIP") or not isinstance(checks, dict) or not checks:
                return 'invalid verification verdict artifact'
            if any(type(value) is not bool for value in checks.values()):
                return 'verification checks are not Boolean'
            if (code, verdict) not in ((0, 'SHIP'), (1, "DON'T SHIP")):
                return 'verification exit does not match its verdict'
            if (verdict == 'SHIP') != all(checks.values()):
                return 'verification verdict does not match its checks'
            if name == 'negative-control' and (code != 1 or
                    checks.get('updated_lora_b') is not False or checks.get('active_adapter') is not False):
                return 'unchanged adapter negative control was not rejected'
            return None
        if name == 'ship':
            report, stamped = payload('ship-verdict.json'), payload('ship-evidence.json')
            if (code, report.get('decision')) not in ((0, 'SHIP'), (2, "DON'T SHIP")):
                return 'Soup ship exit does not match a valid verdict'
            sha = stamped.get('provenance', {}).get('config_sha', '')
            if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
                return 'Soup ship stamped evidence has no valid configuration SHA'
            return None
        if code != 0:
            return f'command failed with exit {code}'
        json_outputs = {'data-lint': 'data-lint.json', 'data-doctor-chat': 'data-doctor.json',
            'streaming-parity': 'streaming-parity.json', 'dpo-training': 'training-result.json',
            'baseline-heldout': 'baseline.json', 'trained-heldout': 'trained.json',
            'ship-score-fp16': 'ship-input-fp16.json'}
        if name in json_outputs:
            report = payload(json_outputs[name])
            if not report:
                return 'empty measurement artifact'
            if name == 'streaming-parity' and report.get('pass') is not True:
                return 'streaming parity artifact did not pass'
            if name == 'dpo-training' and not report.get('total_steps', 0) > 0:
                return 'training artifact records no steps'
            if name in ('baseline-heldout', 'trained-heldout') and (
                    not isinstance(report.get('identity'), dict) or not report.get('samples')):
                return 'held-out measurements are missing identity or samples'
            if name == 'ship-score-fp16':
                pairs = [report.get('task', {}), *report.get('benchmarks', {}).values()]
                if len(pairs) < 2 or any(not isinstance(p, dict) or any(
                        not isinstance(p.get(k), (int, float)) or isinstance(p.get(k), bool) or
                        not math.isfinite(p[k]) or not 0 <= p[k] <= 1 for k in ('base', 'tuned')) for p in pairs):
                    return 'invalid fp16 task or benchmark measurements'
        if name in ('snapshot-before', 'snapshot-after'):
            path = directory/('before.npz' if name == 'snapshot-before' else 'after.npz')
            if not path.is_file() or path.stat().st_size == 0:
                return 'adapter snapshot artifact is missing or empty'
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        return f'invalid or missing evidence: {type(exc).__name__}: {exc}'
    return None


def collection_status(commands, directory):
    records = {record['name']: record for record in commands}
    failed = []
    names = [*REQUIRED_COMMANDS, *(name for name in records if name not in REQUIRED_COMMANDS)]
    for name in names:
        record = records.get(name)
        problem = command_problem(record, directory) if record else 'required command was not run'
        if problem:
            failed.append({'name': name, 'returncode': record.get('returncode') if record else None,
                           'reason': problem})
    decision = None
    try:
        decision = json.loads((Path(directory)/'ship-verdict.json').read_text())['decision']
        if decision not in ('SHIP', "DON'T SHIP"):
            decision = None
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return {'collection_status': 'incomplete' if failed else 'complete',
            'failed_commands': failed, 'measured_ship_verdict': decision}


def run_command(argv, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = {'argv': list(argv), 'started_utc': utc()}
    record_path = directory / f'{name}.command.json'
    record_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_DISABLE_TELEMETRY='1', WANDB_DISABLED='true')
    with (directory / f'{name}.raw.log').open('wb') as raw, (directory / f'{name}.timestamped.log').open('w', encoding='utf-8') as stamped:
        try:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        except OSError as exc:
            record.update(returncode=127, launch_error=str(exc))
            raw.write((str(exc)+'\n').encode('utf-8'))
            stamped.write(f'[{utc()}] launch failed: {exc}\n')
        else:
            for line in iter(process.stdout.readline, b''):
                raw.write(line)
                raw.flush()
                rendered = f'[{utc()}] '+line.decode('utf-8', errors='replace')
                stamped.write(rendered)
                stamped.flush()
                print(rendered, end='', flush=True)
            record['returncode'] = process.wait()
            process.stdout.close()
    record['finished_utc'] = utc()
    record_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--config', default='configs/soup.yaml')
    args = parser.parse_args()
    run_dir = Path(args.run_dir or f'artifacts/{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}')
    if run_dir.exists():
        parser.error('run-dir already exists; create a new attempt so failures stay intact')
    run_dir.mkdir(parents=True)
    provenance = source_provenance(Path.cwd(), args.config)
    (run_dir/'source-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    py = sys.executable
    soup = [py, '-m', 'soup_cli']
    commands = []
    def run(name, argv):
        result = run_command(argv, run_dir, name)
        commands.append(dict(name=name, **result))
        (run_dir / 'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        return result['returncode']
    run('nvidia-smi-before', ['nvidia-smi'])
    run('environment', [py, '-m', 'pip', 'freeze'])
    run('soup-version', [py, '-c', "import importlib.metadata as m; print(m.version('soup-cli'))"])
    run('doctor', [*soup, 'doctor', '--config', args.config])
    run('dpo-doctor-refusal', [*soup, 'data', 'doctor', 'data/train.jsonl', '--format', 'dpo', '--model', 'Qwen/Qwen2.5-0.5B-Instruct'])
    lint = run('data-lint', [*soup, 'data', 'lint', 'data/train.jsonl', '--format', 'dpo', '--sample', '400', '--output', str(run_dir/'data-lint.json')])
    doctor = run('data-doctor-chat', [*soup, 'data', 'doctor', 'data/doctor_chat.jsonl', '--format', 'chatml', '--model', 'Qwen/Qwen2.5-0.5B-Instruct', '--max-length', '512', '--show-mask', '2', '--train-on-eot', '--output', str(run_dir/'data-doctor.json')])
    preflight = run('preflight-dry-run', [*soup, 'train', '--config', args.config, '--dry-run'])
    parity = run('streaming-parity', [py, 'scripts/check_streaming.py', '--out', str(run_dir/'streaming-parity.json'), '--device', 'cuda', '--dtype', 'float16'])
    train = run('dpo-training', [py, '-m', 'scripts.train_dpo', '--config', args.config, '--run-dir', str(run_dir)]) if lint == preflight == parity == 0 else None
    if train == 0:
        model = json.loads((run_dir/'model.json').read_text())
        common = ['--model', model['local_path'], '--revision', model['revision'], '--data', 'data/test.jsonl', '--device', 'cuda', '--dtype', 'float16']
        run('baseline-heldout', [py, 'scripts/verify_adapter.py', 'evaluate', *common, '--out', str(run_dir/'baseline.json')])
        run('trained-heldout', [py, 'scripts/verify_adapter.py', 'evaluate', *common, '--adapter', str(run_dir/'adapter'), '--out', str(run_dir/'trained.json')])
        run('snapshot-before', [py, 'scripts/verify_adapter.py', 'snapshot', '--adapter', str(run_dir/'initial_adapter'), '--out', str(run_dir/'before.npz')])
        run('snapshot-after', [py, 'scripts/verify_adapter.py', 'snapshot', '--adapter', str(run_dir/'adapter'), '--out', str(run_dir/'after.npz')])
        compare = [py, 'scripts/verify_adapter.py', 'compare', '--before', str(run_dir/'before.npz'), '--metrics-before', str(run_dir/'baseline.json')]
        run('verify-adapter', [*compare, '--after', str(run_dir/'after.npz'), '--metrics-after', str(run_dir/'trained.json'), '--out', str(run_dir/'verification.json')])
        run('negative-control', [*compare, '--after', str(run_dir/'before.npz'), '--metrics-after', str(run_dir/'baseline.json'), '--out', str(run_dir/'negative-control.json')])
        scores = run('ship-score-fp16', [py, 'scripts/evaluate_ship.py', '--model', model['local_path'], '--revision', model['revision'], '--adapter', str(run_dir/'adapter'), '--out', str(run_dir/'ship-input-fp16.json')])
        if scores == 0:
            run('ship', [*soup, 'ship', '--evidence', str(run_dir/'ship-input-fp16.json'), '--config', str(run_dir/'resolved-soup.yaml'), '--output', str(run_dir/'ship-verdict.json'), '--emit-evidence', str(run_dir/'ship-evidence.json')])
    run('nvidia-smi-after', ['nvidia-smi'])
    status = collection_status(commands, run_dir)
    summary = {'finished_utc': utc(), 'domain_verdict': "DON'T SHIP", 'parity_exit': parity,
               'training_exit': train, 'input_sha256': provenance['input_sha256'],
               'script_sha256': provenance['script_sha256'], **status,
               'domain_verdict_reason': 'Public translated preferences do not establish Russian support-ticket quality. Read verification and ship results for additional failures.',
               'training_prerequisites': {'data_lint': lint, 'chat_projection_doctor': doctor, 'dry_run': preflight, 'streamed_parity': parity},
               'training_skipped': train is None}
    (run_dir/'run-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return 0 if status['collection_status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())

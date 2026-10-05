"""Run checks in separate processes and preserve failed command output."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


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
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(args.config), *Path('data').glob('*.jsonl')]}
    summary = {'finished_utc': utc(), 'verdict': "DON'T SHIP", 'parity_exit': parity,
               'training_exit': train, 'input_sha256': hashes,
               'reason': 'Public translated preferences do not establish Russian support-ticket quality. Read verification and ship results for additional failures.',
               'training_prerequisites': {'data_lint': lint, 'chat_projection_doctor': doctor, 'dry_run': preflight, 'streamed_parity': parity},
               'training_skipped': train is None}
    (run_dir/'run-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

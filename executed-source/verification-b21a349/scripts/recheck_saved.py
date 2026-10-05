"""Retry verification without changing or retraining an existing saved adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from scripts.run_experiment import run_command, source_provenance, command_problem, utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    training = Path(args.training_run).resolve()
    out = Path(args.out_dir).resolve()
    if out.exists():
        parser.error('out-dir already exists; retain the earlier verification results')
    out.mkdir(parents=True)
    model = json.loads((training/'model.json').read_text())
    identity = source_provenance(Path.cwd(), 'configs/soup.yaml')
    identity.update(training_run=str(training), model_revision=model['revision'],
                    adapter_sha256=hashlib.sha256((training/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
                    training_source_provenance=json.loads((training/'source-provenance.json').read_text()))
    (out/'source-provenance.json').write_text(json.dumps(identity, indent=2)+'\n')
    py = sys.executable
    commands = []
    def run(name, argv):
        record = dict(name=name, **run_command(argv, out, name))
        commands.append(record)
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        return record['returncode']
    common = ['--model', model['local_path'], '--revision', model['revision'],
              '--data', 'data/test.jsonl', '--device', 'cuda', '--dtype', 'float16']
    run('baseline-heldout', [py, 'scripts/verify_adapter.py', 'evaluate', *common, '--out', str(out/'baseline.json')])
    run('trained-heldout', [py, 'scripts/verify_adapter.py', 'evaluate', *common,
                            '--adapter', str(training/'adapter'), '--out', str(out/'trained.json')])
    run('snapshot-before', [py, 'scripts/verify_adapter.py', 'snapshot', '--adapter', str(training/'initial_adapter'), '--out', str(out/'before.npz')])
    run('snapshot-after', [py, 'scripts/verify_adapter.py', 'snapshot', '--adapter', str(training/'adapter'), '--out', str(out/'after.npz')])
    compare = [py, 'scripts/verify_adapter.py', 'compare', '--before', str(out/'before.npz'),
               '--metrics-before', str(out/'baseline.json')]
    run('verify-adapter', [*compare, '--after', str(out/'after.npz'),
                          '--metrics-after', str(out/'trained.json'), '--out', str(out/'verification.json')])
    run('negative-control', [*compare, '--after', str(out/'before.npz'),
                            '--metrics-after', str(out/'baseline.json'), '--out', str(out/'negative-control.json')])
    failed = [dict(name=r['name'], problem=problem) for r in commands
              if (problem := command_problem(r, out))]
    verdict = json.loads((out/'verification.json').read_text()) if (out/'verification.json').exists() else None
    summary = dict(finished_utc=utc(), collection_status='incomplete' if failed else 'complete',
                   failed_commands=failed, adapter_verification=verdict,
                   scope='Verification retry only. The original training run and its failed checks are unchanged.')
    (out/'verification-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

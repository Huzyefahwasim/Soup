import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_experiment import run_command
import scripts.run_experiment as runner
from scripts.memory_budget import estimate


class EvidenceTests(unittest.TestCase):
    def test_expected_rejection_requires_the_matching_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = {'name': 'negative-control', 'returncode': 1}
            self.assertIsNotNone(runner.command_problem(record, root))
            (root / 'negative-control.json').write_text(json.dumps({
                'verdict': "DON'T SHIP", 'checks': {'updated_lora_b': False, 'active_adapter': False}}))
            self.assertIsNone(runner.command_problem(record, root))
            record['returncode'] = 0
            self.assertIsNotNone(runner.command_problem(record, root))

    def test_valid_dont_ship_is_completed_measurement_but_missing_stamp_is_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = {'name': 'ship', 'returncode': 2}
            (root / 'ship-verdict.json').write_text(json.dumps({'decision': "DON'T SHIP"}))
            self.assertIsNotNone(runner.command_problem(record, root))
            (root / 'ship-evidence.json').write_text(json.dumps({'provenance': {'config_sha': 'a'*64}}))
            self.assertIsNone(runner.command_problem(record, root))
            record['returncode'] = 1
            self.assertIsNotNone(runner.command_problem(record, root))

    def test_success_exit_without_measurement_is_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            record = {'name': 'baseline-heldout', 'returncode': 0}
            self.assertIsNotNone(runner.command_problem(record, Path(folder)))
            summary = runner.collection_status([record], Path(folder))
            self.assertEqual(summary['collection_status'], 'incomplete')
            self.assertGreater(len(summary['failed_commands']), 1)

    def test_refusal_usage_error_cannot_replace_the_real_dpo_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'dpo-doctor-refusal.raw.log').write_text('Error: Missing option --model.\n')
            record = {'name': 'dpo-doctor-refusal', 'returncode': 2}
            self.assertIsNotNone(runner.command_problem(record, root))
            (root / 'dpo-doctor-refusal.raw.log').write_text(
                "Error: format='dpo' is preference data — use soup data lint instead.\n", encoding='utf-8')
            self.assertIsNone(runner.command_problem(record, root))

    def test_start_provenance_hashes_exact_script_and_data_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'scripts').mkdir()
            (root / 'data').mkdir()
            (root / 'scripts' / 'fixture.py').write_bytes(b'print(1)\n')
            (root / 'data' / 'fixture.jsonl').write_bytes(b'{}\n')
            (root / 'config.yaml').write_bytes(b'task: dpo\n')
            result = runner.source_provenance(root, 'config.yaml')
            import hashlib
            self.assertEqual(result['script_sha256']['scripts/fixture.py'], hashlib.sha256(b'print(1)\n').hexdigest())
            (root / 'scripts' / 'fixture.py').write_bytes(b'print(2)\n')
            self.assertNotEqual(result['script_sha256'], runner.source_provenance(root, 'config.yaml')['script_sha256'])

    def test_missing_executable_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_command(['this-executable-does-not-exist-soup-test'], Path(folder), 'missing')
            self.assertEqual(result['returncode'], 127)
            self.assertIn('launch_error', result)
            self.assertTrue((Path(folder) / 'missing.raw.log').read_bytes())

    def test_failed_command_keeps_raw_output_and_exit_status(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_command([sys.executable, '-c', "import sys; sys.stdout.buffer.write(b'failure evidence\\n'); raise SystemExit(7)"], Path(folder), 'failed')
            self.assertEqual(result['returncode'], 7)
            self.assertEqual((Path(folder) / 'failed.raw.log').read_bytes(), b'failure evidence\n')
            saved = json.loads((Path(folder) / 'failed.command.json').read_text())
            self.assertEqual(saved['returncode'], 7)
            self.assertIn('Z', saved['finished_utc'])
            self.assertIn('failure evidence', (Path(folder) / 'failed.timestamped.log').read_text())

    def test_dpo_logits_budget_doubles_pair_rows(self):
        values = dict(hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
                      vocab_size=151936, num_attention_heads=14, num_key_value_heads=2,
                      tie_word_embeddings=True)
        result = estimate(values, batch=1, sequence=512, rank=8)
        self.assertEqual(result['logit_elements'], 2 * 512 * 151936)
        self.assertEqual(result['lora_parameters'], 540672)
        self.assertGreater(result['conservative_total_gib'], 2)
        self.assertLess(result['conservative_total_gib'], 6)


if __name__ == '__main__':
    unittest.main()

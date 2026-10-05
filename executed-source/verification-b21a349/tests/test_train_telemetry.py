"""Telemetry regressions; these fixtures do not execute CUDA training."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_dpo import dumps, merge_gradient_stats, write


class TrainingTelemetryTests(unittest.TestCase):
    def test_nonfinite_training_history_remains_strict_valid_json(self):
        history = [{'loss': 0.69, 'grad_norm': float('inf')},
                   {'loss': float('nan'), 'nested': [float('-inf'), 2.0]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.json'
            write(path, history)
            result = json.loads(path.read_text(), parse_constant=lambda value: self.fail(value))
        self.assertEqual(result[0]['grad_norm'], 'Infinity')
        self.assertEqual(result[1]['loss'], 'NaN')
        self.assertEqual(result[1]['nested'], ['-Infinity', 2.0])
        self.assertEqual(result[0]['loss'], 0.69)

    def test_logger_preserves_nonfinite_in_nested_metrics(self):
        result = json.loads(dumps({'step': 33, 'metrics': {'grad_norm': float('inf')}}),
                            parse_constant=lambda value: self.fail(value))
        self.assertEqual(result, {'step': 33, 'metrics': {'grad_norm': 'Infinity'}})

    def test_successful_gradient_aggregation_retains_nonfinite_failure(self):
        gradients = {}
        merge_gradient_stats(gradients, {'layer.0': {'finite': True, 'max_abs': 0.2}})
        merge_gradient_stats(gradients, {'layer.0': {'finite': False, 'max_abs': 0.1}})
        self.assertFalse(gradients['layer.0']['finite'])
        self.assertEqual(gradients['layer.0']['calls'], 2)
        self.assertEqual(gradients['layer.0']['max_abs'], 0.2)


if __name__ == '__main__':
    unittest.main()

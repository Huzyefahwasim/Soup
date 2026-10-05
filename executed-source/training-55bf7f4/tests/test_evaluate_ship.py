"""Check score production with real Soup scorers and deterministic text fixtures."""
import importlib
import importlib.util
import unittest


class ShipProducerTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.evaluate_ship'),
                             'The fp16 ship score producer does not exist yet')
        return importlib.import_module('scripts.evaluate_ship')

    def test_task_outputs_and_correct_score_are_preserved(self):
        from soup_cli.eval.custom import EvalTask
        producer = self.module()
        tasks = [EvalTask(prompt='one', expected='1'), EvalTask(prompt='two', expected='2')]
        report = producer.measure_side(lambda prompt: '1', tasks, [], model_id='fixture')
        self.assertEqual(report['task_accuracy'], 0.5)
        self.assertEqual([row['output'] for row in report['task_results']], ['1', '1'])
        self.assertEqual([row['matched'] for row in report['task_results']], [True, False])
        self.assertEqual(report['generation_errors'], 0)

    def test_swallowed_bundled_generator_error_cannot_become_a_score(self):
        producer = self.module()
        def broken(_prompt):
            raise RuntimeError('fixture inference failed')
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            producer.measure_side(broken, [], ['mini_format_json'], model_id='fixture')

    def test_evidence_uses_each_side_without_fabricating_a_ship_verdict(self):
        producer = self.module()
        report = producer.build_evidence(
            {'task_accuracy': 0.25, 'benchmarks': {'fixture': 0.75}},
            {'task_accuracy': 0.5, 'benchmarks': {'fixture': 0.5}}, {'dtype': 'float16'})
        self.assertEqual(report['task'], {'mode': 'metric', 'base': 0.25, 'tuned': 0.5})
        self.assertEqual(report['benchmarks'], {'fixture': {'base': 0.75, 'tuned': 0.5}})
        self.assertNotIn('decision', report)


if __name__ == '__main__':
    unittest.main()

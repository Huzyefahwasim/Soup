"""The shipping gate must reject plausible but inert training artifacts."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class VerificationTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "scripts" / "verify_adapter.py"
        self.assertTrue(path.exists(), "adapter verification has not been implemented")
        spec = importlib.util.spec_from_file_location("verify_adapter", path)
        self.verify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.verify)
        self.before = {"x.lora_A.weight": np.ones((2, 2)),
                       "x.lora_B.weight": np.zeros((2, 2))}
        self.after = {"x.lora_A.weight": np.ones((2, 2)),
                      "x.lora_B.weight": np.full((2, 2), 0.01)}
        identity = {"model": "fixture", "revision": "abc", "data_sha256": "def",
                    "dtype": "float32", "max_length": 512, "max_prompt_length": 256,
                    "truncation_mode": "keep_start"}
        self.base = {"identity": identity, "adapter": None, "samples": [
            {"id": str(i), "chosen": -4.0, "rejected": -5.0} for i in range(50)]}
        self.trained = {"identity": dict(identity), "adapter": "trained", "reload_exact": True,
                        "effect": {"probe_delta": 0.1, "repeat_noise": 0.0}, "samples": [
            {"id": str(i), "chosen": -3.0, "rejected": -5.0,
             "base_chosen": -4.0, "base_rejected": -5.0} for i in range(50)]}

    def result(self):
        return self.verify.compare_evidence(self.before, self.after, self.base, self.trained)

    def test_active_reloaded_adapter_with_heldout_gain_passes(self):
        self.assertEqual(self.result()["verdict"], "SHIP")

    def test_same_checkpoint_negative_control_fails(self):
        self.after = self.before
        self.assertEqual(self.result()["verdict"], "DON'T SHIP")

    def test_changed_weights_without_probability_effect_fail(self):
        self.trained["effect"]["probe_delta"] = 0.0
        self.assertFalse(self.result()["checks"]["active_adapter"])

    def test_nonfinite_saved_tensor_fails(self):
        self.after["x.lora_B.weight"][0, 0] = np.nan
        self.assertFalse(self.result()["checks"]["finite_tensors"])

    def test_partial_or_silently_dropped_reload_fails(self):
        self.trained["reload_exact"] = False
        self.assertFalse(self.result()["checks"]["portable_reload"])

    def test_off_adapter_must_reproduce_independent_base(self):
        self.trained["samples"][0]["base_chosen"] = -2.0
        self.assertFalse(self.result()["checks"]["base_control"])

    def test_improvement_cannot_be_manufactured_from_different_rows(self):
        self.trained["samples"][0]["id"] = "different"
        self.assertFalse(self.result()["checks"]["matched_data"])

    def test_missing_provenance_cannot_pass_as_matching_data(self):
        self.base["identity"] = self.trained["identity"] = {}
        self.assertFalse(self.result()["checks"]["matched_data"])

    def test_heldout_regression_fails_despite_active_adapter(self):
        for row in self.trained["samples"]:
            row["chosen"] = -6.0
        self.assertFalse(self.result()["checks"]["heldout_improvement"])

    def test_snapshot_archive_round_trip_avoids_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            self.verify.write_snapshot(path, self.after, {"r": 8})
            tensors, metadata = self.verify.read_snapshot(path)
            self.assertEqual(metadata, {"r": 8})
            np.testing.assert_array_equal(tensors["x.lora_B.weight"], np.full((2, 2), 0.01))

    def test_prompt_cap_matches_trl_keep_start_without_scoring_prompt(self):
        class TokenizedChat:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, return_dict=True):
                ids = [11, 12, 13, 14, 15] if add_generation_prompt else [11, 12, 13, 14, 15, 20, 21]
                return {"input_ids": ids} if return_dict else ids
        ids, boundary = self.verify.response_tokens(TokenizedChat(), "prompt", "answer", 3, 5)
        self.assertEqual(ids, [11, 12, 13, 20, 21])
        self.assertEqual(boundary, 3)

    def test_explicit_return_types_match_trl_despite_tokenizer_dictionary_default(self):
        class DefaultDictionaryChat:
            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, return_dict=True):
                self.calls.append((add_generation_prompt, return_dict))
                ids = [11, 12, 13] if add_generation_prompt else [11, 12, 13, 20, 21]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)} if return_dict else ids

        tokenizer = DefaultDictionaryChat()
        ids, boundary = self.verify.response_tokens(tokenizer, "prompt", "answer", 256, 512)
        self.assertEqual((ids, boundary), ([11, 12, 13, 20, 21], 3))
        self.assertEqual(tokenizer.calls, [(True, False), (False, True)])

    def test_single_batch_token_lists_are_unwrapped_like_trl(self):
        class SingleBatchChat:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, return_dict=True):
                ids = [[11, 12, 13]] if add_generation_prompt else [[11, 12, 13, 20, 21]]
                return {"input_ids": ids} if return_dict else ids

        ids, boundary = self.verify.response_tokens(SingleBatchChat(), "prompt", "answer", 256, 512)
        self.assertEqual((ids, boundary), ([11, 12, 13, 20, 21], 3))

    def test_real_token_boundary_merge_is_still_rejected(self):
        # Synthetic token IDs represent a prompt-final token merged with the response.
        # They are a contract fixture, not measured Qwen tokenizer output.
        class BoundaryMergeChat:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, return_dict=True):
                ids = [11, 12, 13] if add_generation_prompt else [11, 12, 99, 21]
                return {"input_ids": ids} if return_dict else ids

        with self.assertRaisesRegex(ValueError, "prompt prefix"):
            self.verify.response_tokens(BoundaryMergeChat(), "prompt", "answer", 256, 512)

    def test_streaming_gate_catches_wrong_backward_despite_equal_forward(self):
        path = ROOT / "scripts" / "check_streaming.py"
        self.assertTrue(path.exists(), "streaming parity check has not been implemented")
        spec = importlib.util.spec_from_file_location("check_streaming", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.parity_report(np.array([1.0]), np.array([1.0]),
                                     {"layer0": np.array([0.1])},
                                     {"layer0": np.array([0.0])}, "float32")
        self.assertTrue(result["forward_pass"])
        self.assertFalse(result["backward_pass"])

    def test_disabled_reference_control_rejects_policy_used_as_reference(self):
        path = ROOT / "scripts" / "check_streaming.py"
        spec = importlib.util.spec_from_file_location("check_streaming", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.reference_report(np.array([1.0]), np.array([1.0]),
                                         np.array([0.0]), "float32")
        self.assertFalse(result["reference_pass"])

    def streaming(self):
        path = ROOT / "scripts" / "check_streaming.py"
        spec = importlib.util.spec_from_file_location("check_streaming", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_dpo_loss_discrepancy_cannot_pass_despite_equal_gradients(self):
        result = self.streaming().loss_report(0.7, 0.6, "float16")
        self.assertFalse(result["loss_pass"])

    def test_identical_finite_loss_passes(self):
        self.assertTrue(self.streaming().loss_report(0.693, 0.693, "float16")["loss_pass"])

    def test_one_wrong_poststep_tensor_cannot_hide_in_large_matching_state(self):
        left = {"lora_A": np.ones(10000), "lora_B": np.array([0.1])}
        right = {"lora_A": np.ones(10000), "lora_B": np.array([0.2])}
        self.assertFalse(self.streaming().optimizer_report(left, right, "float16")["optimizer_state_pass"])

    def test_matching_poststep_states_pass_and_report_exactness(self):
        states = {"lora_A": np.array([1.0]), "lora_B": np.array([0.01])}
        result = self.streaming().optimizer_report(states, states, "float16")
        self.assertTrue(result["optimizer_state_pass"])
        self.assertTrue(result["optimizer_state_bit_exact"])

    def test_missing_poststep_tensor_fails(self):
        self.assertFalse(self.streaming().optimizer_report(
            {"lora_A": np.array([1.0])}, {"lora_A": np.array([1.0]), "lora_B": np.array([0.1])},
            "float16")["optimizer_state_pass"])

    def passing_probe(self):
        return dict(forward_pass=True, backward_pass=True, loss_pass=True,
                    reference_pass=True, all_expected_gradients=True, base_frozen=True,
                    decoder_base_on_meta=True, optimizer_changed_parameters=True,
                    negative_control_rejected=True, updated_parameters_finite=True,
                    resident_optimizer_changed_parameters=True, optimizer_state_pass=True,
                    optimizer_state_expected=True)

    def test_loss_gate_vetoes_overall_probe_even_when_other_checks_pass(self):
        result = self.passing_probe()
        result["loss_pass"] = False
        self.assertFalse(self.streaming().probe_pass(result))

    def test_poststep_gate_vetoes_overall_probe_even_when_other_checks_pass(self):
        result = self.passing_probe()
        result["optimizer_state_pass"] = False
        self.assertFalse(self.streaming().probe_pass(result))


if __name__ == "__main__":
    unittest.main()

"""Offline checks for preference-data filtering, splitting, and provenance."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"


def load_preparer():
    if not SCRIPT.exists():
        raise AssertionError("Dataset preparation implementation is missing")
    spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_rows(count, prefix="train"):
    for index in range(count):
        yield {
            "prompt": f"Человек: Объясните пример номер {index} из раздела {prefix}.",
            "chosen": "Помощник: Проверим условия и объясним нужные действия.",
            "rejected": "Ассистент: Сделайте всё сразу, дополнительные условия не нужны.",
        }


class DataPreparationTests(unittest.TestCase):
    def test_rejects_invalid_pairs_and_removes_completion_role_markers(self):
        prep = load_preparer()
        rows = list(source_rows(9))
        rows[0]["chosen"] = "  "
        rows[1]["rejected"] = rows[1]["chosen"]
        rows[2]["prompt"] = "Человек: Первый вопрос\n\nПомощник: Ответ\n\nЧеловек: Второй вопрос"
        rows[3]["chosen"] = "а" * 1025
        rows[4]["rejected"] = 42
        rows[5]["chosen"] = "Assistant: English language answer only."
        result, diagnostics = prep.select_pairs(rows, sizes=(1, 1, 1))
        self.assertEqual(sum(map(len, result.values())), 3)
        self.assertEqual(diagnostics["train"]["accepted_unique"], 3)
        for pairs in result.values():
            for entry in pairs:
                self.assertFalse(entry["pair"]["chosen"].startswith("Помощник:"))
                self.assertFalse(entry["pair"]["rejected"].startswith("Ассистент:"))

    def test_normalized_duplicates_cannot_cross_splits(self):
        prep = load_preparer()
        train = list(source_rows(12))
        train.append({**train[0], "prompt": " ЧЕЛОВЕК:  ОБЪЯСНИТЕ  ПРИМЕР НОМЕР 0 ИЗ РАЗДЕЛА TRAIN. "})
        result, diagnostics = prep.select_pairs(train, sizes=(5, 2, 3))
        self.assertEqual([len(result[key]) for key in ("train", "validation", "test")], [5, 2, 3])
        all_prompts = [entry["pair"]["prompt"].casefold() for split in result.values() for entry in split]
        self.assertEqual(len(all_prompts), len(set(all_prompts)))
        self.assertEqual(diagnostics["train"]["rejected"]["duplicate_prompt"], 1)
        split_hashes = [{entry["prompt_sha256"] for entry in result[key]} for key in result]
        self.assertTrue(all(a.isdisjoint(b) for i, a in enumerate(split_hashes) for b in split_hashes[i + 1:]))

    def test_rebuild_is_deterministic_and_serializes_only_training_fields(self):
        prep = load_preparer()
        result, stats = prep.select_pairs(source_rows(30), sizes=(8, 2, 3), seed=42)
        repeat, _ = prep.select_pairs(source_rows(30), sizes=(8, 2, 3), seed=42)
        self.assertEqual(result, repeat)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = prep.write_outputs(output, result, stats, seed=42)
            row = json.loads((output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(set(row), {"prompt", "chosen", "rejected"})
            self.assertEqual(manifest["split_counts"], {"train": 8, "validation": 2, "test": 3})
            self.assertEqual(len(manifest["records"]), 13)
            self.assertIsNone(manifest["records"][0]["chosen_score"])
            self.assertTrue(manifest["files"]["train.jsonl"]["sha256"])

    def test_insufficient_unique_data_fails_instead_of_repeating_pairs(self):
        prep = load_preparer()
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            prep.select_pairs(source_rows(2), sizes=(2, 1, 1))

    def test_rejects_obvious_personal_contacts_and_dangerous_request_topics(self):
        prep = load_preparer()
        rows = list(source_rows(6))
        rows[0]["prompt"] = "Человек: Напишите клиенту по адресу ivan.petrov@example.org."
        rows[1]["prompt"] = "Человек: Позвоните на номер +7 999 123 45 67 для обсуждения."
        rows[2]["prompt"] = "Человек: Как изготовить бомбу в домашних условиях?"
        result, diagnostics = prep.select_pairs(rows, sizes=(1, 1, 1))
        self.assertEqual(diagnostics["train"]["accepted_unique"], 3)
        self.assertEqual(diagnostics["train"]["rejected"]["contact_or_secret"], 2)
        self.assertEqual(diagnostics["train"]["rejected"]["sensitive_topic"], 1)
        self.assertEqual(len(result["test"]), 1)

    def test_project_holdouts_come_from_the_actual_russian_train_release(self):
        prep = load_preparer()
        splits, _ = prep.select_pairs(source_rows(12), sizes=(4, 2, 3))
        self.assertEqual({entry["source_split"] for split in splits.values() for entry in split}, {"train"})
        source_indices = [entry["source_index"] for split in splits.values() for entry in split]
        self.assertEqual(len(source_indices), len(set(source_indices)))


if __name__ == "__main__":
    unittest.main()

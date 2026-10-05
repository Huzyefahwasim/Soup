"""Prepare a small, reproducible Russian preference dataset from pinned public data.

This does not synthesize labels. It keeps the source chosen/rejected ordering and
records the translation and annotation limitations in data/manifest.json.
Only downloading/reading the source Parquet files needs optional pyarrow.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import re
import unicodedata
import urllib.request


SOURCE_ID = "d0rj/rlhf-reward-datasets-ru"
SOURCE_REVISION = "ec1b66da5885fb4af7cc90096036c3cd4ee0dab5"
SOURCE_FILES = {
    "train": {
        "path": "data/train-00000-of-00001-c259716327f16b71.parquet",
        "sha256": "c5d8b4f28b4875f73a8c4c8a5b08b22704caa15404677bb38ef1c2ed61821352",
        "rows": 76256,
    },
    "test": {
        "path": "data/test-00000-of-00001-1b75995d104c69e6.parquet",
        "sha256": "220ad4dd16fb7a23d817ccfd275a80c2974c3dd58f8a5f387497abc8cb207aae",
        "rows": 5103,
    },
}
ROLE_PREFIX = re.compile(r"\A(?:Помощник|Ассистент|Assistant)\s*:\s*", re.IGNORECASE)
USER_PREFIX = re.compile(r"\A(?:Человек|Human|Пользователь)\s*:\s*", re.IGNORECASE)
HISTORY_ROLE = re.compile(r"(?:^|\n)\s*(?:Человек|Помощник|Ассистент|Human|Assistant|Пользователь)\s*:", re.IGNORECASE)
CONTACT_OR_SECRET = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)"
    r"|\bAKIA[A-Z0-9]{16}\b|\bsk-[A-Za-z0-9_-]{20,}"
    r"|(?:password|api[_ -]?key|access[_ -]?token)\s*[:=]\s*\S+",
    re.IGNORECASE,
)
SENSITIVE_TOPIC = re.compile(
    r"уби(?:ть|йств)|самоуб|бомб|взрывчат|взлом|порн|расист|изнасил",
    re.IGNORECASE,
)
MAX_TEXT_CHARS = 1024
MAX_PAIR_CHARS = 1024


def normalized_prompt(text):
    """Match unicode/case/whitespace variants, including Russian ё/е."""
    value = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return " ".join(value.split())


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def russian_fraction(text):
    letters = [character for character in text if character.isalpha()]
    return sum("а" <= character.lower() <= "я" or character.lower() == "ё" for character in letters) / max(len(letters), 1)


def clean_pair(row):
    if not all(isinstance(row.get(key), str) for key in ("prompt", "chosen", "rejected")):
        return None, "missing_or_non_string"
    pair = {key: row[key].strip() for key in ("prompt", "chosen", "rejected")}
    pair["prompt"] = USER_PREFIX.sub("", pair["prompt"], count=1).strip()
    for key in ("chosen", "rejected"):
        pair[key] = ROLE_PREFIX.sub("", pair[key], count=1).strip()
    if not all(pair.values()):
        return None, "empty"
    if normalized_prompt(pair["chosen"]) == normalized_prompt(pair["rejected"]):
        return None, "equal_responses"
    if HISTORY_ROLE.search(pair["prompt"]):
        return None, "multi_turn_transcript"
    if any(len(value) > MAX_TEXT_CHARS for value in pair.values()) or max(
        len(pair["prompt"]) + len(pair["chosen"]),
        len(pair["prompt"]) + len(pair["rejected"]),
    ) > MAX_PAIR_CHARS:
        return None, "too_long"
    if any(russian_fraction(value) < 0.5 for value in pair.values()):
        return None, "non_russian"
    if CONTACT_OR_SECRET.search("\n".join(pair.values())):
        return None, "contact_or_secret"
    if SENSITIVE_TOPIC.search("\n".join(pair.values())):
        return None, "sensitive_topic"
    return pair, None


def select_candidates(rows, count, source_split, seed, excluded_prompts=()):
    """Choose the lowest seeded hash ranks over a bounded, row-wise pass."""
    seen = set(excluded_prompts)
    candidates = []
    rejected = Counter()
    diagnostics = {"scanned": 0, "accepted_unique": 0}
    for source_index, row in enumerate(rows):
        diagnostics["scanned"] += 1
        pair, reason = clean_pair(row)
        if reason:
            rejected[reason] += 1
            continue
        normalized = normalized_prompt(pair["prompt"])
        if normalized in seen:
            rejected["duplicate_prompt"] += 1
            continue
        seen.add(normalized)
        diagnostics["accepted_unique"] += 1
        rank = int(sha256_text(f"{seed}:{source_split}:{source_index}"), 16)
        entry = {
            "pair": pair,
            "source_split": source_split,
            "source_index": source_index,
            "prompt_sha256": sha256_text(normalized),
            "pair_sha256": sha256_text(json.dumps(pair, ensure_ascii=False, sort_keys=True)),
            "chosen_score": None,
            "rejected_score": None,
        }
        candidate = (-rank, -source_index, entry)
        if len(candidates) < count:
            heapq.heappush(candidates, candidate)
        elif candidate[:2] > candidates[0][:2]:
            heapq.heapreplace(candidates, candidate)
    diagnostics["rejected"] = dict(sorted(rejected.items()))
    if len(candidates) != count:
        raise ValueError(f"Insufficient eligible {source_split} pairs: need {count}, found {len(candidates)}")
    selected = [item[2] for item in sorted(candidates, key=lambda item: (-item[0], -item[1]))]
    return selected, diagnostics


def select_pairs(train_rows, sizes=(400, 50, 50), seed=42):
    train_count, validation_count, test_count = sizes
    if any(size <= 0 for size in sizes):
        raise ValueError("Split sizes must be positive")
    training_pool, train_stats = select_candidates(train_rows, sum(sizes), "train", seed)
    return {
        "train": training_pool[:train_count],
        "validation": training_pool[train_count:train_count + validation_count],
        "test": training_pool[train_count + validation_count:],
    }, {"train": train_stats}


def write_outputs(output, splits, diagnostics, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    files = {}
    for split_name, entries in splits.items():
        filename = f"{split_name}.jsonl"
        path = output / filename
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for output_index, entry in enumerate(entries):
                handle.write(json.dumps(entry["pair"], ensure_ascii=False) + "\n")
                records.append({"output_file": filename, "output_index": output_index, **{key: value for key, value in entry.items() if key != "pair"}})
        files[filename] = {"sha256": file_sha256(path), "rows": len(entries)}
    doctor_path = output / "doctor_chat.jsonl"
    with doctor_path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in splits["train"]:
            pair = entry["pair"]
            handle.write(json.dumps({"messages": [{"role": "user", "content": pair["prompt"]}, {"role": "assistant", "content": pair["chosen"]}]}, ensure_ascii=False) + "\n")
    files[doctor_path.name] = {"sha256": file_sha256(doctor_path), "rows": len(splits["train"]), "purpose": "Chosen-only ChatML projection for soup doctor; not DPO training data"}
    prompt_hashes = [entry["prompt_sha256"] for entries in splits.values() for entry in entries]
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise ValueError("Normalized prompt overlap detected")
    manifest = {
        "schema_version": 1,
        "source": {
            "dataset_id": SOURCE_ID,
            "revision": SOURCE_REVISION,
            "url": f"https://huggingface.co/datasets/{SOURCE_ID}/tree/{SOURCE_REVISION}",
            "declared_license": "MIT",
            "declared_upstream": "yitingxie/rlhf-reward-datasets",
            "language_origin": "Russian translation; translator/model undocumented in source card",
            "source_files": SOURCE_FILES,
            "scores_available": False,
            "used_source_splits": ["train"],
            "original_test_status": "Excluded: observed English text despite Russian monolingual dataset-card metadata",
            "annotation_provenance": "Source chosen/rejected labels preserved; no independent re-annotation or quality audit",
        },
        "seed": seed,
        "selection": "SHA256 rank of seed:source_split:source_index over every eligible Russian source train row; deduplicate normalized prompt before ranking; lowest 500 ranks partitioned in order into 400 train, 50 validation and 50 project test; original English source test excluded",
        "filters": {
            "single_turn_only": True,
            "max_text_chars": MAX_TEXT_CHARS,
            "max_prompt_plus_completion_chars": MAX_PAIR_CHARS,
            "min_cyrillic_fraction_of_letters": 0.5,
            "normalized_prompt": "NFKC + casefold + ё to е + collapse whitespace",
            "contacts_secrets": "Conservative lexical rejection of literal emails, long phone/number sequences, key shapes, explicit password/token assignments",
            "sensitive_topics": "Conservative lexical rejection of homicide, self-harm, explosives, hacking, pornography, racism and sexual violence terms; may exclude benign discussion",
        },
        "modifications": "Strip one leading user/assistant role marker; trim boundary whitespace; no response rewriting, label generation, artificial typos or scores",
        "split_counts": {name: len(entries) for name, entries in splits.items()},
        "normalized_prompt_overlap": 0,
        "filter_diagnostics": diagnostics,
        "files": files,
        "limitations": [
            "Translated general assistant preferences, not real support tickets; no organization policies or support outcomes.",
            "Sparse upstream card; MIT is uploader declaration and the translation method is unspecified.",
            "Original source test contains English examples; all project splits are held-out partitions of the Russian source train, not the official source test.",
            "Some labels/responses can be poor; contacts/topic regexes are conservative screening, not a complete content audit.",
            "No human feedback scores exist in this source; null scores explicitly mean unavailable.",
            "Exact normalized prompt deduplication does not establish semantic novelty or remove all near duplicates.",
            "This small public demonstration cannot support a production SHIP decision.",
        ],
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def download_source(split, cache):
    info = SOURCE_FILES[split]
    destination = Path(cache) / SOURCE_REVISION / Path(info["path"]).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and file_sha256(destination) == info["sha256"]:
        return destination
    url = f"https://huggingface.co/datasets/{SOURCE_ID}/resolve/{SOURCE_REVISION}/{info['path']}"
    request = urllib.request.Request(url, headers={"User-Agent": "russian-preference-takehome/1.0"})
    partial = destination.with_suffix(".part")
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if file_sha256(partial) != info["sha256"]:
        raise ValueError(f"Source checksum mismatch for {split}; refusing to prepare unverified data")
    partial.replace(destination)
    return destination


def parquet_rows(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("Source preparation needs pyarrow; install the project's requirements first") from error
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["prompt", "chosen", "rejected"]):
        yield from batch.to_pylist()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "data")
    parser.add_argument("--cache", type=Path, default=root / ".cache" / "preference-source")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_path = download_source("train", args.cache)
    splits, diagnostics = select_pairs(parquet_rows(train_path), seed=args.seed)
    manifest = write_outputs(args.output, splits, diagnostics, seed=args.seed)
    print(json.dumps({"source": SOURCE_ID, "revision": SOURCE_REVISION, "splits": manifest["split_counts"], "normalized_prompt_overlap": 0, "filter_diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()

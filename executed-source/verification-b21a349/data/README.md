# Russian preference-pair demonstration

These files contain **500 public Russian preference pairs**: 400 training, 50
validation, and 50 test. The data is translated general assistant dialogue. It is
**not a support-ticket dataset**, has no company policy context, and cannot by
itself justify deploying a support model.

## Source and attribution

Dataset author/publisher: **d0rj**. Dataset:
[d0rj/rlhf-reward-datasets-ru](https://huggingface.co/datasets/d0rj/rlhf-reward-datasets-ru/tree/ec1b66da5885fb4af7cc90096036c3cd4ee0dab5),
frozen at revision `ec1b66da5885fb4af7cc90096036c3cd4ee0dab5`.
Its dataset card declares **MIT** and identifies
[yitingxie/rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets)
as the translated source. Preserve this attribution and the source license
declaration when redistributing these adapted rows.

The Russian card does not identify the translator, translation model, translation
quality checks, or annotators. The named upstream card also provides little
provenance. MIT here is the publisher's explicit dataset-card declaration; the
repository supplies no separate copyright notice or license file. We do not
claim to have independently audited the licensing chain, labels, or translation.

The source release has 76,256 `train` and 5,103 `test` rows, stored as
Parquet, with three string fields: `prompt`, `chosen`, `rejected`. Original
preference order is preserved. No preference scores are available; the manifest
records scores as `null`, rather than fabricating them. The data is downloaded at
the frozen revision and checked against its public LFS SHA-256 values. The
downloaded official `test` file contains English examples despite the Russian
monolingual card metadata. It is excluded: all three project splits come from
the actual Russian `train` file. Our 50-row test is a project holdout, not the
official upstream test split.

## Rebuild

After installing the project requirements, run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\prepare_data.py
```

On Linux/Colab the equivalent is:

```sh
python scripts/prepare_data.py
```

Source files are cached in `.cache/preference-source/` and are not part of the
deliverable. The preparation scans Parquet in batches of 256 rows, without
loading the full source into a pandas frame. Each eligible row gets a deterministic
SHA-256 rank derived from seed `42`, original source split, and source row index.
The lowest 500 ranks from original source `train` are partitioned in order into
400 training, 50 validation, and 50 held-out test rows. Normalized prompts are
unique before partitioning, so they cannot overlap across project splits.

`train.jsonl`, `validation.jsonl`, and `test.jsonl` each contain only:

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

`manifest.json` contains immutable source revision/file hashes, selection rules,
filter counts, source row indices, absent scores, normalized prompt hashes, and
output checksums. `doctor_chat.jsonl` is a chosen-only ChatML projection of the
400 training rows for the Soup doctor compatibility check. It is not DPO data.

## Filters and limitations

Preparation rejects missing/non-string/empty values, equal responses, serialized
multi-turn transcripts, and exact normalized prompt duplicates. Normalization
uses NFKC, case folding, `ё` → `е`, and collapsed whitespace. Train, validation,
and test have zero overlap under that rule. This does not prove semantic novelty
or remove every paraphrase.

Only single-turn Russian pairs are kept. One leading `Человек:` user marker and
one leading `Помощник:`/`Ассистент:` response marker are removed; text is otherwise
preserved. Each field is limited to 1,024 characters, and prompt plus either
response is limited to 1,024 characters. Character limits are a screening bound,
not a tokenizer-length guarantee; the training/evaluation runner must inspect
lengths with the model tokenizer.

Conservative regular expressions reject literal emails, long phone/number
sequences, obvious credential shapes, and terms associated with violence,
self-harm, explosives, hacking, pornography, racism, or sexual violence. They may
remove benign educational examples, and are not a complete PII or safety audit.
Per-reason counts are recorded in the manifest. No artificial typos, fabricated
support context, synthetic rejected answers, or external teacher labels are added.

Some source preferences may reward poor answers or contain factual errors.
Public translated dialogue also lacks the slang, truncation, context omissions,
policy decisions, escalation outcomes, and authentic operator feedback needed
for support deployment. A successful run demonstrates reproducible preparation
and training mechanics; the production verdict remains **DON'T SHIP** until
tested on authorized, representative support data with a reviewed preference
rubric.

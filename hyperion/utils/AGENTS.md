# AGENTS.md

This directory contains foundational metadata, trial, score, list, math, plotting, and Kaldi-style utility classes used across Hyperion. Changes here often affect data prep, IO, recipes, NumPy models, PyTorch datasets, and evaluation scripts.

Use this with the repository-level `AGENTS.md`. The most relevant docs are:

- `docs/info_tables.rst`
- `docs/hyper_dataset.rst`
- `docs/trials.rst`
- `docs/utils.rst`

## Core Responsibilities

Important utility families:

- `InfoTable` and child metadata tables: `SegmentSet`, `RecordingSet`, `ClassInfo`, `FeatureSet`, `VADSet`, `ImageSet`, `VideoSet`, `DiarizationSet`.
- Dataset container: `HyperDataset`.
- Verification metadata: `TrialKey`, `TrialNdx`, `SparseTrialKey`, `SparseTrialNdx`, `TrialScores`, `SparseTrialScores`, `EnrollmentMap`.
- Kaldi-style lists and matrix helpers: `SCPList`, `Utt2Info`, `kaldi_matrix`, `kaldi_io_funcs`.
- List/math/text/time helpers used throughout the repo.

Be conservative: this directory defines shared contracts.

## `InfoTable` Contract

`InfoTable` wraps a pandas `DataFrame` and standardizes table behavior.

Core rules:

- every table has an `id` column,
- `id` is also the pandas index,
- subclasses should preserve `InfoTable` behavior for `load`, `save`, `filter`, `split`, `cat`, indexing, and copying,
- DataFrame slices that still contain `id` should remain valid table objects where possible.

When adding a new table type:

- derive from `InfoTable`,
- validate required columns early,
- preserve `id` dtype behavior,
- implement file-format-specific load/save only if the generic table behavior is insufficient,
- document alignment semantics with other tables.

## `HyperDataset` Contract

`HyperDataset` ties `SegmentSet` to optional related manifests.

Alignment rules:

- `segments` is the required anchor.
- `recordings` aligns through `segments.recording`; if no `recording` column exists, segment id is often used as the recording id.
- `features`, `vads`, and `diarizations` align directly by segment id.
- `classes` are keyed by segment column name, such as `speaker`, `gender`, or `language`.
- `enrollments` and `trials` align by model id and segment id.

Preserve lazy loading and path/object duality: most `HyperDataset` arguments can be an already loaded object or a path.

When mutating datasets, make sure `clean(...)` semantics continue to remove orphaned auxiliary rows.

## Trial and Score Objects

Verification classes preserve row/column alignment:

- `TrialKey`: target/non-target labels for `(model, segment)` pairs.
- `TrialNdx`: trial mask without labels.
- `TrialScores`: score matrix plus score mask.
- sparse variants store equivalent information more compactly.
- `EnrollmentMap`: model-to-enrollment-segment mapping.

Do not treat these as plain arrays. Preserve:

- `model_set` order,
- `seg_set` order,
- mask semantics,
- target/non-target exclusivity,
- alignment behavior with keys, ndx, and scores,
- file compatibility with existing HDF5, TXT, CSV, TSV, and NIST-style conventions.

## File Format Compatibility

Many utilities support multiple formats:

- CSV/TSV tables,
- HDF5 files,
- Kaldi `.scp` and `.ark`-style manifests,
- NIST-style enrollment/trial files,
- YAML dataset manifests.

When changing load/save behavior:

- keep old files readable where practical,
- infer separators/extensions consistently with existing code,
- create parent directories before writing,
- avoid changing default column names on disk,
- preserve special compatibility behavior such as `EnrollmentMap` writing `modelid`.

## Metadata and ID Alignment

Most bugs in this layer are id alignment bugs.

Before changing code, identify which ids are being aligned:

- segment id,
- recording id,
- class id,
- model id,
- enrollment segment id,
- feature/VAD/diarization id.

For filters, splits, merges, and concatenation, verify that associated masks, matrices, and auxiliary fields are transformed in the same order as the ids.

## Style Rules

- Prefer pandas operations for table transformations.
- Prefer NumPy operations for matrix/mask transformations.
- Use `PathLike` from `hyperion.utils.misc` for path-accepting APIs.
- Use `Path` internally for filesystem operations.
- Use `logging` for warnings/progress.
- Raise `ValueError` or `TypeError` for malformed user data when assertions would hide a useful error message.
- Keep docstrings explicit about required columns, alignment rules, and return types.

## Tests

Relevant tests:

- `tests/hyperion/utils`
- `tests/hyperion/metrics`
- `tests/hyperion/io` for file compatibility touching Kaldi/HDF5 readers or writers.

Use targeted tests for changed utility classes. For id alignment changes, add or run tests that exercise sorting, filtering, splitting, merging, save/load, and equality.

Useful examples:

```bash
pytest tests/hyperion/utils/test_trial_scores.py
pytest tests/hyperion/utils/test_trial_key.py
pytest tests/hyperion/utils/test_sparse_trial_scores.py
pytest tests/hyperion/utils/test_scp_list.py
```

## Common Pitfalls

- Losing the `id` column while preserving the DataFrame index.
- Reordering ids without reordering masks, score matrices, or quality measures.
- Breaking lazy loading in `HyperDataset`.
- Saving a table with a new column name that existing recipes do not read.
- Filtering segments without cleaning class, feature, VAD, enrollment, and trial tables.
- Treating missing labels, empty strings, and `NaN` interchangeably without checking downstream behavior.

# AGENTS.md

This directory contains dataset preparation classes. Their job is to convert dataset-specific metadata and media layouts into the Hyperion metadata format used by `HyperDataset`, `InfoTable` subclasses, CLIs, recipes, and training/evaluation code.

Use this file together with the repository-level `AGENTS.md`. The most relevant docs are:

- `docs/hyper_dataset.rst`
- `docs/info_tables.rst`
- `docs/data_prep.rst`

## Purpose

Each data prep class should:

1. Parse one external speech dataset layout.
2. Build Hyperion metadata tables such as `RecordingSet`, `SegmentSet`, `ClassInfo`, `EnrollmentMap`, `TrialKey`, or `TrialNdx`.
3. Save a `HyperDataset` bundle under `output_dir`.
4. Register itself as a `hyperion-prepare-data` subcommand through `DataPrep.registry`.

The typical final call is:

```python
dataset = HyperDataset(
    segments=segments,
    recordings=recordings,
    classes=classes,
    enrollments=enrollments,
    trials=trials,
    sparse_trials=False,
)
dataset.save(self.output_dir)
```

## Hyperion Metadata Directory Format

A prepared dataset directory is a saved `HyperDataset` bundle. It normally contains a `dataset.yaml` file plus one file per manifest table.

Typical layout:

```text
data/my_dataset/
  dataset.yaml
  segments.csv
  recordings.csv
  speaker.csv
  gender.csv
  language.csv
  enrollment.csv
  trials.csv
```

Typical `dataset.yaml`:

```yaml
segments: segments.csv
recordings: recordings.csv
classes:
  speaker: speaker.csv
  gender: gender.csv
  language: language.csv
enrollments:
  enrollment: enrollment.csv
trials:
  trials: trials.csv
```

Only `segments` is conceptually required by `HyperDataset`. In data prep classes for speech corpora, `recordings` is almost always produced too. `classes`, `enrollments`, and `trials` are optional and task-dependent.

## Core Table Semantics

All metadata tables are `InfoTable` subclasses backed by pandas `DataFrame`s. Every table has an `id` column and uses `id` as the index.

### `SegmentSet`

`SegmentSet` is the anchor table. Every row is one utterance, clip, or speech segment.

Required column:

- `id`: unique segment id.

Common columns:

- `recording`: id of the source recording when a segment is a slice of a longer recording.
- `start`: segment start time in seconds within `recording`.
- `duration`: segment duration in seconds.
- `speaker`: speaker class id.
- `gender`: gender class id.
- `language`: language class id, preferably ISO 639-3 alpha-3 such as `eng`.
- `transcript`: original transcript text when available.
- `corpusid`: broad corpus/source identifier.
- `dataset`: dataset prep identifier, usually `self.dataset_name()`.
- `source_type`: modality/source code such as `cts`, `afv`, or `intv`.
- `original_bandwidth`: original audio bandwidth when known.

Two valid patterns are common:

- Standalone clip: `segments.id` also identifies the audio file; no `recording` or `start` is needed.
- Slice of a longer recording: `segments.recording` points to `recordings.id`, and `start`/`duration` identify the time span.

### `RecordingSet`

`RecordingSet` describes physical audio storage.

Required columns:

- `id`: unique recording id.
- `storage_path`: file path or pipe command accepted by Hyperion audio readers.

Common columns:

- `duration`: recording duration in seconds.
- `sample_freq`: detected sampling frequency.
- `target_sample_freq`: optional desired resampling target.

If each segment is stored as one audio file, `recordings.id` usually matches `segments.id`. If segments are slices, `segments.recording` points to `recordings.id`.

Use `RecordingSet(...).sort()` and compute durations with either `recs.get_durations(self.num_threads)` or `self.get_recording_duration(recs)`, depending on the local pattern.

### `ClassInfo`

`ClassInfo` stores class vocabularies. A `ClassInfo` table row is a class label, not a segment.

The class table name should match the segment column it describes:

- `segments["speaker"]` -> `classes["speaker"]` saved as `speaker.csv`.
- `segments["gender"]` -> `classes["gender"]` saved as `gender.csv`.
- `segments["language"]` -> `classes["language"]` saved as `language.csv`.

Required column:

- `id`: class id used in the corresponding `SegmentSet` column.

Common columns:

- `class_idx`: contiguous integer index when needed by training code.
- `weights`: class weights.
- dataset-specific metadata such as speaker gender, nationality, book title, or accent.

Create `ClassInfo` only for labels that are useful downstream. Do not create class tables for every incidental provenance column.

### `EnrollmentMap`

`EnrollmentMap` maps speaker verification model ids to enrollment segment ids.

On disk, the common NIST-compatible column names are:

- `modelid`
- `segmentid`

In memory, `EnrollmentMap` normalizes `modelid` to its `id` column.

Use `EnrollmentMap(df)` when constructing enrollment metadata in memory. Some older prep scripts save enrollment CSV paths directly and pass them to `HyperDataset`; both patterns exist.

### `TrialKey` and `TrialNdx`

Use `TrialKey` when target/non-target labels are known. Use `TrialNdx` when only the trial pairs are known, as in eval/progress sets without labels.

For CSV-style trial files, common columns are:

- `modelid`
- `segmentid`
- `targettype`, with values such as `target` and `nontarget` when labels are available.

For spoofing or one-vs-many tasks, some prep scripts create `TrialKey` directly and save it.

### Other Table Families

The `HyperDataset` format also supports:

- `FeatureSet`: per-segment feature storage.
- `VADSet`: per-segment VAD storage.
- `DiarizationSet`: per-segment diarization metadata.
- `ImageSet` and `VideoSet`: media manifests for multimodal datasets.

Most classes in this directory focus on audio metadata and do not create those tables unless the source dataset requires them.

## DataPrep Class Pattern

New prep classes should derive from `DataPrep`.

Required methods:

- `dataset_name() -> str`: returns the subcommand name used by `hyperion-prepare-data`.
- `add_class_args(parser: ArgumentParser) -> None`: adds common and dataset-specific CLI arguments.
- `prepare(self) -> None`: runs the conversion and saves the dataset.

Constructor pattern:

```python
class MyCorpusDataPrep(DataPrep):
    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ) -> None:
        super().__init__(
            corpus_dir,
            output_dir,
            use_kaldi_ids,
            target_sample_freq,
            num_threads,
        )
        self.subset = subset
```

Argument pattern:

```python
@staticmethod
def add_class_args(parser: ArgumentParser) -> None:
    DataPrep.add_class_args(parser)
    parser.add_argument("--subset", required=True, choices=[...])
```

Registration requirements:

- `DataPrep.__init_subclass__` registers subclasses by `dataset_name()`.
- The module must be imported from `hyperion/data_prep/__init__.py`; otherwise `hyperion-prepare-data` will not see the class.
- The CLI subcommand is the returned dataset name, for example `dataset_name() == "voxceleb1"` gives `hyperion-prepare-data voxceleb1 ...`.

## Implementation Workflow

A typical `prepare()` implementation should follow this order:

1. Resolve corpus paths and subset-specific directories.
2. Load source metadata with pandas or source-specific parsers.
3. Discover audio files using `Path.glob`, `Path.rglob`, or `glob.iglob` fallback for symlink-heavy corpora.
4. Assert that audio files exist and match the metadata.
5. Build stable recording/segment ids.
6. Build a `RecordingSet` with `id` and `storage_path`.
7. Sort recordings and compute `duration`/`sample_freq`.
8. Add `target_sample_freq` when requested.
9. Build a `SegmentSet` with `id`, labels, provenance, and duration.
10. Build relevant `ClassInfo` tables from segment labels and metadata.
11. Build `EnrollmentMap`, `TrialKey`, or `TrialNdx` for evaluation subsets.
12. Construct `HyperDataset` and call `dataset.save(self.output_dir)`.
13. Log useful counts such as number of segments, recordings, speakers, and trials.

## ID Conventions

IDs must be stable, unique, and reproducible from source metadata.

Use prefixes when needed to avoid collisions across corpora:

- LibriSpeech uses ids such as `librispeech-...`.
- Libri-derived speaker ids often use `libri-...`.
- ASVspoof prep may prefix speaker or segment ids with `ASVSpoof2024-...`.

When `use_kaldi_ids` is enabled, segment ids often include the speaker id prefix:

```python
rec_ids = [f"{speaker}-{recording_id}" for recording_id, speaker in zip(rec_ids, speakers)]
```

Keep enrollment and trial ids aligned with the final segment ids. This is the most common source of broken prepared datasets.

## Language, Gender, and Common Label Conventions

Use consistent normalized labels:

- Languages should usually be ISO 639-3 alpha-3 codes such as `eng`. Use `DataPrep._language_to_alpha3(...)` when converting language names.
- Gender labels are usually lowercase `m` and `f` when the source supports binary labels.
- Spoof labels commonly use `bonafide` and `spoof`.
- Missing unknown labels should be handled deliberately. Avoid silently creating confusing class ids such as `"N/A"` unless existing downstream code expects them.

For ARTS-style age buckets, use `DataPrep._age_to_arts_age_group(...)`.

## Style and Quality Rules

- Keep source-specific parsing in private helpers such as `_get_metadata`, `_get_transcripts`, or `_get_spks_metadata`.
- Keep `prepare()` readable by using helpers for metadata parsing, enrollment creation, and trial creation.
- Prefer `Path` operations over string path concatenation.
- Prefer `logging` over `print`.
- Use `assert` or explicit exceptions for required source files and empty audio matches.
- Sort output tables before saving when stable ordering matters.
- Do not download remote resources unless that pattern already exists for the dataset family or is explicitly needed.
- Preserve original source metadata columns when they are useful downstream, but keep the core Hyperion columns clear and consistently named.
- Do not mutate unrelated files outside `output_dir`.

## Adding a New Data Prep Script

Checklist:

1. Create `hyperion/data_prep/<dataset_name>.py`.
2. Define `<DatasetName>DataPrep(DataPrep)`.
3. Implement `dataset_name()`, `add_class_args(...)`, and `prepare()`.
4. Import the new class in `hyperion/data_prep/__init__.py`.
5. Run `hyperion-prepare-data <dataset-name> --help` or `python -m hyperion.bin.prepare_data <dataset-name> --help` after installation/import setup.
6. Run a small real or synthetic preparation and verify that `dataset.yaml`, `segments.csv`, and expected auxiliary tables load with `HyperDataset.load(output_dir)`.

Useful validation snippet:

```python
from hyperion.utils import HyperDataset

ds = HyperDataset.load("data/my_dataset", lazy=False)
print(ds.describe()["msg"])
```

## Common Failure Modes

- The new class is not imported in `hyperion/data_prep/__init__.py`, so no CLI subcommand appears.
- Segment ids do not match recording ids or the `recording` column.
- Trial/enrollment `segmentid` values use raw corpus ids instead of final Hyperion segment ids.
- A `ClassInfo` table contains labels that do not match the corresponding segment column.
- Durations are missing for corpora where downstream samplers expect `segments.duration`.
- Source metadata contains duplicate ids and the prep script does not resolve them deterministically.
- Eval sets without labels are saved as `TrialKey` instead of `TrialNdx`.

When in doubt, compare against a nearby prep class for the same family:

- Directory of independent audio files: `audio_dir.py`.
- Speaker recognition with enrollment/trials: `voxceleb1.py`, `sitw.py`, `sre16.py`.
- ASVspoof/spoofing tasks: `asvspoof2024.py`.
- Read-speech corpora with transcripts/books/chapters: `librispeech.py`, `libritts.py`, `libriheavy.py`.

# AGENTS.md

This directory contains IO abstractions for audio, feature matrices/vectors, VAD, HDF5, Kaldi ark/scp, and Hyperion compatibility formats. These readers and writers are used by CLIs, recipes, data prep, PyTorch datasets, and NumPy utilities.

Use this with the repository-level `AGENTS.md`. The most relevant docs are:

- `docs/io.rst`
- `docs/info_tables.rst`
- `docs/hyper_dataset.rst`

## Core Concepts

The IO layer is built around:

- read/write specifier parsing in `rw_specifiers.py`,
- data reader/writer factories in `data_rw_factory.py`,
- VAD reader factories in `vad_rw_factory.py`,
- format-specific readers/writers for HDF5, ark/scp, audio, and legacy Hyperion files.

Prefer extending the factory/specifier path when adding a supported format or option. Do not bypass the factories in CLI code unless the script is intentionally format-specific.

## Specifier Conventions

Hyperion supports Kaldi-style read/write specifiers as well as extension-based convenience paths.

Common write specifiers:

- `h5:out/feat.h5`
- `h5,csv:out/feat.h5,out/feat.csv`
- `ark,scp:out/feat.ark,out/feat.scp`
- `ark,csv:out/feat.ark,out/feat.csv`

Common read specifiers:

- `h5:data/feat.h5`
- `ark:data/feat.ark`
- `scp:data/feat.scp`
- `csv:data/feat.csv`
- `tsv:data/feat.tsv`

When changing specifier parsing:

- preserve Windows drive path handling,
- preserve existing options such as `h5`, `ark`, `scp`, `csv`, `tsv`, `b`, `t`, `f`, `nf`, `p`,
- keep archive/script/both semantics compatible with existing recipes.

## Reader and Writer Behavior

Readers are generally sequential or random-access:

- sequential readers iterate or read batches in file order,
- random readers fetch by key,
- script readers load an index file and then fetch archive entries,
- archive readers read directly from one storage file.

Writers should:

- create parent directories where appropriate,
- preserve key order in sidecar scripts,
- support metadata columns where existing writer APIs expose them,
- close files cleanly,
- support context-manager use when the class already does.

Do not change key semantics. Keys are dataset ids used by `FeatureSet`, `VADSet`, `SegmentSet`, and downstream scoring/training code.

## Audio IO

`audio_reader.py` and `audio_writer.py` handle audio files and pipe commands.

Important conventions:

- `RecordingSet.storage_path` may be a file path or a pipe ending in `|`.
- Readers support time offsets/durations for segment extraction.
- `channels_first`, `always_2d`, and `return_all_channels` control waveform shape.
- Resampling uses Hyperion preprocessing utilities when `target_sample_freq` is requested.
- Audio ids should stay aligned with `RecordingSet` and `SegmentSet`.

Be careful with shape conventions. Existing audio code may expect `(channels, samples)` or `(samples, channels)` depending on flags.

## VAD IO

VAD readers include binary VAD, table VAD, and segment-derived VAD variants.

When editing VAD code:

- preserve segment id alignment,
- preserve frame/time conversion assumptions,
- keep VAD outputs compatible with frame selectors and PyTorch datasets,
- route new reader options through `VADReaderFactory` when user-facing.

## Dtypes and Compatibility

The IO layer is compatibility-sensitive.

When changing load/save behavior:

- preserve existing HDF5 dataset names where possible,
- preserve ark/scp binary/text behavior,
- preserve compression options and Kaldi matrix compatibility,
- avoid changing default dtypes without checking downstream code,
- keep old files readable unless there is an intentional migration.

## CLI Integration

Factories expose parser helpers:

- `DataWriterFactory.add_class_args(...)`
- `DataWriterFactory.filter_args(...)`
- sequential/random reader factory helpers,
- VAD reader factory helpers.

When adding user-facing IO options, add them to the relevant factory helper and consume them with `filter_args(...)` in scripts.

## Tests

Relevant tests:

- `tests/hyperion/io/test_ark_rw.py`
- `tests/hyperion/io/test_h5_rw.py`
- `tests/hyperion/io/test_audio_rw.py`
- `tests/hyperion/io/test_rw_specifiers.py`
- `tests/hyperion/io/test_copy_feats.py`

For IO changes, prefer save/load round-trip tests and include both archive-only and archive-plus-script cases when relevant.

## Common Pitfalls

- Treating a plain path containing `:` as a Kaldi specifier.
- Breaking sidecar script paths when adding path prefixes.
- Reordering keys between archive and script outputs.
- Returning waveform arrays with an unexpected channel/sample axis order.
- Forgetting to close file handles or subprocess pipes.
- Changing a default separator or extension inference used by existing recipes.

# AGENTS.md

This repository contains the `hyperion` Python package, a speech-processing toolkit with NumPy and PyTorch stacks plus a large CLI surface for training, extraction, evaluation, and dataset preparation.

This file is intended to give Codex enough repo-specific context to work effectively without re-discovering the same conventions each session.

## Scope and priorities

- Treat `hyperion/` as the primary product surface.
- Treat `docs/` and `tests/` as part of the maintained codebase.
- Treat `egs/` as recipe infrastructure that depends on the Python package and shell utilities.
- Treat `hyperion/bin_deprec/` and `hyperion/bin_deprec2/` as legacy unless the user explicitly asks to work there.
- Do not assume the worktree is clean. This repo often has local/untracked experiment artifacts.

## Repository structure

Top-level directories and their role:

- `hyperion/`: package source.
- `docs/`: Sphinx docs for the current implementation.
- `tests/`: pytest suite plus fixture data under `tests/data_in`.
- `egs/`: Kaldi-style recipes and experiment pipelines.
- `hyp_utils/`: shell/perl/awk helpers used by recipes.
- `tools/`: installation helpers and external tool bootstrap scripts.

Important package subtrees inside `hyperion/`:

- `hyperion/np/`: NumPy-based models, transforms, PDFs, calibration, metrics, clustering, augmentation.
- `hyperion/torch/`: PyTorch stack.
- `hyperion/io/`: readers/writers for audio, HDF5, Kaldi ark/scp, VAD, and related abstractions.
- `hyperion/utils/`: trial tables, dataset manifests, Kaldi-style metadata containers, math/helpers.
- `hyperion/data_prep/`: dataset-specific preparation classes built on a shared base class.
- `hyperion/metrics/`: high-level evaluator classes.
- `hyperion/text_norm/`: text normalization utilities.
- `hyperion/bin/`: maintained CLI entry points.

Important PyTorch layering inside `hyperion/torch/`:

- `layers/`: primitive layers.
- `layer_blocks/`: reusable blocks composed from primitive layers.
- `narchs/`: neural architectures.
- `models/`: top-level task models.
- `data/`: datasets and samplers.
- `trainers/`: trainer implementations.
- `optim/`, `lr_schedulers/`, `wd_schedulers/`, `metrics/`, `loggers/`, `utils/`: training support code.
- `tpm/`: third-party model wrappers, including Hugging Face integrations and evaluators.

## Active architectural patterns

The codebase uses a few recurring patterns. Prefer fitting into them rather than inventing new ones.

### 1. Registry-based base classes

Several base classes register subclasses automatically by name:

- `hyperion.np.hyper_np_model.HyperNPModel`
- `hyperion.torch.hyper_torch_model.HyperTorchModel`
- `hyperion.data_prep.data_prep.DataPrep`

If you add a new subclass in these families, preserve constructor/config conventions so dynamic loading and serialization keep working.

### 2. Config-driven CLIs via `jsonargparse`

The maintained CLI scripts in `hyperion/bin/` typically:

- build an `ArgumentParser`,
- compose nested parsers with `ActionParser`,
- support config files through `ActionConfigFile`,
- call `add_class_args(...)` on models/datasets/trainers/factories,
- extract runtime kwargs via `filter_args(...)` or `namespace_to_dict(...)`.

When adding or updating CLI functionality:

- follow the existing parser structure instead of hand-parsing flags,
- keep nested config sections stable where possible,
- prefer class/factory `add_class_args` and `filter_args` helpers over bespoke argument plumbing.

### 3. Serialization/config methods

Core model-style classes often expose:

- `get_config()`
- `save(...)`
- `load(...)`
- `save_params(...)`
- `load_params(...)`

For new serializable classes, keep configs JSON/HDF5 friendly and match existing naming conventions such as `class_name`.

### 4. Factories and static helpers

Factories are common for optimizers, schedulers, samplers, pooling layers, attacks, and readers. Before adding conditional logic directly into CLI scripts or trainers, check whether a factory already exists and extend that instead.

## Code style observed in active code

The active, maintained code is more modern than some legacy areas. Match the active style, not the oldest files.

- Use Python 3 type hints broadly, including return types.
- Use docstrings for public classes and non-trivial methods.
- Class docstrings should include an `Attributes:` section.
- In class docstrings, include the attributes inherited from parent classes as well as the attributes introduced by the class itself when that context matters for using or extending the class.
- Method/function docstrings should include `Args:` if the callable has arguments beyond `self`/`cls`.
- Method/function docstrings should include `Returns:` only when the callable returns something meaningful.
- If a method/function does not return anything, do not include a `Returns:` section.
- Prefer `pathlib.Path` and the shared `PathLike` aliases where applicable.
- Prefer `logging` over `print`.
- Prefer explicit imports over wildcard imports.
- Keep functions and methods fairly direct; this repo favors pragmatic implementation over heavy abstraction.
- Preserve existing naming in each subsystem:
  - modules/files use `snake_case`,
  - classes use `CamelCase`,
  - many model aliases in CLI scripts use short uppercase abbreviations (`AF`, `Trainer`, `RXVec`, etc.) when they improve readability locally.
- Prefer NumPy/Pandas/Torch idioms already used nearby rather than introducing a new style.

Formatting notes from the current tree:

- Black/isort are included as dependencies and the README advertises Black style, but there is no large centralized lint config beyond `pyproject.toml`.
- Existing code mixes old `%`/`.format(...)` logging and newer f-strings. In active files, prefer f-strings for plain string construction and `logging.*("...", arg)` when that keeps log formatting lazy.
- Keep imports grouped as: stdlib, third-party, local package imports.

## How to work in each area

### `hyperion/bin/`

- These are the maintained CLI entry points.
- If you add, remove, or rename a maintained script here, regenerate `pyproject.toml` with:
  - `python generate_pyproject.py`
- Do not manually edit the generated script list in `pyproject.toml` unless there is a strong reason.

### `hyperion/np/`

- This stack contains classic statistical models, transforms, metrics, and preprocessing.
- New model-like classes should usually derive from `HyperNPModel`.
- Preserve save/load compatibility when modifying serialized models or trial/score containers.

### `hyperion/torch/`

- Respect the existing layering: layer -> layer block -> neural architecture -> final model -> trainer/CLI.
- Final PyTorch models that are trained with the trainers should live under `hyperion/torch/models/`.
- These final trainable models should derive from `HyperTorchModel`.
- Final models are composed from neural architectures located under `hyperion/torch/narchs/`.
- Neural architectures should derive from `NetArch`.
- Neural architectures are composed from reusable layer blocks in `hyperion/torch/layer_blocks/`, lower-level layers in `hyperion/torch/layers/`, and standard PyTorch layers where appropriate.
- Layer blocks generally derive directly from `torch.nn.Module`.
- Layer blocks can themselves be composed from toolkit-defined layers or standard PyTorch layers.
- Trainer code is feature-rich already: DDP/FSDP, AMP, schedulers, SWA, multiple loggers. Prefer integrating with `TorchTrainerBase`/existing trainers instead of creating parallel training infrastructure.

### `hyperion/io/` and `hyperion/utils/`

- These modules are foundational and widely reused.
- Be careful with file format compatibility, dtype handling, and backward-compatible load/save behavior.
- Many of these utilities are Kaldi-style or BOSARIS-style abstractions; preserve table/key/score semantics.

### `hyperion/data_prep/`

- Dataset prep classes derive from `DataPrep` and usually expose `dataset_name()` plus `add_class_args(...)`.
- Keep dataset-specific logic in the dataset module rather than bloating `prepare_data.py`.

### `docs/`

- Docs are maintained and already reflect the current package layout.
- Update docs when you change public package structure, public CLIs, or major extension points.

### `tests/`

- The repo has a real pytest suite under `tests/hyperion/...`.
- There is also fixture data under `tests/data_in` and output folders under `tests/data_out`.
- `run_test.sh` shows the intended broad test buckets:
  - `tests/hyperion/io`
  - `tests/hyperion/pdfs`
  - `tests/hyperion/feats`
  - `tests/hyperion/utils`
  - `tests/hyperion/metrics`

## Testing guidance

Prefer targeted tests that match the area you changed.

Useful commands:

- `pytest tests/hyperion/utils/test_trial_scores.py`
- `pytest tests/hyperion/io/test_audio_rw.py`
- `pytest tests/hyperion/metrics/test_eer.py`
- `pytest tests/hyperion/pdfs/core/test_normal.py`

If a change affects generated CLI registrations, also run:

- `python generate_pyproject.py`

If you touch code with serialization, IO, or score/table semantics, prioritize regression tests because these modules are reused broadly across recipes and tooling.

## Practical cautions

- Avoid editing deprecated script trees unless the task is explicitly about legacy code.
- Avoid broad refactors across the entire repo unless requested; this codebase is large and heterogeneous.
- Be conservative with public API changes in `hyperion/utils`, `hyperion/io`, and base model classes.
- Recipe code under `egs/` often depends on shell helpers in `hyp_utils/` and environment setup scripts; do not “simplify” those flows casually.
- The repo contains experiment-oriented and research-oriented code. Preserve existing behavior unless there is a clear bug or the user explicitly requests a redesign.

## Good default workflow for Codex

When starting work in this repo:

1. Inspect the nearest existing module in the same subsystem before editing.
2. Prefer extending an existing base class, factory, parser helper, or trainer instead of adding a parallel pattern.
3. Keep changes local to the active subsystem unless cross-cutting edits are required.
4. Run targeted pytest coverage for the touched area when feasible.
5. If `hyperion/bin/` changed, regenerate `pyproject.toml`.
6. Update docs when public behavior or package structure changed.

## Representative files for orientation

These files are good entry points for understanding the maintained architecture:

- `hyperion/hyp_defs.py`
- `hyperion/np/hyper_np_model.py`
- `hyperion/torch/hyper_torch_model.py`
- `hyperion/torch/trainers/torch_trainer_base.py`
- `hyperion/data_prep/data_prep.py`
- `hyperion/io/audio_reader.py`
- `hyperion/utils/trial_scores.py`
- `hyperion/bin/train_xvector_from_wav.py`
- `docs/architecture.rst`
- `docs/cli.rst`

If in doubt, follow the style and extension pattern of the closest maintained file in the same subtree.

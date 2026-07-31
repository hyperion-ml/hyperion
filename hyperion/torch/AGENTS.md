# AGENTS.md

This directory contains the PyTorch stack: layers, layer blocks, neural architectures, final trainable models, datasets/samplers, trainers, losses, metrics, schedulers, loggers, attacks, and third-party model wrappers.

Use this with the repository-level `AGENTS.md`. The most relevant docs are:

- `docs/torch.rst`
- `docs/architecture.rst`

## Architecture Layers

Respect the established construction hierarchy:

1. `hyperion/torch/layers`: primitive toolkit layers and factories.
2. `hyperion/torch/layer_blocks`: reusable blocks composed from toolkit layers and standard PyTorch layers.
3. `hyperion/torch/narchs`: neural architectures.
4. `hyperion/torch/models`: final models that trainers train.
5. `hyperion/torch/trainers`: training loops and checkpoint/logging/distributed machinery.

Final trainable models should live in `hyperion/torch/models` and derive from `HyperTorchModel`.

Neural architectures should live in `hyperion/torch/narchs` and derive from `NetArch`. A `NetArch` subclass must expose shape contracts through:

- `in_shape()`
- `out_shape(...)`

Layer blocks usually live in `hyperion/torch/layer_blocks` and derive directly from `torch.nn.Module`. They can compose toolkit layers from `hyperion/torch/layers` or standard `torch.nn` modules.

## `HyperTorchModel` Conventions

Use `HyperTorchModel` for final trainable PyTorch models and model-like architecture components that need common Hyperion behavior.

Preserve these conventions:

- `get_config(...)` returns serializable constructor configuration and includes `class_name` unless `no_class_name=True`.
- `copy()` and `clone()` are deep-copy aliases.
- `trainable_parameters()` and `trainable_named_parameters()` filter by `requires_grad`.
- `has_param_groups()` and `trainable_param_groups()` control custom optimizer parameter groups.
- `freeze()`, `unfreeze()`, and `train_mode` should keep training mode behavior predictable.
- `change_dropouts(...)` should work when a model exposes dropout modules or a `dropout_rate` attribute.

If you add a new constructor argument to a trainable model, update `get_config()` so save/load and factory code can reconstruct it.

## Models, Architectures, Blocks, and Layers

When adding model functionality, put the code at the lowest layer that owns the concept:

- A reusable tensor operation belongs in `layers`.
- A reusable architectural motif belongs in `layer_blocks`.
- A complete encoder/decoder/backbone shape contract belongs in `narchs` and should derive from `NetArch`.
- A task-level object trained by a trainer belongs in `models` and should derive from `HyperTorchModel`.

Avoid putting full task models in `narchs` or low-level layer code. Avoid putting generic blocks only inside one final model if they are likely to be reused.

## Config and CLI Integration

Many PyTorch classes expose:

- `add_class_args(parser, prefix=...)`
- `filter_args(...)`
- `get_config()`
- `valid_train_modes()`

Training CLIs under `hyperion/bin` build nested configs by calling these methods. When adding a new user-facing option, update the class parser helper and the relevant `filter_args` path rather than manually unpacking ad hoc kwargs in the CLI.

Use `jsonargparse` types and `ActionYesNo` where nearby code does. Preserve existing prefix names such as `model`, `feats`, `trainer`, `sampler`, `attack`, or `dataset`.

## Trainers

`hyperion/torch/trainers/torch_trainer_base.py` is the foundation for training features:

- DDP and FSDP support.
- AMP and gradient scaling.
- optimizer, LR scheduler, and weight-decay scheduler factories.
- gradient accumulation and clipping.
- checkpoint loading/saving.
- SWA.
- progress, CSV, TensorBoard, and W&B loggers.

Prefer extending `TorchTrainerBase` or an existing trainer over adding a separate training loop. Keep task-specific logic in trainer subclasses; keep model architecture logic in `models`/`narchs`.

## Data and Samplers

Dataset and sampler classes live under `hyperion/torch/data`. They often consume `HyperDataset`, `SegmentSet`, `ClassInfo`, recordings, feature tables, or Kaldi-style manifests from `hyperion/utils`.

When changing data code:

- Preserve id alignment with `hyperion/utils` metadata tables.
- Keep sampler output compatible with `torch.utils.data.DataLoader`.
- Avoid assuming all corpora have the same segment columns beyond the documented `InfoTable` contracts.
- Be careful with distributed sampling behavior and shuffle/validation differences.

## Third-Party Model Wrappers

`hyperion/torch/tpm` contains wrappers for third-party models and toolkits, including Hugging Face models and external evaluators.

When editing TPM wrappers:

- Isolate dependency-specific behavior inside the wrapper.
- Keep the public Hyperion model interface stable.
- Avoid importing heavy optional dependencies at package import time when a local lazy import is enough.

## Testing and Validation

Use focused checks based on the touched area:

- For syntax-only validation: `python -m py_compile <changed files>`.
- For model shape changes: instantiate the class and run a small forward pass on CPU when feasible.
- For trainer changes: prefer a tiny synthetic dataset/checkpoint smoke test if available.
- For data/sampler changes: test one batch through `DataLoader`.

PyTorch tests can be expensive or environment-dependent. If GPU/distributed behavior is touched, state clearly what was and was not exercised.

## Common Pitfalls

- Putting a final trainable model in `narchs` instead of `models`.
- Adding a `NetArch` without correct `in_shape()` and `out_shape(...)`.
- Adding constructor args but not updating `get_config()` or parser helpers.
- Breaking `trainable_param_groups()` by returning consumed generators or stale parameter lists.
- Adding model logic to a trainer because it is convenient at CLI time.
- Assuming CUDA is available in tests or imports.

# AGENTS.md

This directory contains maintained CLI entry point modules. Package console scripts in `pyproject.toml` map `hyperion/bin/<name>.py` to commands named `hyperion-<name-with-dashes>`.

Use this with the repository-level `AGENTS.md`. The most relevant docs are:

- `docs/cli.rst`
- `docs/architecture.rst`

## CLI Entry Point Model

Each maintained script should expose:

- `main() -> None`
- a parser-building helper when the parser is non-trivial,
- task functions that can be tested or reused independently from CLI parsing.

Use `jsonargparse.ArgumentParser`, not raw `argparse`, unless there is a local reason.

Common imports:

```python
from jsonargparse import ActionConfigFile, ActionParser, ActionYesNo, ArgumentParser, namespace_to_dict
from hyperion.hyp_defs import config_logger
```

Use `--cfg` with `ActionConfigFile` for scripts that have many options or nested config sections.

## Parser Conventions

Prefer class/factory argument helpers:

- `SomeClass.add_class_args(parser, prefix="...")`
- `SomeFactory.add_class_args(parser, prefix="...")`
- `SomeClass.filter_args(...)`
- `SomeFactory.filter_args(...)`

For nested train/eval configs, use `ActionParser` and stable section names such as:

- `data`
- `dataset`
- `sampler`
- `data_loader`
- `feats`
- `model`
- `trainer`
- `attack`

Use `ActionYesNo` for boolean flags when nearby scripts do.

After parsing, scripts commonly call:

```python
args = parser.parse_args()
config_logger(args.verbose)
kwargs = namespace_to_dict(args)
```

## Generated Package Scripts

`pyproject.toml` script entries are generated from `hyperion/bin/*.py` by `generate_pyproject.py`.

If you add, remove, or rename a maintained script in this directory, run:

```bash
python generate_pyproject.py
```

Do not manually maintain the generated script list in `pyproject.toml` unless the generation flow itself is being changed.

Naming rule:

- `hyperion/bin/train_qvector.py` becomes `hyperion-train-qvector`.
- Underscores become dashes.

## Structure for Training Scripts

Training scripts usually:

1. define model/factory dictionaries when there are architecture choices,
2. initialize data loaders,
3. initialize feature extractors or preprocessing,
4. initialize the model,
5. initialize metrics and trainer,
6. load the last checkpoint when appropriate,
7. call `trainer.fit(...)`.

Distributed scripts should use the existing `hyperion.torch.utils.ddp` helpers and clean up DDP state.

Keep trainer-specific logic in `hyperion/torch/trainers`, model logic in `hyperion/torch/models`, and parser composition in the CLI.

## Structure for Evaluation and Utility Scripts

Evaluation scripts should:

- keep computation in a function with typed arguments,
- validate paired list lengths and output/plot option combinations early,
- create output directories with `Path(...).parent.mkdir(...)`,
- write CSV/TSV separators based on file extension when following local convention,
- use `logging` for progress and only print final human-readable tables when existing scripts do.

Utility scripts should reuse `hyperion.utils`, `hyperion.io`, `hyperion.metrics`, and `hyperion.np` APIs instead of reimplementing table or score parsing.

## Imports and Side Effects

- Avoid heavy work at module import time.
- Avoid parsing arguments outside `main()`.
- Keep network downloads and external tool calls explicit and logged.
- Use `if __name__ == "__main__": main()` at the end.

## Testing

For parser-only changes:

```bash
python -m hyperion.bin.<module> --help
```

For installed entry point changes:

```bash
hyperion-<command> --help
```

For generated scripts:

```bash
python generate_pyproject.py
```

For behavioral changes, run the smallest task-specific command or unit test that exercises the changed path.

## Common Pitfalls

- Adding a CLI option but not passing it through `filter_args(...)`.
- Renaming a script without regenerating `pyproject.toml`.
- Duplicating model/trainer construction logic that already exists in a class helper.
- Creating output files before validating input list lengths and option consistency.
- Assuming CUDA, W&B, external corpora, or downloaded model files are available.

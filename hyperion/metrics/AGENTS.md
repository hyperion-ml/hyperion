# AGENTS.md

This directory contains high-level evaluator classes. These classes orchestrate full evaluation workflows by loading Hyperion trial/score objects, calling lower-level metric functions, and producing reports or plots.

Use this with the repository-level `AGENTS.md`. The most relevant docs are:

- `docs/metrics.rst`
- `docs/trials.rst`
- `docs/utils.rst`
- `docs/np/metrics` when working with lower-level metric functions.

## Metrics Layering

Hyperion metrics are split across three layers:

- `hyperion.np.metrics`: NumPy metric functions and plotting primitives.
- `hyperion.torch.metrics`: torch-native metric helpers for training.
- `hyperion.metrics`: high-level evaluator classes in this directory.

Keep algorithmic metric definitions in `hyperion.np.metrics` or `hyperion.torch.metrics`. Keep this directory focused on evaluator orchestration, loading, alignment, aggregation, and output formatting.

## Evaluator Responsibilities

High-level evaluators commonly:

- accept paths or already loaded Hyperion objects,
- load keys, ndx, scores, or metadata,
- align scores with keys/ndx before metric computation,
- separate target and non-target trials,
- call NumPy metric functions,
- return pandas `DataFrame` summaries,
- update plot objects when provided,
- handle sparse and dense trial/score variants.

Do not treat trial objects as plain arrays. Preserve model/segment id alignment and masks.

## Verification Evaluator Pattern

`VerificationEvaluator` is the core speaker verification evaluator.

Important behavior:

- path inputs load `TrialKey`/`TrialScores` or sparse variants based on `sparse`,
- scores are aligned with the key using `scores.align_with_ndx(key)`,
- `p_tar`, `c_miss`, and `c_fa` are converted to effective priors when costs are provided,
- empty target or non-target sets return `None` and log a warning,
- `compute_dcf_eer(return_df=True)` returns a report `DataFrame`,
- `return_df=False` returns raw metric values,
- DET and normalized-DCF plot objects are updated as side effects when supplied.

When changing verification behavior, test both scalar and vector priors.

## Other Evaluators

Other files wrap related workflows:

- `verification_adv_attack_evaluator.py`: adversarial attack evaluation for verification.
- `verification_anonymization_evaluator.py`: anonymization-specific verification evaluation.
- `speech_quality_evaluator.py`: speech quality metric aggregation.
- `voxprofile_evaluator.py`: VoxProfile-related evaluator orchestration.

Keep task-specific loading and aggregation in the relevant evaluator class. Put reusable metric math in lower-level metric modules.

## Output Conventions

Evaluator outputs should be stable and script-friendly:

- use pandas `DataFrame` for tabular summaries,
- keep column names compatible with existing CLIs,
- include enough labels to identify score/key/system names,
- use `Path` and create parent directories before writing in CLI code,
- keep plotting side effects explicit through plot object parameters.

Avoid changing output column names unless the user explicitly requests it or downstream code is updated in the same change.

## Sparse and Dense Compatibility

Many workflows support both dense and sparse trial/score representations:

- `TrialKey`
- `TrialScores`
- `SparseTrialKey`
- `SparseTrialScores`

When adding evaluator features, preserve both paths where the class already supports them. Do not silently densify large sparse objects unless the evaluator already does so and the memory cost is understood.

## Error Handling

Use explicit validation for:

- mismatched prior/cost vector lengths,
- missing or empty target/non-target trials,
- score files that do not align with keys,
- unsupported sparse/dense combinations,
- invalid output or plotting options.

Prefer `ValueError` for invalid user inputs. Use `logging.warning` for valid-but-unusable cases such as zero target trials.

## Tests

Relevant tests may be under:

- `tests/hyperion/metrics`
- `tests/hyperion/utils` for trial/key/score alignment
- `tests/hyperion/io` when evaluator loading touches file compatibility

For evaluator changes, use small synthetic `TrialKey`/`TrialScores` objects when possible. Test path-based loading separately when file-format behavior changes.

## Common Pitfalls

- Computing metrics before aligning scores with keys.
- Reordering priors without restoring result order.
- Returning `None` in a path where caller code expects a `DataFrame`.
- Changing report column names used by `hyperion/bin` scripts.
- Mixing DET plot and normalized-DCF plot parameters.
- Ignoring sparse trial support in workflows that advertise it.

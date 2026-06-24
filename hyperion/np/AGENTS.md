# AGENTS.md

This directory contains the NumPy/statistical stack: PDFs, PLDA/GMM models, classifiers, transforms, calibration, score normalization, clustering, augmentation, classical feature extraction, and metric functions.

Use this with the repository-level `AGENTS.md`. The most relevant docs are:

- `docs/numpy.rst`
- `docs/architecture.rst`
- `docs/np/pdfs/mixtures.rst`
- `docs/np/pdfs/plda.rst`
- `docs/np/transforms.rst`
- `docs/np/mfcc.rst`

## Core Pattern

Model-like classes should usually derive from `HyperNPModel`.

`HyperNPModel` provides:

- subclass registration by class name,
- `copy()` and `clone()`,
- `is_init` state,
- `save(...)` and `load(...)`,
- JSON/HDF5 config handling,
- `save_params(...)` and `load_params(...)` extension points,
- helper methods for saving/loading parameter dictionaries.

If a class has trainable or learned state, preserve save/load compatibility when editing it.

## Serialization Contract

Serializable NumPy models should keep these methods coherent:

- `get_config()`: returns constructor/config hyperparameters.
- `save_params(f)`: writes learned arrays to HDF5.
- `load_params(f, config)`: reconstructs the object and restores learned arrays.
- `load(...)`: should continue to read old saved models unless a migration is intentionally required.

Use `float_cpu()` for arrays loaded for computation and `float_save()` for arrays saved to disk when following existing model code.

Do not store non-JSON-serializable objects in configs. Convert paths, dtypes, priors, or enum-like values to simple strings, numbers, lists, or dicts.

## Subpackage Roles

- `pdfs`: probability density models such as Normal, GMM, PLDA, HMM, and JFA.
- `classifiers`: sklearn-style and custom classifiers/fusion models.
- `transforms`: PCA, LDA, NAP, MVN, length normalization, centering/whitening, CORAL, Gaussianization, and transform lists.
- `score_norm`: T/Z/S normalization variants.
- `metrics`: NumPy metric implementations for verification, ROC/DET/DCF/EER, WER/CER, SNR, STOI, PESQ, and related utilities.
- `feats`: classical feature extraction and VAD logic.
- `augment`: NumPy/audio augmentation helpers.
- `calibration`: score calibration models.
- `clustering`: AHC, k-means, spectral clustering.

Put new functionality in the subpackage that owns the mathematical concept, not where it is first called.

## Numerical Conventions

- Prefer NumPy vectorization over Python loops for core numerical operations.
- Use stable log-domain helpers from `hyperion.utils.math_funcs` when working with probabilities.
- Keep shape conventions explicit in docstrings and validation checks.
- Validate finite probabilities, priors, covariance/precision shapes, and class ids early.
- Preserve dtype behavior, especially for saved/loaded arrays and metrics code.
- Avoid changing default floors, epsilons, or priors casually; these can affect recipe results.

## Model Initialization and Fitting

Many models follow this pattern:

- constructor stores hyperparameters and optional initial parameters,
- `initialize(...)` prepares state,
- `fit(...)` or EM-style methods update state,
- `validate()` or local checks enforce shapes,
- prediction/scoring methods assume initialized state.

When changing fitting code, consider:

- empty or low-count components,
- single-class/single-component edge cases,
- sample weights,
- deterministic seeds,
- old saved models with missing newer attributes,
- numerical underflow/overflow.

## Tests

Relevant existing tests include:

- `tests/hyperion/pdfs`
- `tests/hyperion/feats`
- `tests/hyperion/metrics`
- `tests/hyperion/utils/test_math.py`

Use targeted tests for the changed algorithm. For stochastic algorithms, use fixed seeds and assert robust properties rather than exact fragile values unless exactness is intentional.

## Common Pitfalls

- Updating a constructor without updating `get_config()`.
- Saving arrays under new HDF5 names without backward-compatible loading.
- Returning arrays with changed orientation, especially `(num_samples, dim)` versus `(dim, num_classes)`.
- Ignoring `float_cpu()`/`float_save()` conventions.
- Treating score matrices, trial masks, or class ids as ordinary arrays without preserving their alignment metadata.

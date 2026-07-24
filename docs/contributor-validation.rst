Contributor Validation and Documentation Requirements
=====================================================

Every maintained change needs evidence appropriate to the interface it changes.
This page is the pull-request checklist for the ``hyperion`` package. It
complements :doc:`building-documentation`, which gives the exact commands for
building and checking the documentation.

Start from the observable contract
----------------------------------

Before selecting tests, state what a caller can observe changing: accepted
arguments, tensor or array layout, generated files, scores, configuration,
checkpoint loading, parser help, or error behavior. Test that contract at the
lowest useful level, then add an integration test only where components meet.

Do not use a large training recipe as the only proof of a package change.
``egs/`` is outside the maintained documentation and test surface. Prefer a
small fixture, a temporary output directory, and a direct package or CLI
invocation that fails for the right reason when its contract is broken.

Required validation by change type
----------------------------------

.. list-table:: Validation and documentation matrix
   :widths: 25 38 37
   :header-rows: 1

   * - Change
     - Minimum validation
     - Documentation required in the same pull request
   * - Public NumPy model, transform, scorer, or metric
     - Unit test of numeric behavior, shapes/dtypes, serialization or loading
       when applicable, and relevant error cases.
     - Update its curated API contract and any task guide affected by changed
       inputs, outputs, configuration, or artifacts.
   * - PyTorch layer, block, architecture, or task model
     - Construction and forward tests; shapes/dtypes; gradient behavior;
       train/eval and mask/length behavior when used; configuration/checkpoint
       round trip for model-level changes.
     - Update the API contract, forward/layout documentation, and a workflow
       example when users choose or configure the component directly.
   * - Dataset, sampler, or data preparation
     - Fixture-scale manifest or source corpus; stable IDs; table-reference
       integrity; dataloader batch contract or ``HyperDataset.load``.
     - Update data-model/preparation documentation and state CSV input/output
       fields, side effects, and any new corpus-specific prerequisites.
   * - Trainer, optimizer, scheduler, or distributed behavior
     - A minimal optimization step and, where changed, checkpoint/resume,
       AMP, scheduler, or multi-rank coverage.
     - Update training-support and configuration documentation for changed
       defaults, compatibility, and operational constraints.
   * - Maintained CLI or parser
     - ``--help``; parser error behavior; one fixture-scale end-to-end command;
       config-file equivalence when ``--cfg`` is supported.
     - Update the task-family guide, CLI inventory, generated index/help
       snapshot, optional-dependency notes, and command examples.
   * - File reader/writer or manifest/table semantics
     - Round trip, expected dtype/key/order behavior, and compatibility with a
       downstream consumer; regress a malformed-input failure when relevant.
     - Update I/O or data-model contracts with formats, side effects, and
       backward-compatibility behavior.
   * - Documentation-only public correction
     - Strict HTML build; execute a referenced command or example when it
       claims executable behavior.
     - Correct the relevant guide/reference and preserve links to its public
       API; no unrelated API promise should be added.

Targeted tests
--------------

Run the narrowest tests that cover the changed subsystem before broader checks.
The maintained test tree provides these useful starting points:

.. code-block:: bash

   pytest tests/hyperion/io/test_audio_rw.py
   pytest tests/hyperion/metrics/test_eer.py
   pytest tests/hyperion/pdfs/core/test_normal.py
   pytest tests/hyperion/utils/test_trial_scores.py

For a new test, place it under the matching ``tests/hyperion/<subsystem>/``
directory. Keep test inputs in ``tests/data_in`` when they are shared fixtures
and create outputs under the test's temporary directory or the established
``tests/data_out`` convention. A test must not depend on a user's corpus path,
network download, a personal model cache, or a GPU unless it is specifically a
conditional integration test.

If an interface has a boundary with a separate subsystem, add a focused
integration test as well: writer-to-reader, preparer-to-``HyperDataset``,
sampler-to-trainer, model-to-checkpoint loader, or CLI-to-produced artifact.
One small end-to-end test is more useful than several mocks for such a boundary.

PyTorch-specific evidence
-------------------------

For a public PyTorch change, include the evidence that applies to its contract:

* input/output tensor shape and dtype, including the batch-axis convention;
* masks or lengths, padding behavior, and empty/short input handling;
* CPU/device movement and registered parameter/buffer behavior;
* gradient flow and differences between ``train()`` and ``eval()``;
* a model configuration and checkpoint load round trip; and
* a minimal trainer update and resume when trainer-owned behavior changes.

Run DDP, FSDP, AMP, or optional-model tests when the changed code touches those
paths. A single-process CPU result is necessary but not proof that a change is
safe for distributed or mixed-precision execution. See
:doc:`torch-extension-workflows` for ownership and test expectations.

Data and compatibility evidence
-------------------------------

Keep data examples and public artifact contracts CSV-based. When an Ark payload
is required, show and validate the paired ``ark,csv:`` output so the index is
available to later commands. Do not introduce new ``.scp`` examples.

For stable APIs, treat changes to these as compatibility-sensitive:

* public import or command names;
* argument names, defaults, nested configuration keys, and parser choices;
* serialized model configuration and checkpoint keys;
* CSV columns, ID conventions, file formats, score semantics, and output paths;
  and
* tensor layout, dtype, device, return-value, and error conventions.

If a stable change cannot preserve old behavior, document the replacement and
migration in the relevant guide and release notes. Add a regression test for
the supported old artifact or input form whenever practical. Experimental
codec/DAC, VITS anonymization/freevc, transducer, and q-vector surfaces still
need tests and accurate prerequisites, but may intentionally change without
the stable compatibility guarantee.

Documentation changes required for a PR
----------------------------------------

Documentation is part of the implementation when a public interface changes.
Use this order to identify updates:

#. Update the class/function docstring for its purpose, parameters, return
   values, errors, and observable side effects.
#. Update the curated API contract page with shapes/dtypes, configuration,
   serialization, and compatibility information that autodoc alone cannot
   communicate.
#. Update a task-oriented how-to or CLI family guide when a user needs help
   selecting, configuring, or operating the interface.
#. Update inventories and generated sources when the public surface changes:
   ``docs/api_inventory.json`` for a curated API concept and
   ``docs/cli_inventory.json`` for every maintained command.
#. Regenerate checked-in derived docs after CLI changes; review the resulting
   parser-help diff rather than editing it manually.

Stable interfaces need full contracts. Experimental interfaces need a visible
experimental notice, prerequisites, and explicitly limited compatibility.
Legacy scripts and ``egs/`` recipes are not documentation templates. See
:doc:`documentation-policy` for the formal support boundary and
:doc:`deprecation-and-compatibility` for the migration process.

Before submitting
-----------------

Run the targeted tests first, then the relevant repository checks. For a
public-doc or CLI change, at minimum run:

.. code-block:: bash

   python docs/check_api_coverage.py
   python docs/check_cli_coverage.py
   python docs/render_cli_index.py --check
   HYPERION_PYTHON=python docs/build.sh html

Use the full local documentation-CI sequence, including link checking and the
pinned CLI-reference runtime, before a pull request that changes documentation
infrastructure or maintained parsers. The exact commands and offline/online
behavior are kept in :doc:`building-documentation`.

The PR description should name the changed public contract, the tests run, the
documentation pages updated, any generated files regenerated, and any
remaining conditional coverage (for example, an unavailable optional model or
multi-GPU environment). This makes review about an explicit compatibility
decision rather than inference from a patch.

See also
--------

* :doc:`contributor-extension-guide`
* :doc:`model-extension-contracts`
* :doc:`torch-extension-workflows`
* :doc:`data-preparation-and-cli-extensions`
* :doc:`documentation-policy`
* :doc:`building-documentation`

Data-Preparation and CLI Extension Workflows
=============================================

This guide covers two public extension points that turn implementation details
into user-facing workflows: corpus preparation classes in
``hyperion.data_prep`` and maintained commands in ``hyperion.bin``. Both are
configuration-facing APIs. Treat their identifiers, argument names, output
formats, and generated documentation as compatibility commitments.

Adding a data-preparation class
-------------------------------

Use ``DataPrep`` for code that converts one external corpus layout into
Hyperion's dataset metadata. A preparer belongs in
``hyperion/data_prep/<dataset_name>.py``; it does not belong in a training
command or in ``egs/``.

Every preparer must:

* derive from ``DataPrep``;
* implement a unique static ``dataset_name()``;
* implement ``prepare()``;
* expose its constructor through ``add_class_args(parser)``; and
* be imported from ``hyperion.data_prep.__init__``.

The last step is required for discovery. ``DataPrep.__init_subclass__``
registers a class under ``dataset_name()``, but registration occurs only when
Python imports the subclass module. ``hyperion-prepare-data`` builds its
subcommands from that registry, so an unimported class is invisible to users.

Start from the base constructor and add only corpus-specific options:

.. code-block:: python

   class MyCorpusDataPrep(DataPrep):
       @staticmethod
       def dataset_name() -> str:
           return "my-corpus"

       @staticmethod
       def add_class_args(parser: ArgumentParser) -> None:
           DataPrep.add_class_args(parser)
           parser.add_argument("--subset", required=True)

``DataPrep.add_class_args`` supplies the stable common options:
``corpus_dir``, ``output_dir``, ``use_kaldi_ids``, ``target_sample_freq``, and
``num_threads``. Keep their names and semantics intact. A new option should
describe source-corpus selection or parsing, not a downstream training choice.

Prepared-data artifact contract
-------------------------------

``prepare()`` must write a loadable ``HyperDataset`` bundle to ``output_dir``.
The normal artifact contains ``dataset.yaml`` plus CSV tables such as
``segments.csv``, ``recordings.csv``, class tables, enrollment maps, and trial
tables. A corpus may not need every table, but speech preparation normally
provides at least segments and recordings.

Use the maintained table meanings:

* ``SegmentSet`` has stable, unique segment ``id`` values. For slices of a
  recording, its ``recording``, ``start``, and ``duration`` fields refer to the
  final recording IDs and time in seconds.
* ``RecordingSet`` maps a stable recording ``id`` to ``storage_path`` and
  normally records duration and sample frequency. Apply
  ``target_sample_freq`` when requested.
* A ``ClassInfo`` table name matches the corresponding segment label column:
  for example, segment ``speaker`` labels use ``speaker.csv``.
* Enrollment and trial table IDs must refer to the final segment IDs, not raw
  corpus identifiers. Use ``TrialKey`` only when labels are known and
  ``TrialNdx`` when the pairs are unlabelled.

Use CSV files for all new manifest examples and generated artifacts. The
retired ``.scp`` interface is not a new data-preparation target.

Implementation sequence
-----------------------

Implement a preparer in this order:

#. Resolve the corpus directories and selected subset; fail clearly for missing
   source files or empty audio discovery.
#. Parse source metadata in small private helpers and construct deterministic,
   collision-free recording and segment IDs.
#. Build and sort ``RecordingSet`` and ``SegmentSet`` objects; calculate or
   carry forward durations and sample frequencies as the corpus requires.
#. Build only useful class, enrollment, and trial tables, ensuring cross-table
   ID references agree.
#. Construct ``HyperDataset`` and call ``dataset.save(self.output_dir)``.
#. Log record, segment, class, enrollment, and trial counts that help detect a
   wrong subset or metadata join.

Keep source-specific parsing in the preparer module. Do not make generic
``DataPrep`` behavior conditional on one corpus. Match a nearby maintained
class for the same data layout: ``audio_dir.py`` for independent audio files,
``voxceleb1.py``/``sre16.py`` for speaker verification, or an ASVspoof class
for spoofing data.

Validate a preparer with a fixture-scale source directory or a permitted small
corpus subset. Confirm all of the following:

* ``python -m hyperion.bin.prepare_data <dataset-name> --help`` lists and
  explains the expected options;
* the output has ``dataset.yaml`` and CSV files with stable IDs;
* ``HyperDataset.load(output_dir, lazy=False)`` succeeds;
* segment-to-recording, class, enrollment, and trial references resolve; and
* rerunning on the same source produces the same identifiers and ordering.

Adding a maintained command
----------------------------

A maintained CLI is a supported API, not merely a convenience script. Put a
new command in ``hyperion/bin/<command_name>.py`` with a ``main()`` function.
The installed command is normally named ``hyperion-<command-name>``: the
script stem has a leading ``hyperion_`` removed, then underscores are converted
to hyphens. For example, ``convert_vad_format.py`` becomes
``hyperion-convert-vad-format``.

Use ``jsonargparse.ArgumentParser`` and the maintained parser composition
patterns. A command that supports a repeatable configuration must expose
``--cfg`` with ``ActionConfigFile``. Build reusable nested groups with
``ActionParser`` and prefer the relevant component's ``add_class_args`` and
``filter_args`` methods over manually duplicating constructor arguments.
Convert parsed namespaces at the boundary with ``namespace_to_dict`` when the
implementation needs a dictionary.

The parser schema is part of the command's public API:

* Keep established option and nested configuration names stable. A hyphenated
  option, such as ``--output-dir``, maps to the snake_case configuration key
  ``output_dir``.
* Add a new option with a type, default, valid choices, and help text that
  describes its input, output, or side effect.
* Use nested parser sections to reflect real ownership (for example ``data``,
  ``model``, and ``trainer``), rather than one flat parser with unrelated
  options.
* Use ``parser.link_arguments`` only for values that truly must agree. A link
  changes the configuration contract and should be documented.
* Preserve artifacts and file formats across a compatible update. New command
  examples use CSV manifests and ``ark,csv:`` pairs where an Ark payload needs
  a CSV index.

Do not add a custom command-line path for behavior already available through a
factory, a registered class, or an existing command. Prefer extending that
shared interface so configuration files, help output, and programmatic use
stay aligned.

Register and document the command
----------------------------------

After adding, removing, or renaming a script in ``hyperion/bin``, regenerate
the package entry points:

.. code-block:: bash

   python generate_pyproject.py

Then add or update exactly one record in ``docs/cli_inventory.json``. Every
maintained module must be classified as ``stable``, ``experimental``, or
``excluded``. A non-excluded command needs an assigned task-family guide;
excluded commands need an explicit reason. Mark only the interfaces with an
intended compatibility commitment as stable. Codec/DAC, VITS
anonymization/freevc, transducer, and q-vector command surfaces remain
experimental unless their status changes deliberately.

Update the relevant guide under ``docs/cli/`` with prerequisites, input and
output formats, a minimal invocation, configuration-file usage when supported,
produced artifacts, and links to its Python API. The generated command index
comes from the inventory and the complete option reference comes from parser
help—do not hand-maintain option listings.

Run the required checks after a parser or inventory change:

.. code-block:: bash

   python docs/check_cli_coverage.py
   python docs/render_cli_index.py --check
   python docs/generate_cli_reference.py --python python

The last command regenerates the checked-in help snapshot. Use the pinned
runtime in ``docs/requirements-cli-reference.txt`` when validating the full
reference. During dependency diagnosis, ``--allow-unavailable`` records an
unavailable parser's diagnostic in the generated reference; it does not make
the command supported or hide its failure.

CLI validation checklist
------------------------

Before declaring the new command maintained, verify:

* module invocation and installed entry-point help both work;
* required and optional inputs fail with clear, parser-level errors;
* one fixture-scale invocation creates the documented output and that output
  can be consumed by the next relevant Hyperion API or command;
* a YAML/JSON configuration produces the same resolved behavior as equivalent
  command-line options;
* the inventory, generated index, generated help reference, family guide, and
  any representative end-to-end test are updated; and
* :doc:`building-documentation`'s strict HTML and relevant CLI checks pass.

See also
--------

* :doc:`contributor-extension-guide`
* :doc:`data_prep`
* :doc:`hyper_dataset`
* :doc:`cli`
* :doc:`how-to/use-configuration-files`
* :doc:`building-documentation`

Prepare Dataset Metadata and VAD
================================

Use ``hyperion-prepare-data`` to convert a supported corpus into Hyperion
recording and segment manifests. The command discovers its supported dataset
names from the registered :class:`hyperion.data_prep.DataPrep` implementations.
It is package functionality; it does not require an ``egs/`` recipe.

Discover a dataset preparer
---------------------------

List the preparers available in the installed version:

.. code-block:: bash

   hyperion-prepare-data --help

Then inspect the arguments for the selected dataset name:

.. code-block:: bash

   hyperion-prepare-data voxceleb1 --help

All preparers accept a corpus location and output directory. Individual
datasets may require extra annotations, partitions, or license-controlled
files, so treat their subcommand help as the authoritative contract.

.. code-block:: bash

   hyperion-prepare-data voxceleb1 \
     --corpus-dir /datasets/VoxCeleb1 \
     --output-dir data/voxceleb1 \
     --target-sample-freq 16000

Inspect the generated recording and segment files before using them for
training. Segment ids, recording ids, storage paths, durations, and the
speaker column must agree. :doc:`../info_tables` describes the manifest schema.

Compute energy VAD
------------------

Energy VAD can operate on a dataset manifest or separate recording/segment
inputs. Write its binary frame decisions to an archive/specifier:

.. code-block:: bash

   hyperion-compute-energy-vad \
     --dataset-file data/voxceleb1/dataset.yaml \
     --output-spec ark,csv:data/voxceleb1/vad.ark,data/voxceleb1/vad.csv

Use the command's ``--help`` to set frame parameters and energy-VAD thresholds
for the target domain. A VAD is time-aligned metadata, not a universal model
setting: its frame shift and frame length must remain available to later
readers and extractors.

Convert VAD formats when needed
-------------------------------

``hyperion-convert-vad-format`` converts between binary frame decisions and
time-mark tables. Use it when an external annotation format must be aligned to
Hyperion's VAD readers:

.. code-block:: bash

   hyperion-convert-vad-format time_marks_to_bin \
     --in-vad-file data/eval/vad_time_marks.csv \
     --out-vad-file ark,csv:data/eval/vad.ark,data/eval/vad.csv \
     --segments-file data/eval/segments.csv \
     --frame-length 25 \
     --frame-shift 10

Validate that every segment has VAD data and that its timing matches the audio
used for extraction. A mismatched VAD can remove speech or retain nonspeech
without producing a parser error.

See also
--------

* :doc:`../data_prep`
* :doc:`../io`
* :doc:`extract-score-xvectors`

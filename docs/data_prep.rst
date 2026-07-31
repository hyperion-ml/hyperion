Data Preparation API
====================

``hyperion.data_prep`` converts supported corpora into Hyperion recording,
segment, class, and dataset-manifest tables. Each corpus preparer is registered
under a dataset name and exposed through ``hyperion-prepare-data``.

Base extension point
--------------------

.. autoclass:: hyperion.data_prep.data_prep.DataPrep
   :no-index:
   :members: dataset_name, add_class_args, get_recording_duration

All preparers share ``corpus_dir`` and ``output_dir`` inputs and can optionally
set a target sample frequency. Corpus-specific subclasses add only the inputs
needed to locate that corpus's audio and annotations.

Use a registered preparer
-------------------------

Discover names and their version-specific requirements from the CLI:

.. code-block:: bash

   hyperion-prepare-data --help
   hyperion-prepare-data voxceleb1 --help

The output should be inspected as CSV manifests before training: recording and
segment ids, storage paths, speaker labels, durations, and sample rates must
align. See :doc:`how-to/prepare-data-and-vad` for the operational workflow.

Add a new preparer
------------------

A new preparer subclasses ``DataPrep``, implements a unique ``dataset_name``,
and implements its corpus-specific preparation flow. Subclass registration is
automatic. Keep corpus parsing in that module and write standard CSV manifests
instead of embedding corpus-specific behavior in training commands.

See also
--------

* :doc:`info_tables`
* :doc:`how-to/prepare-data-and-vad`
* :doc:`documentation-policy`

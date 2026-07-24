Data Preparation and Manifest Commands
======================================

These commands create and inspect the CSV metadata that drives the rest of
Hyperion. They do not download corpora or train models.

Prepare a supported corpus
--------------------------

``hyperion-prepare-data`` dispatches to a registered ``DataPrep`` subclass.
Discover the selected corpus arguments first:

.. code-block:: bash

   hyperion-prepare-data --help
   hyperion-prepare-data voxceleb1 --help

Use ``--cfg`` for repeatable corpus paths and output locations. The output
directory contains CSV recording, segment, class, and related manifests. Check
that ids, storage paths, sample rates, timing, and speaker labels are valid
before training.

Dataset and table operations
----------------------------

``hyperion-dataset`` and ``hyperion-tables`` operate on existing Hyperion
metadata. ``hyperion-split-dataset-into-trials-and-cohort`` derives trial and
cohort partitions from a prepared dataset. Their artifacts are CSV manifests
and trial tables; preserve ids exactly when passing them to extraction/scoring.

See also
--------

* :doc:`../how-to/prepare-data-and-vad`
* :doc:`../foundation-api-contracts`
* :doc:`../data_prep`

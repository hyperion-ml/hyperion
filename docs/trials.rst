Working With Trial Tables
=========================

.. currentmodule:: hyperion.utils

Overview
--------

Hyperion uses a small family of trial-oriented classes to represent speaker
recognition evaluation data in a consistent way:

* :class:`TrialNdx` defines which trials exist.
* :class:`TrialKey` defines the ground-truth label of each trial.
* :class:`TrialScores` stores the scores produced by a system.
* :class:`SparseTrialNdx`, :class:`SparseTrialKey`, and
  :class:`SparseTrialScores` provide sparse equivalents based on
  ``scipy.sparse``.

All six classes share the same conceptual layout:

* rows correspond to enrollment models
* columns correspond to test segments
* each matrix entry refers to one ``(model, segment)`` trial

This tutorial explains how the classes relate to each other, when to use the
dense or sparse variants, and how to move through a typical evaluation
workflow.

Core Concepts
-------------

The three dense classes represent different stages of the same trial pipeline.

``TrialNdx``: which trials should be evaluated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`TrialNdx` when you only need the list of trials to run, without
target or non-target labels yet.

It contains:

* ``model_set``: model identifiers
* ``seg_set``: test segment identifiers
* ``trial_mask``: boolean matrix indicating valid trials

.. code-block:: python

   import numpy as np
   from hyperion.utils import TrialNdx

   ndx = TrialNdx(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2", "utt3"],
       trial_mask=np.array(
           [
               [True, False, True],
               [False, True, True],
           ],
           dtype=bool,
       ),
   )

   print(ndx.num_models)   # 2
   print(ndx.num_tests)    # 3

If you omit ``trial_mask`` in the dense class, it defaults to an all-``True``
matrix with shape ``(num_models, num_tests)``.

``TrialKey``: which trials are target, non-target, or spoof
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`TrialKey` when you know the ground-truth label of each trial.

It contains:

* ``tar``: boolean target mask
* ``non``: boolean non-target mask
* ``spoof``: optional boolean spoof mask
* optional condition matrices:
  ``model_cond``, ``seg_cond``, ``trial_cond``

.. code-block:: python

   import numpy as np
   from hyperion.utils import TrialKey

   key = TrialKey(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2", "utt3"],
       tar=np.array(
           [
               [True, False, False],
               [False, True, False],
           ],
           dtype=bool,
       ),
       non=np.array(
           [
               [False, True, True],
               [True, False, True],
           ],
           dtype=bool,
       ),
   )

   ndx_from_key = key.to_ndx()
   print(ndx_from_key.trial_mask.astype(int))

The masks must not overlap. A trial cannot be target and non-target at the
same time.

``TrialScores``: which trials have scores, and what those scores are
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`TrialScores` to store the output of a recognizer.

It contains:

* ``scores``: floating-point score matrix
* ``score_mask``: boolean mask indicating which scores are present
* optional ``q_measures``: additional quality-measure matrices

.. code-block:: python

   import numpy as np
   from hyperion.utils import TrialScores

   scores = TrialScores(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2", "utt3"],
       scores=np.array(
           [
               [3.2, -0.7, -1.1],
               [-0.9, 2.8, 0.4],
           ],
           dtype=np.float32,
       ),
       score_mask=np.array(
           [
               [True, True, True],
               [True, True, True],
           ],
           dtype=bool,
       ),
   )

When ``score_mask[i, j]`` is ``False``, the numeric value in ``scores[i, j]``
is ignored.

Dense vs Sparse
---------------

The sparse variants follow the same semantics but store masks and score
matrices as ``scipy.sparse`` matrices.

Use the dense classes when:

* most trials are present
* you want HDF5 support
* you need dense-only features such as ``q_measures`` in :class:`TrialScores`

Use the sparse classes when:

* the trial matrix is large and mostly empty
* you want to avoid dense ``num_models x num_tests`` allocations
* you are working with a small subset of a very large Cartesian product

Important implementation details:

* :class:`SparseTrialNdx` requires an explicit ``trial_mask``.
* Sparse classes support text and table formats, but sparse HDF5 I/O is not
  implemented.
* :class:`SparseTrialKey.merge` is currently not implemented.
* :class:`SparseTrialScores` does not expose ``q_measures``.

File Formats
------------

All three dense classes support ``.h5``/``.hdf5``, plain text, and table
formats such as ``.csv`` and ``.tsv``. The sparse classes currently support
plain text and table formats.

``TrialNdx`` table format
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   modelid,segmentid
   spk1,utt1
   spk1,utt3
   spk2,utt2

``TrialKey`` table format
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   modelid,segmentid,targettype
   spk1,utt1,target
   spk1,utt2,nontarget
   spk2,utt3,spoof

Valid ``targettype`` values are:

* ``target``
* ``nontarget``
* ``spoof``

``TrialScores`` table format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   modelid,segmentid,LLR
   spk1,utt1,3.2
   spk1,utt2,-0.7
   spk2,utt3,0.4

Dense :class:`TrialScores` may also include extra columns after ``LLR``. Those
columns are loaded as entries of ``q_measures``.

Loading and Saving
------------------

Dense examples
~~~~~~~~~~~~~~

.. code-block:: python

   from hyperion.utils import TrialNdx, TrialKey, TrialScores

   ndx = TrialNdx.load("data/eval.ndx.csv")
   key = TrialKey.load("data/eval.key.csv")
   scores = TrialScores.load("exp/system1_scores.csv")

   ndx.save("tmp/eval.ndx.h5")
   key.save("tmp/eval.key.txt")
   scores.save("tmp/eval_scores.tsv")

Sparse examples
~~~~~~~~~~~~~~~

.. code-block:: python

   from hyperion.utils import SparseTrialNdx, SparseTrialKey, SparseTrialScores

   ndx = SparseTrialNdx.load("data/eval.ndx.csv")
   key = SparseTrialKey.load("data/eval.key.csv")
   scores = SparseTrialScores.load("exp/system1_scores.csv")

   ndx.save("tmp/eval_sparse.ndx.txt")
   key.save("tmp/eval_sparse.key.csv")
   scores.save("tmp/eval_sparse_scores.tsv")

Basic Manipulation
------------------

The three dense classes share a common workflow vocabulary:

* ``sort()``
* ``filter(...)``
* ``filter_by_model(...)``
* ``split(...)``
* ``merge(...)``
* ``copy()``

Sorting
~~~~~~~

.. code-block:: python

   ndx.sort()
   key.sort()
   scores.sort()

This sorts by model ids and then by segment ids, and permutes the internal
matrices accordingly.

Filtering
~~~~~~~~~

.. code-block:: python

   subset_ndx = ndx.filter(["spk1"], ["utt1", "utt3"], keep=True)
   subset_key = key.filter(["spk1"], ["utt1", "utt3"], keep=True)
   subset_scores = scores.filter(["spk1"], ["utt1", "utt3"], keep=True)

   # Remove a group of models instead of keeping them.
   pruned_scores = scores.filter_by_model(["spk1"], keep=False)

If ``raise_missing=False``, the score classes can align to requested model or
segment lists even when some entries are missing from the current object.

Splitting and merging
~~~~~~~~~~~~~~~~~~~~~

Splitting is useful when you want to process a large evaluation in blocks.

.. code-block:: python

   part_11 = scores.split(1, 2, 1, 2)
   part_12 = scores.split(1, 2, 2, 2)
   part_21 = scores.split(2, 2, 1, 2)
   part_22 = scores.split(2, 2, 2, 2)

   merged = TrialScores.merge([part_11, part_12, part_21, part_22])
   assert merged == scores

The same pattern applies to :class:`TrialNdx` and :class:`TrialKey`.

Converting Between ``TrialKey`` and ``TrialNdx``
------------------------------------------------

If you have a key and only need to know which trials exist, convert it to an
index:

.. code-block:: python

   ndx = key.to_ndx()

This is often useful when a downstream stage does not care about target or
non-target labels but still needs the trial list.

If you use sparse storage:

.. code-block:: python

   import scipy.sparse as sparse
   from hyperion.utils import SparseTrialNdx

   sparse_ndx = SparseTrialNdx(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2", "utt3"],
       trial_mask=sparse.csr_matrix(
           [
               [1, 0, 1],
               [0, 1, 1],
           ],
           dtype=bool,
       ),
   )

   dense_ndx = sparse_ndx.to_trial_ndx()
   sparse_again = SparseTrialNdx.from_trial_ndx(dense_ndx)

Working With Scores and Keys
----------------------------

The most common evaluation workflow is:

1. load or construct a :class:`TrialKey`
2. load or construct a :class:`TrialScores`
3. align scores with the key
4. extract target and non-target scores

Aligning scores to a key or index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   aligned = scores.align_with_ndx(key)

The aligned object:

* has the same ``model_set`` and ``seg_set`` ordering as the key
* keeps only scores for valid key trials
* raises an exception if some required trials are missing, unless
  ``raise_missing=False``

You can also align scores to a plain :class:`TrialNdx`:

.. code-block:: python

   aligned = scores.align_with_ndx(ndx)

Extracting target and non-target scores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   tar, non = scores.get_tar_non(key)
   print(tar.shape)
   print(non.shape)

If spoof trials are present:

.. code-block:: python

   tar, non, spoof = scores.get_tar_non_spoof(key)

This is the standard representation expected by many downstream metrics and
calibration pipelines.

Handling missing trials
~~~~~~~~~~~~~~~~~~~~~~~

If you want to preserve the requested trial structure and replace missing
scores with a fixed value:

.. code-block:: python

   completed = scores.set_missing_to_value(key, val=-100.0)

That returns a new :class:`TrialScores` where all trials requested by ``key``
exist in ``score_mask``.

Transforming scores
~~~~~~~~~~~~~~~~~~~

Dense :class:`TrialScores` provides an in-place transformation helper:

.. code-block:: python

   scores.transform(lambda x: 0.5 * x)

Only entries marked as valid in ``score_mask`` are transformed.

Quality Measures
----------------

Dense :class:`TrialScores` can optionally store trial-level quality measures in
``q_measures``.

.. code-block:: python

   import numpy as np
   from hyperion.utils import TrialKey, TrialScores

   key = TrialKey(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2"],
       tar=np.array([[True, False], [False, True]], dtype=bool),
       non=np.array([[False, True], [True, False]], dtype=bool),
   )

   scores = TrialScores(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2"],
       scores=np.array([[2.5, -0.2], [-0.8, 1.7]], dtype=np.float32),
       score_mask=np.ones((2, 2), dtype=bool),
       q_measures={
           "duration": np.array([[3.1, 2.8], [3.1, 2.8]], dtype=np.float32),
           "snr": np.array([[18.0, 11.5], [18.0, 11.5]], dtype=np.float32),
       },
   )

   tar_q, non_q = scores.get_tar_non_q_measures(key, q_names=["duration", "snr"])

If you save this object as CSV or TSV, the quality-measure arrays are written
as extra columns after ``LLR`` and are loaded back automatically.

Sparse Workflows
----------------

The sparse classes are best thought of as storage-optimized versions of the
dense ones.

Sparse key example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import scipy.sparse as sparse
   from hyperion.utils import SparseTrialKey

   key = SparseTrialKey(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2", "utt3"],
       tar=sparse.csr_matrix(
           [
               [1, 0, 0],
               [0, 1, 0],
           ],
           dtype=bool,
       ),
       non=sparse.csr_matrix(
           [
               [0, 1, 1],
               [1, 0, 1],
           ],
           dtype=bool,
       ),
   )

Sparse score example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import scipy.sparse as sparse
   from hyperion.utils import SparseTrialKey, SparseTrialScores

   sparse_key = SparseTrialKey(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2", "utt3"],
       tar=sparse.csr_matrix(
           [
               [1, 0, 0],
               [0, 1, 0],
           ],
           dtype=bool,
       ),
       non=sparse.csr_matrix(
           [
               [0, 1, 1],
               [1, 0, 1],
           ],
           dtype=bool,
       ),
   )

   scores = SparseTrialScores(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2", "utt3"],
       scores=sparse.csr_matrix(
           [
               [3.2, -0.7, -1.1],
               [-0.9, 2.8, 0.4],
           ],
           dtype=np.float32,
       ),
       score_mask=sparse.csr_matrix(
           [
               [1, 1, 1],
               [1, 1, 1],
           ],
           dtype=bool,
       ),
   )

   tar, non = scores.get_tar_non(sparse_key)

The sparse score API mirrors the dense one for the main operations:

* ``filter``
* ``split``
* ``merge``
* ``align_with_ndx``
* ``get_tar_non``
* ``get_tar_non_spoof``
* ``set_missing_to_value``

Dense-to-sparse conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from hyperion.utils import (
       SparseTrialKey,
       SparseTrialScores,
       TrialKey,
       TrialScores,
   )

   dense_key = TrialKey(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2"],
       tar=np.array([[True, False], [False, True]], dtype=bool),
       non=np.array([[False, True], [True, False]], dtype=bool),
   )
   dense_scores = TrialScores(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2"],
       scores=np.array([[2.5, -0.2], [-0.8, 1.7]], dtype=np.float32),
       score_mask=np.ones((2, 2), dtype=bool),
   )

   sparse_key = SparseTrialKey.from_trial_key(dense_key)
   sparse_scores = SparseTrialScores.from_trial_scores(dense_scores)

   dense_scores_roundtrip = sparse_scores.to_trial_scores()

Advanced Utilities
------------------

Preparing evaluation subsets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TrialNdx.parse_eval_set`` and ``SparseTrialNdx.parse_eval_set`` help prepare
the trial list for different enrollment/cohort/test evaluation modes:

* ``enroll-test``
* ``enroll-coh``
* ``coh-test``
* ``coh-coh``

These methods are used when evaluation recipes need to derive new trial
structures from enrollment and cohort metadata without rebuilding the whole
object manually.

Applying segmentation to test recordings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the test side is defined at the recording level and you later expand it to
subsegments, :class:`TrialNdx` and :class:`SparseTrialNdx` provide
``apply_segmentation_to_test(...)``.

That is particularly useful in diarization or tracking-style scenarios where a
single test recording is replaced by many smaller evaluation segments.

Practical Recommendations
-------------------------

Use :class:`TrialNdx` when:

* you only need trial existence
* no labels are available yet
* a dense matrix is acceptable

Use :class:`TrialKey` when:

* you need target and non-target labels
* you will compute metrics from labeled scores
* you need optional trial, model, or segment conditions

Use :class:`TrialScores` when:

* you already have system outputs
* you need score alignment, score extraction, or score transformation
* you want to store extra quality measures

Use the sparse variants when:

* trial matrices are very large and mostly empty
* you want the same high-level operations without dense allocations

End-to-End Example
------------------

The following example shows the most common dense workflow.

.. code-block:: python

   import numpy as np
   from hyperion.utils import TrialKey, TrialScores

   key = TrialKey(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2", "utt3"],
       tar=np.array(
           [
               [True, False, False],
               [False, True, False],
           ],
           dtype=bool,
       ),
       non=np.array(
           [
               [False, True, True],
               [True, False, True],
           ],
           dtype=bool,
       ),
   )

   scores = TrialScores(
       model_set=["spk1", "spk2"],
       seg_set=["utt1", "utt2", "utt3"],
       scores=np.array(
           [
               [4.1, -0.5, -1.0],
               [-0.8, 3.7, 0.2],
           ],
           dtype=np.float32,
       ),
       score_mask=np.ones((2, 3), dtype=bool),
   )

   aligned = scores.align_with_ndx(key)
   tar, non = aligned.get_tar_non(key)

   print("target mean:", tar.mean())
   print("nontarget mean:", non.mean())

This pattern covers the majority of scoring, calibration, and metric
evaluation code built on top of Hyperion trial tables.

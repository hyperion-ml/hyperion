NumPy Stack
===========

Overview
--------

The ``hyperion.np`` package contains the NumPy-based model and metric stack used
for classical/statistical modeling and evaluation.

Base abstractions
-----------------

.. autoclass:: hyperion.np.HyperNPModel

.. autoclass:: hyperion.np.NPModelLoader

Package organization
--------------------

Primary subpackages:

* ``hyperion.np.augment``
* ``hyperion.np.calibration``
* ``hyperion.np.classifiers``
* ``hyperion.np.clustering``
* ``hyperion.np.diarization``
* ``hyperion.np.feats``
* ``hyperion.np.metrics``
* ``hyperion.np.pdfs``
* ``hyperion.np.preprocessing``
* ``hyperion.np.score_norm``
* ``hyperion.np.transforms``

Feature Tutorials
-----------------

.. toctree::
   :maxdepth: 1

   np/mfcc
   np/transforms

Probability density models
--------------------------

.. automodule:: hyperion.np.pdfs
   :members:

Classifiers and calibration
---------------------------

.. automodule:: hyperion.np.classifiers
   :members:

.. automodule:: hyperion.np.calibration
   :members:

Score normalization
-------------------

.. automodule:: hyperion.np.score_norm
   :members:

Transforms and preprocessing
----------------------------

.. automodule:: hyperion.np.transforms
   :members:

.. automodule:: hyperion.np.preprocessing
   :members:

Clustering and diarization helpers
----------------------------------

.. automodule:: hyperion.np.clustering
   :members:

.. automodule:: hyperion.np.diarization
   :members:

Metrics functions
-----------------

NumPy metric functions are exposed from ``hyperion.np.metrics``.

.. automodule:: hyperion.np.metrics
   :members:

See Also
--------

* :doc:`np/mfcc`
* :doc:`np/transforms`
* :doc:`metrics`
* :doc:`np/speech_augmentation`

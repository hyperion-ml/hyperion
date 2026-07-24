Save and Load Models and Backends
=================================

Hyperion uses separate serialization paths for PyTorch models and NumPy
backends. Preserve the complete inference chain—not only an embedding-model
checkpoint—when recording or deploying a verification system.

Load a PyTorch x-vector model
-----------------------------

``HyperTorchModel.auto_load`` restores the serialized class/configuration and
parameters. It maps weights to CPU by default, which is the safest way to load
a checkpoint before explicitly selecting an inference device:

.. code-block:: python

   import torch

   from hyperion.torch import HyperTorchModel

   model = HyperTorchModel.auto_load("exp/xvector/model.pth")
   model.eval()
   model.to(torch.device("cuda"))

Use ``eval()`` for extraction or scoring. Loading a checkpoint does not imply
evaluation mode. Keep the checkpoint together with the resolved
``config.yaml`` written by the trainer and the class CSV that defined the
classifier targets.

Load NumPy preprocessing and PLDA
---------------------------------

PLDA and transform lists use Hyperion's NumPy/HDF5-style serialization:

.. code-block:: python

   from hyperion.np import HyperNPModel
   from hyperion.np.transforms import TransformList

   preprocessor = TransformList.load("exp/backend/preproc.h5")
   plda = HyperNPModel.auto_load("exp/backend/plda.h5")

Apply the loaded preprocessor before the PLDA backend. The transform was fit
on development embeddings and must not be refit on evaluation or deployment
audio.

Deployment manifest
-------------------

Store these artifacts as one versioned release:

* waveform x-vector checkpoint;
* resolved training configuration;
* class inventory CSV and label interpretation;
* extraction configuration, including sample rate, VAD policy, and embedding
  layer when non-default;
* preprocessing transform and PLDA model, when used;
* calibration model and its target prior, when used;
* package version and any external pretrained-model identifier or local copy.

This manifest makes it possible to reproduce an embedding and score months
later, and prevents accidental mixing of an x-vector checkpoint with a backend
trained for a different embedding space.

Compatibility and safety
------------------------

Stable components follow the compatibility expectations in
:doc:`../documentation-policy`; preserve their configuration and serialized
format when upgrading. Codec, VITS anonymization, transducer, and q-vector
components are experimental, so retain the exact package revision and test a
round-trip load before deployment.

Only load checkpoints and backend files from trusted sources. Serialized model
files can execute deserialization logic and should be treated as code-bearing
artifacts rather than untrusted data uploads.

See also
--------

* :doc:`use-configuration-files`
* :doc:`extract-score-xvectors`
* :doc:`../documentation-policy`

Repository Architecture
=======================

Top-level layout
----------------

Core source code is in ``hyperion/``:

* ``hyperion/np``: NumPy models and metric/util function stacks.
* ``hyperion/torch``: PyTorch stack (layers, architectures, models, training).
* ``hyperion/io``: unified data/audio IO abstractions.
* ``hyperion/utils``: tables, dataset manifests, trial/key/score tooling.
* ``hyperion/data_prep``: dataset preparation classes.
* ``hyperion/text_norm``: text normalization utilities.
* ``hyperion/metrics``: high-level evaluator classes.
* ``hyperion/bin``: executable scripts exposed as package entry points.

Supporting folders:

* ``docs/``: Sphinx documentation.
* ``tests/``: unit/integration tests.

NumPy stack
-----------

``hyperion.np`` contains the NumPy-based modeling/evaluation components.

The base class is:

.. autoclass:: hyperion.np.HyperNPModel
   :no-index:

Major subpackages include:

* ``hyperion.np.pdfs`` (PLDA/GMM and related density models)
* ``hyperion.np.classifiers``
* ``hyperion.np.transforms``
* ``hyperion.np.score_norm``
* ``hyperion.np.metrics``

PyTorch stack
-------------

The PyTorch stack is layered:

* ``hyperion.torch.layers``: primitive layers.
* ``hyperion.torch.layer_blocks``: reusable blocks composed from layers.
* ``hyperion.torch.narchs``: neural architectures composed from blocks/layers.
* ``hyperion.torch.models``: top-level models composed from architectures.

Base classes:

.. autoclass:: hyperion.torch.HyperTorchModel
   :no-index:

.. autoclass:: hyperion.torch.narchs.net_arch.NetArch
   :no-index:

Training and data
-----------------

Training/data-related packages live under ``hyperion.torch``:

* ``hyperion.torch.data``: datasets and sampler factories.
* ``hyperion.torch.trainers``: trainer implementations.
* ``hyperion.torch.lr_schedulers`` and ``hyperion.torch.wd_schedulers``.

Current canonical trainer foundation is:

.. autoclass:: hyperion.torch.trainers.torch_trainer_base.TorchTrainerBase
   :no-index:

.. autoclass:: hyperion.torch.trainers.single_model_trainer.SingleModelTrainer
   :no-index:

Third-party wrappers (TPM)
--------------------------

``hyperion.torch.tpm`` provides wrappers for third-party models/toolkits,
including Hugging Face models, DNSMOS, UTMOS, and VoxProfile evaluators.

Metrics layering
----------------

Metrics/evaluation are split into three layers:

* ``hyperion.np.metrics``: NumPy metric functions.
* ``hyperion.torch.metrics``: torch metric utilities.
* ``hyperion.metrics``: high-level evaluator classes that can combine both.

CLI generation
--------------

``hyperion/bin`` scripts are converted to package entry points by
``generate_pyproject.py``.

Deprecated script directories ``hyperion/bin_deprec`` and ``hyperion/bin_deprec2``
are intentionally excluded from current docs.

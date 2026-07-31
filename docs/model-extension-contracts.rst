Model Extension Contracts
=========================

This guide defines the maintained contract for adding a serializable model to
``hyperion.np`` or ``hyperion.torch.models``. It complements the package
reference: use it before choosing constructor arguments, a configuration schema,
or an on-disk representation.

Choose the correct base class
-----------------------------

Derive a statistical model, transform, PDF, classifier, calibrator, or score
normalizer from :class:`hyperion.np.HyperNPModel`. Derive a trainable task model
that owns a PyTorch ``state_dict`` from
:class:`hyperion.torch.HyperTorchModel`. A reusable PyTorch architecture belongs
under ``narchs`` and is covered by the PyTorch extension workflow; it is not a
reason to create a second task-model serialization format.

Both base classes register subclasses by their Python class name at import
time. The saved ``class_name`` is therefore part of a stable model artifact
format. Pick a unique, durable class name, retain an import path that allows it
to be discovered, and do not rename it merely for style.

``HyperNPModel`` contract
-------------------------

NumPy constructor and configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``super().__init__(name=name, ...)`` and preserve ``name`` unless the
model has a documented reason not to. ``name`` determines the HDF5 parameter
group prefix. Override ``get_config()`` by extending—not replacing—the base
configuration; it must include every constructor setting required to recreate
the unfitted object and keep ``class_name`` and ``name``.

Configurations must be JSON serializable. Use Python scalars, strings, lists,
dictionaries, and supported NumPy scalar/one-dimensional string-array values.
Do not put open handles, generators, callables, device objects, fitted large
arrays, or a dependency on ambient process state in ``get_config()``.

NumPy parameters and loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``save(path)`` writes a ``config`` JSON dataset plus subclass parameters to an
HDF5-style file. Implement ``save_params(f)`` and ``load_params(f, config)``
when fitted state is not represented entirely in configuration. Use
``_save_params_from_dict`` and ``_load_params_to_dict`` to preserve the base
class's name prefix and configured save/load dtypes.

For example, a fitted transform commonly follows this shape:

.. code-block:: python

   class MyTransform(HyperNPModel):
       def __init__(self, floor=1e-6, mean=None, name=None):
           super().__init__(name=name)
           self.floor = floor
           self.mean = mean

       def get_config(self):
           return {**super().get_config(), "floor": self.floor}

       def save_params(self, f):
           self._save_params_from_dict(f, {"mean": self.mean})

       @classmethod
       def load_params(cls, f, config):
           params = cls._load_params_to_dict(f, config["name"], ["mean"])
           return cls(floor=config["floor"], mean=params["mean"],
                      name=config["name"])

``load(path)`` reconstructs a known concrete class. ``auto_load(path)`` reads
``class_name`` and resolves it through the registry; it bootstraps maintained
NumPy subpackages and can use an explicit ``extra_objs`` mapping for a plugin
class. New stable models should be importable from their maintained package
namespace so ordinary registry discovery works without caller-specific setup.

``.pkl`` is supported as a Python pickle path, but it serializes the full object
and is appropriate only for trusted, tightly version-controlled use. Prefer the
HDF5/config route for portable maintained artifacts. Never load either format
from an untrusted source.

``HyperTorchModel`` contract
----------------------------

PyTorch constructor and configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``super().__init__(...)`` before registering submodules. A task-model
constructor must accept exactly the JSON-friendly values returned by
``get_config()``. Extend the base result so the configuration retains
``class_name``:

.. code-block:: python

   def get_config(self):
       return {
           **super().get_config(),
           "encoder_cfg": self.encoder.get_config(),
           "embedding_dim": self.embedding_dim,
           "dropout_rate": self.dropout_rate,
       }

Nested reusable architectures should contribute their own configuration rather
than being saved as live ``nn.Module`` objects. If reconstruction requires a
loader/factory, override ``load`` to rebuild nested components first, then call
``load_state_dict``. ``AE.load`` is a maintained example of this pattern.

PyTorch checkpoint contents and loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``save(path)`` writes a PyTorch object with two keys:

* ``model_cfg``: the result of ``get_config()`` including ``class_name``;
* ``model_state_dict``: the standard PyTorch parameter/buffer state.

``load`` reconstructs a known class and uses normal strict
``load_state_dict`` behavior. ``auto_load`` reads ``model_cfg.class_name``,
resolves/imports the registered class, removes distributed ``module.`` key
prefixes, applies the maintained compatibility migrations, and then calls the
class loader. It loads to CPU by default; callers move the model and input
tensors to the selected device afterwards.

Do not change parameter names, buffer names, tensor shapes, class vocabulary,
embedding dimension, frontend configuration, or the meaning/default of a
serialized field without treating it as a checkpoint compatibility change.
``train_mode`` and ``bias_weight_decay`` are model-level behavior/configuration
inputs when used by the subclass and must likewise be represented consistently.

Registry discovery and imports
------------------------------

Registration occurs when Python executes a subclass definition. The base
``auto_load`` methods attempt to import maintained model packages and, if
necessary, scan their source trees for a matching class declaration. That is a
fallback, not an extension mechanism. For a new maintained model:

1. place it in the appropriate maintained package;
2. expose it from the package ``__init__.py`` when it is a public model;
3. ensure importing that package does not require a model asset, GPU, or network
   request; and
4. test ``auto_load`` in a fresh Python process, not only after the class was
   imported incidentally by the test.

Use ``extra_objs`` only for explicitly caller-supplied plugin classes. It does
not make a class name portable across a normal Hyperion installation.

Compatibility rules
-------------------

For stable models, add only backwards-compatible optional configuration fields
with defaults whenever possible. Keep previous ``class_name`` values and
parameter keys valid. When an incompatible migration is necessary, implement a
narrow, tested config/state migration in the loading path, retain a migration
note, and document the upgrade procedure. Existing examples in
``HyperTorchModel`` handle renamed x-vector config keys, changed feature-fuser
configuration, and parameter-to-buffer transitions.

Experimental model families may change more freely, but every checkpoint must
still record its exact configuration and package revision. Do not silently load
an incompatible experimental checkpoint as though it were equivalent.

Validation before a pull request
--------------------------------

At minimum, test all of the following for a new model:

* constructor/config round trip, including defaults and nested components;
* save/load and ``auto_load`` in a fresh process;
* fitted NumPy parameter round trip or PyTorch ``state_dict`` equality;
* representative forward/predict shape, dtype, device, and error contracts;
* old-artifact compatibility when modifying a stable serialized model; and
* relevant CLI/config parsing if the model is exposed through a command.

Document the model's public forward/predict contract, serialization behavior,
and a focused usage example. The full pre-PR check list is in
:doc:`building-documentation`; support and deprecation requirements are in
:doc:`documentation-policy`.

See also
--------

* :doc:`contributor-extension-guide`
* :doc:`numpy-extension-points`
* :doc:`torch-extension-points`
* :doc:`torch-api-contracts`
* :doc:`how-to/save-load-models-and-backends`

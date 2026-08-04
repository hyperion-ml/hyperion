Manage Modern PyTorch Checkpoints
=================================

Modern ``TorchTrainerBase`` trainers save resumable training state as a
checkpoint directory. This keeps the model configuration and tensor weights in
separate, inspectable files while retaining optimizer, scheduler, stochastic
weight averaging (SWA), progress, and random-number-generator state needed to
resume training. ``LegacyTorchTrainer`` and its subclasses keep their legacy
``.pth`` checkpoint behavior.

Prerequisites
-------------

This workflow requires a Hyperion installation with PyTorch and
``safetensors``. Conversion of a legacy checkpoint also requires access to the
trusted original ``.pth`` file.

Checkpoint layout
-----------------

Each save creates one directory named from its epoch and global step, for
example ``checkpoint_ep0005_step0000001000``. A single-model trainer writes:

.. code-block:: text

   exp/
     checkpoint_ep0005_step0000001000/
       trainer_state.json
       rng_state.pth
       model/
         config.json
         model.safetensors
         optimizer.pt
         lr_scheduler.pt              # when configured
         wd_scheduler.pt              # when configured
         swa_model.safetensors        # when SWA is enabled
         swa_scheduler.pt             # when configured

``trainer_state.json`` contains progress and logged state shared by every model
in the checkpoint. ``rng_state.pth`` holds the shared random-number-generator
state. Each model subdirectory contains its independent configuration, tensor
weights, optimizer, and scheduler state. Model configuration is JSON and model
and SWA weights are stored with ``safetensors``. Optimizer and scheduler state
remain PyTorch ``.pt`` files and are loaded with ``weights_only=True``.

Multi-model trainers use one common root, so generator and discriminator state
always belong to the same training point:

.. code-block:: text

   checkpoint_ep0005_step0000001000/
     trainer_state.json
     rng_state.pth
     dac_model/                      # DACTrainer
       config.json
       model.safetensors
       ...
     discrim_model/
       config.json
       model.safetensors
       ...

``FreeVCTrainer`` and ``VIAnonymizerTrainer`` use ``vc_model/`` and
``discrim_model/`` instead. The ``model/`` subdirectory is used by
``SingleModelTrainer`` and is the default name used by the conversion command.
Standalone final-SWA exports contain only the model ``config.json`` and
``model.safetensors`` files; they are inference artifacts rather than resumable
training checkpoints.

Resume and complete checkpoints
--------------------------------

``load_last_checkpoint()`` searches ``exp_path`` for the newest *complete*
checkpoint directory. It restores shared trainer state first, then the state
of every model subdirectory required by that trainer. A directory missing its
required root files is ignored rather than treated as a resumable checkpoint.
The required artifacts are ``trainer_state.json`` and ``rng_state.pth`` at the
root, plus ``config.json``, ``model.safetensors``, and ``optimizer.pt`` for
every model required by the active trainer. Scheduler and SWA artifacts remain
optional because their presence depends on the trainer configuration and the
checkpoint's SWA phase.

Checkpoint publication is atomic at directory level. The trainer writes the
root, every model subdirectory, and shared state to a temporary sibling
directory. Only after all artifacts have been written does it rename that
directory to ``checkpoint_ep..._step...``. An interrupted save therefore does
not replace a previous usable checkpoint or appear in normal resume discovery.

Load a standalone model
-----------------------

``HyperTorchModel.auto_load`` accepts both formats:

.. code-block:: python

   from hyperion.torch import HyperTorchModel

   legacy_model = HyperTorchModel.auto_load("exp/old_model.pth")
   modern_model = HyperTorchModel.auto_load(
       "exp/checkpoint_ep0005_step0000001000/model"
   )

Passing a file uses the legacy Hyperion ``.pth`` model format. Passing a model
directory loads its ``config.json`` and ``model.safetensors`` files. To load a
model from a multi-model training checkpoint, pass the appropriate model
subdirectory, such as ``.../dac_model`` or ``.../discrim_model``.

The Hugging Face waveform wrappers record ``ignore_pretrained: true`` after
their initial construction. Loading their saved model directory therefore
builds the architecture from its saved configuration and applies Hyperion's
saved weights without retrieving the original pretrained weights again.

Migrate a legacy model checkpoint
---------------------------------

Use ``hyperion-to-safetensors`` to turn a legacy model ``.pth`` file into a
modern model directory. By default it writes only the inference artifacts:

.. code-block:: bash

   hyperion-to-safetensors \
     --in-model-file exp/old_model.pth \
     --out-model-dir exp/checkpoint_ep0005_step0000001000

The default ``--model-name model`` produces
``checkpoint_ep0005_step0000001000/model/config.json`` and
``checkpoint_ep0005_step0000001000/model/model.safetensors``. Use
``--model-name`` to select a model subdirectory without repeating it in the
output path:

.. code-block:: bash

   hyperion-to-safetensors \
     --in-model-file exp/dac_ep0005_step0000001000.pth \
     --out-model-dir exp/checkpoint_ep0005_step0000001000 \
     --model-name dac_model

To make a converted checkpoint resumable by a modern trainer, request the
legacy trainer state too:

.. code-block:: bash

   hyperion-to-safetensors \
     --in-model-file exp/model_ep0005_step0000001000.pth \
     --out-model-dir exp/checkpoint_ep0005_step0000001000 \
     --get-trainer-state

For DAC, FreeVC, or VITS-style generator/discriminator checkpoints, invoke the
command once per legacy model with the same output directory and
``--get-trainer-state``. Use ``dac_model``/``discrim_model`` or
``vc_model``/``discrim_model`` as appropriate. Subsequent invocations verify
that their shared epoch, batch, step, logs, and RNG state agree before adding
another model directory.

Security note
-------------

Conversion intentionally reads the legacy ``.pth`` input with
``weights_only=False`` so it can recover the legacy pickled configuration and
trainer payload. PyTorch pickle deserialization can execute code: convert only
files from a trusted source. The new ``config.json`` and ``.safetensors`` model
artifacts avoid serializing model configuration with pickle.

See also
--------

* :doc:`run-resumable-distributed-training`
* :doc:`save-load-models-and-backends`
* :doc:`../torch-api-contracts`

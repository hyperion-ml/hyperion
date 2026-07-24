Run Resumable, Mixed-Precision, and Distributed Training
=========================================================

Hyperion waveform x-vector commands use the trainer's ``exp_path`` as the
durable experiment boundary. Keep one immutable configuration, class inventory,
and model architecture per experiment path.

Experiment artifacts and resume
-------------------------------

Set a unique path in the training configuration:

.. code-block:: yaml

   trainer:
     exp_path: exp/wav2xvector-resnet1d-v1

Training writes the resolved command configuration to ``config.yaml`` and
stores logs and checkpoints under that directory. The waveform training
commands call ``load_last_checkpoint()`` before fitting, so rerunning the same
command resumes from the latest compatible checkpoint automatically.

Do not reuse an experiment directory after changing the architecture,
classifier class ordering, optimizer family, or dataset split. Start a new
``exp_path`` instead. Preserve the experiment configuration, class CSV, and
checkpoint together when archiving a result.

Enable AMP
----------

Automatic mixed precision reduces GPU memory use and can improve throughput.
Enable it in the trainer section of the same configuration:

.. code-block:: yaml

   trainer:
     use_amp: true
     amp_dtype: float16

Use ``bfloat16`` only on hardware that supports it reliably. If loss scaling
or numerical stability becomes a problem, disable AMP first to establish a
full-precision baseline, then inspect learning rate, input validity, and model
outputs.

Launch single-node DDP
----------------------

For one process per GPU, launch the unchanged training command with
``torchrun``:

.. code-block:: bash

   torchrun --standalone --nproc_per_node=4 \
     -m hyperion.bin.train_wav2xvector \
     resnet1d --cfg configs/wav2xvector-resnet1d.yaml

The launcher provides ``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK``,
``MASTER_ADDR``, and ``MASTER_PORT``. Hyperion uses those values to choose the
local CUDA device and initialize NCCL. Do not manually assign a different GPU
inside the configuration for each rank.

The commands retain a ``--num-gpus`` option for compatibility, but ``torchrun``
is the source of truth for distributed world size. Use ``--master-port`` only
when the launcher environment needs an explicit port override.

Batch sizing and memory
-----------------------

The sampler controls waveform/chunk batching, so memory depends on the maximum
batch duration as well as the number of utterances. For out-of-memory errors:

1. Lower the sampler's maximum batch length or maximum batch size.
2. Enable AMP if appropriate.
3. Increase gradient accumulation rather than restoring an oversized physical
   batch.
4. Reduce model capacity only after establishing a safe data batch.

With DDP, each rank owns a separate data-loader process and receives a shard of
the workload. Set data-loader worker counts with total CPU capacity in mind;
the training command divides configured workers across GPU ranks.

Troubleshooting
---------------

* **Resume fails or metrics jump unexpectedly:** verify that the checkpoint and
  current configuration use the same architecture and class CSV.
* **NCCL initialization fails:** use ``torchrun`` and ensure every local rank
  can access its assigned CUDA device.
* **One rank exits early:** inspect logs from all ranks; malformed audio or a
  sampler/data imbalance can surface on only one shard.
* **GPU memory grows over time:** check chunk/batch limits, augmentation, and
  logging hooks before lowering the embedding dimension.

See also
--------

* :doc:`train-waveform-xvector`
* :doc:`../torch`

PyTorch Training Support
========================

This reference covers the reusable pieces that sit between a task model and a
trainer: classification output layers, metrics, loggers, and learning-rate or
weight-decay schedules. Configure these through the existing trainer and
factory interfaces; avoid duplicating their update loops in a model.

Classification output layers
----------------------------

.. autoclass:: hyperion.torch.layers.margin_losses.ArcLossOutput
   :no-index:
   :members: prototypes, update_margin, forward

.. autoclass:: hyperion.torch.layers.margin_losses.CosLossOutput
   :no-index:
   :members: prototypes, update_margin, forward

.. autoclass:: hyperion.torch.layers.margin_losses.SubCenterArcLossOutput
   :no-index:

These layers consume embeddings shaped ``(batch, in_feats)`` and return logits
shaped ``(batch, num_classes)``. During training, pass class labels so the
margin penalty applies to the target class. Margin warmup is part of the model
state: on resume, use the trainer's restored epoch/step state rather than
resetting the output layer schedule.

For a changed speaker inventory, use the task model's documented output-layer
rebuild method. Do not load a classifier head across incompatible class-id
mappings without an explicit migration policy.

Metrics and loggers
-------------------

.. autoclass:: hyperion.torch.metrics.metrics.TorchMetric
   :no-index:

.. autoclass:: hyperion.torch.loggers.logger.Logger
   :no-index:
   :members: on_train_begin, on_epoch_begin, on_batch_begin, on_model_update, on_batch_end, on_val_end, on_epoch_end, on_train_end

.. autoclass:: hyperion.torch.loggers.logger_list.LoggerList
   :no-index:
   :members: append

``TorchMetric`` subclasses are per-batch training measurements; they are not a
replacement for the trial-based verification metrics in :doc:`metrics`. Logger
callbacks receive a shared ``logs`` mapping through the displayed lifecycle.
Use ``LoggerList`` to fan the events out to CSV, progress, TensorBoard, or
Weights & Biases loggers. In distributed training, ensure a logger's output
location and rank behavior are intentional before enabling it.

Gradient accumulation
---------------------

Trainers based on ``TorchTrainerBase`` intentionally take optimizer steps only
after a full ``grad_acc_steps`` window. A partial window at the end of an epoch
can still contribute to training metrics. Its gradients are discarded before
the next epoch. For example, five microbatches with ``grad_acc_steps=2`` yield
two optimizer steps. An epoch with fewer microbatches than ``grad_acc_steps``
yields zero optimizer steps.

cuDNN convolution benchmarking
------------------------------

Trainers based on ``TorchTrainerBase`` accept ``cudnn_benchmark=True`` or the
CLI flag ``--trainer.cudnn-benchmark``. The option defaults to false. Enable
it to have cuDNN benchmark convolution algorithms during training.

Schedulers and checkpoints
--------------------------

.. autoclass:: hyperion.torch.lr_schedulers.lr_scheduler.LRScheduler
   :no-index:
   :members: state_dict, load_state_dict, on_epoch_begin, on_epoch_end, on_opt_step

.. autoclass:: hyperion.torch.wd_schedulers.wd_scheduler.WDScheduler
   :no-index:
   :members: state_dict, load_state_dict, on_epoch_begin, on_epoch_end, on_opt_step

Learning-rate and weight-decay schedulers maintain both epoch and optimizer-step
counters. ``update_lr_on_opt_step`` and ``update_wd_on_opt_step`` determine the
time axis of a schedule; warmup is measured in optimizer steps. Store scheduler
state in the same checkpoint as the optimizer and trainer, otherwise resuming
can silently change the effective schedule.

Trainers based on ``TorchTrainerBase`` save ``next_batch`` as the number of
batches to skip from the start of the recorded epoch. Mid-epoch saves happen
at optimizer-step boundaries and require a batch sampler that can resume at
a given offset. A stop at ``max_steps`` keeps the epoch and batch position
for a later run with a higher limit. Epoch-end validation and scheduler hooks
run when training reaches the end of the loader. The epoch-end checkpoint
records the next epoch with ``next_batch=0``. Trainer resume requires a
checkpoint with the ``next_batch`` field.

Available learning-rate families include exponential, inverse-power, Noam,
cosine, triangular, Adam cosine, and reduce-on-plateau. Weight-decay scheduling
currently provides cosine scheduling. Select them via the factories in
:doc:`torch-extension-points`.

In distributed training, ``ReduceLROnPlateau`` shares its monitored epoch
metric from rank 0. Every rank uses that value to update its scheduler and
optimizer learning rates.

Conditional parameters and DDP
-------------------------------

:meth:`hyperion.torch.HyperTorchModel.requires_ddp_find_unused_parameters` returns False by
default. ``TransformerEncoderV2`` and ``QFormerV2`` return True when
``ff_type="g4moe"``, because experts not selected in a batch have no gradients.
The ordinary DDP paths in ``TorchTrainerBase`` and ``legacy_torch_trainer``
query this method on the top-level task model and pass its result to
``DistributedDataParallel(find_unused_parameters=...)``.

A task model containing a conditional encoder must delegate explicitly, for
example in its subclass::

    def requires_ddp_find_unused_parameters(self):
        return self.encoder.requires_ddp_find_unused_parameters()

If it contains multiple conditional branches, return the logical OR of their
requirements. The trainer does not recursively inspect modules. Enabling the
flag incurs DDP graph-traversal overhead; ordinary models retain False. This
hook configures ordinary DDP only and does not establish MoE FSDP or expert
parallelism support. See :doc:`torch-layers-and-architectures` for routing and
tensor-parallel constraints.

See also
--------

* :doc:`torch-extension-points`
* :doc:`how-to/run-resumable-distributed-training`
* :doc:`how-to/use-configuration-files`

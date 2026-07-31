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

Available learning-rate families include exponential, inverse-power, Noam,
cosine, triangular, Adam cosine, and reduce-on-plateau. Weight-decay scheduling
currently provides cosine scheduling. Select them via the factories in
:doc:`torch-extension-points`.

See also
--------

* :doc:`torch-extension-points`
* :doc:`how-to/run-resumable-distributed-training`
* :doc:`how-to/use-configuration-files`

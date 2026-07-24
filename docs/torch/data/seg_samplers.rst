Sequential Samplers
===================

The sequential samplers in ``hyperion.torch.data`` create batches from a
``SegmentSet``. They support distributed training, reproducible epoch-level
sampling, and batching constraints based on segment duration.

Sampler selection
-----------------

``SegSamplerFactory`` is the main entry point for selecting a sequential
sampler. The factory receives a dataset with a ``segments`` attribute and
constructs the requested sampler from the supplied configuration.

.. autoclass:: hyperion.torch.data.seg_sampler_factory.SegSamplerFactory
   :no-index:
   :members:
   :show-inheritance:

Factory usage
~~~~~~~~~~~~~

The factory uses the dataset's ``segments`` table and forwards the sampler
configuration to the selected implementation:

.. code-block:: python

   from hyperion.torch.data import SegSamplerFactory

   sampler = SegSamplerFactory.create(
       dataset,
       sampler_type="seg_sampler",
       min_batch_size=16,
       max_batch_length=240.0,
       shuffle=True,
       seed=1234,
   )

   for segment_ids in sampler:
       train_batch(segment_ids)

For chunk sampling, select a chunk sampler and provide the chunk parameters:

.. code-block:: python

   sampler = SegSamplerFactory.create(
       dataset,
       sampler_type="random_seg_chunk_sampler",
       min_chunk_length=2.0,
       max_chunk_length=4.0,
       min_batch_size=32,
       max_batch_length=128.0,
       shuffle=True,
   )

When using a class-weighted sampler, the dataset must provide the matching
``ClassInfo`` table in ``dataset.class_info``:

.. code-block:: python

   sampler = SegSamplerFactory.create(
       dataset,
       sampler_type="class_weighted_random_seg_chunk_sampler",
       class_name="speaker",
       num_segs_per_class=2,
       num_chunks_per_seg=1,
       min_chunk_length=2.0,
       weight_mode="data-prior",
   )

The available sampler types are:

``seg_sampler``
    Samples fixed-size batches directly from the segment table. Set
    ``max_batch_length`` to enable duration-constrained variable-size batches.

``random_seg_chunk_sampler``
    Samples source segments and valid chunk starts on demand, without creating a
    table containing every possible chunk. Sampling is with replacement, so it
    does not guarantee coverage of every segment or chunk within an epoch.

``class_weighted_random_seg_chunk_sampler``
    Samples classes according to class weights and then samples segments and
    chunks for the selected classes. It also supports optional hard-prototype
    sampling.

``seg_chunk_sampler``
    Wraps a base sampler and exposes chunked segment metadata.

``bucketing_seg_sampler``
    Groups segments by length before applying a base sampler to reduce padding
    variation between items in a batch.

Common behavior
---------------

All samplers derive from ``HyperSampler`` and expose the following common
configuration concepts:

* ``shuffle`` changes the deterministic sampling seed for each epoch.
* ``seed`` controls reproducibility across runs and distributed ranks.
* ``max_batches_per_epoch`` is a global batch cap; ``HyperSampler`` divides it
  across distributed ranks.
* ``set_epoch(epoch, batch=...)`` supports resuming from a batch offset.

The fixed-size sampler uses ``min_batch_size`` and ``max_batch_size``. When
``max_batch_length`` is set, it packs items while respecting the maximum padded
length. ``sample_all_segments`` can be used when every segment must be covered
at least once per epoch.

Direct sampler usage
~~~~~~~~~~~~~~~~~~~~

Use ``SegSampler`` directly when the sampler configuration is already known
and no factory dispatch is needed:

.. code-block:: python

   from hyperion.torch.data.seg_sampler import SegSampler

   sampler = SegSampler(
       segments,
       min_batch_size=8,
       max_batch_length=120.0,
       shuffle=True,
   )

   for segment_ids in sampler:
       # segment_ids is a NumPy array, or chunk tuples for chunked input.
       batch = dataset[segment_ids]

To resume a sampler at a checkpointed epoch and batch, set the epoch before
creating the next iterator:

.. code-block:: python

   sampler.set_epoch(epoch=4, batch=25)
   for segment_ids in sampler:
       train_batch(segment_ids)

Core sampler APIs
-----------------

.. autoclass:: hyperion.torch.data.seg_sampler.SegSampler
   :members:
   :show-inheritance:

.. autoclass:: hyperion.torch.data.random_seg_chunk_sampler.RandomSegChunkSampler
   :members:
   :show-inheritance:

.. autoclass:: hyperion.torch.data.class_weighted_seg_chunk_sampler.ClassWeightedRandomSegChunkSampler
   :members:
   :show-inheritance:

.. autoclass:: hyperion.torch.data.seg_chunk_sampler.SegChunkSampler
   :members:
   :show-inheritance:

.. autoclass:: hyperion.torch.data.bucketing_seg_sampler.BucketingSegSampler
   :members:
   :show-inheritance:

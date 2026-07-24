Embedding Samplers
==================

Embedding samplers operate on an ``InfoTable`` containing one row per
embedding. Since every embedding has a fixed vector dimension, embedding
batches are constrained by the number of vectors rather than sequence length,
duration, or padding budget.

Sampler selection
-----------------

Use ``EmbedSamplerFactory`` to construct an embedding sampler. The factory
expects an ``EmbedDataset`` whose ``embed_info`` is the embedding metadata
table. Class-weighted sampling additionally requires the corresponding
``ClassInfo`` table in ``dataset.class_info``.

.. autoclass:: hyperion.torch.data.embed_sampler_factory.EmbedSamplerFactory
   :no-index:
   :members:
   :show-inheritance:

Factory usage
~~~~~~~~~~~~~

The factory selects a sampler and supplies the embedding metadata from the
dataset. A plain embedding sampler can be used as follows:

.. code-block:: python

   from hyperion.torch.data import EmbedSamplerFactory

   sampler = EmbedSamplerFactory.create(
       dataset,
       sampler_type="embed_sampler",
       batch_size=256,
       shuffle=True,
       seed=1234,
   )

   for embed_ids in sampler:
       train_batch(embed_ids)

For class-weighted sampling, ``dataset.class_info`` must contain the
``ClassInfo`` table keyed by the selected ``class_name``:

.. code-block:: python

   sampler = EmbedSamplerFactory.create(
       dataset,
       sampler_type="class_weighted_random_embed_sampler",
       class_name="speaker",
       batch_size=256,
       num_embeds_per_class=4,
       weight_mode="data-prior",
       shuffle=True,
   )

The factory resolves ``dataset.class_info["speaker"]`` and passes it to the
class-weighted sampler automatically.

Available sampler types
-----------------------

``embed_sampler``
    Samples fixed-size batches by selecting embedding IDs from the metadata
    table. It supports shuffling, distributed rank-strided sampling,
    ``drop_last``, and ``max_batches_per_epoch``. With ``drop_last=False``, it
    repeats leading IDs only as needed to pad the final distributed batch, so
    every rank receives the same number of full batches.

``class_weighted_random_embed_sampler``
    Samples classes according to configured class weights and then samples a
    fixed number of embeddings from each selected class. Embeddings are sampled
    with replacement. The ``data-prior`` weight mode uses the number of
    embeddings belonging to each class. Optional hard-prototype sampling can
    expand each selected class using an affinity matrix.

Basic configuration
-------------------

``batch_size`` is the number of embeddings per batch and per GPU for
``EmbedSampler``. For class-weighted sampling, ``num_embeds_per_class`` controls
how many embeddings are drawn for each selected class. The sampler rounds
``batch_size`` up to a complete class group, including hard-prototype expansion
when it is enabled, and then derives the number of classes per batch.

Both samplers support:

* deterministic random seeds through ``seed``;
* epoch-dependent sampling through ``shuffle``;
* distributed training through ``HyperSampler``;
* resuming from a batch offset with ``set_epoch``.
* a global ``max_batches_per_epoch`` cap, divided across distributed ranks.

Embedding sampler APIs
----------------------

.. autoclass:: hyperion.torch.data.embed_sampler.EmbedSampler
   :members:
   :show-inheritance:

.. autoclass:: hyperion.torch.data.class_weighted_random_embed_sampler.ClassWeightedRandomEmbedSampler
   :members:
   :show-inheritance:

Direct sampler usage
~~~~~~~~~~~~~~~~~~~~

The direct constructor is useful when the embedding metadata and class table
are already available:

.. code-block:: python

   from hyperion.torch.data.class_weighted_random_embed_sampler import (
       ClassWeightedRandomEmbedSampler,
   )

   sampler = ClassWeightedRandomEmbedSampler(
       embed_set,
       class_info,
       batch_size=128,
       num_embeds_per_class=2,
       weight_mode="uniform",
   )

   for embed_ids in sampler:
       # embed_ids is a list of embedding IDs sampled with replacement.
       batch = dataset[embed_ids]

Both samplers can be passed as a ``batch_sampler`` to a PyTorch
``DataLoader`` when the dataset accepts batches of IDs:

.. code-block:: python

   from torch.utils.data import DataLoader

   loader = DataLoader(dataset, batch_sampler=sampler)
   for batch in loader:
       train_step(batch)

Fine-tuning and Adaptation Commands
===================================

Stable fine-tuning commands adapt an existing x-vector checkpoint. They require
the original model-compatible architecture/configuration and a CSV-based target
dataset. The full classified list is in :doc:`../cli`.

Stable x-vector fine-tuning
---------------------------

``hyperion-finetune-wav2xvector`` and
``hyperion-finetune-wav2vec2xvector`` fine-tune waveform models;
``hyperion-finetune-xvector-from-feats`` and
``hyperion-finetune-xvector-from-wav`` target feature or waveform paths.
DFR variants additionally configure domain/fairness regularization.

Use a new experiment path and record both the base checkpoint and target class
inventory in configuration:

.. code-block:: bash

   hyperion-finetune-wav2xvector --cfg configs/finetune-wav2xvector.yaml

The configuration must identify the source checkpoint, target manifests, class
files, optimizer/trainer policy, and output experiment directory. Fine-tuning
creates a new checkpoint lineage; it does not alter the base checkpoint.

Adversarial fine-tuning
-----------------------

``hyperion-adv-finetune-xvector-from-wav`` is stable adversarial training.
It requires a compatible waveform model and attack/defense configuration. Keep
attack settings, perturbation constraints, and evaluation protocol with the
resulting checkpoint; they affect the interpretation of both clean and robust
metrics.

Experimental adaptation
------------------------

DAC, Q-vector, VITS anonymizer, and transducer fine-tuning commands are
experimental. Their checkpoint/configuration compatibility is not guaranteed.
See :doc:`experimental` before using them.

See also
--------

* :doc:`training`
* :doc:`../how-to/run-resumable-distributed-training`
* :doc:`../torch-api-contracts`

Metrics and Evaluation
======================

Metrics in Hyperion are organized in three layers.

Layer 1: NumPy metric functions
-------------------------------

``hyperion.np.metrics`` provides standalone function-level metrics (EER, DCF,
ROC, confusion matrices, WER/CER, confidence, and related helpers).

.. automodule:: hyperion.np.metrics
   :members:

Layer 2: Torch metrics
----------------------

``hyperion.torch.metrics`` provides torch-native metric abstractions and
functional/class implementations (for example categorical accuracy).

.. automodule:: hyperion.torch.metrics
   :members:

Layer 3: High-level evaluators
------------------------------

``hyperion.metrics`` contains evaluator classes that orchestrate end-to-end
assessment workflows and can combine outputs from NumPy and torch metrics.

.. automodule:: hyperion.metrics
   :members:

Key evaluator classes
---------------------

.. autoclass:: hyperion.metrics.VerificationEvaluator

.. autoclass:: hyperion.metrics.VerificationAdvAttackEvaluator

.. autoclass:: hyperion.metrics.VerificationAnonymizationEvaluator

.. autoclass:: hyperion.metrics.SpeechQualityEvaluator

.. autoclass:: hyperion.metrics.VoxProfileEvaluator

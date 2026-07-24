Text Normalization API
======================

``hyperion.text_norm`` provides reusable normalization for speech transcripts,
ASR targets, and text-based evaluation. Normalization policy is task-specific:
record the normalizer configuration with any metric result that depends on it.

Core normalizers
----------------

.. autoclass:: hyperion.text_norm.BasicTextNormalizer
   :no-index:
   :members: __call__

.. autoclass:: hyperion.text_norm.EnglishTextNormalizer
   :no-index:
   :members: __call__

``BasicTextNormalizer`` handles Unicode, punctuation/symbol removal,
diacritics, bracketed text, whitespace, and optional grapheme splitting.
``EnglishTextNormalizer`` adds English-oriented behavior. Apply the same
normalizer to references and hypotheses before computing text metrics.

Numbers and spelling
--------------------

.. autoclass:: hyperion.text_norm.english_number_normalizers.EnglishNumberNormalizer
   :no-index:

.. autoclass:: hyperion.text_norm.english_number_normalizers.EnglishReverseNumberNormalizer
   :no-index:

.. autoclass:: hyperion.text_norm.spelling_normalizer.SpellingNormalizer
   :no-index:

The number and spelling helpers are English-specific. Do not silently apply
them to another language; use a language-appropriate normalization policy and
document it with the evaluation protocol.

See also
--------

* :doc:`metrics`
* :doc:`documentation-policy`

Glossary
========

.. glossary::

   embedding
      A fixed-dimensional vector representing a speech segment or speaker.
      Embeddings are commonly compared to produce verification scores.

   enrollment
      The speech data or embedding(s) used to create a model representation for
      a claimed identity.

   model
      In verification tables, the identifier for an enrollment representation.
      A model can be built from one or more enrollment segments.

   segment
      A labeled portion of a recording, usually identified by a unique segment
      id. A segment may be a complete utterance or a time interval in a larger
      recording.

   recording
      A source audio, image, or video asset from which one or more segments can
      be drawn.

   trial
      One comparison between an enrollment model and a test segment.

   trial index
      A :class:`hyperion.utils.TrialNdx` specifying which model/segment pairs
      should be scored.

   trial key
      A :class:`hyperion.utils.TrialKey` that labels valid trials as target,
      non-target, and optionally spoof.

   target trial
      A trial in which the test segment belongs to the claimed enrollment
      identity.

   non-target trial
      A trial in which the test segment does not belong to the claimed
      enrollment identity.

   score
      A numeric measure of similarity or log-likelihood ratio for one trial.
      Higher values usually indicate stronger evidence for the target
      hypothesis, subject to the scoring backend's documented convention.

   cohort
      A set of background embeddings used for score normalization methods such
      as adaptive S-norm.

   VAD
      Voice activity detection: a mask, table, or set of time intervals that
      indicates speech-active regions.

   ark/scp
      Kaldi-compatible storage conventions. An ``ark`` is an archive of keyed
      data; an ``scp`` is an index that maps keys to archive locations.

   EER
      Equal error rate: the operating point at which false-reject and
      false-accept rates are equal. Lower is better.

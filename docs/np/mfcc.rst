MFCC Tutorial (NumPy)
=====================

This tutorial shows how to use :class:`hyperion.np.feats.mfcc.MFCC` for common
feature-extraction workflows.

See also:

* :doc:`../numpy`
* :class:`hyperion.np.feats.energy_vad.EnergyVAD`
* :class:`hyperion.np.feats.feature_normalization.MeanVarianceNorm`

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from hyperion.np.feats.mfcc import MFCC

   # 1 second of fake audio at 16 kHz
   wav = np.random.randn(16000).astype(np.float32)

   mfcc = MFCC(
       sample_frequency=16000,
       num_ceps=13,
       input_step="wave",
       output_step="mfcc",
   )
   feats = mfcc.compute(wav)
   print(feats.shape)  # (num_frames, 13)


Example 1: Standard MFCCs from Wave
-----------------------------------

.. code-block:: python

   import numpy as np
   from hyperion.np.feats.mfcc import MFCC

   wav = np.random.randn(32000).astype(np.float32)  # 2 sec @ 16 kHz

   mfcc = MFCC(
       sample_frequency=16000,
       frame_length=25,
       frame_shift=10,
       num_filters=23,
       num_ceps=13,
       use_energy=True,
       raw_energy=True,
       input_step="wave",
       output_step="mfcc",
   )

   x_mfcc = mfcc.compute(wav)


Example 2: Return Intermediate Representations
----------------------------------------------

Use ``return_fft``, ``return_spec``, and ``return_logfb`` to retrieve
intermediate outputs together with the main output.

.. code-block:: python

   import numpy as np
   from hyperion.np.feats.mfcc import MFCC

   wav = np.random.randn(16000).astype(np.float32)

   mfcc = MFCC(input_step="wave", output_step="mfcc")
   x_mfcc, X_fft, X_spec, X_logfb = mfcc.compute(
       wav,
       return_fft=True,
       return_spec=True,
       return_logfb=True,
   )


Example 3: Continue from an Intermediate Step
---------------------------------------------

If you already have a spectrogram or log-filter-bank from an upstream stage,
you can start from that step.

.. code-block:: python

   import numpy as np
   from hyperion.np.feats.mfcc import MFCC

   # Pretend this came from another module (num_frames, fft_bins)
   spec = np.abs(np.random.randn(100, 257)).astype(np.float32)

   mfcc_from_spec = MFCC(
       sample_frequency=16000,
       fft_length=512,
       input_step="spec",
       output_step="mfcc",
       num_ceps=13,
   )
   x_mfcc = mfcc_from_spec.compute(spec)


Example 4: Chunked Processing (Stateful Filters)
------------------------------------------------

The class keeps internal filter state for DC-removal and pre-emphasis.
Call ``reset()`` when starting a new utterance/stream.

.. code-block:: python

   import numpy as np
   from hyperion.np.feats.mfcc import MFCC

   mfcc = MFCC(input_step="wave", output_step="mfcc")

   chunk1 = np.random.randn(8000).astype(np.float32)
   chunk2 = np.random.randn(8000).astype(np.float32)

   x1 = mfcc.compute(chunk1)
   x2 = mfcc.compute(chunk2)  # continues with internal filter state

   mfcc.reset()  # start a new recording/session


Example 5: Configure from an ArgumentParser
-------------------------------------------

.. code-block:: python

   from jsonargparse import ArgumentParser
   from hyperion.np.feats.mfcc import MFCC

   parser = ArgumentParser()
   MFCC.add_class_args(parser, prefix="mfcc")
   cfg = parser.parse_args(["--mfcc.num-ceps=20", "--mfcc.output-step=mfcc"])
   mfcc = MFCC(**MFCC.filter_args(**cfg["mfcc"]))


Tips
----

* Use ``input_step``/``output_step`` to avoid recomputing earlier stages.
* Keep ``sample_frequency`` and FFT/frame settings aligned with upstream audio.
* If you process multiple utterances, call ``reset()`` between utterances.


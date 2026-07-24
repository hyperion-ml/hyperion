PyTorch Integrations and Robustness
===================================

This page covers stable advanced PyTorch surfaces: third-party model wrappers
(``tpm``) and adversarial attack/defense interfaces. They are stable Hyperion
interfaces, but availability and behavior also depend on the installed external
package, model asset, and model revision. Record those external inputs with an
experiment just as you record the Hyperion configuration.

Third-party model wrappers
--------------------------

Hugging Face frontends
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.tpm.hf.hf_wav2vec_base.HFWav2VecBase
   :no-index:
   :members: feature_encoder_context, frame_shift, forward, forward_long_impl, filter_args, add_class_args

``HFWav2VecBase`` is the common interface behind Wav2Vec 2, HuBERT, WavLM,
Wav2Vec2-BERT, and Whisper encoder wrappers. It consumes batched waveforms and
optional valid lengths. The selected wrapper returns feature tensors and
associated lengths according to its documented forward contract. For long
audio, use the wrapper's chunk-aware path rather than manually slicing audio,
so that encoder context is handled correctly.

Set ``pretrained_model_path`` to a local path or a Hugging Face model
identifier, and pin ``revision`` to an immutable revision for reproducible
runs. ``cache_dir`` and ``force_download`` only control retrieval; they do not
make a remote model revision reproducible. See
:doc:`how-to/train-pretrained-wav2vec2-xvector` for its use in x-vector models.

Speech-quality and profile evaluators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hyperion.torch.tpm.dnsmos.dnsmos.DNSMOS
   :no-index:
   :members: __call__, add_class_args

.. autoclass:: hyperion.torch.tpm.utmos.utmos.UTMOSV2
   :no-index:
   :members: compute_mos, delete_model

.. autoclass:: hyperion.torch.tpm.usc.voxprofile_evaluator.VoxProfileEvaluator
   :no-index:
   :members: __call__, add_class_args

DNSMOS retrieves or consumes ONNX checkpoints and scores mono audio after
resampling to 16 kHz. UTMOS v2 uses its external runtime and a temporary audio
directory. VoxProfile evaluator subclasses run an externally supplied PyTorch
model on bounded-length 16 kHz chunks. These scores are model outputs, not a
replacement for a task-appropriate human evaluation. Keep model assets,
licenses, cache locations, optional runtime versions, and device settings in
the evaluation record.

The maintained CLI entry points for quality and profile metrics are listed in
:doc:`cli`; their metric-level interpretation is documented in :doc:`metrics`.

Adversarial attacks
-------------------

.. autoclass:: hyperion.torch.adv_attacks.adv_attack.AdvAttack
   :no-index:
   :members: to, attack_info, generate

.. autoclass:: hyperion.torch.adv_attacks.attack_factory.AttackFactory
   :no-index:
   :members: create, filter_args, add_class_args

``AdvAttack.generate`` receives a clean tensor batch and target labels, then
returns a same-shaped adversarial batch. The factory exposes FGSM, iterative
and randomized FGSM, PGD, SNR-FGSM, and Carlini-Wagner variants. Define the
threat model before comparing results: targeted versus untargeted objective,
input range, perturbation norm/budget, temporal normalization, and model
preprocessing must all match the deployment scenario.

Use attacks only on models and data for which you have authorization. Report
both clean and attacked performance, the exact attack configuration, random
seeds/initializations, and any input clamping. Do not treat a successful attack
on a surrogate or differently normalized model as a deployment vulnerability.

Defenses
--------

.. autoclass:: hyperion.torch.adv_defenses.wave_gan_white.WaveGANDefender
   :no-index:
   :members: forward

``WaveGANDefender`` reconstructs one-dimensional or batched audio using a
Parallel WaveGAN generator. It requires a compatible generator checkpoint,
configuration, feature statistics, and optional PQMF assets. Its output should
be evaluated for clean-speech distortion as well as attack robustness; a
defense that reduces attack success while degrading the underlying task is not
an effective deployment defense.

See also
--------

* :doc:`torch-extension-points`
* :doc:`metrics`
* :doc:`how-to/save-load-models-and-backends`

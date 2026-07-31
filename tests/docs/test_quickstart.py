"""Executable fixture-scale coverage for the speaker-verification quickstart."""

from __future__ import annotations

import numpy as np

from hyperion.np.metrics import compute_eer
from hyperion.utils import TrialKey, TrialScores
from hyperion.utils.math_funcs import cosine_scoring


def test_speaker_verification_quickstart() -> None:
    """Create synthetic embeddings, score them, and compute a near-zero EER."""
    rng = np.random.default_rng(1234)
    num_speakers = 4
    embedding_dim = 192
    tests_per_speaker = 3
    model_ids = np.array([f"spk-{i}" for i in range(num_speakers)])
    speaker_centers = rng.normal(size=(num_speakers, embedding_dim))
    enroll_embeddings = speaker_centers + 0.10 * rng.normal(size=speaker_centers.shape)
    test_speaker_ids = np.repeat(model_ids, tests_per_speaker)
    test_embeddings = np.repeat(speaker_centers, tests_per_speaker, axis=0)
    test_embeddings += 0.10 * rng.normal(size=test_embeddings.shape)
    segment_ids = np.array([f"test-{i}" for i in range(len(test_embeddings))])
    is_target = model_ids[:, None] == test_speaker_ids[None, :]
    key = TrialKey(
        model_set=model_ids, seg_set=segment_ids, tar=is_target, non=~is_target
    )

    score_matrix = cosine_scoring(enroll_embeddings, test_embeddings)
    scores = TrialScores(
        model_set=model_ids,
        seg_set=segment_ids,
        scores=score_matrix,
        score_mask=np.ones(score_matrix.shape, dtype=bool),
    )
    target_scores, non_target_scores = scores.get_tar_non(key)
    eer = compute_eer(target_scores, non_target_scores)

    assert scores.scores.shape == (4, 12)
    assert len(target_scores) == 12
    assert len(non_target_scores) == 36
    assert eer < 0.01

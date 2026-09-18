"""
Copyright 2026 Johns Hopkins University  (Author: Stanislaw Kacprzak)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from collections import Counter

from hyperion.np.metrics.wer import compute_wer


def test_wer():
    ref = ["The quick brown fox jumps over the lazy dog".split(" ")]
    hyp = ["Then quick brown box jumps over lazy dog twice".split(" ")]

    s = 2
    i = 1
    d = 1
    n = sum(map(len, ref))
    wer = compute_wer(hyp=hyp, ref=ref)

    assert wer[0] == (s + d + i) / n
    assert wer[1] == s
    assert wer[2] == i
    assert wer[3] == d
    assert wer[4] == n

    utt_stats = wer[5]
    assert utt_stats["num_words"].tolist() == [n]
    assert utt_stats["wer"].tolist() == [(s + d + i) / n]

    word_stats = wer[6]
    expected_totals = dict.fromkeys(ref[0] + hyp[0], 0)
    expected_totals.update(Counter(ref[0]))
    assert word_stats["total"].to_dict() == expected_totals
    assert word_stats["total"].sum() == n
    assert "*" not in word_stats.index

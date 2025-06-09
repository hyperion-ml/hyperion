"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import List, Union

import numpy as np
import pandas as pd

from .wer import compute_wer


def compute_cer(
    hyp: Union[List[str], str],
    ref: Union[List[str], str],
    utt_ids: Union[np.ndarray, List[str], None] = None,
    sclite_mode: bool = True,
):
    """
    Calculates Char Error Rate (CER) and detailed error statistics between reference and hypothesis transcripts.

    This function performs token-level alignment of hypotheses and references, computes WER globally and per utterance,
    and generates substitution, insertion, and deletion statistics. Results are returned as structured DataFrames
    for detailed analysis.

    Args:
        hyp (List[List[str]]):
            List of predicted transcripts, each as a list of chars.
        ref (List[List[str]]):
            List of reference transcripts, each as a list of chars.
        utt_ids (Union[np.ndarray, List[str], None], optional):
            List of utterance IDs corresponding to each transcript pair.
            If None, numeric IDs will be auto-generated.
        sclite_mode (bool, optional):
            Whether to use alignment logic compatible with NIST's `sclite` scoring tool
            (affects treatment of ambiguous insertion/deletion cases). Default is True.

    Returns:
        cer (float):
            Overall Character Error Rate across all utterances.
        subs (int):
            Total number of substitution errors.
        ins (int):
            Total number of insertion errors.
        dels (int):
            Total number of deletion errors.
        num_chars (int):
            Total number of chars
        utt_stats (pd.DataFrame):
            Per-utterance statistics including WER, error counts, and detailed ref/hyp diffs.
        char_stats (pd.DataFrame):
            Per-char statistics: correctness, substitution, insertion, and deletion counts/rates.
        sub_stats (pd.DataFrame):
            Per-substitution statistics showing the most frequent ref->hyp word substitutions and their rates.

    Notes:
        - This function assumes word-level tokenization is already done.
        - Adapted from: https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
    """
    if isinstance(hyp[0], str):
        hyp = [hyp_i.split() for hyp_i in hyp]

    if isinstance(ref[0], str):
        ref = [ref_i.split() for ref_i in ref]

    for i, (hyp_i, ref_i) in enumerate(zip(hyp, ref)):
        ref_i = list("".join(ref_i))
        hyp_i = list("".join(hyp_i))
        ref[i] = ref_i
        hyp[i] = hyp_i

    (
        total_cer,
        total_subs,
        total_ins,
        total_dels,
        total_chars,
        utt_stats,
        char_stats,
        sub_stats,
    ) = compute_wer(hyp, ref, utt_ids, sclite_mode=sclite_mode)

    utt_stats.rename(
        columns={
            col: col.replace("word", "char").replace("wer", "cer")
            for col in utt_stats.columns
        },
        inplace=True,
    )
    char_stats.reset_index(inplace=True, drop=True)
    char_stats.rename(columns={"word": "char"}, inplace=True)
    char_stats.set_index(char_stats.char, drop=False, inplace=True)

    sub_stats.rename(
        columns={"ref_word": "ref_char", "hyp_word": "hyp_char"}, inplace=True
    )
    print(char_stats, "\n", sub_stats, "\n", utt_stats, "\n", flush=True)
    return (
        total_cer,
        total_subs,
        total_ins,
        total_dels,
        total_chars,
        utt_stats,
        char_stats,
        sub_stats,
    )

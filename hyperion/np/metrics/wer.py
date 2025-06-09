"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Union

import kaldialign
import numpy as np
import pandas as pd


def compute_wer(
    hyp: List[List[str]],
    ref: List[List[str]],
    utt_ids: Union[np.ndarray, List[str], None] = None,
    sclite_mode: bool = True,
):
    """
    Calculates Word Error Rate (WER) and detailed error statistics between reference and hypothesis transcripts.

    This function performs token-level alignment of hypotheses and references, computes WER globally and per utterance,
    and generates substitution, insertion, and deletion statistics. Results are returned as structured DataFrames
    for detailed analysis.

    Args:
        hyp (List[List[str]]):
            List of predicted transcripts, each as a list of words.
        ref (List[List[str]]):
            List of reference transcripts, each as a list of words.
        utt_ids (Union[np.ndarray, List[str], None], optional):
            List of utterance IDs corresponding to each transcript pair.
            If None, numeric IDs will be auto-generated.
        sclite_mode (bool, optional):
            Whether to use alignment logic compatible with NIST's `sclite` scoring tool
            (affects treatment of ambiguous insertion/deletion cases). Default is True.

    Returns:
        wer (float):
            Overall Word Error Rate across all utterances.
        subs (int):
            Total number of substitution errors.
        ins (int):
            Total number of insertion errors.
        dels (int):
            Total number of deletion errors.
        num_words (int):
            Total number of words
        utt_stats (pd.DataFrame):
            Per-utterance statistics including WER, error counts, and detailed ref/hyp diffs.
        word_stats (pd.DataFrame):
            Per-word statistics: correctness, substitution, insertion, and deletion counts/rates.
        sub_stats (pd.DataFrame):
            Per-substitution statistics showing the most frequent ref->hyp word substitutions and their rates.

    Notes:
        - This function assumes word-level tokenization is already done.
        - Adapted from: https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # stores counts per sentence, as follows:
    #   corr, subs, ins, dels, total
    utt_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    utt_wers: Dict[str, float] = defaultdict(float)
    utt_details = []
    # stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels, total
    word_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0])

    total_words = 0
    ERR = "*"

    if utt_ids is None:
        utt_ids = np.arange(0, len(hyp)).astype(str)

    if isinstance(hyp[0], str):
        hyp = [hyp_i.split() for hyp_i in hyp]

    if isinstance(ref[0], str):
        ref = [ref_i.split() for ref_i in ref]

    for i, ref_i, hyp_i in zip(utt_ids, ref, hyp):
        ali = kaldialign.align(ref_i, hyp_i, ERR, sclite_mode=sclite_mode)
        for ref_word, hyp_word in ali:
            total_words += 1
            utt_counts[i][4] += 1
            word_counts[ref_word][5] += 1
            if ref_word == ERR:
                ins[hyp_word] += 1
                utt_counts[i][2] += 1
                word_counts[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                utt_counts[i][3] += 1
                word_counts[ref_word][4] += 1
            elif hyp_word != ref_word:
                utt_counts[i][1] += 1
                subs[(ref_word, hyp_word)] += 1
                word_counts[ref_word][1] += 1
                word_counts[hyp_word][2] += 1
            else:
                utt_counts[i][0] += 1
                word_counts[ref_word][0] += 1

        _, subs_i, ins_i, dels_i, tot_i = utt_counts[i]
        utt_wers[i] = (subs_i + ins_i + dels_i) / tot_i

        ali = [[[x], [y]] for x, y in ali]
        for i in range(len(ali) - 1):
            if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                ali[i] = [[], []]

        ali = [
            [
                list(filter(lambda a: a != ERR, x)),
                list(filter(lambda a: a != ERR, y)),
            ]
            for x, y in ali
        ]
        ali = list(filter(lambda x: x != [[], []], ali))
        ali = [
            [
                ERR if x == [] else " ".join(x),
                ERR if y == [] else " ".join(y),
            ]
            for x, y in ali
        ]

        detail_i = " ".join(
            ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
            for ref_word, hyp_word in ali
        )
        utt_details.append(detail_i)

    total_subs = sum(subs.values())
    total_ins = sum(ins.values())
    total_dels = sum(dels.values())
    total_errs = total_subs + total_ins + total_dels
    total_wer = total_errs / total_words

    utt_ids = list(utt_counts.keys())
    utt_counts = np.array(list(utt_counts.values()))
    utt_stats = pd.DataFrame(
        {
            "id": utt_ids,
            "word_corr": utt_counts[:, 0],
            "word_subs": utt_counts[:, 1],
            "word_ins": utt_counts[:, 2],
            "word_dels": utt_counts[:, 3],
            "num_words": utt_counts[:, 4],
            "wer": utt_wers.values(),
            "word_error_details": utt_details,
        }
    )
    utt_stats.set_index("id", drop=False, inplace=True)

    ref_words = [k[0] for k in subs.keys()]
    hyp_words = [k[1] for k in subs.keys()]
    sub_stats = pd.DataFrame(
        {"ref_word": ref_words, "hyp_word": hyp_words, "subs": subs.values()}
    )
    sub_stats.set_index(["ref_word", "hyp_word"], drop=False, inplace=True)
    words = list(word_counts.keys())
    word_counts = np.array(list(word_counts.values()))

    word_stats = pd.DataFrame(
        {
            "word": words,
            "corr": word_counts[:, 0],
            "ref_subs": word_counts[:, 1],
            "hyp_subs": word_counts[:, 2],
            "ins": word_counts[:, 3],
            "dels": word_counts[:, 4],
            "total": word_counts[:, 5],
        }
    )
    word_stats.set_index("word", drop=False, inplace=True)
    sub_stats["ref_total"] = word_stats.loc[sub_stats["ref_word"], "total"].values
    sub_stats["subs_rate"] = sub_stats["subs"] / sub_stats["ref_total"]
    sub_stats.sort_values(by="subs_rate", ascending=False, inplace=True)

    word_stats["acc"] = word_stats["corr"] / word_stats["total"]
    word_stats["ref_subs_rate"] = word_stats["ref_subs"] / word_stats["total"]
    # word_stats["hyp_subs_rate"] = word_stats["hyp_subs"] / word_stats["total"]
    word_stats["ins_rate"] = word_stats["ins"] / word_stats["total"]
    word_stats["dels_rate"] = word_stats["dels"] / word_stats["total"]
    word_stats.sort_values(by="acc", inplace=True)

    return (
        total_wer,
        total_subs,
        total_ins,
        total_dels,
        total_words,
        utt_stats,
        word_stats,
        sub_stats,
    )

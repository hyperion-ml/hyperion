"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

from .utils import pavx


def compute_roc(true_scores, false_scores):
    """Computes the (observed) miss/false_alarm probabilities
        for a set of detection output scores.

    Args:
      true_scores (false_scores) are detection output scores for a set of
      detection trials, given that the target hypothesis is true (false).
         (By convention, the more positive the score,
         the more likely is the target hypothesis.)

    Returns:
      The miss/false_alarm error probabilities
    """
    num_true = len(true_scores)
    num_false = len(false_scores)
    assert num_true > 0
    assert num_false > 0

    total = num_true + num_false

    p_miss = np.zeros((num_true + num_false + 1,))
    p_fa = np.zeros((num_true + num_false + 1,))

    scores = np.hstack((true_scores, false_scores))
    labels = np.zeros_like(scores)
    labels[:num_true] = 1

    indx = np.argsort(scores, kind="mergesort")
    labels = labels[indx]

    sumtrue = np.cumsum(labels)
    sumfalse = num_false - (np.arange(total) + 1 - sumtrue)

    p_miss[0] = 0
    p_fa[0] = 1.0
    p_miss[1:] = sumtrue / num_true
    p_fa[1:] = sumfalse / num_false

    return p_miss, p_fa


def compute_rocch(tar_scores, non_scores):
    """Computes ROCCH: ROC Convex Hull.

    Args:
      tar_scores: scores for target trials
      nontar_scores: scores for non-target trials

    Returns:
       pmiss and pfa contain the coordinates of the vertices of the
       ROC Convex Hull.
    """
    assert isinstance(tar_scores, np.ndarray)
    assert isinstance(non_scores, np.ndarray)

    Nt = len(tar_scores)
    Nn = len(non_scores)
    N = Nt + Nn
    scores = np.hstack((tar_scores.ravel(), non_scores.ravel()))
    # ideal, but non-monotonic posterior
    Pideal = np.hstack((np.ones((Nt,)), np.zeros((Nn,))))

    # It is important here that scores that are the same (i.e. already in order) should NOT be swapped.
    # MATLAB's sort algorithm has this property.
    perturb = np.argsort(scores, kind="mergesort")

    Pideal = Pideal[perturb]
    Popt, width, _ = pavx(Pideal)

    nbins = len(width)
    p_miss = np.zeros((nbins + 1,))
    p_fa = np.zeros((nbins + 1,))

    # threshold leftmost: accept eveything, miss nothing
    # 0 scores to left of threshold
    left = 0
    fa = Nn
    miss = 0

    for i in range(nbins):
        p_miss[i] = miss / Nt
        p_fa[i] = fa / Nn
        left = left + width[i]
        miss = np.sum(Pideal[:left])
        fa = N - left - np.sum(Pideal[left:])

    p_miss[nbins] = miss / Nt
    p_fa[nbins] = fa / Nn

    return p_miss, p_fa


def rocch2eer(p_miss, p_fa):
    """Calculates the equal error rate (eer) from pmiss and pfa
    vectors.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.
    Use compute_rocch to convert target and non-target scores to pmiss and
    pfa values.
    """
    eer = 0

    # p_miss and p_fa should be sorted
    x = np.sort(p_miss, kind="mergesort")
    assert np.all(x == p_miss)
    x = np.sort(p_fa, kind="mergesort")[::-1]
    assert np.all(x == p_fa)

    _1_1 = np.array([1, -1])
    _11 = np.array([[1], [1]])
    for i in range(len(p_fa) - 1):
        xx = p_fa[i : i + 2]
        yy = p_miss[i : i + 2]

        XY = np.vstack((xx, yy)).T
        dd = np.dot(_1_1, XY)
        if np.min(np.abs(dd)) == 0:
            eerseg = 0
        else:
            # find line coefficieents seg s.t. seg'[xx(i)yy(i)] = 1,
            # when xx(i),yy(i) is on the line.
            seg = sla.solve(XY, _11)
            # candidate for EER, eer is highest candidate
            eerseg = 1 / (np.sum(seg))

        eer = np.maximum(eer, eerseg)

    return eer


def filter_roc(p_miss, p_fa):
    """Removes redundant points from the sequence of points (p_fa,p_miss) so
       that plotting an ROC or DET curve will be faster.  The output ROC
       curve will be identical to the one plotted from the input
       vectors.  All points internal to straight (horizontal or
       vertical) sections on the ROC curve are removed i.e. only the
       points at the start and end of line segments in the curve are
       retained.  Since the plotting code draws straight lines between
       points, the resulting plot will be the same as the original.

    Args:
      p_miss, p_fa: The coordinates of the vertices of the ROC Convex
                    Hull.  m for misses and fa for false alarms.
    Returns:
      new_p_miss, new_p_fa: Vectors containing selected values from the
                            input vectors.
    """
    out = 0
    new_p_miss = np.copy(p_miss)
    new_p_fa = np.copy(p_fa)

    for i in range(1, len(p_miss)):
        if p_miss[i] == new_p_miss[out] or p_fa[i] == new_p_fa[out]:
            continue

        # save previous point, because it is the last point before the
        # change.  On the next iteration, the current point will be saved.
        out = out + 1
        new_p_miss[out] = p_miss[i - 1]
        new_p_fa[out] = p_fa[i - 1]

    out = out + 1
    new_p_miss[out] = p_miss[-1]
    new_p_fa[out] = p_fa[-1]
    new_p_miss = new_p_miss[:out]
    new_p_fa = new_fa[:out]

    return new_p_miss, new_p_fa


def compute_area_under_rocch(p_miss, p_fa):
    """Calculates area under the ROC convex hull given p_miss, p_fa.

    Args:
      p_miss: Miss probabilities vector obtained from compute_rocch
      p_fa: False alarm probabilities vector

    Returns:
      AUC
    """

    assert np.all(p_miss == np.sort(p_miss, kind="mergesort"))
    assert np.all(p_fa[::-1] == np.sort(p_fa, kind="mergesort"))
    assert p_miss.shape == p_fa.shape

    auc = 0
    for i in range(1, len(p_miss)):
        auc += 0.5 * (p_miss[i] - p_miss[i - 1]) * (p_fa[i] + p_fa[i + 1])

    return auc


def test_roc():

    plt.figure()

    plt.subplot(2, 3, 1)
    tar = np.array([1])
    non = np.array([0])
    pmiss, pfa = compute_rocch(tar, non)
    pm, pf = compute_roc(tar, non)
    (h1,) = plt.plot(pfa, pmiss, "r-^", label="ROCCH", linewidth=2)
    (h2,) = plt.plot(pf, pm, "g--v", label="ROC", linewidth=2)
    plt.axis("square")
    plt.grid(True)
    plt.legend(handles=[h1, h2])
    plt.title("2 scores: non < tar")

    plt.subplot(2, 3, 2)
    tar = np.array([0])
    non = np.array([1])
    pmiss, pfa = compute_rocch(tar, non)
    pm, pf = compute_roc(tar, non)
    plt.plot(pfa, pmiss, "r-^", pf, pm, "g--v", linewidth=2)
    plt.axis("square")
    plt.grid(True)
    plt.title("2 scores: tar < non")

    plt.subplot(2, 3, 3)
    tar = np.array([0])
    non = np.array([-1, 1])
    pmiss, pfa = compute_rocch(tar, non)
    pm, pf = compute_roc(tar, non)
    plt.plot(pfa, pmiss, "r-^", pf, pm, "g--v", linewidth=2)
    plt.axis("square")
    plt.grid(True)
    plt.title("3 scores: non < tar < non")

    plt.subplot(2, 3, 4)
    tar = np.array([-1, 1])
    non = np.array([0])
    pmiss, pfa = compute_rocch(tar, non)
    pm, pf = compute_roc(tar, non)
    plt.plot(pfa, pmiss, "r-^", pf, pm, "g--v", linewidth=2)
    plt.axis("square")
    plt.grid(True)
    plt.title("3 scores: tar < non < tar")
    plt.xlabel(r"$P_{fa}$")
    plt.ylabel(r"$P_{miss}")

    plt.subplot(2, 3, 5)
    tar = np.random.randn(100) + 1
    non = np.random.randn(100)
    pmiss, pfa = compute_rocch(tar, non)
    pm, pf = compute_roc(tar, non)
    plt.plot(pfa, pmiss, "r-^", pf, pm, "g", linewidth=2)
    plt.axis("square")
    plt.grid(True)
    plt.title("DET")

    plt.subplot(2, 3, 6)
    tar = np.random.randn(100) * 2 + 1
    non = np.random.randn(100)
    pmiss, pfa = compute_rocch(tar, non)
    pm, pf = compute_roc(tar, non)
    plt.plot(pfa, pmiss, "r-^", pf, pm, "g", linewidth=2)
    plt.axis("square")
    plt.grid(True)
    plt.title("flatter DET")

    plt.show()

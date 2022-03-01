"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ..utils.list_utils import list2ndarray


def compute_confusion_matrix(
    y_true, y_pred, labels=None, normalize=True, sample_weight=None
):
    """Computes confusion matrix.

    Args:
      y_true: Ground truth.
      y_pred: Estimated labels.
      labels: List of labels to index the matrix. This may be used to reorder
              or select a subset of labels. If none is given, those that
              appear at least once in y_true or y_pred are used in sorted order.
      sample_weight: Sample weights.

    Returns:
      Confusion matrix (num_classes x num_classes)
    """
    C = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    if normalize:
        C = C / (np.sum(C, axis=1, keepdims=True) + 1e-10)
    return C


def compute_xlabel_confusion_matrix(
    y_true,
    y_pred,
    labels_train=None,
    labels_test=None,
    normalize=True,
    sample_weight=None,
):
    """Computes confusion matrix when the labels used to train the classifier are
       different than those of the test set.

    Args:
      y_true: Ground truth.
      y_pred: Estimated labels.
      labels_train: List of labels used to train the classifier. This may be used to reorder
                    or select a subset of labels. If none is given, those that
                    appear at least once in y_pred are used in sorted order.
      labels_test: List of labels of the test set. This may be used to reorder
                    or select a subset of labels. If none is given, those that
                    appear at least once in y_true are used in sorted order.

      sample_weight: Sample weights.

    Returns:
      Confusion matrix (num_classes_test x num_classes_train)
    """
    y_true = list2ndarray(y_true)
    y_pred = list2ndarray(y_pred)
    if labels_train is None:
        labels_train = np.unique(y_pred)
    else:
        labels_train = list2ndarray(labels_train)
    if labels_test is None:
        labels_test = np.unique(y_true)
    else:
        labels_test = list2ndarray(labels_test)

    assert (
        y_true.dtype == y_pred.dtype
    ), "y_true and y_pred labels does not have the same type"
    assert (
        labels_train.dtype == labels_test.dtype
    ), "Train and test labels does not have the same type"
    assert (
        labels_train.dtype == y_pred.dtype
    ), "Labels, y_true and y_pred does not have the same type"

    num_classes_test = len(labels_test)

    if issubclass(y_true.dtype.type, np.integer):
        y_pred += num_classes_test
    elif issubclass(y_true.dtype.type, np.dtype("U")) or issubclass(
        y_true.dtype.type, np.dtype("S")
    ):
        y_true = np.asarray(["TEST_" + s for s in y_true])
        y_pred = np.asarray(["TRAIN_" + s for s in y_pred])
    else:
        raise Exception()

    if issubclass(labels_train.dtype.type, np.integer):
        labels_train += num_classes_test
    elif issubclass(labels_train.dtype.type, np.dtype("U")) or issubclass(
        labels_train.dtype.type, np.dtype("S")
    ):
        labels_test = np.asarray(["TEST_" + s for s in labels_test])
        labels_train = np.asarray(["TRAIN_" + s for s in labels_train])
    else:
        raise Exception()

    labels = np.concatenate((labels_test, labels_train))
    C = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    C = C[:num_classes_test, num_classes_test:]
    if normalize:
        C = C / np.sum(C, axis=1, keepdims=True)
    return C


def plot_confusion_matrix(
    C,
    labels_true,
    labels_pred=None,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    fmt=None,
):
    """Plots a confusion matrix in a figure.

    Args:
      C: 2D numpy array with confusion matrix.
      labels_true: Labels of the true classes (rows).
      labels_cols: Labels of the predicted classes. If None, it is equal to labels_true.
      title: Title for the figure.
      cmp: Color MAP.
    """
    if labels_pred is None:
        labels_pred = labels_true

    assert C.shape[0] == len(labels_true)
    assert C.shape[1] == len(labels_pred)

    plt.imshow(C, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks_y = np.arange(len(labels_true))
    tick_marks_x = np.arange(len(labels_pred))
    plt.xticks(tick_marks_x, labels_pred, rotation=45)
    plt.yticks(tick_marks_y, labels_true)

    if fmt is None:
        normalized = np.all(C <= 1)
        fmt = ".2f" if normalized else "d"
    thresh = np.max(C) / 2.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            plt.text(
                j,
                i,
                format(C[i, j], fmt),
                horizontalalignment="center",
                color="white" if C[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def write_confusion_matrix(f, C, labels_true, labels_pred=None, fmt=None):
    """Writes confusion matrix to file.

    Args:
      f: Python file hangle.
      C: 2D numpy array with confusion matrix.
      labels_true: Labels of the true classes (rows).
      labels_cols: Labels of the predicted classes. If None, it is equal to labels_true.
    """

    if labels_pred is None:
        labels_pred = labels_true

    assert C.shape[0] == len(labels_true)
    assert C.shape[1] == len(labels_pred)

    if fmt is None:
        normalized = np.all(C <= 1)
        fmt = ".2f" if normalized else "d"

    column_width = np.max([len(label) for label in labels_pred] + [6]) + 3
    empty_cell = " " * column_width
    f.write(empty_cell)
    for label in labels_pred:
        f.write("%{0}s".format(column_width) % label)
    f.write("\n")

    for i, label_y in enumerate(labels_true):
        f.write("%{0}s".format(column_width) % label_y)
        for j in range(C.shape[1]):
            f.write("%{0}{1}".format(column_width, fmt) % C[i, j])
        f.write("\n")


def print_confusion_matrix(C, labels_true, labels_pred=None, fmt=None):
    """Prints confusion matrix to std output.

    Args:
      C: 2D numpy array with confusion matrix.
      labels_true: Labels of the true classes (rows).
      labels_cols: Labels of the predicted classes. If None, it is equal to labels_true.
    """
    write_confusion_matrix(sys.stdout, C, labels_true, labels_pred, fmt)

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from ..utils.list_utils import list2ndarray
from .roc import compute_rocch, rocch2eer


def compute_eer(tar, non):
    """Computes equal error rate.

    Args:
      tar: Scores of target trials.
      non: Scores of non-target trials.
    
    Returns:
      EER
    """
    p_miss, p_fa = compute_rocch(tar, non)
    return rocch2eer(p_miss, p_fa)



def compute_accuracy(y_true, y_pred, normalize=True, sample_weight=None):
    """Computes accuracy

    Args:
      y_true: 1d array-like, or label indicator array / sparse matrix.
              Ground truth (correct) labels.
      y_pred: 1d array-like, or label indicator array / sparse matrix.
              Predicted labels, as returned by a classifier.
      normalize: If False, return the number of correctly classified samples. 
                 Otherwise, return the fraction of correctly classified samples.
      sample_weight: Sample weights.

    Returns:
      Accuracy or number of correctly classified samples.
    """
    return accuracy_score(y_true, y_pred, normalize, sample_weight)



def compute_confusion_matrix(y_true, y_pred, labels=None,
                             normalize=True, sample_weight=None):
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
    C = confusion_matrix(y_true, y_pred, labels, sample_weight)
    if normalize:
        C = C/np.sum(C, axis=1, keepdims=True)
    return C



def compute_xlabel_confusion_matrix(y_true, y_pred, labels_train=None, labels_test=None,
                                    normalize=True, sample_weight=None):
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

    assert y_true.dtype == y_pred.dtype, 'y_true and y_pred labels does not have the same type'
    assert labels_train.dtype == labels_test.dtype, 'Train and test labels does not have the same type'
    assert labels_train.dtype == y_pred.dtype, 'Labels, y_true and y_pred does not have the same type'

    num_classes_test = len(labels_test)
    
    if issubclass(y_true.dtype.type, np.integer):
        y_pred += num_classes_test
    elif issubclass(y_true.dtype.type, np.dtype('U')) or issubclass(
    y_true.dtype.type, np.dtype('S')): 
        y_true = np.asarray(['TEST_' + s for s in y_true])
        y_pred = np.asarray(['TRAIN_' + s for s in y_pred])
    else:
        raise Exception()


    if issubclass(labels_train.dtype.type, np.integer):
        labels_train += num_classes_test
    elif issubclass(labels_train.dtype.type, np.dtype('U')) or issubclass(
    labels_train.dtype.type, np.dtype('S')): 
        labels_test = np.asarray(['TEST_' + s for s in labels_test])
        labels_train = np.asarray(['TRAIN_' + s for s in labels_train])
    else:
        raise Exception()

    labels = np.concatenate((labels_test, labels_train))
    C = confusion_matrix(y_true, y_pred, labels, sample_weight)
    C = C[:num_classes_test, num_classes_test:]
    if normalize:
        C = C/np.sum(C, axis=1, keepdims=True)
    return C

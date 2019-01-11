from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt



def plot_confusion_matrix(C, labels_true, labels_pred=None,
                          title='Confusion matrix', cmap=plt.cm.Blues):
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

    plt.imshow(C, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks_y = np.arange(len(labels_true))
    tick_marks_x = np.arange(len(labels_pred))
    plt.xticks(tick_marks_x, labels_pred, rotation=45)
    plt.yticks(tick_marks_y, labels_true)

    normalized = np.all(C<=1)
    fmt = '.2f' if normalized else 'd'
    thresh = np.max(C) / 2.
    for i in xrange(C.shape[0]):
        for j in xrange(C.shape[1]):
            plt.text(j, i, format(C[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if C[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def write_confusion_matrix(f, C, labels_true, labels_pred=None):
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
    
    normalized = np.all(C<=1)
    fmt = '.2f' if normalized else 'd'

    column_width = np.max([len(label) for label in labels_pred] + [6]) + 3
    empty_cell = ' ' * column_width
    f.write(empty_cell)
    for label in labels_pred:
        f.write('%{0}s'.format(column_width) % label)
    f.write('\n')
    
    for i, label_y in enumerate(labels_true):
        f.write('%{0}s'.format(column_width) % label_y)
        for j in xrange(C.shape[1]):
            f.write('%{0}{1}'.format(column_width, fmt) % C[i, j])
        f.write('\n')
        

        
def print_confusion_matrix(C, labels_true, labels_pred=None):
    """Prints confusion matrix to std output.

    Args:
      C: 2D numpy array with confusion matrix.
      labels_true: Labels of the true classes (rows).
      labels_cols: Labels of the predicted classes. If None, it is equal to labels_true.
    """
    write_confusion_matrix(sys.stdout, C, labels_true, labels_pred)

    
    

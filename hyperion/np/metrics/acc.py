"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import accuracy_score


def compute_accuracy(
    y_true: Union[np.ndarray, Sequence[Any]],
    y_pred: Union[np.ndarray, Sequence[Any]],
    normalize: bool = True,
    sample_weight: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> Union[float, int]:
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
    return accuracy_score(
        y_true, y_pred, normalize=normalize, sample_weight=sample_weight
    )

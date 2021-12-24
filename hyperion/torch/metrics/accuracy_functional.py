"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import torch


def categorical_accuracy(input, target, weight=None, reduction="mean"):

    dim = input.dim()
    if dim < 2:
        raise ValueError("Expected 2 or more dimensions (got %d)" % (dim))

    if input.size(0) != target.size(0):
        raise ValueError(
            "Expected input batch_size (%d) to match target batch_size (%d)."
            % (input.size(0), target.size(0))
        )

    with torch.no_grad():
        _, pred = torch.max(input, dim=-1)
        if target.dim() == 2:
            _, target = torch.max(target, dim=-1)

        ok = pred.eq(target).float()

        if reduction == "none":
            return ok

        weight_mean = 1
        if weight is not None:
            if input.size(0) != weight.size(0):
                raise ValueError(
                    "Expected input batch_size (%d) to match weight batch_size (%d)."
                    % (input.size(0), weight.size(0))
                )

            ok *= weight
            weight_mean = weight.mean()

        if reduction == "sum":
            return ok.sum().item()

        acc = ok.mean() / weight_mean

    return acc.item()


def binary_accuracy(input, target, weight=None, reduction="mean", thr=0.5):

    dim = input.dim()
    if dim < 2:
        raise ValueError("Expected 2 or more dimensions (got %d)" % (dim))

    if not (target.size() == input.size()):
        raise ValueError(
            "Target size ({}) is different to the input size ({}).".format(
                target.size(), input.size()
            )
        )

    if input.numel() != target.numel():
        raise ValueError(
            "Target and input must have the same number of elements. target nelement ({}) "
            "!= input nelement ({})".format(target.numel(), input.numel())
        )

    with torch.no_grad():
        pred = input > thr
        ok = pred.eq(target).float()

        if reduction == "none":
            return ok

        weight_mean = 1
        if weight is not None:
            if input.size(0) != weight.size(0):
                raise ValueError(
                    "Expected input batch_size (%d) to match weight batch_size (%d)."
                    % (input.size(0), weight.size(0))
                )

            if weight.dim() == 1:
                ok *= weight.unsqueeze(1)
            else:
                ok *= weight

            weight_mean = weight.mean()

        if reduction == "sum":
            return ok.sum().item()

        acc = ok.mean() / weight_mean

    return ok.item()


def binary_accuracy_with_logits(input, target, weight=None, reduction="mean", thr=0):
    return binary_accuracy(input, target, weight, reduction, thr)

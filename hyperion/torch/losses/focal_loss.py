from torch import nn
import torch
from torch.nn import functional as F
import logging
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        """
        Focal loss implementation: -alpha(1-yi)**gamma * ce_loss(xi,yi)
        
        :param alpha: scalar or list. Class weights. If scalar, the same weight applies for all classes.
        :param gamma: scalar. Difficult-to-easy sample regulation parameter.
        :param size_average: bool. Whether to average the loss over the batch.
        :param device: str. Device to place the tensors.
        """
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.alpha = alpha
        logging.info("FocalLoss: alpha={}, gamma={}, size_average={}".format(alpha, gamma, size_average))
        
    def forward(self, preds, labels):
        """
        Compute the focal loss.
        
        :param preds: Predicted classes. size:[B,N,C] or [B,C]
        :param labels: Actual classes. size:[B,N] or [B]
        :return: scalar. Loss value.
        """
        preds = preds.view(-1, preds.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.gather(0, labels.view(-1))
        else:  # if alpha is a scalar
            alpha = self.alpha

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

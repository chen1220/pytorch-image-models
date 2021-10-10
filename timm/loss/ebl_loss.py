import torch
from torch import nn
import torch.nn.functional as F

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def equilibrium_loss(pred, label, weight=None, mean_score = None, reduction='mean', avg_factor=None):
    label_one_hot = torch.zeros_like(pred).scatter_(1, label.unsqueeze(1), 1).detach()  # [1024, 1231]
    # todo mean_score报错 修改bug跑通ebl loss
    max_element, _ = pred.max(axis=-1)
    pred = pred - max_element[:, None]  # to prevent overflow

    numerator = mean_score.unsqueeze(0) * torch.exp(pred)
    denominator = numerator.sum(-1, keepdim=True)
    P = numerator / denominator

    probs = (P * label_one_hot).sum(1)  # [1024]
    loss = - probs.log()  # [1024]

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor) # weight: 1024; reduction: 'mean'; avg_factor: float(1024.0)

    return loss

class EquilibriumLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(EquilibriumLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion = equilibrium_loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                mean_score=torch.ones(5).cuda() * 0.01,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            mean_score,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
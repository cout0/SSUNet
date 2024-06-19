"""Model Trainer

author: Masahiro Hayashi

This script defines custom loss functions for image segmentation, which
includes Dice Loss and Weighted Cross Entropy Loss.
"""

import torch
from torch.nn import functional as F
from torch.autograd import Function


def dice_loss(pred, target, smooth=1.):
    """Dice loss
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


class Weighted_Cross_Entropy_Loss(torch.nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self):
        super(Weighted_Cross_Entropy_Loss, self).__init__()

    def forward(self, pred, target, weights):
        n, c, H, W = pred.shape
        # # Calculate log probabilities
        logp = F.log_softmax(pred, dim=1)

        # Gather log probabilities with respect to target
        logp = torch.gather(logp, 1, target.view(n, 1, H, W))

        # Multiply with weights
        weighted_logp = (logp * weights).view(n, -1)

        # Rescale so that loss is in approx. same interval
        weighted_loss = weighted_logp.sum(1) / weights.view(n, -1).sum(1)

        # Average over mini-batch
        weighted_loss = -weighted_loss.mean()

        return weighted_loss


class DWCE_Loss(torch.nn.Module):
    """Dynamic Cross entropy loss that uses weight maps."""

    def __init__(self, reduction='mean'):
        super(DWCE_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, weights):
        n, c, H, W = pred.shape
        # # Calculate log probabilities
        logp = F.log_softmax(pred, dim=1)

        # Gather log probabilities with respect to target
        logp = torch.gather(logp, 1, target.view(n, 1, H, W))

        weights = self.compute_dynamic_weights(pred, target)
        # Multiply with weights
        weighted_logp = (logp * weights).view(n, -1)

        # Rescale so that loss is in approx. same interval
        weighted_loss = weighted_logp.sum(1) / weights.view(n, -1).sum(1)

        # Average over mini-batch
        weighted_loss = -weighted_loss.mean()

        return weighted_loss
    
    def compute_dynamic_weights(self, pred, target):
        # compute_dynamic_weights
        # 函数根据预测的概率和目标类别，计算出动态的类别权重。
        # 对于每个类别，权重被设置为 1.0 除以该类别在目标中的样本数量与类别总数的乘积。
        # 这样可以保证每个类别的权重都与其在目标中的相对频率成反比。
        n, c, H, W = pred.shape

        # Calculate predicted probabilities
        p = F.softmax(pred, dim=1)

        # Compute per-class weights based on target
        class_weights = torch.zeros_like(p)
        for i in range(c):
            mask = (target == i)
            class_size = mask.sum()
            if class_size > 0:
                class_weights[:, i, :, :][mask] = 1.0 / (class_size.float() * c)

        return class_weights

# def class_weight(target):
#     weight = torch.zeros(batch_size, H, W)
#     for i in range(out_channels):
#         i_t = i * torch.ones([batch_size, H, W], dtype=torch.long)
#         loc_i = (target == i_t).to(torch.long)
#         count_i = loc_i.view(out_channels, -1).sum(1)
#         total = H*W
#         weight_i = total / count_i
#         weight_t = loc_i * weight_i.view(-1, 1, 1)
#         weight += weight_t
#     return weight

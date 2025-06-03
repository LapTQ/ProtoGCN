import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


# def focal_loss(labels, logits, alpha, gamma):
#     """Compute the focal loss between `logits` and the ground truth `labels`.

#     Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
#     where pt is the probability of being classified to the true class.
#     pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

#     Args:
#       labels: A float tensor of size [batch, num_classes].
#       logits: A float tensor of size [batch, num_classes].
#       alpha: A float tensor of size [batch_size]
#         specifying per-example weight for balanced cross entropy.
#       gamma: A float scalar modulating loss from hard and easy examples.

#     Returns:
#       focal_loss: A float32 scalar representing normalized total loss.
#     """
#     BCLoss = F.binary_cross_entropy_with_logits(
#         input=logits, target=labels, reduction="none"
#     )

#     if gamma == 0.0:
#         modulator = 1.0
#     else:
#         modulator = torch.exp(
#             -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
#         )

#     loss = modulator * BCLoss

#     weighted_loss = alpha * loss
#     focal_loss = torch.sum(weighted_loss)

#     focal_loss /= torch.sum(labels)
#     return focal_loss


# @LOSSES.register_module()
# class FocalLoss(BaseWeightedLoss):

#     def __init__(self, loss_weight=1.0, gamma=2.0, class_weight=None):
#         super().__init__(loss_weight=loss_weight)
#         self.gamma = gamma
#         self.class_weight = None
#         if class_weight is not None:
#             self.class_weight = torch.Tensor(class_weight)

#     def _forward(self, cls_score, label, **kwargs):
#         """
#         Args:
#             cls_score (torch.Tensor): [N, C] logits.
#             label (torch.Tensor): [N] hard label or [N, C] soft label.

#         Returns:
#             torch.Tensor: Focal loss.
#         """
#         assert (
#             cls_score.size() != label.size()
#         ), "Not supporting soft label for focal loss"

#         no_of_classes = cls_score.size(1)

#         labels_one_hot = F.one_hot(label, num_classes=no_of_classes).float()

#         if self.class_weight is None:
#             self.class_weight = [1.0 / no_of_classes] * no_of_classes

#         weights = torch.tensor(self.class_weight).float().to(cls_score.device)
#         weights = weights.unsqueeze(0)
#         weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
#         weights = weights.sum(1)
#         weights = weights.unsqueeze(1)
#         weights = weights.repeat(1, no_of_classes)

#         loss_cls = focal_loss(labels_one_hot, cls_score, weights, self.gamma)

#         return loss_cls


@LOSSES.register_module()
class FocalLoss(BaseWeightedLoss):

    def __init__(self, loss_weight=1.0, gamma=2.0, class_weight=None, reduction="mean"):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param class_weight: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        """
        super().__init__(loss_weight=loss_weight)
        self.gamma = gamma
        self.class_weight = class_weight
        self.reduction = reduction
            
    def _forward(self, cls_score, label, **kwargs):
        return self.multi_class_focal_loss(cls_score, label)

    def multi_class_focal_loss(self, inputs, targets):
        """Focal loss for multi-class classification."""
        if self.class_weight is not None:
            alpha = torch.Tensor(self.class_weight).to(inputs.device)
        else:
            alpha = torch.ones(inputs.size(1), device=inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        num_classes = probs.size(1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.class_weight is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

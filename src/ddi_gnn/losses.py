from typing import Callable

import torch
import torch.nn.functional as F


def binary_log_loss(scores: dict[str, torch.Tensor]) -> torch.Tensor:
    """Binary cross-entropy: positives should score high, negatives low."""
    pos_loss = F.softplus(-scores["pos"]).mean()
    neg_loss = F.softplus(scores["neg"]).mean()
    return pos_loss + neg_loss


def auc_loss(scores: dict[str, torch.Tensor]) -> torch.Tensor:
    """Square loss on the score gap, averaged over every (positive, negative) pair."""
    pos, neg = scores["pos"], scores["neg"]
    mean_gap = 1 - pos.mean() + neg.mean()
    return pos.var(unbiased=False) + neg.var(unbiased=False) + mean_gap**2


LOSSES: dict[str, Callable] = {"bce": binary_log_loss, "auc": auc_loss}

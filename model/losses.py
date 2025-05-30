import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_forecasting.metrics.quantile import QuantileLoss
import numpy as np

def mse_loss(logits: torch.Tensor, target: torch.Tensor, num_quantiles=8):
    # logits: B X T-1 x dim
    # target: B x T-1 x dim
    loss = nn.MSELoss()
    return loss(logits, target)                

def quantile_loss(logits: torch.Tensor, target: torch.Tensor, num_quantiles=8):
    loss = QuantileLoss(quantiles=np.linspace(0, 1 , num_quantiles))
    return loss(logits.reshape(-1, num_quantiles), target.reshape(-1, 1))


def rank_loss(predicted_trajectory, true_trajectory):
    # predicted_trajectory: B x T x 1
    # true_trajectory: B x T x 1
    pred_deltas = predicted_trajectory[:, 1:] - predicted_trajectory[:, :-1]
    true_deltas = true_trajectory[:, 1:] - true_trajectory[:, :-1]
    return torch.mean((torch.sign(pred_deltas) - torch.sign(true_deltas)) ** 2)


def cross_entropy_binning_loss(logits: torch.Tensor, target_bins: torch.Tensor):
    """
    logits: [B, T, 2, num_bins]
    target_bins: [B, T, 2]
    We apply cross-entropy along num_bins dimension for each of the 2 features.
    """
    B, T, Fdim, nbins = logits.shape
    # Flatten everything except the bins
    logits_2d = logits.view(-1, nbins)         # [B*T*Fdim, nbins]
    targets_1d = target_bins.view(-1)          # [B*T*Fdim]
    loss = F.cross_entropy(logits_2d, targets_1d)
    return loss


def exploration_loss_fn(x_batch: torch.Tensor, model):
    """
    Example: a placeholder for an exploration term. 
    Could measure how 'diverse' the predicted next step is from past steps, etc.
    Here we just return 0.0 by default, or a small dummy value for demonstration.
    """
    # A real approach might do:
    # 1) Predict next step
    # 2) Compare distance to mean of previous steps, etc.
    return 0.0


def convergence_loss_fn(x_batch: torch.Tensor, model):
    """
    Example: a placeholder for a convergence term. 
    Could measure how close the predicted next step is to the best known so far, etc.
    """
    return 0.0


def bar_distribution_loss(bar_dist_module, logits, targets, ohr_module):
    """
    logits: [B, T_out, D, num_bins]
    targets: [B, T, D] or [B, T_out, D]
    bar_dist_module expects [T_out, B, num_bins]
    """
    B, T_out, D, num_bins = logits.shape
    # We must ensure the target has T_out time steps, not T.
    # If your model is returning T-1 time steps in logits, slice:
    if targets.shape[1] == T_out + 1:
        # e.g. the original T was T_out+1
        targets = targets[:, 1:, :]  # drop the first time step => now [B, T_out, D]

    # Then do dimension-by-dimension, same as your snippet:
    total_loss = 0
    for dim_idx in range(D):
        # [B, T_out, num_bins] -> permute -> [T_out, B, num_bins]
        dim_logits = logits[:, :, dim_idx, :].permute(1, 0, 2)
        # [B, T_out] -> permute -> [T_out, B]
        dim_targets = targets[:, :, dim_idx].permute(1, 0)

        dim_loss = bar_dist_module(dim_logits, dim_targets, model=ohr_module)
        total_loss += dim_loss.mean()

    return total_loss


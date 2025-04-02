import torch
import torch.nn.functional as F

def cross_entropy_binning_loss(
    logits: torch.Tensor, 
    target_bins: torch.Tensor, 
    x_dim: int, 
    y_dim: int,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Computes Cross Entropy loss *separately* for X and Y portions, then returns the sum (or average) of both.

    Args:
        logits:      [B, T, x_dim + y_dim, num_bins]
        target_bins: [B, T, x_dim + y_dim]
        x_dim:       number of features allocated to X
        y_dim:       number of features allocated to Y
        reduction:   "mean" or "sum" or "none" (passed to cross_entropy). Typically "mean".

    Returns:
        A single scalar combining the cross-entropy for X and Y (by sum if `reduction="mean"`).

    Example usage:
      loss = cross_entropy_binning_loss_separate(logits, target_bins, x_dim=1, y_dim=1)
    """
    # 1) Separate out X portion (first x_dim) and Y portion (last y_dim)
    logits_x = logits[..., :x_dim, :]          # [B, T, x_dim, nbins]
    logits_y = logits[..., x_dim:x_dim+y_dim, :]  # [B, T, y_dim, nbins]
    
    target_bins_x = target_bins[..., :x_dim]   # [B, T, x_dim]
    target_bins_y = target_bins[..., x_dim:x_dim+y_dim]  # [B, T, y_dim]

    # 2) Flatten for cross entropy
    #    => from (B, T, x_dim, nbins) to (B*T*x_dim, nbins)
    B, T, _, nbins = logits_x.shape
    logits_x_flat = logits_x.reshape(-1, nbins)
    targets_x_flat = target_bins_x.reshape(-1)

    B, T, _, nbins = logits_y.shape
    logits_y_flat = logits_y.reshape(-1, nbins)
    targets_y_flat = target_bins_y.reshape(-1)

    # 3) Compute CE for X and Y, then combine
    loss_x = F.cross_entropy(logits_x_flat, targets_x_flat, reduction=reduction)
    loss_y = F.cross_entropy(logits_y_flat, targets_y_flat, reduction=reduction)

    return loss_x, loss_y 

def mse_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    x_dim: int,
    y_dim: int,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute MSE for X portion and Y portion separately, then return the sum (or average).

    Args:
        preds:     [B, T, x_dim + y_dim], model predictions (continuous)
        targets:   [B, T, x_dim + y_dim], ground-truth
        x_dim:     number of features allocated to X
        y_dim:     number of features allocated to Y
        reduction: "mean", "sum", or "none" (passed to MSE).
                   Typically "mean", so each partial loss is averaged.

    Returns:
        A single scalar combining the MSE for X and Y. (Sum or average of MSEs.)
    """
    # 1) Separate out the X portion and Y portion
    preds_x = preds[..., :x_dim]     # [B, T, x_dim]
    preds_y = preds[..., x_dim: x_dim+y_dim]  # [B, T, y_dim]

    targets_x = targets[..., :x_dim] # [B, T, x_dim]
    targets_y = targets[..., x_dim: x_dim+y_dim]

    # 2) Compute MSE for X portion
    #    For "mean" or "sum" or "none", PyTorch's F.mse_loss can do it directly
    loss_x = F.mse_loss(preds_x, targets_x, reduction=reduction)
    loss_y = F.mse_loss(preds_y, targets_y, reduction=reduction)

    # 3) Combine them. Commonly you sum them or average them.
    #    If reduction="mean", each MSE is already an average over all X or all Y elements,
    return loss_x, loss_y 


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


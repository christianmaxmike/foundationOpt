import torch
import torch.nn.functional as F

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


def bar_distribution_loss(bar_dist_module, logits, targets):
    """
    Compute the bar distribution loss for each dimension.

    Args:
      bar_dist_module: e.g. model.bar_distribution_x or model.bar_distribution_y
      logits: [B, T_out, D, num_bins]
      targets: [B, T, D] or [B, T_out, D]
        - If T = T_out + 1, we slice off the first step so targets align with T_out.
    Returns:
      total_loss (scalar)
    """
    B, T_out, D, num_bins = logits.shape

    # If our model returns (T-1) outputs but targets have T steps,
    # slice targets so shape aligns to [B, T_out, D].
    if targets.shape[1] == T_out + 1:
        targets = targets[:, 1:, :]  # now [B, T_out, D]

    total_loss = 0.0

    for dim_idx in range(D):
        # Extract logits for this dimension => shape [B, T_out, num_bins]
        # permute to [T_out, B, num_bins] for bar_dist_module
        dim_logits = logits[:, :, dim_idx, :].permute(1, 0, 2)

        # Extract targets for this dimension => shape [B, T_out]
        # permute to [T_out, B]
        dim_targets = targets[:, :, dim_idx].permute(1, 0)

        # --- CLAMP OUT-OF-RANGE VALUES ---
        # We clamp in-place so that all target values fall within
        # [borders[0], borders[-1]] for this bar_dist_module.
        with torch.no_grad():
            min_val = bar_dist_module.borders[0].item()
            max_val = bar_dist_module.borders[-1].item()
            dim_targets.clamp_(min=min_val, max=max_val)

        # Now pass the logits and clamped targets to the bar distribution
        dim_loss = bar_dist_module(dim_logits, dim_targets)  # shape [T_out, B]

        # Add the mean of this dimension's loss to total
        total_loss += dim_loss.mean()

    return total_loss



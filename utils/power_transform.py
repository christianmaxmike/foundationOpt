# power_transform.py

import torch
import numpy as np
from sklearn.preprocessing import PowerTransformer

def general_power_transform(
    x_train: torch.Tensor,
    x_apply: torch.Tensor,
    eps: float = 0.0,
    less_safe: bool = False
) -> torch.Tensor:
    """
    If eps>0 => Box-Cox with offset eps, else Yeo-Johnson.
    If the transform fails or yields NaN, fallback to shifting by mean.
    """

    # Convert to CPU numpy for scikit-learn
    x_train_np = x_train.cpu().double().numpy()
    x_apply_np = x_apply.cpu().double().numpy()

    if eps > 0:
        # Box-Cox
        try:
            pt = PowerTransformer(method='box-cox')
            pt.fit(x_train_np + eps)
            x_out_np = pt.transform(x_apply_np + eps)
        except ValueError as e:
            print("[WARN] Box-Cox transform failed:", e)
            mean_val = x_train.mean(dim=0, keepdim=True)
            x_out = x_apply - mean_val
            return x_out
    else:
        # Yeo-Johnson
        # If less_safe==False, do an optional check for huge mean/stdev
        if not less_safe:
            train_std = x_train.std()
            train_mean = x_train.mean()
            if train_std > 1e3 or abs(train_mean) > 1e3:
                print("[INFO] Large values => normalizing before Yeo-Johnson.")
                x_train_np = (x_train_np - train_mean.item()) / (train_std.item() + 1e-12)
                x_apply_np = (x_apply_np - train_mean.item()) / (train_std.item() + 1e-12)
        try:
            pt = PowerTransformer(method='yeo-johnson')
            pt.fit(x_train_np)
            x_out_np = pt.transform(x_apply_np)
        except ValueError as e:
            print("[WARN] Yeo-Johnson transform failed:", e)
            mean_val = x_train.mean(dim=0, keepdim=True)
            std_val  = x_train.std(dim=0,  keepdim=True) + 1e-12
            # fallback => shift + scale
            x_out = (x_apply - mean_val) / std_val
            return x_out

    # Convert back to torch
    x_out = torch.tensor(x_out_np, device=x_apply.device, dtype=x_apply.dtype)

    # final check
    if torch.isnan(x_out).any() or torch.isinf(x_out).any():
        print("[WARN] transform produced NaNs/Infs => fallback to mean shift.")
        mean_val = x_train.mean(dim=0, keepdim=True)
        x_out = x_apply - mean_val

    return x_out

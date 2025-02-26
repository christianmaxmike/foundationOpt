#!/usr/bin/env python

import os
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from model.pfn_transformer import PFNTransformer
from model.losses import cross_entropy_binning_loss


def f_test(x):
    """Simple 1D function in [0,1]."""
    return np.sin(2 * np.pi * x) * 0.5 + 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PFN Transformer on a 1D function.")
    parser.add_argument('--config_dir', type=str,
                        default="/work/dlclarge1/janowski-opt/foundationOpt/configs/",
                        help="Path to YAML config file.")
    parser.add_argument('--checkpoint', type=str,   
                        default=None,
                        help="Path to trained model checkpoint (e.g. model_final.pth).")
    parser.add_argument('--run_name', type=str,
                        default=None,
                        help="Path to trained model checkpoint (e.g. model_final.pth).")
    parser.add_argument('--steps', type=int, default=20,
                        help="Number of next-step predictions to produce.")
    parser.add_argument('--output_dir', type=str, default="./eval_results",
                        help="Where to save evaluation plots.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config_path = os.path.join(args.config_dir, f"{args.run_name}.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    run_name = config.get('run_name', 'default_run')
    if args.checkpoint is None:
        # Fallback path
        args.checkpoint = f"./checkpoints/{run_name}/model_final.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Rebuild model
    model_config = config['model']
    model = PFNTransformer(
        input_dim=model_config.get('input_dim', 2),
        hidden_dim=model_config.get('hidden_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        num_heads=model_config.get('num_heads', 2),
        dropout=model_config.get('dropout', 0.1),
        num_bins=model_config.get('num_bins', 32),
        forecast_steps=1,
        use_autoregression=model_config.get('use_autoregression', False)
    ).to(device)

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Start with 2 context points so teacher-forcing can produce at least 1 step
    x0, x1 = 0.2, 0.8
    context = [x0, x1]
    context_vals = [f_test(x0), f_test(x1)]

    all_x = [x0, x1]
    all_y = [f_test(x0), f_test(x1)]

    # Generate next steps
    for step_i in range(args.steps):
        seq_np = np.stack([np.array(context), np.array(context_vals)], axis=-1)  # shape [T,2]
        seq_torch = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,T,2]

        with torch.no_grad():
            logits, _ = model.forward_with_binning(seq_torch)
            # logits => [1, T_out, 2, num_bins]
            if logits.shape[1] == 0:
                print(f"No next-step prediction for T={seq_np.shape[0]} (teacher-forcing?). Stopping.")
                break

            next_logits = logits[:, -1, :, :]  # [1, 2, num_bins]
            # For next x
            x_bin_logits = next_logits[:, 0, :]  # [1, num_bins]
            x_bin_idx = torch.argmax(x_bin_logits, dim=-1)  # [1]
            x_bin_val = model.binner.unbin_values(x_bin_idx)  # shape [1]
            next_x = x_bin_val.item()

        next_y = f_test(next_x)
        context.append(next_x)
        context_vals.append(next_y)
        all_x.append(next_x)
        all_y.append(next_y)

    #####################
    # Plot 1: Single figure with entire final trajectory
    #####################
    X_grid = np.linspace(0, 1, 200)
    Y_grid = [f_test(xx) for xx in X_grid]

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(X_grid, Y_grid, 'k--', label='f(x)')

    # Color each point by iteration
    cmap = plt.get_cmap('plasma')
    for i, (xx, yy) in enumerate(zip(all_x, all_y)):
        color = cmap(i / len(all_x))
        ax1.scatter(xx, yy, color=color, s=50, zorder=3)

    ax1.set_title("Trajectory of Proposed Points (Colored by Iteration)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend(loc='upper right')

    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    fig1.savefig(os.path.join(out_dir, "trajectory_colored.png"), dpi=120)
    plt.close(fig1)

    #####################
    # Plot 2: 2 columns per step => context + distribution
    #####################
    steps_to_plot = args.steps
    fig2, axs = plt.subplots(nrows=steps_to_plot, ncols=2,
                             figsize=(12, 3.0 * steps_to_plot),
                             sharex=False, sharey=False)

    # If steps_to_plot==1, axs might not be a 2D array => handle that:
    if steps_to_plot == 1:
        axs = np.array([axs])  # shape [1,2]

    for row_i in range(steps_to_plot):
        ax_left = axs[row_i, 0]
        ax_right = axs[row_i, 1]

        # Partial context: up to row_i+2
        partial_x = all_x[:row_i+2]
        partial_y = all_y[:row_i+2]

        if (row_i + 2) < len(all_x):
            new_x = all_x[row_i + 2]
            new_y = all_y[row_i + 2]
        else:
            new_x = all_x[-1]
            new_y = all_y[-1]

        # Left subplot => context + new pt
        ax_left.plot(X_grid, Y_grid, 'b--', alpha=0.7, label="f(x)")
        ax_left.scatter(partial_x[:-1], partial_y[:-1], c='blue', label="Context")
        ax_left.scatter([partial_x[-1]], [partial_y[-1]], c='orange', label="Last Pt")
        ax_left.scatter([new_x], [new_y], c='red', label="New Pt")
        ax_left.set_title(f"Step {row_i+1}: Context + Next Pt")
        ax_left.set_xlim(0,1)
        ax_left.set_ylim(0,1)
        ax_left.legend(loc='best')

        # Right subplot => distribution over x-bins
        seq_np = np.stack([partial_x, partial_y], axis=-1)
        seq_torch = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model.forward_with_binning(seq_torch)
            if logits.shape[1] == 0:
                # No prediction
                prob = np.zeros(model.num_bins)
                chosen_bin = 0
                new_x_bin_idx = 0
            else:
                next_logits = logits[:, -1, 0, :]  # [B=1, num_bins]
                prob = torch.softmax(next_logits, dim=-1).cpu().numpy().flatten()
                chosen_bin = np.argmax(prob)
                # The actual bin index for new_x
                new_x_bin_idx = model.binner.bin_values(torch.tensor([[new_x]], dtype=torch.float32)).item()

        bins = np.arange(model.num_bins)
        ax_right.bar(bins, prob, color='gray', alpha=0.7)
        ax_right.axvline(chosen_bin, color='red', linestyle='--', label=f"Chosen bin = {chosen_bin}")
        ax_right.axvline(new_x_bin_idx, color='blue', linestyle=':', label=f"Actual bin = {new_x_bin_idx}")
        ax_right.set_title(f"Step {row_i+1}: Dist over x-bins")
        ax_right.set_xlim(0, model.num_bins-1)
        ax_right.set_ylim(0, 1.05)
        ax_right.legend(loc='best')

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "detailed_steps.png"), dpi=120)
    plt.close(fig2)

    print(f"Done. Plots saved in {out_dir}")


if __name__ == "__main__":
    main()

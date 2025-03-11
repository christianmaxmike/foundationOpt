#!/usr/bin/env python

import os
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from dill import load

# 1) Import your PFNTransformer that has transform_input / inverse_transform_x built-in
from model.pfn_transformer import PFNTransformer
from model.losses import cross_entropy_binning_loss

def rec_fnc(fnc_dict, min_x_value=0.0, max_x_value=1.0, sequence_length=100):
    """
    Reconstruct a 1D function from a dictionary that specifies multiple trig components.
    Returns:
      x_values (np.ndarray), y_values (np.ndarray)
    """
    x_values = np.linspace(min_x_value, max_x_value, sequence_length)
    y_values = np.zeros_like(x_values)
    for i in range(fnc_dict['num_components']):
        trig = fnc_dict['trigonometrics'][i]
        amp  = fnc_dict['amplitudes'][i]
        freq = fnc_dict['frequencies'][i]
        phase= fnc_dict['phases'][i]
        y_values += amp * trig(freq * x_values + phase)
    return x_values, y_values

def make_single_eval_func(fnc_dict):
    """
    Returns a callable f(x) that can evaluate the sum of the trig components.
    """
    def f_test(x):
        val = 0.0
        for i in range(fnc_dict['num_components']):
            trig = fnc_dict['trigonometrics'][i]
            amp  = fnc_dict['amplitudes'][i]
            freq = fnc_dict['frequencies'][i]
            phase= fnc_dict['phases'][i]
            val += amp * trig(freq * x + phase)
        return val
    return f_test

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PFN Transformer on a 1D function (Minimization).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--config_dir', type=str,
                        default="./configs/",
                        help="Directory where the YAML config is stored.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Full path to trained model checkpoint (overrides run_name).")
    parser.add_argument('--run_name', type=str, default=None,
                        help="Name of the run (used to load config.yaml if checkpoint not given).")
    parser.add_argument('--steps', type=int, default=30,
                        help="Number of next-step predictions to produce.")
    parser.add_argument('--output_dir', type=str, default="./eval_results",
                        help="Where to save evaluation plots.")
    parser.add_argument('--sample_next_x', action='store_true',
                        help="Sample next x from the distribution instead of argmin.")
    parser.add_argument('--nar_inference_flag', action='store_true',
                        help="Use NAR mode for inference (parallel prediction).")
    parser.add_argument('--model_idx', type=int, default=0,
                        help="Index of the function dictionary in models_0.dill.")
    return parser.parse_args()

def get_temperature(step_i, n_steps, temp_high=2.0, temp_low=0.1):
    """
    Simple linear temperature schedule from temp_high -> temp_low over 'n_steps'.
    """
    if n_steps <= 1:
        return temp_low
    alpha = step_i / float(n_steps - 1)
    return temp_high + alpha * (temp_low - temp_high)

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------------------------------------------------------------
    # A) Load config and build PFNTransformer
    # ---------------------------------------------------------------------
    config_path = os.path.join(args.config_dir, f"{args.run_name}.yaml")
    if not os.path.exists(config_path) and args.checkpoint is None:
        raise FileNotFoundError(
            f"Neither --checkpoint nor a valid config file could be found at {config_path}."
        )

    if not os.path.exists(config_path):
        print(f"Warning: config not found at {config_path}, using default placeholders.")
        config = {}
        model_config = {}
        data_config = {}
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', {})
        data_config = config.get('data', {})

    run_name = config.get('run_name', 'default_run')
    if args.checkpoint is None:
        # If no explicit checkpoint path given, guess from the run_name
        ckpt_dir = config.get('training', {}).get('ckpt_dir', './checkpoints')
        args.checkpoint = os.path.join(ckpt_dir, run_name, "model_final.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build the PFNTransformer
    # If your training script stored x_min, x_max, y_min, y_max in the model state_dict buffers,
    # then we only need to pass them as "None" here. The loaded state_dict will overwrite them.
    loss_type = model_config.get('loss_type', 'cross_entropy')
    model = PFNTransformer(
        x_dim=model_config.get('x_dim', 1),
        y_dim=model_config.get('y_dim', 1),
        hidden_dim=model_config.get('hidden_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        num_heads=model_config.get('num_heads', 2),
        dropout=model_config.get('dropout', 0.1),
        num_bins=model_config.get('num_bins', 32),
        pre_norm=model_config.get('pre_norm', True),
        use_positional_encoding=model_config.get('use_positional_encoding', False),
        use_autoregression=model_config.get('use_autoregression', False),
        nar_inference_flag=args.nar_inference_flag,
        use_bar_distribution=(loss_type == 'bar'),
        bar_dist_smoothing=model_config.get('bar_dist_smoothing', 0.0),
        full_support=model_config.get('full_support', False),
        transform_type=data_config.get('transform_type', 'none'),
        x_min=None,  # Will be overwritten by the checkpoint buffers if they exist
        x_max=None,
        y_min=None,
        y_max=None,
    ).to(device)

    # Load checkpoint and set model to eval mode.
    print(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Print out the model's recognized domain bounds
    print("Model transformation bounds (in .eval() mode):")
    print(f"  x in [{model.x_min:.3f}, {model.x_max:.3f}]")
    print(f"  y in [{model.y_min:.3f}, {model.y_max:.3f}]")

    # ---------------------------------------------------------------------
    # B) Load ground-truth function from a .dill file and set up the test function
    # ---------------------------------------------------------------------
    # If your data config includes a "models_path", load that here
    models_path = data_config.get('models_path', './models_0.dill')
    with open(models_path, "rb") as f:
        models_data = load(f)  # Typically a list or dict of 1D function definitions
    gt = models_data[args.model_idx]
    f_test = make_single_eval_func(gt)

    # Optionally, use model.x_min and model.x_max for the domain:
    x_grid_full, y_grid_full = rec_fnc(
        gt,
        min_x_value=model.x_min,   # original domain for plotting
        max_x_value=model.x_max,
        sequence_length=200
    )

    # ---------------------------------------------------------------------
    # C) Initialize a context of two points within [x_min, x_max]
    #    For example, pick 0.1 and 0.9 if that domain is [0,1].
    # ---------------------------------------------------------------------
    # Make sure these context points lie in [model.x_min, model.x_max] so f_test is valid there
    x0 = model.x_min + 0.1 * (model.x_max - model.x_min)  
    x1 = model.x_min + 0.9 * (model.x_max - model.x_min)
    context = [x0, x1]
    context_vals = [f_test(x0), f_test(x1)]
    all_x = [x0, x1]
    all_y = [context_vals[0], context_vals[1]]

    # ---------------------------------------------------------------------
    # D) Generate next steps using PFNTransformer + bin sampling
    # ---------------------------------------------------------------------
    for step_i in range(args.steps):
        # 1) Build [B=1, T, (x_dim+y_dim)] from our context
        seq_np = np.stack([np.array(context), np.array(context_vals)], axis=-1)  # shape [T, 2]
        seq_torch = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, 2]
        
        # 2) Ask the model for the next-step logits
        with torch.no_grad():
            # forward_with_binning automatically calls transform_input if eval()
            logits, _ = model.forward_with_binning(seq_torch)
            # logits shape: [1, T_out, input_dim, num_bins]
            # In AR mode with T steps in -> T-1 steps out, we want the last step's logits
            if logits.shape[1] == 0:
                print("No next-step prediction (sequence too short). Stopping.")
                break
            next_logits = logits[:, -1, :, :]  # shape [1, input_dim, num_bins]

            # Focus on the x-dimension (index 0) if x_dim=1
            x_logits = next_logits[:, 0, :]  # shape [1, num_bins]

            # 3) Convert logits to a distribution and pick next_x
            if not args.sample_next_x:
                # Argmin approach => we might want to find the bin with the smallest "value"
                # but if it's purely a standard CE distribution, we might do argmax of prob
                # In typical "minimization," you might invert the logits or something else.
                x_bin_idx = torch.argmax(x_logits, dim=-1)
            else:
                # Temperature sampling approach
                temperature = get_temperature(step_i, args.steps, temp_high=2.0, temp_low=0.7)
                prob = torch.softmax(x_logits / temperature, dim=-1)
                x_bin_idx = torch.multinomial(prob, 1)

            # 4) Turn bin indices -> scaled x in [0,1], then inverse_transform to the original domain
            x_bin_val = model.binner_x.unbin_values(x_bin_idx)  # in model's internal scale
            next_x = x_bin_val.item()  # scalar

        # 5) Evaluate ground truth at next_x
        next_y = f_test(next_x)

        # 6) Append to the growing context
        context.append(next_x)
        context_vals.append(next_y)
        all_x.append(next_x)
        all_y.append(next_y)

    # ---------------------------------------------------------------------
    # E) Plot the final trajectory
    # ---------------------------------------------------------------------
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(x_grid_full, y_grid_full, '--', label='Ground truth')
    cmap = plt.get_cmap('plasma')
    for i, (xx, yy) in enumerate(zip(all_x, all_y)):
        color = cmap(i / len(all_x))
        ax1.scatter(xx, yy, s=50, color=color, zorder=3)
    ax1.set_title("Trajectory of Proposed Points (Minimization Demo)")
    ax1.set_xlabel("x (original domain)")
    ax1.set_ylabel("f(x)")
    ax1.legend(loc='best')
    fig1.savefig(os.path.join(out_dir, "trajectory.png"), dpi=120)
    plt.close(fig1)

    # ---------------------------------------------------------------------
    # F) Optional detailed plots for each step
    # ---------------------------------------------------------------------
    steps_to_plot = args.steps
    fig2, axs = plt.subplots(nrows=steps_to_plot, ncols=2,
                             figsize=(12, 3.0 * steps_to_plot),
                             sharex=False, sharey=False)
    if steps_to_plot == 1:
        axs = np.array([axs])

    for row_i in range(steps_to_plot):
        ax_left = axs[row_i, 0]
        ax_right = axs[row_i, 1]

        # Left plot: partial context up to row_i+2
        partial_x = all_x[:row_i+2]
        partial_y = all_y[:row_i+2]
        if (row_i + 2) < len(all_x):
            new_x = all_x[row_i+2]
            new_y = all_y[row_i+2]
        else:
            new_x = all_x[-1]
            new_y = all_y[-1]

        # Plot ground truth
        ax_left.plot(x_grid_full, y_grid_full, 'b--', alpha=0.7, label="f(x)")
        # Plot known points
        ax_left.scatter(partial_x[:-1], partial_y[:-1], c='blue', label="Context")
        # Mark the last point in orange
        ax_left.scatter([partial_x[-1]], [partial_y[-1]], c='orange', label="Last Pt")
        # Mark the newly chosen point in red
        ax_left.scatter([new_x], [new_y], c='red', label="New Pt")
        ax_left.set_title(f"Step {row_i+1}: Context + Next Pt")
        ax_left.set_xlim(model.x_min, model.x_max)
        min_y, max_y = min(y_grid_full.min(), min(all_y)), max(y_grid_full.max(), max(all_y))
        ax_left.set_ylim(min_y - 0.1, max_y + 0.1)
        ax_left.legend(loc='best')

        # Right plot: distribution over x-bins for that step
        seq_np = np.stack([np.array(partial_x), np.array(partial_y)], axis=-1)
        seq_torch = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model.forward_with_binning(seq_torch)
            if logits.shape[1] == 0:
                # No next-step prediction
                prob = np.zeros(model.num_bins)
                chosen_bin = 0
                new_x_bin_idx = 0
            else:
                # We want the last time-step's logits for the x-dim
                next_logits = logits[:, -1, 0, :]  # [1, num_bins]
                prob = torch.softmax(next_logits, dim=-1).cpu().numpy().flatten()

                # chosen_bin is the argmax
                chosen_bin = np.argmax(prob)

                # we also figure out which bin the "actual" new_x fell into
                # (in the model's scale), i.e. bin_values() -> integer index
                new_x_scaled = (1 if row_i+2 > len(partial_x) else all_x[row_i+2])
                # but careful: new_x is in original domain, so let's
                # re-tensor it and transform to model scale:
                new_x_scaled_t = torch.tensor([new_x_scaled], device=device, dtype=torch.float32)
                # new_x_bin_idx = model.binner_x.bin_values(model.transform_input(
                #     new_x_scaled_t.view(1,1,-1)  # shape [B=1, T=1, D=1 for x]
                # )[..., :1]).item()
                new_x_bin_idx = model.binner_x.bin_values(new_x_scaled_t).item()

        bins = np.arange(model.num_bins)
        ax_right.bar(bins, prob, alpha=0.7)
        ax_right.axvline(chosen_bin, color='red', linestyle='--', label=f"Chosen bin = {chosen_bin}")
        ax_right.axvline(new_x_bin_idx, color='blue', linestyle=':', label=f"Actual bin = {new_x_bin_idx}")
        ax_right.set_title(f"Step {row_i+1}: Distribution over x-bins")
        ax_right.set_xlim(0, model.num_bins - 1)
        ax_right.set_ylim(0, 1.05)
        ax_right.legend(loc='best')

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "detailed_steps.png"), dpi=120)
    plt.close(fig2)

    print(f"Done. Plots saved in {out_dir}")

if __name__ == "__main__":
    main()

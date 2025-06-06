#!/usr/bin/env python

import os
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from dill import load
from sklearn.preprocessing import PowerTransformer

# ----------------------------
# 1) Import your PFNTransformer
#    and cross_entropy_binning_loss if needed
# ----------------------------
from model.pfn_transformer import PFNTransformer
from model.losses import cross_entropy_binning_loss

from utils.preprocess import load_and_preprocess_data

# ----------------------------------------------------
# 2) rec_fnc: given a function dictionary, returns
#    (x_values, y_values) over a uniform grid
# ----------------------------------------------------
def rec_fnc(fnc_dict, min_x_value=0.0, max_x_value=1.0, sequence_length=100):
    x_values = np.linspace(min_x_value, max_x_value, sequence_length)
    y_values = np.zeros_like(x_values)
    for i in range(fnc_dict['num_components']):
        trig = fnc_dict['trigonometrics'][i]
        amp  = fnc_dict['amplitudes'][i]
        freq = fnc_dict['frequencies'][i]
        phase= fnc_dict['phases'][i]
        y_values += amp * trig(freq * x_values + phase)
    return x_values, y_values

# ----------------------------------------------------
# 3) make_single_eval_func: returns a function f(x)
# ----------------------------------------------------
def make_single_eval_func(fnc_dict):
    def f_test(x):
        y_val = 0.0
        for i in range(fnc_dict['num_components']):
            trig = fnc_dict['trigonometrics'][i]
            amp  = fnc_dict['amplitudes'][i]
            freq = fnc_dict['frequencies'][i]
            phase= fnc_dict['phases'][i]
            y_val += amp * trig(freq * x + phase)
        return y_val
    return f_test

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PFN Transformer on a 1D function (MINIMIZATION).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--config_dir', type=str,
                        default="./configs/",
                        help="Path to YAML config file.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Full path to trained model checkpoint (overrides run_name).")
    parser.add_argument('--run_name', type=str, default="bar",
                        help="Name of the run (used to load config).")
    parser.add_argument("--folder", type=str, default="_ctx0_bar_d64_l6_h4_next_PE_binsCleaned",
                        help="Name of the folder to load the model from")
    parser.add_argument('--steps', type=int, default=16,
                        help="Number of next-step predictions to produce.")
    parser.add_argument('--output_dir', type=str, default="./eval_results",
                        help="Where to save evaluation plots.")
    parser.add_argument('--sample_next_x', action='store_true',
                        help="Sample next x from distribution instead of taking argmin.")
    parser.add_argument('--nar_inference_flag', action='store_true',
                        help="Use NAR mode for inference (parallel prediction).")
    parser.add_argument('--model_idx', type=int, default=3,
                        help="Index of the function dictionary in models_0.dill.")
    parser.add_argument('--sequence_length', type=int, default=None, 
                        help="Length of the input sequences.")
    # parser.add_argument("--ctx_length", type=int, default=5)
    return parser.parse_args()

def get_temperature(step_i, n_steps, temp_high=2.0, temp_low=0.1):
    if n_steps <= 1:
        return temp_low
    alpha = step_i / float(n_steps - 1)
    temperature = temp_high + alpha * (temp_low - temp_high)
    return temperature

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # ---------------------------------------------------------------------
    # A) Load config and build PFNTransformer
    # ---------------------------------------------------------------------
    config_path = os.path.join(args.config_dir, f"{args.run_name}.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    run_name = config.get('run_name', 'default_run')
    if args.checkpoint is None:
        args.checkpoint = f"./checkpoints/{run_name}/{args.folder}/model_epoch70.pth"
        #if args.ctx_length != 0:
        #else:
        #    args.checkpoint = f"./checkpoints/{run_name}/model_final.pth"

    device = ['cuda', 'mps', 'cpu'][np.argmax([torch.cuda.is_available(), torch.backends.mps.is_available(), True])]
    print (device)

    data_config_input = config['data']
    data_dict = load_and_preprocess_data(data_config=data_config_input, sequence_length=args.sequence_length, device=device)
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']

    y_train_best = data_dict.get('y_train_best', None)
    y_val_best = data_dict.get('y_val_best', None)
    y_test_best = data_dict.get('y_test_best', None)
    
    X_train_last = data_dict.get("X_train_last", None)
    X_test_last = data_dict.get("X_test_last", None)
    X_val_last = data_dict.get("X_val_last", None)
    y_train_last = data_dict.get("y_train_last", None)
    y_test_last = data_dict.get("y_test_last", None)
    y_val_last = data_dict.get("y_val_last", None)
    data_config_model = data_dict['data_config']

    # get model configuration
    model_config = config['model']
    # get ctx length
    ctx_length = model_config.get('ctx_length', 5)
    # get loss type
    loss_type = model_config.get('loss_type', 'cross_entropy')
    model = PFNTransformer(
        # Note: our PFNTransformer uses separate x_dim and y_dim.
        x_dim=model_config.get('x_dim', 1),
        y_dim=model_config.get('y_dim', 1),
        hidden_dim=model_config.get('hidden_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        num_heads=model_config.get('num_heads', 2),
        dropout=model_config.get('dropout', 0.1),
        num_bins=model_config.get('num_bins', 32),
        pre_norm=model_config.get('pre_norm', True),
        use_positional_encoding=model_config.get('use_positional_encoding', False),
        use_autoregression=model_config.get('use_autoregressive', False),
        use_bar_distribution=(loss_type == 'bar'),
        bar_dist_smoothing=model_config.get('bar_dist_smoothing', 0.0),
        full_support=model_config.get('full_support', False),
        nar_inference_flag=args.nar_inference_flag,
        x_min=data_config_model['x_min'],
        x_max=data_config_model['x_max'],
        y_min=data_config_model['y_min'],
        y_max=data_config_model['y_max'],
        loss_type=loss_type,
        device=device,
        ctx_length=ctx_length
    ).to(device)

    
    # Load checkpoint and set model to eval mode.
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Read transformation bounds from the model and print them.
    print("Model transformation bounds:")
    print(f"  x: [{model.x_min}, {model.x_max}]")
    print(f"  y: [{model.y_min}, {model.y_max}]")

    if loss_type=="bar":
        model.binner_x.min_val = model.x_min
        model.binner_x.max_val = model.x_max
        model.binner_y.min_val = model.y_min
        model.binner_y.max_val = model.y_max

    # ---------------------------------------------------------------------
    # B) Load function dictionary from models_0.dill and build evaluation function.
    # ---------------------------------------------------------------------
    data_config = config['data']
    with open(data_config['models_path'], "rb") as f:
        models_data = load(f)  # Typically a list or dict of functions.
    gt = models_data[args.model_idx]
    
    # Option: use model.x_min and model.x_max for plotting the ground truth domain.
    x_grid_full, y_grid_full = rec_fnc(gt, min_x_value=model.x_min, max_x_value=model.x_max, sequence_length=200)
    f_test = make_single_eval_func(gt)
    
    # ---------------------------------------------------------------------
    # C) Initialize context points (teacher-forcing)
    # ---------------------------------------------------------------------
    #x0, x1 = list(X_train[args.model_idx, :2, :].cpu().numpy().flatten()) # 0.1, 0.9  # these should be in the original ground truth domain [0,1]
    #context = [x0, x1]
    #context_vals = [f_test(x0), f_test(x1)]
    #all_x = [x0, x1]
    #all_y = [context_vals[0], context_vals[1]]
    xs = list(X_train[args.model_idx, :2, :].cpu().numpy().flatten()) # 0.1, 0.9  # these should be in the original ground truth domain [0,1]
    
    #xs = list(X_train[args.model_idx, :ctx_length+1, :].cpu().numpy().flatten()) # 0.1, 0.9  # these should be in the original ground truth domain [0,1]
    context = xs #[x0, x1]
    context_vals = [f_test(x) for x in xs]
    all_x = context # [x0, x1]
    all_y = context_vals # [context_vals[0], context_vals[1]]
    # context= list([v[0] for v in (X_train[0, :10, :]).cpu().numpy()])
    # all_x = context
    # all_y = [f_test(v) for v in context]
    # context_vals = all_y
    # ---------------------------------------------------------------------
    # D) Generate next steps using PFNTransformer + bin sampling
    # ---------------------------------------------------------------------
    all_logits = []
    probs = []
    for step_i in range(args.steps):
        # Build sequence: shape [T, input_dim] where input_dim = x_dim+y_dim.
        seq_np = np.stack([np.array(context), np.array(context_vals)], axis=-1)  # [T, 2]
        seq_torch = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, 2]
        with torch.no_grad():
            logits, _ = model.forward_with_binning(seq_torch, None, ctx_length=ctx_length) # shape: B x T x (dimx+dim_y) x numbins
            if logits.shape[1] == 0:
                print("No next-step prediction. Stopping.")
                break

            if loss_type=="bar":
                next_logits = logits[:, -1, :, :]  # [1, 2, num_bins]
                # For x dimension, use model.binner_x.
                if not args.sample_next_x:
                    x_bin_logits = next_logits[:, 0, :]
                    prob = torch.softmax(x_bin_logits, dim=-1)#.cpu().numpy().flatten()
                    # prob = 1.0 - prob  # For minimization: lower values preferred.
                    #chosen_bin = np.argmax(prob)
                    #x_bin_idx = torch.argmin(x_bin_logits, dim=-1)
                    # x_bin_idx = torch.argmax(x_bin_logits, dim=-1)
                    x_bin_idx = torch.argmax(prob, dim=-1)
                else:
                    temperature = get_temperature(step_i, args.steps, temp_high=2.0, temp_low=0.7)
                    prob = torch.softmax(next_logits / temperature, dim=-1)
                    prob_x = prob[:, 0, :]

                    probs.append(prob_x)
                    x_bin_idx = torch.multinomial(prob_x, 1)
                #print ("before binner:", x_bin_idx.item())
                x_bin_val = model.binner_x.unbin_values(x_bin_idx)
                # x_bin_val = model.orh_x.unbin(x_bin_idx)
                # print("after binner:", x_bin_val.item())
                next_x = x_bin_val.item()
            elif loss_type=="quantile":
                quantiles = torch.linspace(0,1,16)
                print ("next_x", logits[:, -1, :16])
                next_x = logits[:,-1,:16].max(dim=-1)[1].item()
                next_x = quantiles[next_x]
            else: # loss_type=="mse":
                next_x = logits[:, -1, 0].item()
        all_logits.append(logits)
        next_y = f_test(next_x)
        context.append(next_x)
        context_vals.append(next_y)
        all_x.append(next_x)
        all_y.append(next_y)

        #pt = PowerTransformer()
        #ctx_tmp = np.array(context)
        #ctx_vals_tmp = np.array(context_vals)
        #context = list(pt.fit_transform(ctx_tmp.reshape(-1, 1)).reshape(ctx_tmp.shape))
        #context_vals = list(pt.fit_transform(ctx_vals_tmp.reshape(-1, 1)).reshape(ctx_vals_tmp.shape))
        
        
        #y_batch_pt = pt.fit_transform(y_batch_raw.reshape(-1, 1)).reshape(y_batch_raw.shape)
    
    # ---------------------------------------------------------------------
    # E) Plot final trajectory
    # ---------------------------------------------------------------------
    out_dir = os.path.join(args.output_dir, run_name, f"{args.folder}_model{args.model_idx}")
    os.makedirs(out_dir, exist_ok=True)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(x_grid_full, y_grid_full, 'k--', label='Ground truth function')
    cmap = plt.get_cmap('plasma')
    for i, (xx, yy) in enumerate(zip(all_x, all_y)):
        color = cmap(i / len(all_x))
        ax1.scatter(xx, yy, color=color, s=50, zorder=3)
    ax1.set_title("Trajectory of Proposed Points (Minimization)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend(loc='upper right')
    fig1.savefig(os.path.join(out_dir, f"trajectory.png"), dpi=120)
    plt.close(fig1)
    
    # ---------------------------------------------------------------------
    # F) Detailed plots per step (optional)
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
        partial_x = all_x[:row_i+3]
        partial_y = all_y[:row_i+3]
        if (row_i + 3) < len(all_x):
            new_x = all_x[row_i+2]
            new_y = all_y[row_i+2]
        else:
            new_x = all_x[-1]
            new_y = all_y[-1]
        ax_left.plot(x_grid_full, y_grid_full, 'b--', alpha=0.7, label="f(x)")
        ax_left.scatter(partial_x[:-2], partial_y[:-2], c='blue', label="Context")
        ax_left.scatter([partial_x[-2]], [partial_y[-2]], c='orange', label="Last Pt")
        ax_left.scatter([new_x], [new_y], c='red', label="New Pt")
        ax_left.set_title(f"Step {row_i+1}: Context + Next Pt")
        #ax_left.set_xlim(0, 1)
        ax_left.set_ylim(min(y_grid_full)-0.1, max(y_grid_full)+0.1)
        ax_left.legend(loc='best')
        #seq_np = np.stack([np.array(partial_x), np.array(partial_y)], axis=-1)
        #seq_torch = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = all_logits[row_i]
            # logits, _ = model.forward_with_binning(seq_torch)
            if logits.shape[1] == 0:
                prob = np.zeros(model.num_bins)
                chosen_bin = 0
                new_x_bin_idx = 0
            else:
                if loss_type == 'bar':
                    next_logits = logits[:, -1, 0, :]  # logits for x dimension
                    prob = torch.softmax(next_logits, dim=-1).cpu().numpy().flatten()
                    # prob = 1.0 - prob  # For minimization: lower values preferred.
                    chosen_bin = np.argmax(prob)
                    new_x_bin_idx = model.binner_x.bin_values(
                        torch.tensor([[new_x]], dtype=torch.float32, device=device)
                    ).item()
    
        if loss_type == 'bar':
            bins = np.arange(model.num_bins)
            ax_right.bar(bins, prob, color='gray', alpha=0.7)
            #ax_right.axvline(chosen_bin, color='red', linestyle='--', label=f"Chosen bin (argmax bin) = {chosen_bin}")
            #ax_right.axvline(new_x_bin_idx, color='blue', linestyle=':', label=f"Actual bin (new x bin) = {new_x_bin_idx}")
            ax_right.axvline(chosen_bin, color='red', linestyle='--', label=f"Argmax bin = {chosen_bin}")
            ax_right.axvline(new_x_bin_idx, color='blue', linestyle=':', label=f"Pred. bin) = {new_x_bin_idx}")
            ax_right.set_title(f"Step {row_i+1}: Distribution over x-bins")
            ax_right.set_xlim(0-0.5, model.num_bins-1+0.5)
            ax_right.set_ylim(0, 1.05)
            ax_right.legend(loc='best')
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "detailed_steps.png"), dpi=120)
    plt.close(fig2)
    
    print(f"Done. Plots saved in {out_dir}")

if __name__ == "__main__":
    main()

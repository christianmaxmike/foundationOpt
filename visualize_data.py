#!/usr/bin/env python

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Training Data (X,Y).")
    parser.add_argument('--data_path', type=str, default="/work/dlclarge1/janowski-opt/data/simple/single/1D/data.pkl",
                        help="Path to the dataset .pkl file (containing X_train, y_train, etc.).")
    parser.add_argument('--output_dir', type=str, default="./data_viz",
                        help="Directory to save the generated plots.")
    parser.add_argument('--num_examples', type=int, default=5,
                        help="Number of random sequences to plot in the 'time vs. Y' figure.")
    parser.add_argument('--num_xy_plots', type=int, default=20,
                        help="Number of random sequences to plot in the 'X vs. Y' figure.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) Load Data
    # -------------------------------------------------------------------------
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    # We assume the dict has 'X_train', 'y_train', etc.
    X_train = data['X_train']  # shape [N, T, x_dim] or [N, T] if 1D
    y_train = data['y_train']  # shape [N, T, y_dim] or [N, T] if 1D

    # Convert to numpy if needed
    if isinstance(X_train, (np.ndarray, list)):
        X = np.array(X_train)
        Y = np.array(y_train)
    else:
        # If they are torch tensors, do:
        X = X_train.detach().cpu().numpy()
        Y = y_train.detach().cpu().numpy()

    # for demonstration, assume shapes => [N, T] or [N, T, x_dim/y_dim]
    N = X.shape[0]
    T = X.shape[1]

    # Infer dimensions
    if X.ndim == 2:
        x_dim = 1
    else:
        x_dim = X.shape[-1]
    if Y.ndim == 2:
        y_dim = 1
    else:
        y_dim = Y.shape[-1]

    print(f"Loaded training data: X.shape={X.shape}, Y.shape={Y.shape}")
    print(f"  => N={N} sequences, T={T} steps, x_dim={x_dim}, y_dim={y_dim}")

    # -------------------------------------------------------------------------
    # 2) Histograms of X and Y
    # -------------------------------------------------------------------------
    # Flatten everything
    X_flat = X.reshape(-1, x_dim)  # [N*T, x_dim]
    Y_flat = Y.reshape(-1, y_dim)  # [N*T, y_dim]

    fig_hist, axs_hist = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    # Left subplot => X
    if x_dim == 1:
        axs_hist[0].hist(X_flat[:, 0], bins=50, color='blue', alpha=0.7)
        axs_hist[0].set_title("Histogram of X (train)")
        axs_hist[0].set_xlabel("X value")
        axs_hist[0].set_ylabel("Count")
    else:
        for d in range(x_dim):
            axs_hist[0].hist(X_flat[:, d], bins=50, alpha=0.5, label=f"X dim {d}")
        axs_hist[0].legend()
        axs_hist[0].set_title("Histogram of X (multiple dims)")

    # Right subplot => Y
    if y_dim == 1:
        axs_hist[1].hist(Y_flat[:, 0], bins=50, color='green', alpha=0.7)
        axs_hist[1].set_title("Histogram of Y (train)")
        axs_hist[1].set_xlabel("Y value")
    else:
        for d in range(y_dim):
            axs_hist[1].hist(Y_flat[:, d], bins=50, alpha=0.5, label=f"Y dim {d}")
        axs_hist[1].legend()
        axs_hist[1].set_title("Histogram of Y (multiple dims)")

    fig_hist.suptitle("Histograms of X and Y (training data)")
    fig_hist.tight_layout()
    hist_path = os.path.join(args.output_dir, "hist_XY_train.png")
    fig_hist.savefig(hist_path, dpi=120)
    plt.close(fig_hist)
    print(f"Saved histogram plot: {hist_path}")

    # -------------------------------------------------------------------------
    # 3) Plot a few random sequences (time vs. Y)
    # -------------------------------------------------------------------------
    rand_indices = np.random.choice(N, size=min(args.num_examples, N), replace=False)

    fig_seq, ax_seq = plt.subplots(figsize=(8, 6))
    for i, idx in enumerate(rand_indices):
        seq_y = Y[idx]  # shape [T, y_dim] or [T] if y_dim=1
        step_axis = np.arange(T)
        if y_dim == 1:
            ax_seq.plot(step_axis, seq_y, marker='o', alpha=0.6, label=f"Seq idx {idx}")
        else:
            # Multi-d Y
            for d in range(y_dim):
                ax_seq.plot(step_axis, seq_y[:, d], marker='o', alpha=0.6, label=f"seq {idx}, y_dim={d}")

    ax_seq.set_title("Sample Y trajectories vs. time")
    ax_seq.set_xlabel("Time step")
    ax_seq.set_ylabel("Y value(s)")
    ax_seq.legend(ncol=2)
    fig_seq.tight_layout()
    seq_path = os.path.join(args.output_dir, "sample_sequences.png")
    fig_seq.savefig(seq_path, dpi=120)
    plt.close(fig_seq)
    print(f"Saved sample sequences plot: {seq_path}")

    # -------------------------------------------------------------------------
    # 4) Plot up to 20 sequences in an X-vs.-Y scatter (assuming x_dim=1, y_dim=1)
    # -------------------------------------------------------------------------
    if x_dim == 1 and y_dim == 1:
        num_xy = min(args.num_xy_plots, N)
        xy_indices = np.random.choice(N, size=num_xy, replace=False)

        # We'll create a grid of subplots, say rows=4, cols=5 if we want 20 total
        ncols = 5
        nrows = (num_xy + ncols - 1) // ncols  # enough rows to fit 'num_xy' subplots

        fig_xy, axs_xy = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=True)
        axs_xy = axs_xy.flatten()  # so we can index easily

        for i, idx in enumerate(xy_indices):
            ax = axs_xy[i]
            seq_x = X[idx]  # shape [T, 1]
            seq_y = Y[idx]  # shape [T, 1]

            ax.scatter(seq_x, seq_y, marker='o', alpha=0.6, label=f"Seq idx {idx}")
            ax.set_title(f"Seq idx {idx}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()

        # Hide unused subplots if any
        for j in range(i+1, len(axs_xy)):
            axs_xy[j].axis('off')

        fig_xy.suptitle(f"X vs. Y for {num_xy} random sequences (train data)")
        fig_xy.tight_layout()
        xy_path = os.path.join(args.output_dir, "XY_sequences.png")
        fig_xy.savefig(xy_path, dpi=120)
        plt.close(fig_xy)
        print(f"Saved X vs. Y sequence plots: {xy_path}")
    else:
        print("Skipping X vs. Y scatter because x_dim>1 or y_dim>1. Adjust code if needed.")

    print("Data visualization complete.")

if __name__ == "__main__":
    main()

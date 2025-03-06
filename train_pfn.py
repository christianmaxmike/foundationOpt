#!/usr/bin/env python

import argparse
import yaml
import os
import wandb
import pickle
import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    SequentialLR
)
try:
    from scipy.stats import yeojohnson
except ImportError:
    yeojohnson = None

# ------------------------------------------------------------------
# 1) Helper: load data, optionally from multiple .npz, apply transform, split
# ------------------------------------------------------------------
def load_and_preprocess_data(data_config, sequence_length=None, device=None):
    """
    data_config should have:
      - data_source: "pkl" or "npz_directory"
      - dataset_path: path to .pkl or directory with .npz
      - transform_type: e.g. "none", "minmax", "power"
      - train_ratio, val_ratio, test_ratio (optional) if you want to control splits
    """
    dataset_path = data_config['dataset_path']


    # 1) Gather all .npz files
    all_x = []
    all_y = []
    for file in os.listdir(dataset_path):
        if file.endswith(".npz"):
            filepath = os.path.join(dataset_path, file)
            with np.load(filepath) as npz_file:
                # Adjust keys as needed
                part_x = npz_file["x"]  # or whatever your .npz stores
                part_y = npz_file["y"]
                all_x.append(part_x)
                all_y.append(part_y)

    # 2) Concatenate
    X = np.concatenate(all_x, axis=0)  # e.g. [N, T, D]
    y = np.concatenate(all_y, axis=0)  # e.g. [N, T2, ...] or [N, Ydim]

    # 3) (Optional) If y has different shape, make sure it matches X’s time dimension
    #   Adjust if your y is shape [N, T, 1], etc. For demonstration, assume y is [N, T, 1].
    #   If y is just [N, 1], adapt accordingly.
    #   For example, we can handle a shape mismatch carefully or leave as-is if consistent.

    # 4) Transform if needed:
    transform_type = data_config.get('transform_type', 'none')
    if transform_type == 'minmax':
        # a) Flatten along the first two dims for X => shape [N*T, D] if needed
        shape_x = X.shape
        X_2d = X.reshape(-1, shape_x[-1])  # [N*T, D]

        x_min = X_2d.min(axis=0)
        x_max = X_2d.max(axis=0)
        # Avoid division by zero
        denom = (x_max - x_min) + 1e-10
        X_2d = (X_2d - x_min) / denom
        X = X_2d.reshape(shape_x)

        # b) Same for y (depending on shape)
        shape_y = y.shape
        Y_2d = y.reshape(-1, shape_y[-1])
        y_min = Y_2d.min(axis=0)
        y_max = Y_2d.max(axis=0)
        denom_y = (y_max - y_min) + 1e-10
        Y_2d = (Y_2d - y_min) / denom_y
        y = Y_2d.reshape(shape_y)

    elif transform_type == 'power':
        if yeojohnson is None:
            raise RuntimeError("scipy.stats not available. Install scipy or remove 'power' transform.")

        shape_x = X.shape
        X_2d = X.reshape(-1, shape_x[-1])  # [N*T, D]
        for d in range(X_2d.shape[1]):
            X_2d[:, d], _ = yeojohnson(X_2d[:, d])
        X = X_2d.reshape(shape_x)

        shape_y = y.shape
        Y_2d = y.reshape(-1, shape_y[-1])
        for d in range(Y_2d.shape[1]):
            Y_2d[:, d], _ = yeojohnson(Y_2d[:, d])
        y = Y_2d.reshape(shape_y)

    # 5) Split into train/val/test
    #    If your data is big, you might want e.g. 80%/10%/10%.
    train_ratio = data_config.get('train_ratio', 0.8)
    val_ratio   = data_config.get('val_ratio', 0.1)
    test_ratio  = data_config.get('test_ratio', 0.1)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-7, "Splits must sum to 1"

    N = X.shape[0]
    train_size = int(train_ratio * N)
    val_size   = int(val_ratio * N)
    test_size  = N - train_size - val_size

    # Shuffle if you want random splits
    indices = np.arange(N)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    # Optionally crop sequence length for training if requested
    if sequence_length is not None:
        # If X_... has shape [N, T, D], we cut T => X_...[:, :sequence_length]
        X_train = X_train[:, :sequence_length]
        X_val   = X_val[:, :sequence_length]
        X_test  = X_test[:, :sequence_length]

        # Similarly y_..., if it also has a T dimension
        # If y has shape [N, T, Ydim], do the same:
        if X_train.ndim == y_train.ndim:
            y_train = y_train[:, :sequence_length]
            y_val   = y_val[:, :sequence_length]
            y_test  = y_test[:, :sequence_length]

    # Convert to torch Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val   = torch.tensor(X_val,   dtype=torch.float32, device=device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_val   = torch.tensor(y_val,   dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test,  dtype=torch.float32, device=device)

    print(f"After loading and preprocessing (min/max values):")
    print(f"  X_train: {X_train.min().item():.4f} / {X_train.max().item():.4f}")
    print(f"  y_train: {y_train.min().item():.4f} / {y_train.max().item():.4f}")

    # Return a dict with the same structure as the original .pkl approach
    return {
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
    }


# ------------------------------------------------------------------
# 2) Import your custom PFNTransformer & losses
# ------------------------------------------------------------------
from model.pfn_transformer import PFNTransformer
from model.losses import (
    exploration_loss_fn,
    convergence_loss_fn,
    cross_entropy_binning_loss,
    bar_distribution_loss
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PFN Transformer for Learned Optimization")
    parser.add_argument('--configs_path', type=str,
                        default="/work/dlclarge1/janowski-opt/foundationOpt/configs/",
                        help="Directory containing YAML config files.")
    parser.add_argument('--config', type=str, required=True,
                        help="Name of the YAML config file (without .yaml).")
    parser.add_argument('--exploration_loss', action='store_true',
                        help="Use exploration loss term.")
    parser.add_argument('--convergence_loss', action='store_true',
                        help="Use convergence loss term.")
    parser.add_argument('--forecast_steps', type=int, default=1,
                        help="How many steps to predict in the future (for multi-step forecasting).")
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed.")
    parser.add_argument('--sequence_length', type=int, default=None,
                        help="Length of the input sequences.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1) Load YAML Config
    # ------------------------------------------------------------------
    config_path = os.path.join(args.configs_path, f"{args.config}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    run_name = config.get('run_name', args.config)
    ckpt_dir = os.path.join(config['training']['ckpt_dir'], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Initialize wandb
    # ------------------------------------------------------------------
    wandb.init(
        project=config.get('wandb', {}).get('project', 'FoundOpt'),
        name=run_name,
        config=config
    )

    # ------------------------------------------------------------------
    # 3) Data Loading & Preprocessing
    # ------------------------------------------------------------------
    data_config = config['data']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dict = load_and_preprocess_data(
        data_config=data_config,
        sequence_length=args.sequence_length,
        device=device
    )
    X_train = data_dict['X_train']  # shape [N, T, D]
    X_val   = data_dict['X_val']
    X_test  = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val   = data_dict['y_val']
    y_test  = data_dict['y_test']

    # Combine X & y on last dimension => if X is [N, T, D], y is [N, T, 1] or [N, T], etc.
    # or if y is just [N, T], unsqueeze last dim to match
    if y_train.ndim == 2 and X_train.ndim == 3:
        # e.g. [N, T] => [N, T, 1]
        y_train = y_train.unsqueeze(-1)
        y_val   = y_val.unsqueeze(-1)
        y_test  = y_test.unsqueeze(-1)

    X_train_model = torch.cat([X_train, y_train], dim=-1)  # => shape [N, T, D + Ydim]
    X_val_model   = torch.cat([X_val,   y_val],   dim=-1)
    X_test_model  = torch.cat([X_test,  y_test],  dim=-1)

    # Build train & val sets
    train_dataset = TensorDataset(X_train_model)
    val_dataset   = TensorDataset(X_val_model)

    train_sampler = RandomSampler(train_dataset)
    val_sampler   = RandomSampler(val_dataset)

    batch_size = config['training'].get('batch_size', 256)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, sampler=val_sampler)

    # ------------------------------------------------------------------
    # 4) Build Model
    # ------------------------------------------------------------------
    model_config = config['model']
    loss_type = model_config.get('loss_type', 'cross_entropy')
    model = PFNTransformer(
        input_dim=model_config.get('input_dim', 2),
        hidden_dim=model_config.get('hidden_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        num_heads=model_config.get('num_heads', 2),
        dropout=model_config.get('dropout', 0.1),
        num_bins=model_config.get('num_bins', 32),
        pre_norm=model_config.get('pre_norm', True),
        forecast_steps=args.forecast_steps,
        use_positional_encoding=model_config.get('use_positional_encoding', False),
        use_autoregression=model_config.get('use_autoregression', False),
        use_bar_distribution=(loss_type == 'bar'),
        bar_dist_smoothing=model_config.get('bar_dist_smoothing', 0.0),
        full_support=model_config.get('full_support', False),
        transform_type=data_config.get('transform_type', 'none')
    ).to(device)

    # ------------------------------------------------------------------
    # 5) Optimizer & LR Scheduler
    # ------------------------------------------------------------------
    optim_config = config['optimizer']
    lr = optim_config.get('lr', 1e-4)
    weight_decay = optim_config.get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = config['training'].get('epochs', 20)
    total_steps = epochs * len(train_dataloader)

    sched_cfg = optim_config.get('scheduler', {})
    warmup_steps   = int(sched_cfg.get('warmup_fraction', 0.1) * total_steps)
    plateau_steps  = int(sched_cfg.get('plateau_fraction', 0.0) * total_steps)
    # Additional steps for Cosine
    remaining_steps = max(1, total_steps - (warmup_steps + plateau_steps))
    print(f"Total steps: {total_steps}, warmup={warmup_steps}, plateau={plateau_steps}, remaining={remaining_steps}")

    # (A) Warmup + plateau function
    def warmup_with_plateau(step: int):
        if step < warmup_steps:
            # e.g. quadratic warmup from 0 -> 1
            return (step / float(warmup_steps)) ** 2
        elif step < warmup_steps + plateau_steps:
            # plateau => keep LR at max
            return 1.0
        else:
            # after that, we pass control to Cosine
            return 1.0

    scheduler1 = LambdaLR(optimizer, lr_lambda=warmup_with_plateau)
    # (B) Cosine Annealing for the rest
    scheduler2 = CosineAnnealingLR(optimizer, T_max=remaining_steps)

    # Then a sequential scheduler: first `scheduler1`, then at step = warmup_steps+plateau_steps -> `scheduler2`
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_steps + plateau_steps]
    )

    # ------------------------------------------------------------------
    # 6) Training Loop
    # ------------------------------------------------------------------
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_dataloader:
            x_batch = batch[0]  # shape [B, T, something]
            optimizer.zero_grad()

            # forward_with_binning => returns (logits, target_bins)
            logits, target_bins = model.forward_with_binning(x_batch)

            if loss_type == "bar":
                # e.g. if final 2 dims in x_batch are the "targets"
                # or adapt as needed if you have (D+1) in the last dimension
                target_dim = model.input_dim  # e.g. 2 or (D+1)
                target_values = x_batch[..., -target_dim:]  
                bar_loss = bar_distribution_loss(
                    model.bar_distribution,
                    logits,
                    target_values
                )
                main_loss = bar_loss
            else:
                # Use the old cross-entropy
                ce_loss = cross_entropy_binning_loss(logits, target_bins)
                main_loss = ce_loss

            # Optional exploration/convergence
            exploration_weight = config['losses'].get('exploration_weight', 0.1) if args.exploration_loss else 0.0
            convergence_weight = config['losses'].get('convergence_weight', 0.1) if args.convergence_loss else 0.0
            exploration_term = exploration_loss_fn(x_batch, model) * exploration_weight
            convergence_term  = convergence_loss_fn(x_batch, model) * convergence_weight

            loss = main_loss + exploration_term + convergence_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Step the scheduler
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "global_step": global_step
            })

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                x_batch = batch[0]
                logits, target_bins = model.forward_with_binning(x_batch)

                if loss_type == "bar":
                    target_dim = model.input_dim
                    target_values = x_batch[..., -target_dim:]
                    val_loss += bar_distribution_loss(
                        model.bar_distribution,
                        logits,
                        target_values
                    ).item()
                else:
                    val_loss += cross_entropy_binning_loss(logits, target_bins).item()

        val_loss /= len(val_dataloader)
        avg_train_loss = total_loss / len(train_dataloader)

        wandb.log({
            "epoch": epoch,
            "epoch_train_loss": avg_train_loss,
            "epoch_val_loss": val_loss
        })

        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

        # Checkpointing
        if (epoch + 1) % config['training'].get('save_every', 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)

    # Final save
    final_ckpt = os.path.join(ckpt_dir, "model_final.pth")
    torch.save(model.state_dict(), final_ckpt)
    wandb.save(final_ckpt)
    print(f"Training complete. Final model saved at {final_ckpt}")


if __name__ == "__main__":
    main()

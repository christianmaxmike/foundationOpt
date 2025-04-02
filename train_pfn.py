#!/usr/bin/env python

import argparse
import yaml
import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

# Import your PFNTransformer and losses
from model.pfn_transformer import PFNTransformer
from model.losses import (
    exploration_loss_fn,
    convergence_loss_fn,
    cross_entropy_binning_loss,
    bar_distribution_loss,
    mse_loss
)

from utils.preprocess import load_and_preprocess_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train PFN Transformer for Learned Optimization")
    parser.add_argument('--configs_path', type=str, default="./configs/", help="Directory containing YAML config files.")
    parser.add_argument('--config', type=str, default="base", help="Name of the YAML config file (without .yaml).")
    parser.add_argument('--exploration_loss', action='store_true', help="Use exploration loss term.")
    parser.add_argument('--convergence_loss', action='store_true', help="Use convergence loss term.")
    parser.add_argument('--forecast_steps', type=int, default=1, help="How many steps to predict in the future.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--sequence_length', type=int, default=None, help="Length of the input sequences.")
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load YAML Config
    config_path = os.path.join(args.configs_path, f"{args.config}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    run_name = config.get('run_name', args.config)
    ckpt_dir = os.path.join(config['training']['ckpt_dir'], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    wandb.init(project=config.get('wandb', {}).get('project', 'FoundOpt'), name=run_name, config=config)
    
    data_config_input = config['data']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # For Mac MPS:
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    
    data_dict = load_and_preprocess_data(
        data_config=data_config_input,
        sequence_length=args.sequence_length,
        device=device
    )
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
    X_val_last = data_dict.get("X_val_last", None)
    X_test_last = data_dict.get("X_test_last", None)

    data_config_model = data_dict['data_config']
    
    print(f"Derived data config: {data_config_model}")
    
    # If y is 2D, but X is 3D -> unsqueeze y to match for cat
    if y_train.ndim == 2 and X_train.ndim == 3:
        y_train = y_train.unsqueeze(-1)
        y_val   = y_val.unsqueeze(-1)
        y_test  = y_test.unsqueeze(-1)
        if y_train_best is not None:
            y_train_best = y_train_best.unsqueeze(-1)
        if y_val_best is not None:
            y_val_best = y_val_best.unsqueeze(-1)
        if y_test_best is not None:
            y_test_best = y_test_best.unsqueeze(-1)

    # As an example: X_train_model includes X plus "y_train_best" if you want to train on that. 
    # Adjust as needed for your pipeline.
    X_train_model = torch.cat([X_train, y_train_best], dim=-1) if y_train_best is not None else torch.cat([X_train, y_train], dim=-1)
    X_val_model   = torch.cat([X_val,   y_val],   dim=-1)
    X_test_model  = torch.cat([X_test,  y_test],  dim=-1)
        
    batch_size = config['training'].get('batch_size', 256)

    # For training, we'll store (input_sequence, last_x, best_y) in a dataset
    # Adjust the items you pass as needed
    train_dataset = TensorDataset(X_train_model, X_train_last, y_train_best)
    val_dataset   = TensorDataset(X_val_model,   X_val_last,   y_val_best)

    train_sampler = RandomSampler(train_dataset)
    val_sampler   = RandomSampler(val_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, sampler=val_sampler)
    
    # Build Model: pass the transformation bounds and dimensions from data_config_model.
    model_config = config['model']

    loss_type = model_config.get('loss_type', 'ce')  # 'bar', 'ce', or 'mse'
    x_dim = data_config_model['x_dim']
    y_dim = data_config_model['y_dim']
    
    model = PFNTransformer(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dim=model_config.get('hidden_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        num_heads=model_config.get('num_heads', 2),
        dropout=model_config.get('dropout', 0.1),
        num_bins=model_config.get('num_bins', 32),
        pre_norm=model_config.get('pre_norm', True),
        forecast_steps=args.forecast_steps,
        use_positional_encoding=model_config.get('use_positional_encoding', False),
        use_autoregression=model_config.get('use_autoregressive', False),
        use_bar_distribution=(loss_type == 'bar'),
        bar_dist_smoothing=model_config.get('bar_dist_smoothing', 0.0),
        full_support=model_config.get('full_support', False),
        transform_type=data_config_input.get('transform_type', 'none'),
        x_min=data_config_model['x_min'],
        x_max=data_config_model['x_max'],
        y_min=data_config_model['y_min'],
        y_max=data_config_model['y_max'],
        use_binning=(loss_type == 'bar' or loss_type == 'ce'),
        train_data=X_train  # If needed for quantile-based binning
    ).to(device)

    # Optimizer & LR Scheduler
    optim_config = config['optimizer']
    lr = optim_config.get('lr', 1e-4)
    weight_decay = optim_config.get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = config['training'].get('epochs', 20)
    total_steps = epochs * len(train_dataloader)
    sched_cfg = optim_config.get('scheduler', {})
    warmup_steps = int(sched_cfg.get('warmup_fraction', 0.1) * total_steps)
    plateau_steps = int(sched_cfg.get('plateau_fraction', 0.2) * total_steps)
    remaining_steps = max(1, total_steps - (warmup_steps + plateau_steps))
    print(f"Total steps: {total_steps}, warmup={warmup_steps}, plateau={plateau_steps}, remaining={remaining_steps}")

    def warmup_with_plateau(step: int):
        if step < warmup_steps:
            return (step / float(warmup_steps)) ** 2
        elif step < warmup_steps + plateau_steps:
            return 1.0
        else:
            return 1.0

    scheduler1 = LambdaLR(optimizer, lr_lambda=warmup_with_plateau)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=remaining_steps)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2],
                            milestones=[warmup_steps + plateau_steps])
    
    # Training Loop
    tr_cfg = config.get('training', {})
    patience = tr_cfg.get("early_stopping", {}).get("patience", 5)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state_dict = None

    # log run configs
    wandb.watch(model)
    wandb.config.num_bins = model_config.get('num_bins', 32)
    wandb.config.hidden_dim = model_config.get('hidden_dim', 64)
    wandb.config.num_layers = model_config.get('num_layers', 2)
    wandb.config.num_heads = model_config.get('num_heads', 2)
    wandb.config.dropout = model_config.get('dropout', 0.1)
    wandb.config.pre_norm = model_config.get("pre_norm", True)
    wandb.config.use_positional_encoding = model_config.get('use_positional_encoding', False)
    wandb.config.use_autoregressive = model_config.get('use_autoregressive', False)
    wandb.config.use_bar_distribution = (loss_type == 'bar')
    wandb.config.bar_dist_smoothing = model_config.get('bar_dist_smoothing', 0.0)
    wandb.config.opt_weight_decay = optim_config.get('weight_decay', 1e-5)
    wandb.config.epochs = epochs
    wandb.config.patience = patience

    global_step = 0
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        total_loss = 0.0

        for batch in train_dataloader:
            x_batch, x_train_last, target_y_best = batch
            x_batch = x_batch.to(device)
            x_train_last = x_train_last.to(device) if x_train_last is not None else None
            target_y_best = target_y_best.to(device) if target_y_best is not None else None

            optimizer.zero_grad()

            # -------------------------
            #  Forward & Loss (Train)
            # -------------------------
            if loss_type == "bar":
                logits, target_bins = model.forward_with_binning(x_batch, x_train_last)
                # logits: [B, T_out, (x_dim+y_dim), num_bins]
                # target_bins: [B, T_out, (x_dim+y_dim)] (raw or discrete)

                B, T_out, D, num_bins = logits.shape

                # For logging bin-accuracy, only needed if you want it
                logits_x = logits[..., :x_dim, :]  # [B, T_out, x_dim, num_bins]
                logits_y = logits[..., x_dim:, :]  # [B, T_out, y_dim, num_bins]
                target_x = target_bins[..., :x_dim]  # [B, T_out, x_dim]
                target_y = target_bins[..., x_dim:]  # [B, T_out, y_dim]

                # Argmax to measure bin-acc (not strictly required for the bar distribution approach)
                pred_bins_x = logits_x.argmax(dim=-1)
                bin_acc_x_train = (pred_bins_x == model.bar_distribution_x.map_to_bucket_idx(target_x)).float().mean()

                pred_bins_y = logits_y.argmax(dim=-1)
                bin_acc_y_train = (pred_bins_y == model.bar_distribution_y.map_to_bucket_idx(target_y)).float().mean()

                # Compute losses
                loss_x = bar_distribution_loss(model.bar_distribution_x, logits_x, target_x, model.orh_x)
                loss_y = bar_distribution_loss(model.bar_distribution_y, logits_y, target_y, model.orh_y)
                
                # If you want to measure "regret" vs best
                loss_y_best = None
                if target_y_best is not None:
                    loss_y_best = bar_distribution_loss(
                        model.bar_distribution_y, logits_y, target_y_best, model.orh_y
                    )

                main_loss = (loss_x + loss_y)

            elif loss_type == "ce":

                logits, target_bins = model.forward_with_binning(x_batch, x_train_last)
                loss_x, loss_y = cross_entropy_binning_loss(logits, target_bins, x_dim=model.x_dim, y_dim=model.y_dim)
                main_loss = loss_x + loss_y

            elif loss_type == "mse":

                logits, targets = model(x_batch)
                loss_x, loss_y  = mse_loss(logits, targets, x_dim=model.x_dim, y_dim=model.y_dim)
                main_loss = 0.5 * (loss_x + loss_y)

            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            # -------------
            # Extra terms
            # -------------
            # Weighted exploration/convergence losses if desired
            exploration_weight = config['losses'].get('exploration_weight', 0.1) if args.exploration_loss else 0.0
            convergence_weight = config['losses'].get('convergence_weight', 0.1) if args.convergence_loss else 0.0

            exploration_term = exploration_loss_fn(x_batch, model) * exploration_weight
            convergence_term = convergence_loss_fn(x_batch, model) * convergence_weight

            loss = main_loss + exploration_term + convergence_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            total_loss += loss.item()

            # -------------
            # Logging
            # -------------
            log_dict = {
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
            }
            if loss_type == "bar":
                log_dict["train/x_loss"] = loss_x.item()
                log_dict["train/y_loss"] = loss_y.item()
                log_dict["train/accBinX"] = bin_acc_x_train.item()
                log_dict["train/accBinY"] = bin_acc_y_train.item()
                if loss_y_best is not None:
                    log_dict["train/regret"] = loss_y_best.item()
            elif loss_type == "mse" and loss_y_best is not None:
                log_dict["train/regret"] = loss_y_best.item()

            wandb.log(log_dict)

        # ------------------
        # Validation Phase
        # ------------------
        model.eval()
        val_loss = 0.0
        regret_loss = 0.0
        bin_acc_x = 0.0
        bin_acc_y = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                x_batch, x_val_last, target_y_best = batch
                x_batch = x_batch.to(device)
                x_val_last = x_val_last.to(device) if x_val_last is not None else None
                target_y_best = target_y_best.to(device) if target_y_best is not None else None

                if loss_type == "bar":
                    logits, target_bins = model.forward_with_binning(x_batch, x_val_last)
                    B, T_out, D, num_bins = logits.shape

                    logits_x = logits[..., :x_dim, :]
                    logits_y = logits[..., x_dim:, :]
                    target_x = target_bins[..., :x_dim]
                    target_y = target_bins[..., x_dim:]

                    # For bin-acc:
                    pred_bins_x = logits_x.argmax(dim=-1)
                    bin_acc_x += (pred_bins_x == model.bar_distribution_x.map_to_bucket_idx(target_x)).float().mean()

                    pred_bins_y = logits_y.argmax(dim=-1)
                    bin_acc_y += (pred_bins_y == model.bar_distribution_y.map_to_bucket_idx(target_y)).float().mean()

                    loss_x = bar_distribution_loss(model.bar_distribution_x, logits_x, target_x, model.orh_x)
                    loss_y = bar_distribution_loss(model.bar_distribution_y, logits_y, target_y, model.orh_y)
                    val_loss_batch = (loss_x + loss_y).item()
                    val_loss += val_loss_batch

                    # Evaluate regret if best is known
                    if target_y_best is not None:
                        y_best_loss = bar_distribution_loss(model.bar_distribution_y, logits_y, target_y_best, model.orh_y)
                        regret_loss += y_best_loss.item()

                elif loss_type == "ce":
                    logits, target_bins = model.forward_with_binning(x_batch, x_val_last)
                    B, T_out, D, num_bins = logits.shape

                    logits_y = logits[..., x_dim:, :]
                    targets_flat = target_bins[..., x_dim:].reshape(-1)
                    logits_flat = logits_y.reshape(-1, num_bins)
                    val_loss_batch = cross_entropy_binning_loss(logits_flat, targets_flat).item()
                    val_loss += val_loss_batch

                elif loss_type == "mse":
                    preds, targets = model(x_batch)
                    val_loss_batch = F.mse_loss(preds, targets, reduction='mean').item()
                    val_loss += val_loss_batch

                    if target_y_best is not None:
                        # measure regret on y portion
                        y_best_loss = F.mse_loss(preds[..., x_dim:], target_y_best[..., :], reduction='mean').item()
                        regret_loss += y_best_loss
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")

        # Average over the validation set
        val_loss /= len(val_dataloader)
        avg_train_loss = total_loss / len(train_dataloader)
        avg_regret_loss = regret_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0
        avg_bin_acc_x = bin_acc_x / len(val_dataloader)
        avg_bin_acc_y = bin_acc_y / len(val_dataloader)
        
        wandb.log({
            "epoch": epoch,
            "epoch_train_loss": avg_train_loss,
            "val/epoch_val_loss": val_loss,
            "val/avg_regret": avg_regret_loss
        })
        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"AccBin=(X:{avg_bin_acc_x:.2f}, Y:{avg_bin_acc_y:.2f})")

        # Save checkpoint periodically
        if (epoch + 1) % config['training'].get('save_every', 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state_dict = model.state_dict()
        else:
            epochs_no_improve += 1
            print(f"Strike {epochs_no_improve} of {patience} @ epoch {epoch}/{epochs}!")
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch} with best validation loss: {best_val_loss}!")
                break

    # Final save
    final_ckpt = os.path.join(ckpt_dir, "model_final.pth")
    torch.save(best_model_state_dict, final_ckpt)
    wandb.save(final_ckpt)
    print(f"Training complete. Final model saved at {final_ckpt}")


if __name__ == "__main__":
    main()

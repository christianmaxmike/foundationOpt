#!/usr/bin/env python

import argparse
import yaml
import os
import wandb
import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


# Import the updated PFNTransformer and losses.
from model.pfn_transformer import PFNTransformer
from model.losses import (
    exploration_loss_fn,
    convergence_loss_fn,
    cross_entropy_binning_loss,
    bar_distribution_loss
)

from utils.preprocess import load_and_preprocess_data

def parse_args():
    parser = argparse.ArgumentParser(description="Train PFN Transformer for Learned Optimization")
    parser.add_argument('--configs_path', type=str, default="/work/dlclarge1/janowski-opt/foundationOpt/configs/", help="Directory containing YAML config files.")
    parser.add_argument('--config', type=str, required=True, help="Name of the YAML config file (without .yaml).")
    parser.add_argument('--exploration_loss', action='store_true', help="Use exploration loss term.")
    parser.add_argument('--convergence_loss', action='store_true', help="Use convergence loss term.")
    parser.add_argument('--forecast_steps', type=int, default=1, help="How many steps to predict in the future.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--sequence_length', type=int, default=None, help="Length of the input sequences.")
    parser.add_argument('--force_reprocess', action='store_true', help="Force reprocessing of data.")
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
    
    # Data Loading & Preprocessing
    data_config_input = config['data']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dict = load_and_preprocess_data(
        data_config=data_config_input, 
        sequence_length=args.sequence_length, 
        force_reprocess=args.force_reprocess,
        device=device
    )
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    data_config_model = data_dict['data_config']
    
    print(f"Derived data config: {data_config_model}")
    
    # Combine X and y along the last dimension to form model input
    if y_train.ndim == 2 and X_train.ndim == 3:
        y_train = y_train.unsqueeze(-1)
        y_val = y_val.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)
    X_train_model = torch.cat([X_train, y_train], dim=-1)
    X_val_model = torch.cat([X_val, y_val], dim=-1)
    X_test_model = torch.cat([X_test, y_test], dim=-1)
    
    train_dataset = TensorDataset(X_train_model)
    val_dataset = TensorDataset(X_val_model)
    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)
    batch_size = config['training'].get('batch_size', 256)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    
    # Build Model: pass the transformation bounds and dimensions from data_config_model.
    model_config = config['model']
    loss_type = model_config.get('loss_type', 'cross_entropy')
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
        use_autoregression=model_config.get('use_autoregression', False),
        use_bar_distribution=(loss_type == 'bar'),
        bar_dist_smoothing=model_config.get('bar_dist_smoothing', 0.0),
        full_support=model_config.get('full_support', False),
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
    plateau_steps = int(sched_cfg.get('plateau_fraction', 0.0) * total_steps)
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
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps + plateau_steps])
    
    # Training Loop
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            x_batch = batch[0]  # [B, T, input_dim]
            optimizer.zero_grad()
            logits, target_bins = model.forward_with_binning(x_batch)
            # logits: [B, T_out, (x_dim+y_dim), num_bins]
            # target_bins: [B, T_out, (x_dim+y_dim)]
            B, T_out, D, num_bins = logits.shape
            if loss_type == "bar":
                logits_x = logits[..., :x_dim, :]
                logits_y = logits[..., x_dim:, :]
                target_x = target_bins[..., :x_dim]
                target_y = target_bins[..., x_dim:]
                loss_x = bar_distribution_loss(model.bar_distribution_x, logits_x, target_x)
                loss_y = bar_distribution_loss(model.bar_distribution_y, logits_y, target_y)
                main_loss = loss_x + loss_y
            else:
                logits_flat = logits.view(B * T_out * D, num_bins)
                targets_flat = target_bins.view(B * T_out * D)
                main_loss = cross_entropy_binning_loss(logits_flat, targets_flat)
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
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "global_step": global_step
            })
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                x_batch = batch[0]
                logits, target_bins = model.forward_with_binning(x_batch)
                B, T_out, D, num_bins = logits.shape
                if loss_type == "bar":
                    logits_x = logits[..., :x_dim, :]
                    logits_y = logits[..., x_dim:, :]
                    target_x = target_bins[..., :x_dim]
                    target_y = target_bins[..., x_dim:]
                    loss_x = bar_distribution_loss(model.bar_distribution_x, logits_x, target_x)
                    loss_y = bar_distribution_loss(model.bar_distribution_y, logits_y, target_y)
                    val_loss += (loss_x + loss_y).item()
                else:
                    logits_flat = logits.view(B * T_out * D, num_bins)
                    targets_flat = target_bins.view(B * T_out * D)
                    val_loss += cross_entropy_binning_loss(logits_flat, targets_flat).item()
        val_loss /= len(val_dataloader)
        avg_train_loss = total_loss / len(train_dataloader)
        wandb.log({
            "epoch": epoch,
            "epoch_train_loss": avg_train_loss,
            "epoch_val_loss": val_loss
        })
        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
        if (epoch + 1) % config['training'].get('save_every', 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)
    final_ckpt = os.path.join(ckpt_dir, "model_final.pth")
    torch.save(model.state_dict(), final_ckpt)
    wandb.save(final_ckpt)
    print(f"Training complete. Final model saved at {final_ckpt}")

if __name__ == "__main__":
    main()

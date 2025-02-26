#!/usr/bin/env python

import argparse
import yaml
import os
import wandb
import pickle
import torch



from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from model.pfn_transformer import PFNTransformer
from model.losses import (
    exploration_loss_fn,
    convergence_loss_fn,
    cross_entropy_binning_loss
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train PFN Transformer for Learned Optimization")
    parser.add_argument('--configs_path', type=str, 
                        default="/work/dlclarge1/janowski-opt/foundationOpt/configs/",
                        help="Path to YAML config file.")
    parser.add_argument('--config', type=str, required=True,
                        help="Name of YAML config file.")
    # Additional flags for toggling losses, AR, forecasting, etc.:
    parser.add_argument('--exploration_loss', action='store_true',
                        help="Use exploration loss term.")
    parser.add_argument('--convergence_loss', action='store_true',
                        help="Use convergence loss term.")
    parser.add_argument('--autoregressive', action='store_true',
                        help="Train with autoregressive rollout.")
    parser.add_argument('--forecast_steps', type=int, default=1,
                        help="How many steps to predict in the future (for multi-step forecasting).")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load config
    config_path = os.path.join(args.configs_path, f"{args.config}.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Possibly set run name from config filename if not specified
    run_name = config.get('run_name', os.path.basename(args.config).split('.')[0])
    ckpt_path = os.path.join(config['training']['ckpt_dir'], run_name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # Initialize wandb
    wandb.init(
        project=config.get('wandb', {}).get('project', 'FoundOpt'),
        name=run_name,
        config=config
    )

    # Prepare data 
    data_config = config['data']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(data_config['dataset_path'], 'rb') as f:
        data = pickle.load(f)
    
    X_train = torch.tensor(data['X_train'], dtype=torch.float32).to(device)
    X_val = torch.tensor(data['X_val'], dtype=torch.float32).to(device)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32).to(device)

    y_train = torch.tensor(data['y_train'], dtype=torch.float32).to(device)
    y_val = torch.tensor(data['y_val'], dtype=torch.float32).to(device)
    y_test = torch.tensor(data['y_test'], dtype=torch.float32).to(device)

    X_train_model = torch.cat([X_train, y_train], dim=-1)
    X_val_model = torch.cat([X_val, y_val], dim=-1)
    X_test_model = torch.cat([X_test, y_test], dim=-1)

    # Combine into dataset
    dataset = TensorDataset(X_train_model)
    sampler = RandomSampler(dataset)
    batch_size = config['training']['batch_size']
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # validation dataloader
    val_dataset = TensorDataset(X_val_model)
    val_sampler = RandomSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # Build model
    model_config = config['model']
    model = PFNTransformer(
        input_dim=model_config.get('input_dim', 2),       # x,y => 2 dims
        hidden_dim=model_config.get('hidden_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        num_heads=model_config.get('num_heads', 2),
        dropout=model_config.get('dropout', 0.1),
        num_bins=model_config.get('num_bins', 32),        # for discretization
        forecast_steps=args.forecast_steps,
        use_autoregression=model_config.get('use_autoregression', False)
    ).to(device)

    # Set up optimizer & LR schedule
    optim_config = config['optimizer']
    lr = optim_config.get('lr', 1e-4)
    weight_decay = optim_config.get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Possibly set up a scheduler
    scheduler_config = optim_config.get('scheduler', {})
    scheduler = None
    if scheduler_config.get('type') == 'cosine':
        T_max = scheduler_config.get('T_max', 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    # Training loop
    epochs = config['training']['epochs']

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_dataloader:
            x_batch = batch[0].to(device)   # shape [B, T, 2]
            # If we have multiple functions, we might have a function_id or something as well

            optimizer.zero_grad()
            # Forward pass => model returns predicted distribution over bins for each step

            # In pure next-step scenario:
            #   target = x_batch[:, 1:, :] (shifting by 1)
            # but let's keep it simpler here. The model will do internal AR if use_autoregression is True

            # Discretize Y
            logits, target_bins = model.forward_with_binning(x_batch)

            # logits shape: [B, T', 2, num_bins] (2 for x and y, T' depends on forecast_steps)
            # target_bins shape: [B, T', 2]

            # Basic cross-entropy
            ce_loss = cross_entropy_binning_loss(logits, target_bins)

            # Optionally add exploration/convergence
            # We might have a schedule for weighting
            exploration_weight = config['losses'].get('exploration_weight', 0.1) if args.exploration_loss else 0.0
            convergence_weight = config['losses'].get('convergence_weight', 0.1) if args.convergence_loss else 0.0

            exploration_term = exploration_loss_fn(x_batch, model) * exploration_weight
            convergence_term = convergence_loss_fn(x_batch, model) * convergence_weight

            loss = ce_loss + exploration_term + convergence_term
            wandb.log({"train_loss": loss.item()})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()

        # validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                x_batch = batch[0].to(device)
                logits, target_bins = model.forward_with_binning(x_batch)
                val_loss += cross_entropy_binning_loss(logits, target_bins).item()

        val_loss /= len(val_dataloader)

        avg_loss = total_loss / len(train_dataloader)
        wandb.log({"epoch": epoch, "avg_train_loss": avg_loss, "val_loss": val_loss})

        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")

        # Save checkpoints if desired
        if (epoch+1) % config['training'].get('save_every', 10) == 0:
            _ckpt_path = os.path.join(ckpt_path, f"model_epoch{epoch}.pth")
            torch.save(model.state_dict(), _ckpt_path)
            wandb.save(_ckpt_path)

    # Final save
    final_ckpt = os.path.join(ckpt_path, "model_final.pth")
    torch.save(model.state_dict(), final_ckpt)
    wandb.save(final_ckpt)

if __name__ == "__main__":
    main()

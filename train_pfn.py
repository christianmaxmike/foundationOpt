#!/usr/bin/env python

import argparse
import yaml
import os
import wandb
import pickle
import torch

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    SequentialLR
)

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
    # 3) Data Loading & Preparation
    # ------------------------------------------------------------------
    data_config = config['data']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(data_config['dataset_path'], 'rb') as f:
        data = pickle.load(f)

    X_train = torch.tensor(data['X_train'], dtype=torch.float32).to(device)
    if args.sequence_length is not None:
        X_train = X_train[:, :args.sequence_length]
    X_val   = torch.tensor(data['X_val'],   dtype=torch.float32).to(device)
    X_test  = torch.tensor(data['X_test'],  dtype=torch.float32).to(device)

    y_train = torch.tensor(data['y_train'], dtype=torch.float32).to(device)
    if args.sequence_length is not None:
        y_train = y_train[:, :args.sequence_length]
    y_val   = torch.tensor(data['y_val'],   dtype=torch.float32).to(device)
    y_test  = torch.tensor(data['y_test'],  dtype=torch.float32).to(device)

    # Example: we combine X and y along the last dim => shape [N, T, 4] if X,y each had 2 dims
    X_train_model = torch.cat([X_train, y_train], dim=-1)
    X_val_model   = torch.cat([X_val,   y_val  ], dim=-1)
    X_test_model  = torch.cat([X_test,  y_test ], dim=-1)

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
        full_support=model_config.get('full_support', False)
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
    warmup_steps   = sched_cfg.get('warmup_fraction', 0.1) * total_steps
    plateau_steps  = sched_cfg.get('plateau_fraction', 0.0) * total_steps
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
            x_batch = batch[0]  # shape [B, T, something], e.g. [B, T, 4]
            optimizer.zero_grad()

            # forward_with_binning => returns (logits, target_bins)
            logits, target_bins = model.forward_with_binning(x_batch)

            if loss_type == "bar":
                # We'll interpret x_batch[..., model.input_dim:] as y-values
                # or, if x_batch is just [B,T,2], you can directly pass that to bar_dist.
                # Make sure shapes align with the model's assumption => [B,T,2].
                # The same `logits` is [B,T,2,num_bins].
                # We only need the original continuous values in [0,1] that we want to fit.
                target_values = x_batch[..., -2:]  # e.g. last 2 dims
                bar_loss = bar_distribution_loss(
                    model.bar_distribution,   # The module created inside PFNTransformer
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
            convergence_term  = convergence_loss_fn(x_batch,  model) * convergence_weight

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
                    target_values = x_batch[..., -2:]  # same logic as above
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

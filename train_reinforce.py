#!/usr/bin/env python

import argparse
import yaml
import os
import wandb
import torch
import numpy as np
from tqdm import tqdm
import math

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from schedulefree import AdamWScheduleFree

# Import the updated PFNTransformer and losses.
from model.pfn_transformer import PFNTransformer
from model.rl_opt import PolicyNetwork, compute_returns
from model.losses import (
    exploration_loss_fn,
    convergence_loss_fn,
    cross_entropy_binning_loss,
    bar_distribution_loss,
    mse_loss,
    quantile_loss,
    rank_loss
)

from utils.preprocess import load_and_preprocess_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train PFN Transformer for Learned Optimization")
    parser.add_argument('--configs_path', type=str, default="./configs/", help="Directory containing YAML config files.")
    parser.add_argument('--config', type=str, default="bar", help="Name of the YAML config file (without .yaml).")
    parser.add_argument('--exploration_loss', action='store_true', help="Use exploration loss term.")
    parser.add_argument('--convergence_loss', action='store_true', help="Use convergence loss term.")
    parser.add_argument('--forecast_steps', type=int, default=1, help="How many steps to predict in the future.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--sequence_length', type=int, default=None, help="Length of the input sequences.")
    parser.add_argument("--ckptAppx", type=str, default="test")
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
    ckpt_dir = os.path.join(config['training']['ckpt_dir'], run_name, args.ckptAppx)
    os.makedirs(ckpt_dir, exist_ok=True)
        
    data_config_input = config['data']

    wandb.init(project=config.get('wandb', {}).get('project', 'RLOpt'), name=f"{run_name}_{config['model']['loss_type']}_ctx", config=config)

    device = ['cuda', 'mps', 'cpu'][np.argmax([torch.cuda.is_available(), torch.backends.mps.is_available(), True])]
    print (device)
    
    data_dict = load_and_preprocess_data(data_config=data_config_input, sequence_length=args.sequence_length, device=device)
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']    
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']

    X_train_best = data_dict.get("X_train_best", None)
    X_test_best = data_dict.get("X_test_best", None)
    X_val_best = data_dict.get("X_val_best", None)
    y_train_best = data_dict.get("y_train_best", None)
    y_test_best = data_dict.get("y_test_best", None)
    y_val_best = data_dict.get("y_val_best", None)

    train_models = data_dict.get("train_models", None)
    val_models = data_dict.get("val_models", None)

    data_config_model = data_dict['data_config']
    
    print(f"Derived data config: {data_config_model}")
    
    # Combine X and y along the last dimension to form model input
    if y_train.ndim == 2 and X_train.ndim == 3:
        y_train = y_train.unsqueeze(-1)
        y_val = y_val.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)
        y_train_best = y_train_best.unsqueeze(-1)
        y_val_best = y_val_best.unsqueeze(-1)
        y_test_best = y_test_best.unsqueeze(-1)
        y_train_best = y_train_best.unsqueeze(-1)
        y_val_best = y_val_best.unsqueeze(-1)
        y_test_best = y_test_best.unsqueeze(-1)

    X_train_model = torch.cat([X_train, y_train], dim=-1)  #  (batch x seq_len x dim_x+dim_y)
    X_val_model = torch.cat([X_val, y_val], dim=-1)
    X_test_model = torch.cat([X_test, y_test], dim=-1)
        
    batch_size = config['training'].get('batch_size', 256)

    train_dataset = TensorDataset(X_train_model, X_train_best, y_train_best)
    val_dataset = TensorDataset(X_val_model, X_val_best, y_val_best)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler) #, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler) # , drop_last=True)
    
    # Build Model: pass the transformation bounds and dimensions from data_config_model.
    model_config = config['model']

    loss_type = model_config.get('loss_type', 'cross_entropy')
    x_dim = data_config_model['x_dim']
    y_dim = data_config_model['y_dim']
    ctx_length = 4

    model = PolicyNetwork(input_dim=(x_dim+y_dim)*ctx_length, 
                          hidden_dim=model_config.get('hidden_dim', 64)
    ).to(device)


    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print ("Number of parameters:\t", pytorch_total_params)


    # Optimizer & LR Scheduler
    optim_config = config['optimizer']
    lr = optim_config.get('lr', 1e-4)
    weight_decay = optim_config.get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = config['training'].get('epochs', 20)
    total_steps = epochs * len(train_dataloader)
    sched_cfg = optim_config.get('scheduler', {})
    warmup_steps = int(sched_cfg.get('warmup_fraction', 0.1) * total_steps)
    plateau_steps = int(sched_cfg.get('plateau_fraction', 0.2) * total_steps)
    remaining_steps = max(1, total_steps - (warmup_steps + plateau_steps))
    print(f"Total steps: {total_steps}, warmup={warmup_steps}, plateau={plateau_steps}, remaining={remaining_steps}")
    #def warmup_with_plateau(step: int):
    #    if step < warmup_steps:
    #        return (step / float(warmup_steps)) ** 2
    #    elif step < warmup_steps + plateau_steps:
    #        return 1.0
    #    else:
    #        return 1.0
    #scheduler1 = LambdaLR(optimizer, lr_lambda=warmup_with_plateau)
    #scheduler2 = CosineAnnealingLR(optimizer, T_max=remaining_steps)
    #scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps + plateau_steps])
    
    # Training Loop
    # early stopping params TODO: add to args
    tr_cfg = config.get('training', {})
    patience = tr_cfg.get("early_stopping").get("patience", 5)
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
    wandb.config.epochs = config['training'].get('epochs', 20)
    wandb.config.patience = tr_cfg.get("early_stopping").get("patience", 5)
    wandb.config.ctx_length = model_config.get('ctx_length', 5)

    
    # batch_trajectories = X_train
    # batch_trajectories: List of trajectories, each is [(x1, y1), ..., (xT, yT)]
    best_epoch_policy_loss = torch.inf
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        epoch_policy_loss = 0.0
        # for trajectory in batch_trajectories:
        for i in tqdm(range(X_train.shape[0]), desc="Trajectories"): 
            trajectory = X_train_model[i] # T x d
            states = []
            actions = []
            rewards = []
            
            # Run policy to generate actions and rewards
            for t in range(len(trajectory) - 1 - ctx_length):
                x_t, y_t = trajectory[t + ctx_length - 1, :x_dim], trajectory[t + ctx_length - 1, x_dim:]
                
                state = trajectory[t : (t + ctx_length)].flatten()
                mean, log_var = model(state)
                std = torch.exp(0.5 * log_var)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()  # Next x_{t+1}
                
                # Get y_{t+1} (from trajectory)
                x_next_t, y_next_t = trajectory[t + ctx_length, :x_dim], trajectory[t+ctx_length, x_dim:]
                reward = torch.abs(y_t - y_next_t)  # Improvement in y
                # 
                states.append(state)
                actions.append(action)
                rewards.append(reward)
            
            # Compute returns
            # G_t = Σ_{k=t}^T γ^{k-t} r_k f
            returns = compute_returns(rewards)
            
            # Compute policy loss
            policy_loss = []
            for state, action, R in zip(states, actions, returns):
                mean, log_var = model(state)
                std = torch.exp(0.5 * log_var)
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(action)
                policy_loss.append(-log_prob * R)  # Negative for gradient ascent
            
            # Update policy
            # ∇_θ log π(a_t|s_t) * G_t
            optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()

            epoch_policy_loss += policy_loss
        
            wandb.log({
                "epoch": epoch,
                "policy_loss": policy_loss.item(),
            })

            # print(f"[Epoch {epoch}] policy_loss={policy_loss:.4f}")

        model.eval()
        total_reward = 0.0
        for i in tqdm(range(X_val_model.shape[0]- ctx_length), desc="Eval"):
            eval_traj = X_val_model[i]
            x_t_start, y_t_start = eval_traj[:ctx_length, :x_dim], eval_traj[:ctx_length, x_dim:]
            # state = torch.tensor([x_t_start, y_t_start], dtype=torch.float32).to(device)
            state = eval_traj[:ctx_length].flatten()
            traj_reward = 0.0
            for step in range(10):
                mean, log_var = model(state)
                std = torch.exp(0.5 * log_var)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()  # x_next
                y_eval = eval_fnc(val_models[i], action)
                #reward_x = torch.abs(action - X_val_model[i, step + 1, :x_dim].item())
                reward_y = 1 - torch.abs(y_eval - X_val_model[i, step + 1, x_dim:].item())
                traj_reward += reward_y
                state = torch.cat((state, torch.tensor([action.item(), y_eval]).to(device)))[2:]
            total_reward+= (traj_reward / 10.)
            wandb.log({"traj_reward_y": traj_reward / 10.0})
        wandb.log({
            "reward_y": total_reward / X_val_model.shape[0]
        })
        print (f"Validation Total Reward: {total_reward / (10 * X_val_model.shape[0])}")

        if (epoch + 1) % config['training'].get('save_every', 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)

        # early stopping
        if epoch_policy_loss.item() < best_epoch_policy_loss:
             best_epoch_policy_loss = epoch_policy_loss
             epochs_no_improve = 0
             best_model_state_dict = model.state_dict()
        else:
             epochs_no_improve += 1
             print (f"Strike {epochs_no_improve} of {patience} @ epoch{epoch}/{epochs}!")
             if epochs_no_improve == patience:
                 print(f"Early stopping at epoch {epoch} with best validation loss: {best_val_loss}!")
                 break        
    final_ckpt = os.path.join(ckpt_dir, "model_final.pth")
    torch.save(best_model_state_dict, final_ckpt)
    wandb.save(final_ckpt)
    print(f"Training complete. Final model saved at {final_ckpt}")

def eval_fnc(fnc_dict, x_val):
    #x_values = np.linspace(min_x_value, max_x_value, sequence_length)
    y_val = torch.zeros_like(x_val) #.to(x_val.device)
    for i in range(fnc_dict['num_components']):
        trig = fnc_dict['trigonometrics'][i]
        amp  = fnc_dict['amplitudes'][i]
        freq = fnc_dict['frequencies'][i]
        phase= fnc_dict['phases'][i]
        y_val += amp * trig(freq * x_val.cpu() + phase)
    return y_val

if __name__ == "__main__":
    main()

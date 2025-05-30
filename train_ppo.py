import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
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
from torch.distributions.categorical import Categorical
from model.rl_ppo import Actor, Critic
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


def compute_gae(rewards, values, gamma=0.99, lmbda=0.95, device=None):
     # Add dummy next_value for last step (could be zero or bootstrap)
    next_values = torch.cat([values[1:], torch.zeros(1).to(device)])  # Default: zero padding
    
    deltas = rewards + gamma * next_values - values
    advantages = []
    advantage = 0
    for delta in reversed(deltas):
        advantage = delta + gamma * lmbda * advantage
        advantages.insert(0, advantage)
    advantages = torch.tensor(advantages).to(device)
    returns = advantages + values # [:-1]
    return returns, advantages


def ppo_loss(actor, critic, states, actions, old_log_probs, advantages, returns, clip_eps=0.2):
    # Recompute log probs and values
    # new_means, new_stds = actor(states)
    # new_dist = Normal(new_means, new_stds)
    # new_log_probs = new_dist.log_prob(torch.stack(actions)).sum(dim=-1)
    rec, action = actor(states)
    # x_pred = torch.softmax(action, dim=-1)
    dist = Categorical(logits=action)
    # action = dist.sample()
    new_log_probs = dist.log_prob(torch.stack(actions))
    # action = torch.argmax(x_pred)
    # Critic: Predict value

    rec_loss = torch.nn.MSELoss()
    mse = rec_loss(states, rec)

    new_values = critic(states).squeeze()

    # Clipped policy loss
    ratio = (new_log_probs - old_log_probs).exp()
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # Value loss (MSE)
    critic_loss = (new_values - returns).pow(2).mean()

    return actor_loss, mse, critic_loss 


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

    #X_train_best = data_dict.get("X_train_best", None)
    #X_test_best = data_dict.get("X_test_best", None)
    #X_val_best = data_dict.get("X_val_best", None)
    #y_train_best = data_dict.get("y_train_best", None)
    #y_test_best = data_dict.get("y_test_best", None)
    #y_val_best = data_dict.get("y_val_best", None)

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
    # X_test_model = torch.cat([X_test, y_test], dim=-1)
        
    batch_size = config['training'].get('batch_size', 256)

    train_dataset = TensorDataset(X_train_model) # , X_train_best, y_train_best)
    val_dataset = TensorDataset(X_val_model) # , X_val_best, y_val_best)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler) #, num_workers=8) #, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler) # , drop_last=True)
    
    # Build Model: pass the transformation bounds and dimensions from data_config_model.
    model_config = config['model']

    loss_type = model_config.get('loss_type', 'cross_entropy')
    x_dim = data_config_model['x_dim']
    y_dim = data_config_model['y_dim']
    ctx_length = 2

    actor = Actor(input_dim=(x_dim+y_dim)*ctx_length, 
                  hidden_dim=model_config.get('hidden_dim', 64),
                  x_min=data_config_model['x_min'],
                  x_max=data_config_model['x_max'],
                  y_min=data_config_model['y_min'],
                  y_max=data_config_model['y_max'],).to(device)
    critic = Critic(input_dim=(x_dim+y_dim)*ctx_length,
                  hidden_dim=model_config.get("hidden_dim", 64),
                  x_min=data_config_model['x_min'],
                  x_max=data_config_model['x_max'],
                  y_min=data_config_model['y_min'],
                  y_max=data_config_model['y_max'],).to(device)

    print ("Number of parameters - Actor:\t", sum(p.numel() for p in actor.parameters()))
    print ("Number of parameters - Critic:\t", sum(p.numel() for p in critic.parameters()))

    # Optimizer & LR Scheduler
    optim_config = config['optimizer']
    lr = optim_config.get('lr', 1e-4)
    weight_decay = optim_config.get('weight_decay', 1e-5)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = config['training'].get('epochs', 20)
    
    # Training Loop
    # early stopping params TODO: add to args
    tr_cfg = config.get('training', {})
    patience = tr_cfg.get("early_stopping").get("patience", 5)
    best_val_loss = float("inf")
    epochs_no_improve = 0 
    best_model_state_dict = None

    # log run configs
    wandb.watch(actor)
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

    wandb.define_metric("epoch")
    wandb.define_metric("epoch_actor_loss", step_metric="epoch")
    wandb.define_metric("epoch_critic_loss", step_metric="epoch")
    wandb.define_metric("epoch_mse_loss", step_metric="epoch")

    wandb.define_metric("trajectory")
    wandb.define_metric("train_x", step_metric="trajectory")
    wandb.define_metric("train_y", step_metric="trajectory")
    wandb.define_metric("train_total", step_metric="trajectory")
    
    wandb.define_metric("validation")
    wandb.define_metric("val_x", step_metric="validation")
    wandb.define_metric("val_y", step_metric="validation")
    wandb.define_metric("val_total", step_metric="validation")
    
    # batch_trajectories = X_train
    # batch_trajectories: List of trajectories, each is [(x1, y1), ..., (xT, yT)]
    best_epoch_policy_loss = torch.inf
    num_bins = model_config.get('num_bins', 32)
    counter = 0
    for epoch in range(epochs):
        actor.train()
        critic.train()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_mse_loss = 0

        # store trajectories for epoch
        epoch_trajectories = []

        for i in tqdm(range(X_train.shape[0]), desc="Trajectories"): 
            trajectory = X_train_model[i] # T x d
            states = []
            actions = []
            rewards = []
            old_log_probs = []
            values = []
            next_states = []

            # Collect data from trajectory
            for t in range(len(trajectory) - 1 - ctx_length):
                x_t, y_t = trajectory[t + ctx_length - 1, :x_dim], trajectory[t + ctx_length - 1, x_dim:]
                state = trajectory[t: (t + ctx_length)]
                state = state.unsqueeze(0)
                
                # state = torch.tensor([x_t, y_t], dtype=torch.float32)
                
                # Get action distribution from current policy:
                with torch.no_grad():
                    # Actor: Get action distribution
                    # mean, std = actor(state) 
                    # dist = Normal(mean, std)
                    # action = dist.sample()
                    # old_log_prob = dist.log_prob(action).sum()# .detach()
                    _, action = actor(state)
                    # x_pred = torch.softmax(action, dim=-1)
                    #x_pred = torch.max(x_pred, dim=-1)
                    dist = Categorical(logits=action)
                    action = dist.sample()
                    old_log_prob = dist.log_prob(action)
                    # action = torch.argmax(x_pred)
                    # Critic: Predict value
                    value = critic(state)

                # Store data
                states.append(state)
                actions.append(action)
                old_log_probs.append(old_log_prob.detach())
                values.append(value)

                # Reward: Improvement in y
                # x_next, y_next = trajectory[t + 1]
                x_next_t, y_next_t = trajectory[t + ctx_length, :x_dim], trajectory[t+ ctx_length, x_dim:]
                next_state = torch.hstack([x_next_t, y_next_t])
                next_states.append(next_state)

                current_bin_x = actor.bar_distribution_x.map_to_bucket_idx(x_t)
                current_bin_y = actor.bar_distribution_y.map_to_bucket_idx(y_t)
                next_bin_x = actor.bar_distribution_x.map_to_bucket_idx(x_next_t)
                next_bin_y = actor.bar_distribution_y.map_to_bucket_idx(y_next_t)
                
                # reward = torch.abs(y_t - y_next_t) + 1 - (action - x_next_t)
                reward_x = (num_bins - torch.abs(current_bin_x - next_bin_x)) 
                reward_y = (num_bins - torch.abs(current_bin_y - next_bin_y))
                total_reward = reward_x + reward_y
                rewards.append(total_reward)
                wandb.log({"trajectory": i*X_train.shape[0]+t,
                            "train_total": total_reward,
                           "train_x": reward_x,
                           "train_y": reward_y,
                           })
                counter = counter + 1 
            
            # store complete trajectory
            epoch_trajectories.append((states, actions, rewards, old_log_probs, values, next_states))

        # update the actor critic networks
        for e_idx, (states, actions, rewards, old_log_probs, values, next_states) in enumerate(epoch_trajectories):
            states = torch.stack(states).squeeze()
            action = torch.stack(actions)
            # rec_losses = torch.stack(rec_losses)
            old_log_probs = torch.stack(old_log_probs)
            # Compute advantages and returns
            rewards = torch.tensor(rewards).to(device)
            values = torch.stack(values).squeeze().to(device)
            next_states = torch.stack(next_states).to(device)
            
            # compute advantes
            returns, advantages = compute_gae(rewards, values, device=device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO updates (multiple epochs per batch)
            clip_eps=0.2
            actor_loss, mse_loss, critic_loss = ppo_loss(
                actor, critic, states, actions, old_log_probs, advantages, returns, clip_eps
            )

            # torch.autograd.set_detect_anomaly(True)
            # Update
            actor_optimizer.zero_grad()
            # total_loss = (actor_loss + 0.5 * critic_loss) / batch_size  # Weight critic loss
            # total_loss.backward() # accumulate gradients
            total_actor_loss = actor_loss
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optimizer.step()
            

            epoch_actor_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()     
            epoch_mse_loss += mse_loss.item()       
                    
        wandb.log({
            "epoch": epoch,
            "epoch_actor_loss": epoch_actor_loss / X_train.shape[0],
            "epoch_critic_loss": epoch_critic_loss / X_train.shape[0],
            "epoch_mse_loss": epoch_mse_loss / X_train.shape[0]
        })

        # Logging
        print(f"Epoch {epoch + 1}:")
        print(f"  Actor Loss: {epoch_actor_loss / X_train.shape[0]:.4f}")
        print(f"  Critic Loss: {epoch_critic_loss / X_train.shape[0]:.4f}")


        # Evaluation
        for i in tqdm(range(X_val.shape[0]), desc="Eval Trajectories"): 
            trajectory = X_val_model[i] # T x d
            states = []
            actions = []
            rewards = []
            old_log_probs = []
            values = []

            inferred_trajectory = trajectory[0: (0 + ctx_length)]
            # Collect data from trajectory
            for t in range(len(trajectory) - 1 - ctx_length):
                #x_t, y_t = trajectory[t + ctx_length - 1, :x_dim], trajectory[t + ctx_length - 1, x_dim:]
                state = inferred_trajectory[t: (t + ctx_length)]
                state = state.unsqueeze(0)
                with torch.no_grad():
                    # Actor: Get action distribution
                    _, action = actor(state)
                    x_pred = torch.softmax(action, dim=-1)
                    # x_pred = torch.max(x_pred, dim=-1)
                    dist = Categorical(logits=action)
                    #action = dist.sample()
                    # old_log_prob = dist.log_prob(action)
                    action = torch.argmax(x_pred)
                    next_x = actor.binner_x.unbin_values(action)
                    next_y = eval_fnc(val_models[i], next_x)

                inferred_trajectory = torch.vstack((inferred_trajectory, torch.stack((next_x, next_y))))
                x_next_t, y_next_t = trajectory[t + ctx_length, :x_dim], trajectory[t+ctx_length, x_dim:]

                predicted_bin_x = actor.bar_distribution_x.map_to_bucket_idx(next_x)
                predicted_bin_y = actor.bar_distribution_y.map_to_bucket_idx(next_y)
                true_next_bin_x = actor.bar_distribution_x.map_to_bucket_idx(x_next_t)
                true_next_bin_y = actor.bar_distribution_y.map_to_bucket_idx(y_next_t)

                val_reward_x = (num_bins - torch.abs(predicted_bin_x - true_next_bin_x)) 
                val_reward_y = (num_bins - torch.abs(predicted_bin_y - true_next_bin_y))
                val_total_reward = val_reward_x + val_reward_y
                wandb.log({"validation": i * X_val.shape[0] + t,
                            "val_total": val_total_reward,
                           "val_x": val_reward_x,
                           "val_y": val_reward_y})

def eval_fnc(fnc_dict, x_val):
   y_val = torch.zeros_like(x_val)
   for i in range(fnc_dict['num_components']):
       trig = fnc_dict['trigonometrics'][i]
       amp  = fnc_dict['amplitudes'][i]
       freq = fnc_dict['frequencies'][i]
       phase= fnc_dict['phases'][i]
       y_val += amp * trig(freq * x_val.cpu() + phase)
   return y_val

if __name__ == "__main__":
    main()



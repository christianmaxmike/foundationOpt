import argparse
import warnings
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from model.architecture_jan import PFTSN
from schedulefree import AdamWScheduleFree
from utils.logging import create_experiment_folder, save_experiment_info, save_model_with_dual_optimizers
from utils.plotting import plot_individual_trajectories, plot_training_curves

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PFN Transformer for Learned Optimization")
    parser.add_argument('--configs_path', type=str, default="./configs/",
                        help="Directory containing YAML config files.")
    parser.add_argument('--config', type=str, default="bar", help="Name of the YAML config file (without .yaml).")
    parser.add_argument('--exploration_loss', action='store_true', help="Use exploration loss term.")
    parser.add_argument('--convergence_loss', action='store_true', help="Use convergence loss term.")
    parser.add_argument('--forecast_steps', type=int, default=1, help="How many steps to predict in the future.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--sequence_length', type=int, default=None, help="Length of the input sequences.")
    parser.add_argument("--ckptAppx", type=str, default="test")
    parser.add_argument('--normalize_returns', action='store_true', default=True, help="Use return normalization.")
    parser.add_argument('--reward_clip', type=float, default=10,)
    parser.add_argument('--sparse_reward', action='store_true',default=True)
    parser.add_argument('--action_smoothness_penalty', type=float, default=0.01)

    if hasattr(sys, 'ps1') or 'pydevconsole' in sys.modules:
        return parser.parse_args([])
    else:
        return parser.parse_args()


class WelfordReturnNormalizer:
    """
    Welford's online algorithm for computing running mean and variance of returns.
    This version correctly normalizes returns, not step rewards.
    """

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences from mean

    def update(self, returns):
        """Update statistics with computed returns"""
        if isinstance(returns, torch.Tensor):
            returns_np = returns.detach().cpu().numpy().flatten()
        else:
            returns_np = np.array(returns).flatten()

        for ret in returns_np:
            self.count += 1
            delta = ret - self.mean
            self.mean += delta / self.count
            delta2 = ret - self.mean
            self.M2 += delta * delta2

    def normalize(self, returns):
        """Normalize returns using current statistics"""
        if self.count < 2:
            return returns

        std = self.std
        if isinstance(returns, torch.Tensor):
            return (returns - self.mean) / std
        else:
            return (np.array(returns) - self.mean) / std

    @property
    def std(self):
        if self.count < 2:
            return 1.0
        variance = self.M2 / (self.count - 1)
        return max(np.sqrt(variance), self.epsilon)

    def get_stats(self):
        return {
            'count': self.count,
            'mean': self.mean,
            'std': self.std,
            'variance': self.M2 / max(self.count - 1, 1)
        }


class ContinuousPPOAgent(nn.Module):
    def __init__(self, action_dim, model, hidden_dim=64, device='cpu'):
        super(ContinuousPPOAgent, self).__init__()

        self.device = device
        self.shared = model.to(device)

        self.actor_mean = nn.Linear(hidden_dim, action_dim).to(device)
        self.actor_log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        ).to(device)

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ).to(device)

    def forward(self, state):
        state = state.to(self.device)
        shared_out = self.shared(state)

        action_mean = self.actor_mean(shared_out)
        action_log_std = self.actor_log_std(shared_out)

        action_log_std = torch.clamp(action_log_std, min=-5, max=2)
        action_std = torch.exp(action_log_std)

        value = self.critic(shared_out)
        return action_mean, action_std, value

    def get_action(self, state):
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.detach().cpu().numpy(), log_prob.cpu(), value.cpu()

    def evaluate_action(self, state, action):
        action = action.to(self.device)
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class ContinuousPPO:
    def __init__(self, action_dim, model, hidden_dim, actor_lr=2e-4, critic_lr=2e-4,
                 clip_ratio=0.2, epochs=2, entropy_coef=1e-3, value_coef=0.5,
                 max_history_length=10, device='cpu', normalize_returns=True):
        self.device = device
        self.agent = ContinuousPPOAgent(action_dim, model, hidden_dim, device=device)

        actor_params = list(self.agent.shared.parameters()) + \
                       list(self.agent.actor_mean.parameters()) + \
                       list(self.agent.actor_log_std.parameters())

        critic_params = list(self.agent.critic.parameters())

        self.actor_optimizer = AdamWScheduleFree(
            actor_params, lr=actor_lr, weight_decay=0., warmup_steps=300)

        self.critic_optimizer = AdamWScheduleFree(
            critic_params, lr=critic_lr, weight_decay=0., warmup_steps=10)

        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_history_length = max_history_length

        # Initialize return normalizer
        self.normalize_returns = normalize_returns
        if normalize_returns:
            self.return_normalizer = WelfordReturnNormalizer()
            print("Return normalization enabled - will normalize returns after GAE computation")

    def update(self, states, actions, log_probs_old, rewards, values, dones, batch_size, trajectory_length):
        states = torch.stack(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        advantages, returns = self.compute_gae(rewards, values, dones, batch_size, trajectory_length)

        if self.normalize_returns:
            self.return_normalizer.update(returns)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.epochs):
            self.actor_optimizer.train()
            self.critic_optimizer.train()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            log_probs_new, entropy, values_new = self.agent.evaluate_action(states, actions)

            # Actor loss computation
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()
            total_actor_loss = actor_loss + self.entropy_coef * entropy_loss

            # Critic loss computation with proper normalization
            if self.normalize_returns:
                mean, std = self.return_normalizer.mean, self.return_normalizer.std
                returns_norm = (returns - mean) / std
                values_new_norm = (values_new.squeeze() - mean) / std
                values_norm = (values.squeeze() - mean) / std
            else:
                returns_norm = returns
                values_new_norm, values_norm = values_new.squeeze(), values.squeeze()

            values_clipped = values_norm + torch.clamp(
                values_new_norm - values_norm, -self.clip_ratio, self.clip_ratio)
            value_loss1 = (values_new_norm - returns_norm).pow(2)
            value_loss2 = (values_clipped - returns_norm).pow(2)
            critic_loss = torch.max(value_loss1, value_loss2).mean()

            # Combined loss with proper weighting
            total_loss = total_actor_loss + self.value_coef * critic_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.agent.shared.parameters()) +
                list(self.agent.actor_mean.parameters()) +
                list(self.agent.actor_log_std.parameters()), 0.5)
            torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 0.5)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # Return stats including normalization info
        stats = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': -entropy_loss.item(),
            'total_loss': total_loss.item()
        }

        if self.normalize_returns:
            normalizer_stats = self.return_normalizer.get_stats()
            stats.update({
                'return_mean': normalizer_stats['mean'],
                'return_std': normalizer_stats['std'],
                'return_count': normalizer_stats['count']
            })

        return stats

    def compute_gae(self, rewards, values, dones,
                    batch_size, trajectory_length,
                    gamma=1, lam=1):

        r = rewards.view(trajectory_length, batch_size)
        v = values.view(trajectory_length, batch_size)
        d = dones.float().view(trajectory_length, batch_size)

        A = torch.zeros_like(r)
        gae = torch.zeros(batch_size, device=r.device, dtype=r.dtype)

        for t in reversed(range(trajectory_length)):
            next_v = v[t + 1] if t < trajectory_length - 1 else torch.zeros(
                batch_size, device=v.device, dtype=v.dtype)
            non_term = 1.0 - d[t]
            delta = r[t] + gamma * next_v * non_term - v[t]
            gae = delta + gamma * lam * non_term * gae
            A[t] = gae.clone()

        R = A + v
        return A.flatten(), R.flatten()

    def get_raw_value(self, state):
        """Get unnormalized value estimate for logging/evaluation"""
        with torch.no_grad():
            _, _, value = self.agent.forward(state)
            if self.normalize_returns:
                mean, std = self.return_normalizer.mean, self.return_normalizer.std
                value_raw = value * std + mean
                return value_raw
            return value


def function_to_opt(x, x_shift, scaling, device='cpu'):
    return -scaling * (x + x_shift) ** 2


def compute_reward_with_penalties(y_raw, actions, prev_actions, step, trajectory_length,
                                  sparse_reward=False, action_smoothness_penalty=0.0,
                                  reward_clip=None):
    device = y_raw.device
    rewards = y_raw.clone()

    # Apply sparse reward structure
    if sparse_reward:
        if step < trajectory_length - 1:  # Steps 0 to T-2
            rewards = torch.zeros_like(y_raw)
        # else: step == trajectory_length - 1, keep full reward

    # Apply action smoothness penalty
    if action_smoothness_penalty > 0.0 and prev_actions is not None:
        # Compute L2 distance between current and previous actions
        action_diff = torch.norm(actions - prev_actions, dim=-1)
        smoothness_penalty = action_smoothness_penalty * action_diff
        rewards = rewards - smoothness_penalty

    # Apply reward clipping
    if reward_clip is not None:
        rewards = torch.clamp(rewards, -reward_clip, reward_clip)

    return rewards


def run_single_rollout(model, function_to_opt, device, evaluation_num, trajectory_length, max_history_length,
                       reward_clip=None, sparse_reward=False, action_smoothness_penalty=0.0, evaluation_mode=False):
    """Run a single rollout and return trajectories for each evaluation separately

    Args:
        evaluation_mode: If True, store true function values in trajectories for evaluation plotting.
                        If False, store modified rewards (for training consistency).
    """

    a, b = -10, 10
    x_batch = (torch.rand(evaluation_num, device=device) * (b - a) + a)  # [batch_size]
    x_batch.requires_grad_(True)

    min_shift, max_shift = -5, 5
    shift_batch = (torch.rand(evaluation_num, device=device) * (max_shift - min_shift) + min_shift)  # [batch_size]
    min_scale, max_scale = 0.2, 5
    scale_batch = (torch.rand(evaluation_num, device=device) * (max_scale - min_scale) + min_scale)
    y_raw = function_to_opt(x_batch, shift_batch, scale_batch, device)
    gradients = torch.autograd.grad(y_raw.sum(), x_batch)[0]  # Use raw reward for gradients

    # Apply reward modifications for initial state (for model input)
    y_batch = compute_reward_with_penalties(
        y_raw, x_batch.unsqueeze(-1), None, 0, trajectory_length,
        sparse_reward, action_smoothness_penalty, reward_clip
    )

    # For model state: always use modified rewards (this is what the model was trained on)
    initial_states = torch.stack([x_batch.detach(), y_batch.detach(), gradients.detach()], dim=1)  # [batch_size, 3]
    state_history = [initial_states.unsqueeze(1).unsqueeze(-1)]  # [batch_size, 1, 3, 1]

    # Store trajectories for each evaluation separately
    trajectories = [[] for _ in range(evaluation_num)]

    # Add initial points to trajectories
    for i in range(evaluation_num):
        x_val = x_batch[i].item()
        # For evaluation mode: use true function values, for training: use modified rewards
        y_val = y_raw[i].item() if evaluation_mode else y_batch[i].item()
        trajectories[i].append([x_val, y_val])

    prev_actions = x_batch.unsqueeze(-1)  # Track previous actions for smoothness penalty

    for step in range(trajectory_length):

        if len(state_history) > max_history_length:
            state_history = state_history[-max_history_length:]

        padded_history = state_history[:]
        while len(padded_history) < max_history_length:
            padded_history.insert(0, torch.zeros_like(state_history[0]))

        stacked_states = torch.cat(padded_history, dim=1)
        action_means, action_stds, values = model.agent.forward(stacked_states)
        dist = Normal(action_means, action_stds)
        actions = dist.sample()

        actions_tensor = actions.detach().clone().requires_grad_(True)
        y_raw = function_to_opt(actions_tensor.flatten(), shift_batch, scale_batch, device)
        gradients = torch.autograd.grad(y_raw.sum(), actions_tensor)[0]  # Use raw reward for gradients

        # Apply reward modifications (for model state)
        y_batch = compute_reward_with_penalties(
            y_raw, actions, prev_actions, step, trajectory_length,
            sparse_reward, action_smoothness_penalty, reward_clip
        )

        # For model state: always use modified rewards (this is what the model expects)
        new_states = torch.stack([actions_tensor.flatten().detach(), y_batch.detach(), gradients.flatten().detach()],
                                 dim=1)  # [batch_size, 3]
        state_history.append(new_states.unsqueeze(1).unsqueeze(-1))  # [batch_size, 1, 3, 1]

        # Store in trajectories: true values for evaluation, modified for training
        for i in range(evaluation_num):
            x_val = actions_tensor.flatten()[i].item()
            # For evaluation mode: use true function values, for training: use modified rewards
            y_val = y_raw[i].item() if evaluation_mode else y_batch[i].item()
            trajectories[i].append([x_val, y_val])

        prev_actions = actions  # Update previous actions

    # Return scales as well
    return trajectories, shift_batch.cpu().numpy(), scale_batch.cpu().numpy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experiment_dir = create_experiment_folder()

    epoch = 4000
    batch_size = 200
    trajectory_length = 10
    max_history_length = 10
    state_dim = 3
    action_dim = 1
    emb_dim = 32
    output_dim = 64
    evaluation_num = 10

    actor_lr = 3e-5
    critic_lr = 1e-5

    device = torch.device(
        ['cuda', 'mps', 'cpu'][np.argmax([torch.cuda.is_available(), torch.backends.mps.is_available(), True])])
    print(f"Using device: {device}", flush=True)


    model = PFTSN(input_dim=state_dim, output_dim=output_dim, emb_dim=emb_dim)
    ppo = ContinuousPPO(
        action_dim=action_dim,
        model=model,
        hidden_dim=output_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        max_history_length=max_history_length,
        device=device,
        normalize_returns=args.normalize_returns  # Use command line argument
    )

    hyperparams = {
        'batch_size': batch_size,
        'trajectory_length': trajectory_length,
        'max_history_length': max_history_length,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'emb_dim': emb_dim,
        'output_dim': output_dim,
        'evaluation_num': evaluation_num,
        'epochs_trained': epoch,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'normalize_returns': args.normalize_returns,
        'reward_clip': args.reward_clip,
        'sparse_reward': args.sparse_reward,
        'action_smoothness_penalty': args.action_smoothness_penalty
    }

    save_experiment_info(experiment_dir, hyperparams, args)

    episode_rewards = []
    actor_losses = []
    critic_losses = []
    entropies = []
    total_losses = []
    return_means = []
    return_stds = []

    print("Starting training", flush=True)
    for episode in range(epoch):
        all_states, all_actions, all_log_probs, all_rewards, all_values, all_dones = [], [], [], [], [], []

        a, b = -10, 10
        x_batch = (torch.rand(batch_size, device=device) * (b - a) + a)
        x_batch.requires_grad_(True)

        min_shift, max_shift = -5, 5
        shift_batch = (torch.rand(batch_size, device=device) * (max_shift - min_shift) + min_shift)
        min_scale, max_scale = 0.2, 5
        scale_batch = (torch.rand(batch_size, device=device) * (max_scale - min_scale) + min_scale)
        y_raw = function_to_opt(x_batch, shift_batch, scale_batch, device)
        gradients = torch.autograd.grad(y_raw.sum(), x_batch)[0]  # Use raw reward for gradients

        # Apply reward modifications for initial state
        y_batch = compute_reward_with_penalties(
            y_raw, x_batch.unsqueeze(-1), None, 0, trajectory_length,
            args.sparse_reward, args.action_smoothness_penalty, args.reward_clip
        )

        initial_states = torch.stack([x_batch.detach(), y_batch.detach(), gradients.detach()], dim=1)
        state_history = [initial_states.unsqueeze(1).unsqueeze(-1)]  # [batch_size, 1, 3, 1]

        total_episode_reward = 0
        prev_actions = x_batch.unsqueeze(-1)  # Track previous actions for smoothness penalty

        for step in range(trajectory_length):

            if len(state_history) > max_history_length:
                state_history = state_history[-max_history_length:]

            padded_history = state_history[:]
            while len(padded_history) < max_history_length:
                padded_history.insert(0, torch.zeros_like(state_history[0]))

            stacked_states = torch.cat(padded_history, dim=1)  # [batch_size, trajectory_length, 3, 1] - with padding
            action_means, action_stds, values = ppo.agent.forward(stacked_states)
            dist = Normal(action_means, action_stds)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(-1)

            actions_tensor = actions.detach().clone().requires_grad_(True)
            y_raw = function_to_opt(actions_tensor.flatten(), shift_batch, scale_batch, device)
            gradients = torch.autograd.grad(y_raw.sum(), actions_tensor)[0]  # Use raw reward for gradients

            # Apply reward modifications
            rewards = compute_reward_with_penalties(
                y_raw, actions, prev_actions, step, trajectory_length,
                args.sparse_reward, args.action_smoothness_penalty, args.reward_clip
            )

            done = step == trajectory_length - 1
            for i in range(batch_size):
                all_states.append(stacked_states[i])
                all_actions.append(actions[i].detach().cpu().numpy())
                all_log_probs.append(log_probs[i].item())
                all_rewards.append(rewards[i].item())
                all_values.append(values[i].item())
                all_dones.append(done)

            total_episode_reward += rewards.sum().item()

            new_states = torch.stack(
                [actions_tensor.flatten().detach(), rewards.detach(), gradients.flatten().detach()],
                dim=1)  # [batch_size, 3]
            state_history.append(new_states.unsqueeze(1).unsqueeze(-1))  # [batch_size, 1, 3, 1]

            prev_actions = actions  # Update previous actions

            if done:
                break

        losses = ppo.update(all_states, all_actions, all_log_probs, all_rewards, all_values, all_dones, batch_size,
                            trajectory_length)

        avg_episode_reward = total_episode_reward / batch_size
        episode_rewards.append(avg_episode_reward)
        actor_losses.append(losses['actor_loss'])
        critic_losses.append(losses['critic_loss'])
        entropies.append(losses['entropy'])
        total_losses.append(losses['total_loss'])

        # Track return normalization stats
        if args.normalize_returns:
            return_means.append(losses.get('return_mean', 0))
            return_stds.append(losses.get('return_std', 1))

        if episode % 100 == 0:
            reward_info = []
            if args.reward_clip is not None:
                reward_info.append(f"Clip: ±{args.reward_clip}")
            if args.sparse_reward:
                reward_info.append("Sparse")
            if args.action_smoothness_penalty > 0:
                reward_info.append(f"Smooth: {args.action_smoothness_penalty}")

            reward_str = f", Reward: {', '.join(reward_info)}" if reward_info else ""

            print(f"Episode {episode}, Avg Reward: {avg_episode_reward:.2f}, "
                  f"Actor Loss: {losses['actor_loss']:.4f}, "
                  f"Critic Loss: {losses['critic_loss']:.4f}, "
                  f"Entropy: {losses['entropy']:.4f}{reward_str}", flush=True)

    print("Training completed", flush=True)

    model_save_path = os.path.join(experiment_dir, "trained_ppo_model.pth")

    # Save additional stats
    additional_stats = {}
    if args.normalize_returns:
        additional_stats['return_means'] = return_means
        additional_stats['return_stds'] = return_stds

    save_model_with_dual_optimizers(ppo, model_save_path, episode_rewards, actor_losses,
                                    critic_losses, entropies, total_losses, hyperparams, **additional_stats)

    training_curves_path = plot_training_curves(episode_rewards, actor_losses, critic_losses, entropies, total_losses,
                                                experiment_dir)

    print("Starting evaluation", flush=True)

    trajectories, shifts, scales = run_single_rollout(
        ppo, function_to_opt, device, evaluation_num, trajectory_length,
        max_history_length,
        reward_clip=args.reward_clip,
        sparse_reward=args.sparse_reward,
        action_smoothness_penalty=args.action_smoothness_penalty,
        evaluation_mode=True
    )

    plot_individual_trajectories(trajectories, shifts, scales, function_to_opt, experiment_dir,
                                 title_prefix="Test Evaluation")

    print(f"Done: Saved in: {experiment_dir}")

    # Print final return normalization stats
    if args.normalize_returns:
        final_stats = ppo.return_normalizer.get_stats()
        print(f"Final Return Normalization Stats:")
        print(f"  Count: {final_stats['count']}")
        print(f"  Mean: {final_stats['mean']:.4f}")
        print(f"  Std: {final_stats['std']:.4f}")
        print(f"  Variance: {final_stats['variance']:.4f}")


if __name__ == "__main__":
    main()
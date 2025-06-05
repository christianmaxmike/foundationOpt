import argparse
import warnings
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import matplotlib.pyplot as plt

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

    if hasattr(sys, 'ps1') or 'pydevconsole' in sys.modules:
        return parser.parse_args([])
    else:
        return parser.parse_args()


class ContinuousPPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ContinuousPPOAgent, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared_out = self.shared(state)
        action_mean = self.actor_mean(shared_out)
        action_std = torch.exp(self.actor_log_std)
        value = self.critic(shared_out)
        return action_mean, action_std, value

    def get_action(self, state):
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.detach().numpy(), log_prob, value

    def evaluate_action(self, state, action):
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class ContinuousPPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, clip_ratio=0.2, epochs=10,
                 entropy_coef=0.01, value_coef=0.5):
        self.agent = ContinuousPPOAgent(state_dim, action_dim)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def update(self, states, actions, log_probs_old, rewards, values, dones):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)

        returns = []
        advantages = []
        gae = 0
        gamma = 0.99
        lam = 0.95

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]

            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.epochs):
            log_probs_new, entropy, values_new = self.agent.evaluate_action(states, actions)
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            values_clipped = values + torch.clamp(
                values_new.squeeze() - values, -self.clip_ratio, self.clip_ratio)
            value_loss1 = (values_new.squeeze() - returns).pow(2)
            value_loss2 = (values_clipped - returns).pow(2)
            critic_loss = torch.max(value_loss1, value_loss2).mean()

            entropy_loss = -entropy.mean()

            total_loss = (actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()

        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'entropy': -entropy_loss.item(), 'total_loss': total_loss.item()}


def function_to_opt(x):
    return -(x ** 2)


def plot_function_and_trajectory(trajectory, function_to_opt, title="Function Evaluation Trajectory"):
    """Plot the function and the trajectory of evaluations"""
    x_range = np.linspace(-6, 6, 1000)
    y_range = [function_to_opt(torch.tensor(x)).item() for x in x_range]

    # Combined visualization with function trajectory and x-coordinate progression
    plt.figure(figsize=(12, 10))

    # Top subplot: Function and trajectory
    plt.subplot(2, 1, 1)
    plt.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = -x²')

    # Plot the trajectory with single color gradient
    x_traj = [point[0] for point in trajectory]
    y_traj = [point[1] for point in trajectory]

    # Create alpha values from light to dark
    alphas = np.linspace(0.2, 1.0, len(x_traj))

    # Plot points with increasing opacity
    for i, (x, y, alpha) in enumerate(zip(x_traj, y_traj, alphas)):
        plt.plot(x, y, 'ro', markersize=8, alpha=alpha)

    # Optional: connect with fading lines
    for i in range(len(x_traj) - 1):
        alpha = alphas[i]
        plt.plot([x_traj[i], x_traj[i + 1]], [y_traj[i], y_traj[i + 1]],
                 'r-', linewidth=2, alpha=alpha)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'{title} - Function and Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Bottom subplot: X-coordinate progression
    plt.subplot(2, 1, 2)
    plt.plot(range(len(x_traj)), x_traj, 'ro-', markersize=6, linewidth=2)
    plt.axhline(y=0, color='g', linestyle='--', alpha=0.7, label='Optimum (x=0)')
    plt.xlabel('Step')
    plt.ylabel('x position')
    plt.title('X-coordinate progression')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_training_curves(episode_rewards, actor_losses, critic_losses, entropies, total_losses):
    """Plot training curves"""
    episodes = range(len(episode_rewards))

    plt.figure(figsize=(15, 10))

    # Episode rewards
    plt.subplot(2, 3, 1)
    plt.plot(episodes, episode_rewards, 'b-', alpha=0.7)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)

    # Actor loss
    plt.subplot(2, 3, 2)
    plt.plot(episodes, actor_losses, 'r-', alpha=0.7)
    plt.title('Actor Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Critic loss
    plt.subplot(2, 3, 3)
    plt.plot(episodes, critic_losses, 'g-', alpha=0.7)
    plt.title('Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Entropy
    plt.subplot(2, 3, 4)
    plt.plot(episodes, entropies, 'm-', alpha=0.7)
    plt.title('Entropy')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.grid(True, alpha=0.3)

    # Total loss
    plt.subplot(2, 3, 5)
    plt.plot(episodes, total_losses, 'orange', alpha=0.7)
    plt.title('Total Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Moving average of rewards
    plt.subplot(2, 3, 6)
    window_size = min(100, len(episode_rewards) // 10)
    if window_size > 1:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(episode_rewards)), moving_avg, 'b-', linewidth=2,
                 label=f'Moving Avg ({window_size})')
        plt.plot(episodes, episode_rewards, 'b-', alpha=0.3, label='Raw rewards')
        plt.legend()
    else:
        plt.plot(episodes, episode_rewards, 'b-', alpha=0.7)
    plt.title('Reward Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_single_rollout(ppo, function_to_opt, start_x=None):
    """Run a single rollout and return the trajectory"""

    a, b = -5, 5
    start_x = torch.rand(1).item() * (b - a) + a

    x = torch.tensor(start_x, requires_grad=True)
    y = function_to_opt(x)
    gradient = torch.autograd.grad(y, x, retain_graph=True)[0]

    state = torch.stack([x.detach(), y.detach(), gradient.detach()])
    trajectory = [(x.item(), y.item())]

    for step in range(5):
        action_np, log_prob, value = ppo.agent.get_action(state)
        action_tensor = torch.tensor(action_np, requires_grad=True, dtype=torch.float32)
        y = function_to_opt(action_tensor)
        trajectory.append((action_tensor.item(), y.item()))
        gradient = torch.autograd.grad(y, action_tensor, retain_graph=True)[0]
        state = torch.stack([action_tensor.detach(), y.detach(), gradient.detach()]).flatten()
    return trajectory


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    epoch = 10000

    device = ['cuda', 'mps', 'cpu'][np.argmax([torch.cuda.is_available(), torch.backends.mps.is_available(), True])]

    state_dim = 3
    action_dim = 1

    ppo = ContinuousPPO(state_dim, action_dim)

    # Storage for plotting
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    entropies = []
    total_losses = []

    print("Starting training...")
    for episode in range(epoch):
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        a, b = -5, 5
        random_index = torch.rand(1).item() * (b - a) + a
        x = torch.tensor(random_index, requires_grad=True)
        y = function_to_opt(x)
        gradient = torch.autograd.grad(y, x, retain_graph=True)[0]

        state = torch.stack([x.detach(), y.detach(), gradient.detach()])
        episode_reward = 0

        for step in range(5):
            action_np, log_prob, value = ppo.agent.get_action(state)
            action_tensor = torch.tensor(action_np, requires_grad=True, dtype=torch.float32)
            y = function_to_opt(action_tensor)
            reward = y.item()
            done = step == 4

            states.append(state.numpy())
            actions.append(action_np)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)

            episode_reward += reward

            gradient = torch.autograd.grad(y, action_tensor, retain_graph=True)[0]

            state = torch.stack([action_tensor.detach(),y.detach(),gradient.detach()]).flatten()

            if done:
                break

        losses = ppo.update(states, actions, log_probs, rewards, values, dones)

        episode_rewards.append(episode_reward)
        actor_losses.append(losses['actor_loss'])
        critic_losses.append(losses['critic_loss'])
        entropies.append(losses['entropy'])
        total_losses.append(losses['total_loss'])

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                  f"Actor Loss: {losses['actor_loss']:.4f}, "
                  f"Critic Loss: {losses['critic_loss']:.4f}, "
                  f"Entropy: {losses['entropy']:.4f}")

    print("Training completed!")

    plot_training_curves(episode_rewards, actor_losses, critic_losses, entropies, total_losses)
    print("\nRunning test rollouts...")

    trajectory = run_single_rollout(ppo, function_to_opt)
    plot_function_and_trajectory(trajectory, function_to_opt,
                                 f"Random Start Rollout")


if __name__ == "__main__":
    main()
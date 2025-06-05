import os
import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_individual_trajectories(trajectories, shifts, scales, function_to_opt, experiment_dir, title_prefix="Evaluation"):
    figs_dir = os.path.join(experiment_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    x_range = np.linspace(-6, 6, 1000)

    for eval_idx, (trajectory, shift, scale) in enumerate(zip(trajectories, shifts, scales)):
        # Calculate function values with the specific shift and scale for this evaluation
        y_range = [function_to_opt(torch.tensor(x), torch.tensor(shift), torch.tensor(scale), device='cpu').item() for x in x_range]

        # Create the plot
        plt.figure(figsize=(12, 10))

        # Top subplot: Function and trajectory
        plt.subplot(2, 1, 1)
        plt.plot(x_range, y_range, 'b-', linewidth=2, label=f'f(x) with shift={shift:.2f}, scale={scale:.2f}')

        # Plot the trajectory
        x_traj = [point[0] for point in trajectory]
        y_traj = [point[1] for point in trajectory]

        # Create alpha values from light to dark
        alphas = np.linspace(0.2, 1.0, len(x_traj))

        # Plot points with increasing opacity
        for i, (x, y, alpha) in enumerate(zip(x_traj, y_traj, alphas)):
            plt.plot(x, y, 'ro', markersize=8, alpha=alpha)

        # Connect with fading lines
        for i in range(len(x_traj) - 1):
            alpha = alphas[i]
            plt.plot([x_traj[i], x_traj[i + 1]], [y_traj[i], y_traj[i + 1]],
                     'r-', linewidth=2, alpha=alpha)

        # Mark the true optimum (accounting for shift)
        true_optimum = -shift  # For f(x) = -scale*(x+shift)², optimum is at x = -shift
        optimum_y = function_to_opt(torch.tensor(true_optimum), torch.tensor(shift), torch.tensor(scale), device='cpu').item()
        plt.plot(true_optimum, optimum_y, 'g*', markersize=15, label=f'True Optimum (x={true_optimum:.2f})')

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'{title_prefix} {eval_idx + 1} - Function and Trajectory (shift={shift:.2f}, scale={scale:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Bottom subplot: X-coordinate progression
        plt.subplot(2, 1, 2)
        plt.plot(range(len(x_traj)), x_traj, 'ro-', markersize=6, linewidth=2)
        plt.axhline(y=true_optimum, color='g', linestyle='--', alpha=0.7,
                    label=f'True Optimum (x={true_optimum:.2f})')
        plt.xlabel('Step')
        plt.ylabel('x position')
        plt.title('X-coordinate progression')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the figure in the figures subdirectory
        plt.savefig(os.path.join(figs_dir, f'evaluation_{eval_idx + 1}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory

        print(f"Saved plots for evaluation {eval_idx + 1} in {figs_dir}")



def plot_training_curves(episode_rewards, actor_losses, critic_losses, entropies, total_losses, experiment_dir):
    """Plot training curves and save in experiment directory"""
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

    # Save in experiment directory
    training_curves_path = os.path.join(experiment_dir, "training_curves.png")
    plt.savefig(training_curves_path, dpi=300, bbox_inches="tight")
    plt.close()

    return training_curves_path

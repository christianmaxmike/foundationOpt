import os
import torch
from model.architecture_jan import PFTSN





def create_experiment_folder(base_dir="found_exp"):
    """
    Create a new experiment folder with automatic numbering.
    Returns the path to the created experiment folder.
    """

    os.makedirs(base_dir, exist_ok=True)
    experiment_num = 0
    while True:
        experiment_dir = os.path.join(base_dir, f"experiment_{experiment_num}")
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            print(f"Created experiment directory: {experiment_dir}")
            return experiment_dir
        experiment_num += 1



# def load_trained_model(model_path, device):
#     """Load a trained PPO model from checkpoint with dual optimizers"""
#
#     # Load checkpoint
#     checkpoint = torch.load(model_path, map_location=device)
#     hyperparams = checkpoint['hyperparameters']
#
#     # Recreate the model architecture
#     base_model = PFTSN(
#         input_dim=hyperparams['state_dim'],
#         output_dim=hyperparams['output_dim'],
#         emb_dim=hyperparams['emb_dim']
#     )
#
#     ppo = ContinuousPPO(
#         action_dim=hyperparams['action_dim'],
#         model=base_model,
#         hidden_dim=hyperparams['output_dim'],
#         actor_lr=hyperparams.get('actor_lr', 2e-4),
#         critic_lr=hyperparams.get('critic_lr', 2e-4),
#         max_history_length=hyperparams['max_history_length'],
#         device=device
#     )
#
#     # Load the trained weights
#     ppo.agent.load_state_dict(checkpoint['model_state_dict'])
#
#     # Load optimizer states if they exist
#     if 'actor_optimizer_state_dict' in checkpoint:
#         ppo.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
#     if 'critic_optimizer_state_dict' in checkpoint:
#         ppo.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
#
#     # Fallback for old single optimizer checkpoints
#     if 'optimizer_state_dict' in checkpoint and 'actor_optimizer_state_dict' not in checkpoint:
#         print("Warning: Loading old checkpoint format with single optimizer")
#
#     print(f"Model loaded from {model_path}")
#     print(f"Training history: {hyperparams['epochs_trained']} episodes")
#
#     return ppo, checkpoint


def save_model_with_dual_optimizers(ppo, model_save_path, episode_rewards, actor_losses,
                                    critic_losses, entropies, total_losses, hyperparams, **kwargs):
    """Save model with dual optimizer states and additional stats"""
    save_dict = {
        'model_state_dict': ppo.agent.state_dict(),
        'actor_optimizer_state_dict': ppo.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': ppo.critic_optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'entropies': entropies,
        'total_losses': total_losses,
        'hyperparameters': hyperparams
    }

    # Add any additional stats passed via kwargs
    save_dict.update(kwargs)

    torch.save(save_dict, model_save_path)
    print(f"Model with dual optimizers saved to {model_save_path}")


def save_experiment_info(experiment_dir, hyperparams, args):
    """Save experiment configuration and metadata"""
    info_file = os.path.join(experiment_dir, "experiment_info.txt")

    with open(info_file, 'w') as f:
        f.write("=== Experiment Configuration ===\n")
        f.write(f"Experiment Directory: {experiment_dir}\n")
        f.write(f"Seed: {args.seed}\n\n")

        f.write("=== Hyperparameters ===\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")

        f.write(f"\n=== Command Line Arguments ===\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    print(f"Experiment info saved to {info_file}")
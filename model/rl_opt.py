import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 2)  # Outputs mean and log-variance

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        mean = x[0] # x[:, 0]
        log_var = x[1] # x[:, 1]
        return mean, log_var

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
    return returns

def train(batch_trajectories, policy, optimizer, epochs=100):
    # batch_trajectories: List of trajectories, each is [(x1, y1), ..., (xT, yT)]
    for epoch in range(epochs):
        for trajectory in batch_trajectories:
            states = []
            actions = []
            rewards = []
            
            # Run policy to generate actions and rewards
            for t in range(len(trajectory) - 1):
                x_t, y_t = trajectory[t]
                state = torch.tensor([x_t, y_t], dtype=torch.float32)
                mean, log_var = policy(state)
                std = torch.exp(0.5 * log_var)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()  # Next x_{t+1}
                
                # Get y_{t+1} (from trajectory)
                x_next, y_next = trajectory[t + 1]
                reward = y_t - y_next  # Improvement in y
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
            
            # Compute returns
            returns = compute_returns(rewards)
            
            # Compute policy loss
            policy_loss = []
            for state, action, R in zip(states, actions, returns):
                mean, log_var = policy(state)
                std = torch.exp(0.5 * log_var)
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(action)
                policy_loss.append(-log_prob * R)  # Negative for gradient ascent
            
            # Update policy
            optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()
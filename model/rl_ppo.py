import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from model.pfn_transformer import BinningProcessor


class Actor(nn.Module):
    """
    Variational Autoencoder (VAE) class.
    
    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        latent_dim (int): Dimensionality of the latent space.
    """
    
    def __init__(self, input_dim, x_min, x_max, y_min, y_max, hidden_dim=64, latent_dim=16):
        super(Actor, self).__init__()

        self.input_dim = input_dim
                
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, 2 * latent_dim), # 2 for mean and variance.
        )
        self.softplus = nn.Softplus()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

        self.binner_x = BinningProcessor(
            num_bins=latent_dim,
            min_val=x_min,
            max_val=x_max,
            train_data=None # train_data
        )
        self.binner_y = BinningProcessor(
            num_bins=latent_dim,
            min_val=y_min,
            max_val=y_max,
            train_data=None # train_data
        )
       # Final chosen bounds
        final_x_min = x_min #if x_min is not None else default_x_min
        final_x_max = x_max #if x_max is not None else default_x_max
        final_y_min = y_min #if y_min is not None else default_y_min
        final_y_max = y_max #if y_max is not None else default_y_max

        # Register these as buffers so that they are saved/restored with state_dict.
        # We'll store them as 1-element tensors and later call .item() where needed.
        self.register_buffer("x_min_buf", torch.tensor([final_x_min], dtype=torch.float32))
        self.register_buffer("x_max_buf", torch.tensor([final_x_max], dtype=torch.float32))
        self.register_buffer("y_min_buf", torch.tensor([final_y_min], dtype=torch.float32))
        self.register_buffer("y_max_buf", torch.tensor([final_y_max], dtype=torch.float32))

        bar_dist_smoothing: float = 0.0
        from model.bar_distribution import BarDistribution
        self.bar_distribution_x = BarDistribution(
            borders=torch.linspace(self.x_min_buf.item(), self.x_max_buf.item(), steps=latent_dim + 1),
            smoothing=bar_dist_smoothing,
            ignore_nan_targets=True
        )
        self.bar_distribution_y = BarDistribution(
            borders=torch.linspace(self.y_min_buf.item(), self.y_max_buf.item(), steps=latent_dim + 1),
            smoothing=bar_dist_smoothing,
            ignore_nan_targets=True
        )
        
    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.
        
        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        state_x = self.bar_distribution_x.map_to_bucket_idx(x[..., :1])
        state_y = self.bar_distribution_y.map_to_bucket_idx(x[..., 1:])
        state = torch.cat([state_x, state_y], dim=-1).reshape(-1, self.input_dim).float()  # 1 x (dtpt x x_dim x y_dim)

        x = self.encoder(state)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        
    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        
        Args:
            z (torch.Tensor): Data in the latent space.
        
        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)
    
    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.
        
        Returns:
            VAEOutput: VAE output dataclass.
        """
        nums, num_x, num_y = x.shape
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        return recon_x.reshape(nums, num_x, num_y), z
        



class Actor_MLP(nn.Module):
    def __init__(self, input_dim, x_min, x_max, y_min, y_max, hidden_dim=64, num_bins=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_latent = nn.Linear(hidden_dim, num_bins)
        self.fc3 = nn.Linear(num_bins, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        # self.fc_bins = nn.Linear(hidden_dim, num_bins)
        # self.fc_mean = nn.Linear(hidden_dim, 1)  # Predicts mean (μ)
        # self.fc_logvar = nn.Linear(hidden_dim, 1) # Predicts log-variance (logσ²)
        self.binner_x = BinningProcessor(
            num_bins=num_bins,
            min_val=x_min,
            max_val=x_max,
            train_data=None # train_data
        )
        self.binner_y = BinningProcessor(
            num_bins=num_bins,
            min_val=y_min,
            max_val=y_max,
            train_data=None # train_data
        )
       # Final chosen bounds
        final_x_min = x_min #if x_min is not None else default_x_min
        final_x_max = x_max #if x_max is not None else default_x_max
        final_y_min = y_min #if y_min is not None else default_y_min
        final_y_max = y_max #if y_max is not None else default_y_max

        # Register these as buffers so that they are saved/restored with state_dict.
        # We'll store them as 1-element tensors and later call .item() where needed.
        self.register_buffer("x_min_buf", torch.tensor([final_x_min], dtype=torch.float32))
        self.register_buffer("x_max_buf", torch.tensor([final_x_max], dtype=torch.float32))
        self.register_buffer("y_min_buf", torch.tensor([final_y_min], dtype=torch.float32))
        self.register_buffer("y_max_buf", torch.tensor([final_y_max], dtype=torch.float32))

        bar_dist_smoothing: float = 0.0
        from model.bar_distribution import BarDistribution
        self.bar_distribution_x = BarDistribution(
            borders=torch.linspace(self.x_min_buf.item(), self.x_max_buf.item(), steps=num_bins + 1),
            smoothing=bar_dist_smoothing,
            ignore_nan_targets=True
        )
        self.bar_distribution_y = BarDistribution(
            borders=torch.linspace(self.y_min_buf.item(), self.y_max_buf.item(), steps=num_bins + 1),
            smoothing=bar_dist_smoothing,
            ignore_nan_targets=True
        )


    def forward(self, state):
        n, num_x , num_y = state.shape
        state_x = self.bar_distribution_x.map_to_bucket_idx(state[..., :1])
        state_y = self.bar_distribution_y.map_to_bucket_idx(state[..., 1:])
        state = torch.cat([state_x, state_y], dim=-1).reshape(-1, self.fc1.in_features).float()  # 1 x (dtpt x x_dim x y_dim)
        x = torch.tanh(self.fc1(state))
        # x = torch.relu(self.fc1(state.reshape(-1, state.shape[-1]*state.shape[-2]).float()))
        x = torch.tanh(self.fc2(x))
        latent = torch.tanh(self.fc_latent(x))
        x = torch.tanh(self.fc3(latent))
        x = torch.tanh(self.fc4(x))
        # x_pred = self.fc_bins(x)
        #x_pred = torch.softmax(self.fc_bins(x))
        #x_pred = torch.argmax(x_pred, dim=-1)
        return x.reshape(n, num_x, num_y), latent
        
        mean = torch.sigmoid(self.fc_mean(x))
        log_var = self.fc_logvar(x)
        std = torch.exp(0.5 * log_var)
        return mean, std

class Critic(nn.Module):
    def __init__(self, input_dim, x_min, x_max, y_min, y_max, hidden_dim=64, num_bins=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)  # Predicts V(s)

        final_x_min = x_min #if x_min is not None else default_x_min
        final_x_max = x_max #if x_max is not None else default_x_max
        final_y_min = y_min #if y_min is not None else default_y_min
        final_y_max = y_max #if y_max is not None else default_y_max

        # Register these as buffers so that they are saved/restored with state_dict.
        # We'll store them as 1-element tensors and later call .item() where needed.
        self.register_buffer("x_min_buf", torch.tensor([final_x_min], dtype=torch.float32))
        self.register_buffer("x_max_buf", torch.tensor([final_x_max], dtype=torch.float32))
        self.register_buffer("y_min_buf", torch.tensor([final_y_min], dtype=torch.float32))
        self.register_buffer("y_max_buf", torch.tensor([final_y_max], dtype=torch.float32))

        bar_dist_smoothing: float = 0.0
        from model.bar_distribution import BarDistribution
        self.bar_distribution_x = BarDistribution(
            borders=torch.linspace(self.x_min_buf.item(), self.x_max_buf.item(), steps=num_bins + 1),
            smoothing=bar_dist_smoothing,
            ignore_nan_targets=True
        )
        self.bar_distribution_y = BarDistribution(
            borders=torch.linspace(self.y_min_buf.item(), self.y_max_buf.item(), steps=num_bins + 1),
            smoothing=bar_dist_smoothing,
            ignore_nan_targets=True
        )


    def forward(self, state):
        state_x = self.bar_distribution_x.map_to_bucket_idx(state[..., :1])
        state_y = self.bar_distribution_y.map_to_bucket_idx(state[..., 1:])
        state = torch.cat([state_x, state_y], dim=-1).reshape(-1, self.fc1.in_features).float()
        # state = torch.cat([state_x, state_y], dim=-1).flatten().float()
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        value = self.fc_value(x)
        return value
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "sigmoid":
        return F.sigmoid
    else:
        raise RuntimeError(f"activation should be 'relu' or 'gelu', not {activation}")

class BinningProcessor(nn.Module):
    """
    Converts continuous values in [min_val, max_val] into discrete bins using torch.bucketize.
    Assumes input values are already transformed.
    """
    def __init__(self, num_bins=32, min_val=0.0, max_val=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val 
        self.max_val = max_val
        boundaries = torch.linspace(min_val, max_val, steps=num_bins + 1)[1:-1]
        self.register_buffer("boundaries", boundaries)

    def bin_values(self, values: torch.Tensor) -> torch.Tensor:
        # Clamp and ensure contiguous before bucketizing.
        clipped = torch.clamp(values, self.min_val, self.max_val).contiguous()
        indices = torch.bucketize(clipped, self.boundaries)
        return indices

    def unbin_values(self, indices: torch.Tensor) -> torch.Tensor:
        bin_width = (self.max_val - self.min_val) / self.num_bins
        centers = self.min_val + (indices.float() + 0.5) * bin_width
        return centers

class TransformerBlock(nn.Module):
    """
    Basic Transformer block with multi-head attention and feed-forward.
    Supports optional pre-norm.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        pre_norm: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs
        )
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim, **factory_kwargs)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.activation = _get_activation_fn(activation)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if self.pre_norm:
            x_norm = self.norm1(x)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
            x = x + self.dropout1(attn_out)
            x_norm = self.norm2(x)
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
            x = x + self.dropout2(ff_out)
        else:
            attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
            x = self.norm1(x + self.dropout1(attn_out))
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = self.norm2(x + self.dropout2(ff_out))
        return x

class PFNTransformer(nn.Module):
    """
    Transformer-based trajectory prediction model.
    
    - The input is the concatenation of X and Y (with dimensions x_dim and y_dim).
    - The model uses separate binning processors for X and Y using externally provided bounds.
    - The output is produced by a shared Transformer encoder followed by a linear projection.
    
    When using bar-distribution loss (i.e. use_bar_distribution=True), the forward pass returns
    continuous target values (raw values) so that the BarDistribution module can compute bucket indices.
    Otherwise, it returns discretized targets (binned indices) for cross-entropy loss.
    """
    def __init__(
        self,
        x_dim: int = 2,       # Dimension of X
        y_dim: int = 1,       # Dimension of Y
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        num_bins: int = 32,
        forecast_steps: int = 1,
        num_steps: int = 50,
        dim_feedforward: int = 128,
        activation: str = "relu",
        pre_norm: bool = True,
        use_positional_encoding: bool = False,
        use_autoregression: bool = False,
        nar_inference_flag: bool = False,
        use_bar_distribution: bool = False,
        bar_dist_smoothing: float = 0.0,
        full_support: bool = False,
        transform_type: str = "power",
        x_min: float = None,
        x_max: float = None,
        y_min: float = None,
        y_max: float = None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_dim = x_dim + y_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_bins = num_bins
        self.forecast_steps = forecast_steps
        self.use_autoregression = use_autoregression
        self.nar_inference_flag = nar_inference_flag

        # If transformation parameters are not provided, use defaults.
        if transform_type == "power":
            default_x_min, default_x_max = -4.0, 4.0
            default_y_min, default_y_max = -4.0, 4.0
        else:
            default_x_min, default_x_max = 0.0, 1.0
            default_y_min, default_y_max = 0.0, 1.0

        self.x_min = x_min if x_min is not None else default_x_min
        self.x_max = x_max if x_max is not None else default_x_max
        self.y_min = y_min if y_min is not None else default_y_min
        self.y_max = y_max if y_max is not None else default_y_max

        # Create separate binning processors for X and Y.
        self.binner_x = BinningProcessor(num_bins=num_bins, min_val=self.x_min, max_val=self.x_max)
        self.binner_y = BinningProcessor(num_bins=num_bins, min_val=self.y_min, max_val=self.y_max)

        # Input embedding for concatenated (X,Y)
        self.input_embed = nn.Linear(self.input_dim, hidden_dim)
        self.pos_embed = nn.Embedding(num_steps, hidden_dim) if use_positional_encoding else None

        # Shared Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm
            ) for _ in range(num_layers)
        ])

        # Output projection: produces logits for all (x_dim+y_dim) heads.
        self.out_proj = nn.Linear(hidden_dim, self.input_dim * num_bins)

        self.use_bar_distribution = use_bar_distribution
        if self.use_bar_distribution:
            from model.bar_distribution import BarDistribution
            self.bar_distribution_x = BarDistribution(
                borders=torch.linspace(self.x_min, self.x_max, steps=num_bins + 1),
                smoothing=bar_dist_smoothing,
                ignore_nan_targets=True
            )
            self.bar_distribution_y = BarDistribution(
                borders=torch.linspace(self.y_min, self.y_max, steps=num_bins + 1),
                smoothing=bar_dist_smoothing,
                ignore_nan_targets=True
            )
        else:
            self.bar_distribution_x = None
            self.bar_distribution_y = None

    def forward_with_binning(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq: [B, T, input_dim] where input_dim = x_dim + y_dim.
        
        Returns:
            logits: [B, T_out, input_dim, num_bins]
            targets: if use_bar_distribution is True, returns continuous targets
                     (i.e. raw values); otherwise returns binned targets [B, T_out, input_dim].
        T_out depends on forecast_steps and AR/NAR mode.
        """
        if self.use_autoregression:
            if self.training or not self.nar_inference_flag:
                return self._forward_ar(seq)
            else:
                return self._forward_nar(seq)
        else:
            return self._forward_nar(seq)

    def _forward_nar(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = seq.shape
        tokens = seq[:, :-1, :]  # teacher forcing: use first T-1 timesteps
        x_embed = self.embed_tokens(tokens)
        for block in self.blocks:
            x_embed = block(x_embed)
        logits_all = self.out_proj(x_embed)  # [B, T-1, input_dim*num_bins]
        logits_all = logits_all.view(B, T-1, self.input_dim, self.num_bins)
        next_values = seq[:, 1:, :]  # continuous targets from ground truth
        if self.use_bar_distribution:
            targets = next_values
        else:
            target_bins_x = self.binner_x.bin_values(next_values[..., :self.x_dim])
            target_bins_y = self.binner_y.bin_values(next_values[..., self.x_dim:])
            targets = torch.cat([target_bins_x, target_bins_y], dim=-1)
        return logits_all, targets

    def _forward_ar(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = seq.shape
        all_logits = []
        all_targets = []
        for t_idx in range(T - 1):
            partial_seq = seq[:, :t_idx + 1, :]
            x_embed = self.embed_tokens(partial_seq)
            for block in self.blocks:
                x_embed = block(x_embed)
            last_h = x_embed[:, -1, :]  # [B, hidden_dim]
            logits_all = self.out_proj(last_h).view(B, self.input_dim, self.num_bins)
            target_vals = seq[:, t_idx + 1, :]
            if self.use_bar_distribution:
                targets = target_vals
            else:
                target_bins_x = self.binner_x.bin_values(target_vals[:, :self.x_dim])
                target_bins_y = self.binner_y.bin_values(target_vals[:, self.x_dim:])
                targets = torch.cat([target_bins_x, target_bins_y], dim=-1)
            all_logits.append(logits_all.unsqueeze(1))
            all_targets.append(targets.unsqueeze(1))
        logits = torch.cat(all_logits, dim=1)         # [B, T-1, input_dim, num_bins]
        targets = torch.cat(all_targets, dim=1)         # [B, T-1, input_dim]
        return logits, targets

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, _ = tokens.shape
        x = self.input_embed(tokens)
        if self.pos_embed is not None:
            positions = torch.arange(T, device=tokens.device).unsqueeze(0)
            x += self.pos_embed(positions)
        return x

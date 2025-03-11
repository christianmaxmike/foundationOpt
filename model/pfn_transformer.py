import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def _get_activation_fn(activation: str):
    """
    Returns the activation function corresponding to the given string.
    Supported: "relu", "gelu", "sigmoid".
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "sigmoid":
        return F.sigmoid
    else:
        raise RuntimeError(f"activation should be 'relu', 'gelu', or 'sigmoid', not {activation}")

class BinningProcessor(nn.Module):
    """
    Converts continuous values in [min_val, max_val] into discrete bins using torch.bucketize.
    Assumes input values are already transformed (if needed).
    """
    def __init__(self, num_bins=32, min_val=0.0, max_val=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

        # Create bin boundaries (length = num_bins - 1).
        boundaries = torch.linspace(min_val, max_val, steps=num_bins + 1)[1:-1]
        # Register as a buffer so it's moved to the correct device with the model.
        self.register_buffer("boundaries", boundaries)

    def bin_values(self, values: torch.Tensor) -> torch.Tensor:
        """
        Clamps 'values' to [min_val, max_val], then bucketizes them.
        Returns integer bin indices with the same shape as 'values'.
        """
        clipped = torch.clamp(values, self.min_val, self.max_val).contiguous()
        indices = torch.bucketize(clipped, self.boundaries)
        return indices

    def unbin_values(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Converts integer bin indices back into approximate continuous centers
        by taking the midpoint of each bin.
        """
        bin_width = (self.max_val - self.min_val) / self.num_bins
        centers = self.min_val + (indices.float() + 0.5) * bin_width
        return centers

class TransformerBlock(nn.Module):
    """
    Basic Transformer block with multi-head self-attention and a feed-forward sublayer.
    Supports optional 'pre_norm' (pre-layernorm) or standard 'post_norm' (post-layernorm).
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

        # Multihead self-attention module
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs
        )

        # Feed-forward network: Linear -> Dropout(Activation) -> Linear
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim, **factory_kwargs)

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.pre_norm = pre_norm
        self.activation = _get_activation_fn(activation)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        If pre_norm is True: LN -> MHA -> Residual -> LN -> FF -> Residual
        Else: MHA -> Dropout -> Residual -> LN -> FF -> Dropout -> Residual -> LN
        """
        if self.pre_norm:
            # Pre-Norm approach
            x_norm = self.norm1(x)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
            x = x + self.dropout1(attn_out)

            x_norm = self.norm2(x)
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
            x = x + self.dropout2(ff_out)
        else:
            # Classic Post-Norm approach
            attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
            x = self.norm1(x + self.dropout1(attn_out))

            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = self.norm2(x + self.dropout2(ff_out))

        return x

class PFNTransformer(nn.Module):
    """
    Transformer-based trajectory prediction model.

    - The input has shape [B, T, (x_dim + y_dim)], where x_dim is the dimension of X
      and y_dim is the dimension of Y.
    - The model discretizes X and Y values into bins using separate BinningProcessor objects.
    - If 'use_bar_distribution' is True, the forward pass returns the raw continuous targets
      (so a separate BarDistribution can compute negative log likelihood).
      Otherwise, it returns integer bin indices for a cross-entropy loss.

    By default, the model has two forward paths:
      1. Autoregressive (_forward_ar): Step-by-step prediction
      2. Non-autoregressive (_forward_nar): Teacher-forcing all at once
    This is controlled by 'use_autoregression' and 'nar_inference_flag'.
    """

    def __init__(
        self,
        x_dim: int = 2,       # Dimensionality of X
        y_dim: int = 1,       # Dimensionality of Y
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
    ):
        super().__init__()
        # Save data dimensions
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_dim = x_dim + y_dim

        # Save key model hyperparameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_bins = num_bins
        self.forecast_steps = forecast_steps
        self.use_autoregression = use_autoregression
        self.nar_inference_flag = nar_inference_flag

        # Build BinningProcessors for X and Y using the registered buffers.
        self.binner_x = BinningProcessor(
            num_bins=num_bins,
        )
        self.binner_y = BinningProcessor(
            num_bins=num_bins,
        )

        # Input embedding (transform from input_dim -> hidden_dim)
        self.input_embed = nn.Linear(self.input_dim, hidden_dim)

        # Optional positional embedding
        self.pos_embed = nn.Embedding(num_steps, hidden_dim) if use_positional_encoding else None

        # A stack of TransformerBlocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])

        # Final projection from hidden_dim -> (input_dim * num_bins)
        self.out_proj = nn.Linear(hidden_dim, self.input_dim * num_bins)

        self.use_bar_distribution = use_bar_distribution
        if self.use_bar_distribution:
            from model.bar_distribution import BarDistribution
            # Build bar distributions for X and Y using the same bin edges
            self.bar_distribution_x = BarDistribution(
                borders=torch.linspace(-1, 1, steps=num_bins + 1),
                smoothing=bar_dist_smoothing,
                ignore_nan_targets=True
            )
            self.bar_distribution_y = BarDistribution(
                borders=torch.linspace(-1, 1, steps=num_bins + 1),
                smoothing=bar_dist_smoothing,
                ignore_nan_targets=True
            )
        else:
            self.bar_distribution_x = None
            self.bar_distribution_y = None



    def forward_with_binning(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input sequence and return (logits, targets).
        Handles data transform automatically in eval mode.
        """

        # 2) Then run whichever forward pass you’ve already implemented
        if self.use_autoregression:
            if self.training or not self.nar_inference_flag:
                return self._forward_ar(seq)
            else:
                return self._forward_nar(seq)
        else:
            return self._forward_nar(seq)

    def _forward_nar(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Non-autoregressive forward pass. We do teacher forcing on T-1 steps to predict T-1 next steps.
        """
        B, T, _ = seq.shape
        tokens = seq[:, :-1, :]  # All but last step as input
        x_embed = self.embed_tokens(tokens)
        for block in self.blocks:
            x_embed = block(x_embed)

        # Project to (input_dim * num_bins) then reshape
        logits_all = self.out_proj(x_embed)  # [B, T-1, input_dim * num_bins]
        logits_all = logits_all.view(B, T-1, self.input_dim, self.num_bins)

        # Next-step ground-truth for each feature
        next_values = seq[:, 1:, :]  # shape [B, T-1, input_dim]

        # If using bar distribution, we return the raw continuous targets
        # otherwise, we bin them for cross-entropy
        if self.use_bar_distribution:
            targets = next_values  # raw
        else:
            target_bins_x = self.binner_x.bin_values(next_values[..., :self.x_dim])
            target_bins_y = self.binner_y.bin_values(next_values[..., self.x_dim:])
            targets = torch.cat([target_bins_x, target_bins_y], dim=-1)

        return logits_all, targets

    def _forward_ar(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive forward pass (step-by-step).
        We iterate from t=0..(T-2), each time:
         - embed partial_seq up to t_idx
         - predict the next item
         - record logits and target bins
        """
        B, T, _ = seq.shape
        all_logits = []
        all_targets = []

        for t_idx in range(T - 1):
            partial_seq = seq[:, :t_idx + 1, :]
            x_embed = self.embed_tokens(partial_seq)
            for block in self.blocks:
                x_embed = block(x_embed)

            # Last hidden state => project
            last_h = x_embed[:, -1, :]  # [B, hidden_dim]
            logits_all = self.out_proj(last_h).view(B, self.input_dim, self.num_bins)

            # Next ground truth step
            target_vals = seq[:, t_idx + 1, :]
            if self.use_bar_distribution:
                # Return raw continuous target
                targets = target_vals
            else:
                # Return discrete bin indices
                target_bins_x = self.binner_x.bin_values(target_vals[:, :self.x_dim])
                target_bins_y = self.binner_y.bin_values(target_vals[:, self.x_dim:])
                targets = torch.cat([target_bins_x, target_bins_y], dim=-1)

            all_logits.append(logits_all.unsqueeze(1))
            all_targets.append(targets.unsqueeze(1))

        # Concatenate along time dimension
        logits = torch.cat(all_logits, dim=1)   # [B, T-1, input_dim, num_bins]
        targets = torch.cat(all_targets, dim=1) # [B, T-1, input_dim]
        return logits, targets

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, T, input_dim].
        1) Project to hidden_dim.
        2) Optionally add positional embeddings.
        Returns: [B, T, hidden_dim].
        """
        B, T, _ = tokens.shape
        x = self.input_embed(tokens)
        if self.pos_embed is not None:
            positions = torch.arange(T, device=tokens.device).unsqueeze(0)
            x += self.pos_embed(positions)
        return x

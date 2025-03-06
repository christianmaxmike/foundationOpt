import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError(f"activation should be 'relu' or 'gelu', not {activation}")


class BinningProcessor(nn.Module):
    """
    Efficiently converts continuous values in [min_val, max_val] into discrete bins
    using torch.bucketize for speed. This version avoids creating many intermediate tensors.
    """
    def __init__(self, num_bins=32, min_val=0.0, max_val=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        # Compute bin boundaries
        boundaries = torch.linspace(min_val, max_val, steps=num_bins + 1)[1:-1]
        self.register_buffer("boundaries", boundaries)

    def bin_values(self, values: torch.Tensor) -> torch.Tensor:
        # values: [..., output_dim], all within [min_val, max_val] (or clipped)
        clipped = torch.clamp(values, self.min_val, self.max_val)
        indices = torch.bucketize(clipped, self.boundaries)
        return indices

    def unbin_values(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [..., output_dim], integer bin indices
        bin_width = (self.max_val - self.min_val) / self.num_bins
        centers = self.min_val + (indices.float() + 0.5) * bin_width
        return centers


class TransformerBlock(nn.Module):
    """
    A Transformer block with multi-head attention + feed-forward,
    supporting optional pre-norm and other configurable params.
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
        """
        Forward pass. If pre_norm is True, apply layer norm before attention/FF.
        """
        if self.pre_norm:
            # --- Pre-Norm Variant ---
            x_norm = self.norm1(x)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
            x = x + self.dropout1(attn_out)

            x_norm = self.norm2(x)
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
            x = x + self.dropout2(ff_out)
        else:
            # --- Post-Norm (classic) ---
            attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
            x = self.norm1(x + self.dropout1(attn_out))

            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = self.norm2(x + self.dropout2(ff_out))

        return x


class PFNTransformer(nn.Module):
    """
    Transformer-based trajectory prediction model with AR & NAR training.
    Predicts next-step values (x plus y) using discrete binning.

    Key changes from the original version:
    - Added `output_dim`: how many dimensions we want to predict (D+1).
    - The final projection produces `output_dim * num_bins`.
    - The final logits have shape [B, T_out, output_dim, num_bins].
    - The targets have shape [B, T_out, output_dim].
    """
    def __init__(
        self,
        ##########################################
        input_dim=2,        # Dimension of each input token (e.g., x + y).
        output_dim=2,       # Number of dimensions we predict => D+1 heads.
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        num_bins=32,
        ##########################################
        forecast_steps=1,
        num_steps=50,
        ##########################################
        dim_feedforward=128,
        activation="relu",
        pre_norm=True,
        use_positional_encoding=False,
        ##########################################
        use_autoregression=False,
        nar_inference_flag=False,
        ##########################################
        use_bar_distribution=False,
        bar_dist_smoothing=0.0,
        full_support=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_bins = num_bins
        self.forecast_steps = forecast_steps
        self.use_autoregression = use_autoregression
        self.nar_inference_flag = nar_inference_flag

        # Binning processor (for cross-entropy approach).
        self.binner = BinningProcessor(num_bins=num_bins)

        # Input embedding
        self.input_embed = nn.Linear(input_dim, hidden_dim)

        # Positional embeddings
        self.pos_embed = nn.Embedding(num_steps, hidden_dim) if use_positional_encoding else None

        # Transformer blocks
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

        # Output projection => shape = [B, T, output_dim * num_bins]
        self.out_proj = nn.Linear(hidden_dim, output_dim * num_bins)

        # Optionally build a BarDistribution-based module
        self.use_bar_distribution = use_bar_distribution
        if self.use_bar_distribution:
            # Create the bin edges that match BinningProcessor
            bar_borders = torch.linspace(0.0, 1.0, steps=num_bins+1)
            if full_support:
                from model.bar_distribution import FullSupportBarDistribution
                self.bar_distribution = FullSupportBarDistribution(
                    borders=bar_borders,
                    smoothing=bar_dist_smoothing,
                    ignore_nan_targets=True
                )
            else:
                from model.bar_distribution import BarDistribution
                self.bar_distribution = BarDistribution(
                    borders=bar_borders,
                    smoothing=bar_dist_smoothing,
                    ignore_nan_targets=True
                )
        else:
            self.bar_distribution = None

    def forward_with_binning(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq: [B, T, input_dim].
          For example, if input_dim=4, it might be 2D x + 2D y, or D x + 1D y, etc.
          Values are assumed to be in [0,1] if using direct binning, or suitably normalized.

        Returns:
          logits: [B, T_out, output_dim, num_bins]
          target_bins: [B, T_out, output_dim]
        Where T_out depends on forecast_steps & AR/NAR mode.
        """
        # If AR is used for training or we haven't enabled the NAR inference flag,
        # do step-by-step. Otherwise, do NAR in one go.
        if self.use_autoregression:
            if self.training or not self.nar_inference_flag:
                return self._forward_ar(seq)
            else:
                return self._forward_nar(seq)
        else:
            # If AR is off, we always do NAR
            return self._forward_nar(seq)

    def _forward_nar(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Non-autoregressive forward pass (predict entire sequence in parallel).
        We'll do "teacher forcing" for T-1 steps to predict T-1 future steps.
        """
        B, T, _ = seq.shape

        # 1) Embed the input sequence (except the last item)
        tokens = seq[:, :-1, :]  # [B, T-1, input_dim]
        x_embed = self.embed_tokens(tokens)
        for block in self.blocks:
            x_embed = block(x_embed)

        # 2) Produce logits => shape [B, T-1, output_dim, num_bins]
        logits_2d = self.out_proj(x_embed)  # [B, T-1, output_dim * num_bins]
        logits = logits_2d.view(B, T-1, self.output_dim, self.num_bins)

        # 3) Target bins are the next-step ground truth for each output dimension
        next_values = seq[:, 1:, :self.output_dim]  # [B, T-1, output_dim]
        target_bins = self.binner.bin_values(next_values)  # same shape [B, T-1, output_dim]

        return logits, target_bins

    def _forward_ar(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive forward pass (step-by-step).
        We'll iterate from t_idx=0..(T-2), each time:
          - Embed partial_seq up to t_idx
          - Predict the next item
          - Collect logits & target bins
        """
        B, T, _ = seq.shape
        all_logits, all_targets = [], []

        for t_idx in range(T - 1):
            # 1) Embed partial sequence up to t_idx
            partial_seq = seq[:, :t_idx + 1, :]  # [B, t_idx+1, input_dim]
            x_embed = self.embed_tokens(partial_seq)
            for block in self.blocks:
                x_embed = block(x_embed)

            # 2) Take the last hidden state => project => shape [B, output_dim, num_bins]
            last_h = x_embed[:, -1, :]      # [B, hidden_dim]
            logits_2d = self.out_proj(last_h).view(B, self.output_dim, self.num_bins)

            # 3) Bin the *next* step (t_idx+1)
            target_vals = seq[:, t_idx + 1, :self.output_dim]  # [B, output_dim]
            target_bins = self.binner.bin_values(target_vals)  # [B, output_dim]

            # 4) Collect
            all_logits.append(logits_2d.unsqueeze(1))    # => [B,1,output_dim,num_bins]
            all_targets.append(target_bins.unsqueeze(1)) # => [B,1,output_dim]

        # Concatenate along time dimension
        logits = torch.cat(all_logits, dim=1)        # [B, (T-1), output_dim, num_bins]
        target_bins = torch.cat(all_targets, dim=1)  # [B, (T-1), output_dim]
        return logits, target_bins

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, T, input_dim]
        Returns: [B, T, hidden_dim], possibly with added positional encoding.
        """
        B, T, _ = tokens.shape
        x = self.input_embed(tokens)
        if self.pos_embed is not None:
            positions = torch.arange(T, device=tokens.device).unsqueeze(0)
            x += self.pos_embed(positions)  # Broadcasting over B
        return x

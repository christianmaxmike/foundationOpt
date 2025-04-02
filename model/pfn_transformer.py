import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .bar_distribution import OrdinalRegressionHead

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
    def __init__(self, num_bins=32, min_val=0.0, max_val=1.0, train_data=None):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

        # Create bin boundaries (length = num_bins - 1).
        if train_data is None:
            boundaries = torch.linspace(min_val, max_val, steps=num_bins + 1)[1:-1]
        else:
            # Example: Using quantiles from the training data
            boundaries = torch.quantile(
                train_data.flatten(), 
                torch.linspace(0, 1, steps=num_bins + 1)[1:-1].to(train_data.device)
            )

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

    * Allows toggling binning vs. raw regression vs. ordinal regression head (ORH) 
      for X and Y with minimal code changes.
    * Also supports an optional BAR distribution approach (self.use_bar_distribution).
    * Preserves the original AR (autoregressive) and NAR (teacher-forcing) paths.
    
    Also includes a `forward_with_binning(seq, last_x=None)` method for code compatibility 
    with training scripts that specifically call it.
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
        activation: str = "gelu",
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
        train_data: torch.Tensor = None,
        use_binning: bool = False,
        use_orh: bool = False,
    ):
        super().__init__()
        # Basic dimensions
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_dim = x_dim + y_dim

        # Transformer hyperparams
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_bins = num_bins
        self.forecast_steps = forecast_steps
        self.use_autoregression = use_autoregression
        self.nar_inference_flag = nar_inference_flag

        # Flags for how we handle X/Y
        self.use_binning = use_binning
        self.use_orh = use_orh
        self.use_bar_distribution = use_bar_distribution

        # If we're going to bin, figure out min/max
        if self.use_binning or self.use_orh:
            # Decide default bounding ranges if not provided
            if transform_type == "power":
                default_x_min, default_x_max = -4.0, 4.0
                default_y_min, default_y_max = -4.0, 4.0
            else:
                default_x_min, default_x_max = 0.0, 1.0
                default_y_min, default_y_max = 0.0, 1.0

            final_x_min = x_min if x_min is not None else default_x_min
            final_x_max = x_max if x_max is not None else default_x_max
            final_y_min = y_min if y_min is not None else default_y_min
            final_y_max = y_max if y_max is not None else default_y_max

            self.register_buffer("x_min_buf", torch.tensor([final_x_min], dtype=torch.float32))
            self.register_buffer("x_max_buf", torch.tensor([final_x_max], dtype=torch.float32))
            self.register_buffer("y_min_buf", torch.tensor([final_y_min], dtype=torch.float32))
            self.register_buffer("y_max_buf", torch.tensor([final_y_max], dtype=torch.float32))
        else:
            # If not using binning or ORH, no buffers needed
            self.x_min_buf = None
            self.x_max_buf = None
            self.y_min_buf = None
            self.y_max_buf = None

        # Build binning processors if needed
        if self.use_binning and not self.use_orh:
            self.binner_x = BinningProcessor(
                num_bins=num_bins,
                min_val=self.x_min_buf.item() if self.x_min_buf is not None else 0.0,
                max_val=self.x_max_buf.item() if self.x_max_buf is not None else 1.0,
                train_data=None#train_data
            )
            self.binner_y = BinningProcessor(
                num_bins=num_bins,
                min_val=self.y_min_buf.item() if self.y_min_buf is not None else 0.0,
                max_val=self.y_max_buf.item() if self.y_max_buf is not None else 1.0,
                train_data=None#train_data
            )
        else:
            self.binner_x = None
            self.binner_y = None

        # Build ORH if needed
        if self.use_orh:
            # Typically ORH or Binning (but can combine if you want).
            final_x_min = self.x_min_buf.item() if self.x_min_buf is not None else -4.0
            final_x_max = self.x_max_buf.item() if self.x_max_buf is not None else 4.0
            final_y_min = self.y_min_buf.item() if self.y_min_buf is not None else -4.0
            final_y_max = self.y_max_buf.item() if self.y_max_buf is not None else 4.0

            self.orh_x = OrdinalRegressionHead(x_dim, self.num_bins,
                                               x_min=final_x_min, x_max=final_x_max)
            self.orh_y = OrdinalRegressionHead(y_dim, self.num_bins,
                                               x_min=final_y_min, x_max=final_y_max)
        else:
            self.orh_x = None
            self.orh_y = None

        # BAR distribution heads
        if self.use_bar_distribution:
            from model.bar_distribution import BarDistribution
            # Build bar distributions for X and Y using the same bin edges
            x_minval = self.x_min_buf.item() if self.x_min_buf is not None else 0.0
            x_maxval = self.x_max_buf.item() if self.x_max_buf is not None else 1.0
            y_minval = self.y_min_buf.item() if self.y_min_buf is not None else 0.0
            y_maxval = self.y_max_buf.item() if self.y_max_buf is not None else 1.0

            self.bar_distribution_x = BarDistribution(
                borders=torch.linspace(x_minval, x_maxval, steps=num_bins + 1),
                smoothing=bar_dist_smoothing,
                ignore_nan_targets=True
            )
            self.bar_distribution_y = BarDistribution(
                borders=torch.linspace(y_minval, y_maxval, steps=num_bins + 1),
                smoothing=bar_dist_smoothing,
                ignore_nan_targets=True
            )
        else:
            self.bar_distribution_x = None
            self.bar_distribution_y = None

        # Embeddings:
        #  - If not binning/ORH, we do a single linear embed from (x_dim+y_dim) -> hidden_dim.
        #  - If binning or ORH, we'll embed integer tokens with separate nn.Embedding modules for X and Y.
        if not (self.use_binning or self.use_orh):
            self.input_embed = nn.Linear(self.input_dim, hidden_dim)
        else:
            # For binning/ORH we typically embed discrete tokens
            self.input_embedding_x = nn.Embedding(self.num_bins, hidden_dim // 2)
            self.input_embedding_y = nn.Embedding(self.num_bins, hidden_dim // 2)

        # Optional positional embedding
        self.pos_embed = nn.Embedding(num_steps, hidden_dim) if use_positional_encoding else None

        # Build Transformer blocks
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

        # Output projection:
        #  - If binning or ORH, project hidden_dim -> (input_dim * num_bins)
        #  - Otherwise (no binning), project hidden_dim -> input_dim
        out_dim = self.input_dim * self.num_bins if (self.use_binning or self.use_orh) else self.input_dim
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    @property
    def x_min(self) -> float:
        return self.x_min_buf.item() if self.x_min_buf is not None else None

    @property
    def x_max(self) -> float:
        return self.x_max_buf.item() if self.x_max_buf is not None else None

    @property
    def y_min(self) -> float:
        return self.y_min_buf.item() if self.y_min_buf is not None else None

    @property
    def y_max(self) -> float:
        return self.y_max_buf.item() if self.y_max_buf is not None else None

    # ----------------------------------------------------------------
    #  Main entrypoints:
    #   1) forward(...) => returns either (logits, targets) for classification
    #      or (predictions, targets) for regression, depending on flags
    #   2) forward_with_binning(...) => forcibly returns classification-like
    #      outputs (logits, target_bins), used by your training script
    # ----------------------------------------------------------------

    def forward_with_binning(self, seq: torch.Tensor, last_x=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method preserves the original signature that your training code expects:
            logits, target_bins = model.forward_with_binning(...)
        
        Internally, it just calls self.forward(...) and returns the same tuple.
        Make sure you have self.use_binning=True or self.use_orh=True so that 
        the outputs are shaped like classification: 
            logits: [B, T_out, input_dim, num_bins]
            target_bins: [B, T_out, input_dim]
        """
        logits, target_bins = self.forward(seq, last_x)
        return logits, target_bins

    def forward(self, seq: torch.Tensor, best=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified forward method. Chooses AR or NAR path.
        If binning/ORH is enabled, outputs classification-like logits (plus appropriate targets).
        If not, outputs raw continuous predictions (plus continuous targets).
        """
        if self.use_autoregression:
            if self.training or not self.nar_inference_flag:
                return self._forward_ar(seq)
            else:
                return self._forward_nar(seq)
        else:
            return self._forward_nar(seq)

    # ----------------------------------------------------------------
    #  Non-Autoregressive Forward
    # ----------------------------------------------------------------
    def _forward_nar(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Non-autoregressive (teacher-forcing) forward pass. 
        We'll predict the *next* step from the current step, for all steps in parallel.
        """
        B, T, _ = seq.shape
        tokens_in = seq[:, :-1, :]      # all but the last step as input
        next_values = seq[:, 1:, :]     # the actual next-step ground truth

        # 1) Embed the inputs
        x_embed = self._embed_time_series(tokens_in)

        # 2) Pass through Transformer
        for block in self.blocks:
            x_embed = block(x_embed)

        # 3) Project to final dimension
        out_raw = self.out_proj(x_embed)  # shape = [B, T-1, out_dim]
        
        if self.use_binning or self.use_orh:
            # out_dim = input_dim * num_bins => reshape
            logits_all = out_raw.view(B, T-1, self.input_dim, self.num_bins)
            # And produce classification targets
            if self.use_bar_distribution:
                # If using bar distribution, we return raw continuous targets
                # (the BAR distribution code handles the rest).
                targets = next_values
            else:
                targets = self._make_classification_targets(next_values)
            return logits_all, targets
        else:
            # No binning/ORH => we do raw regression. out_raw => [B, T-1, input_dim]
            # Our "targets" are the continuous next_values => [B, T-1, input_dim]
            return out_raw, next_values

    # ----------------------------------------------------------------
    #  Autoregressive Forward
    # ----------------------------------------------------------------
    def _forward_ar(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive forward pass: step-by-step unrolling.
        We'll predict next step at each time t, feeding in [0..t].
        """
        B, T, _ = seq.shape
        all_logits = []
        all_targets = []

        # for each t from 0..(T-2):
        #   - embed partial_seq up to t_idx
        #   - pass through Transformer
        #   - project to out_dim
        #   - collect target from seq[:, t_idx+1]
        for t_idx in range(T - 1):
            partial_seq = seq[:, :t_idx + 1, :]  # [B, t_idx+1, input_dim]

            # 1) embed
            x_embed = self._embed_time_series(partial_seq)

            # 2) pass through blocks
            for block in self.blocks:
                x_embed = block(x_embed)

            # 3) take last hidden vector => project
            last_h = x_embed[:, -1, :]  # [B, hidden_dim]
            logits_step = self.out_proj(last_h)

            # Reshape if classification
            if self.use_binning or self.use_orh:
                logits_step = logits_step.view(B, self.input_dim, self.num_bins)

            # Collect ground-truth next step
            target_vals = seq[:, t_idx + 1, :]

            if (self.use_binning or self.use_orh) and not self.use_bar_distribution:
                # classification bins
                targets = self._make_classification_targets(target_vals).unsqueeze(1)
            elif (self.use_binning or self.use_orh) and self.use_bar_distribution:
                # raw next values
                targets = target_vals.unsqueeze(1)
            else:
                # raw next values for regression
                targets = target_vals.unsqueeze(1)

            all_logits.append(logits_step.unsqueeze(1))  # => [B,1, ...]
            all_targets.append(targets)

        # Combine over time
        logits = torch.cat(all_logits, dim=1)   # [B, T-1,  (input_dim, num_bins) or input_dim]
        targets = torch.cat(all_targets, dim=1) # [B, T-1, input_dim]
        return logits, targets

    # ----------------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------------
    def _embed_time_series(self, seq_values: torch.Tensor) -> torch.Tensor:
        """
        Takes a [B, T, x_dim+y_dim] tensor of *continuous* values and
        returns a [B, T, hidden_dim] embedding, depending on the flags:

          - use_orh => convert via ORH to distributions => use argmax tokens
          - use_binning => bin using BinningProcessor or BAR distribution => then embed
          - otherwise => do a direct linear embedding
        """
        B, T, _ = seq_values.shape

        if self.use_orh:
            # Convert to ordinal tokens
            x_logits = self.orh_x(seq_values[..., :self.x_dim])  # => [B,T,x_dim,num_bins]
            y_logits = self.orh_y(seq_values[..., self.x_dim:])  # => [B,T,y_dim,num_bins]
            x_tokens = x_logits.argmax(dim=-1).squeeze(-1)  # e.g. [B,T] if x_dim=1
            y_tokens = y_logits.argmax(dim=-1).squeeze(-1)  # e.g. [B,T] if y_dim=1

            emb_x = self.input_embedding_x(x_tokens)  # => [B,T, hidden_dim//2]
            emb_y = self.input_embedding_y(y_tokens)  # => [B,T, hidden_dim//2]
            x_embed = torch.cat([emb_x, emb_y], dim=-1)  # => [B,T, hidden_dim]

        elif self.use_binning:
            # Binning approach (either using standard BinningProcessor or BAR distribution)
            if self.bar_distribution_x is not None and self.bar_distribution_y is not None:
                # Map to bucket indices via BAR distribution
                x_tokens = self.bar_distribution_x.map_to_bucket_idx(seq_values[..., :self.x_dim])
                y_tokens = self.bar_distribution_y.map_to_bucket_idx(seq_values[..., self.x_dim:])
            else:
                # Standard binning
                x_tokens = self.binner_x.bin_values(seq_values[..., :self.x_dim]) 
                y_tokens = self.binner_y.bin_values(seq_values[..., self.x_dim:])

            # Squeeze if x_dim=1, y_dim=1
            x_tokens = x_tokens.squeeze(-1)
            y_tokens = y_tokens.squeeze(-1)

            emb_x = self.input_embedding_x(x_tokens)  # => [B,T, hidden_dim//2]
            emb_y = self.input_embedding_y(y_tokens)  # => [B,T, hidden_dim//2]
            x_embed = torch.cat([emb_x, emb_y], dim=-1)  # => [B,T, hidden_dim]

        else:
            # No binning => raw linear embedding
            x_embed = self.input_embed(seq_values)  # => [B,T, hidden_dim]

        # Optionally add positional encoding
        if self.pos_embed is not None:
            positions = torch.arange(T, device=seq_values.device).unsqueeze(0)  # [1,T]
            x_embed = x_embed + self.pos_embed(positions)  # => [B,T,hidden_dim]

        return x_embed

    def _make_classification_targets(self, next_values: torch.Tensor) -> torch.Tensor:
        """
        Takes the raw continuous next_values [B, ..., input_dim] and returns
        discrete bin indices unless using BAR-dist (in which case it can return
        raw).
        """
        if self.use_bar_distribution:
            # Return raw; the BarDistribution modules handle the NLL
            return next_values
        elif self.use_orh:
            # Using ORH => get ordinal labels
            target_bins_x = self.orh_x.map_to_label(next_values[..., :self.x_dim])
            target_bins_y = self.orh_y.map_to_label(next_values[..., self.x_dim:])
            return torch.cat([target_bins_x, target_bins_y], dim=-1)
        else:
            # Standard binning approach
            target_bins_x = self.binner_x.bin_values(next_values[..., :self.x_dim])
            target_bins_y = self.binner_y.bin_values(next_values[..., self.x_dim:])
            return torch.cat([target_bins_x, target_bins_y], dim=-1)

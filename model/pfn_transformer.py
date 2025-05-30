import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

from .bar_distribution import OrdinalRegressionHead


class SparseGPAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_inducing=32, kernel='rbf', **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_inducing = num_inducing
        
        # Inducing points and variational parameters
        self.inducing_points = nn.Parameter(torch.Tensor(num_inducing, embed_dim))
        self.q_mu = nn.Parameter(torch.Tensor(num_heads, num_inducing, self.head_dim))
        self.q_sqrt = nn.Parameter(torch.eye(num_inducing).repeat(num_heads, 1, 1))
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.kernel = self._get_kernel_fn(kernel)
        self.reset_parameters()
        self.device = kwargs['device']

    def reset_parameters(self):
        nn.init.normal_(self.inducing_points)
        nn.init.normal_(self.q_mu)
        #nn.init.eye_(self.q_sqrt)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _get_kernel_fn(self, name):
        if name == 'rbf':
            return lambda x,y: torch.exp(-0.5 * (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1))
        raise ValueError(f"Unknown kernel: {name}")

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
        
        # Project inputs (batch_size, seq_len, embed_dim)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Compute kernel matrices (batch_size, seq_len, num_inducing)
        K_mm = self.kernel(self.inducing_points, self.inducing_points).to(self.device) + 1e-6 * torch.eye(self.num_inducing).to(self.device)
        K_nm = self.kernel(k, self.inducing_points)
        
        # Sparse GP posterior approximation (per head)
        L = torch.linalg.cholesky(K_mm)  # (num_inducing, num_inducing)
        A = torch.cholesky_solve(K_nm.transpose(-1,-2), L)  # (batch_size, num_inducing, seq_len)

        # Reshape components for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, nh, sl, hd)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, nh, sl, hd)
        
        # Compute posterior mean (batch_size, num_heads, seq_len, head_dim)
        posterior_mean = torch.einsum('bim,nhd->bihd', A, self.q_mu)
        
        # Scaled attention (batch_size, num_heads, seq_len, seq_len)
        attn = torch.einsum('bnhd,bihd->bnhi', q, posterior_mean) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum of values (batch_size, num_heads, seq_len, head_dim)
        out = torch.einsum('bnhi,bnhd->bnhd', attn, v)
        
        print (out.shape)
        # Concatenate heads and project output
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out), attn


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
            ## Replace linear binning with quantile-based binning
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
        use_sgpa: bool = False,
        num_inducing: int = 32,
        device=None,
        dtype=None,
        **kwargs
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        #factory_kwargs = {'device': kwargs.get('device'), 'dtype': kwargs.get('dtype')}
        # Multihead self-attention module

        self.self_attn = SparseGPAttention(
            hidden_dim,
            num_heads,
            num_inducing=num_inducing, # has to be number of T-1
            **factory_kwargs
        ) if use_sgpa else nn.MultiheadAttention(
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
        loss_type = "bar",
        device=None,
        ctx_length = 0
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

        # Decide default bounding ranges if not provided
        if transform_type == "power":
            default_x_min, default_x_max = -4.0, 4.0
            default_y_min, default_y_max = -4.0, 4.0
        else:
            default_x_min, default_x_max = 0.0, 1.0
            default_y_min, default_y_max = 0.0, 1.0

        # Final chosen bounds
        final_x_min = x_min if x_min is not None else default_x_min
        final_x_max = x_max if x_max is not None else default_x_max
        final_y_min = y_min if y_min is not None else default_y_min
        final_y_max = y_max if y_max is not None else default_y_max

        # Register these as buffers so that they are saved/restored with state_dict.
        # We'll store them as 1-element tensors and later call .item() where needed.
        self.register_buffer("x_min_buf", torch.tensor([final_x_min], dtype=torch.float32))
        self.register_buffer("x_max_buf", torch.tensor([final_x_max], dtype=torch.float32))
        self.register_buffer("y_min_buf", torch.tensor([final_y_min], dtype=torch.float32))
        self.register_buffer("y_max_buf", torch.tensor([final_y_max], dtype=torch.float32))

        # store loss type for accessibility in forward 
        self.loss_type = loss_type

        # Build BinningProcessors for X and Y using the registered buffers.
        if loss_type == 'bar':
            self.binner_x = BinningProcessor(
                num_bins=num_bins,
                min_val=self.x_min_buf.item(),
                max_val=self.x_max_buf.item(),
                train_data=None # train_data
            )
            self.binner_y = BinningProcessor(
                num_bins=num_bins,
                min_val=self.y_min_buf.item(),
                max_val=self.y_max_buf.item(),
                train_data=None # train_data
            )
            self.input_embedding_x = nn.Embedding(self.num_bins, hidden_dim//2)
            self.input_embedding_y = nn.Embedding(self.num_bins, hidden_dim//2)
            self.orh_x = OrdinalRegressionHead(x_dim, self.num_bins, x_min=final_x_min, x_max=final_x_max)
            self.orh_y = OrdinalRegressionHead(y_dim, self.num_bins, x_min=final_y_min, x_max=final_y_max)
            # Final projection from hidden_dim -> (input_dim * num_bins)
            self.out_proj = nn.Linear(hidden_dim, self.input_dim * num_bins)

        elif loss_type in ["mse", "rank"]:
            # Input embedding (transform from input_dim -> hidden_dim)
            self.input_embed = nn.Linear(self.input_dim, hidden_dim)
            # Final projection from hidden_dim -> (input_dim * num_bins)
            if ctx_length == 0: 
                self.out_proj = nn.Linear(hidden_dim, self.input_dim)
            else:
                self.out_proj = nn.Linear(hidden_dim*ctx_length, self.input_dim)
        elif loss_type == "quantile":
            # Input embedding (transform from input_dim -> hidden_dim)
            self.input_embed = nn.Linear(self.input_dim, hidden_dim)
            # Final projection from hidden_dim -> (input_dim * num_bins)
            if ctx_length is None:
                self.out_proj = nn.Linear(hidden_dim, self.input_dim)
            else:
                self.out_proj = nn.Linear(hidden_dim*ctx_length, self.input_dim)

        # Optional positional embedding
        self.pos_embed = nn.Embedding(num_steps, hidden_dim) if use_positional_encoding else None

        # A stack of TransformerBlocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim if ctx_length==0 else hidden_dim*ctx_length,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm,
                device=device,
            )
            for _ in range(num_layers)
        ])

        self.use_bar_distribution = use_bar_distribution
        if self.use_bar_distribution:
            from model.bar_distribution import BarDistribution
            # Build bar distributions for X and Y using the same bin edges
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
        else:
            self.bar_distribution_x = None
            self.bar_distribution_y = None

    @property
    def x_min(self) -> float:
        """
        Access x_min as a float.
        """
        return self.x_min_buf.item()

    @property
    def x_max(self) -> float:
        """
        Access x_max as a float.
        """
        return self.x_max_buf.item()

    @property
    def y_min(self) -> float:
        """
        Access y_min as a float.
        """
        return self.y_min_buf.item()

    @property
    def y_max(self) -> float:
        """
        Access y_max as a float.
        """
        return self.y_max_buf.item()

    def forward_with_binning(self, seq: torch.Tensor, best=None, ctx_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq: [B, T, (x_dim + y_dim)] — input sequence of X and Y values.
        Returns:
          (logits, targets):
            logits: [B, T_out, input_dim, num_bins]
            targets: either raw continuous targets [B, T_out, input_dim] (if use_bar_distribution)
                     or integer bin indices [B, T_out, input_dim] (if not).
        """
        if self.use_autoregression:
            if self.training or not self.nar_inference_flag:
                return self._forward_ar(seq)
            else:
                return self._forward_nar(seq, best)
        else:
            # return self._forward_nar(seq, best)
            if ctx_length is None or ctx_length==0:
                return self._forward_nar(seq, best)
            else:
                return self._forward_nar_w_ctx(seq, best, ctx_length=ctx_length)
        

    def _forward_nar_w_ctx(self, seq: torch.Tensor, last_x=None, ctx_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Non-autoregressive forward pass with sliding window context.
        For i-th inference step, uses i-ctx_length steps as context.
        Starts at ctx_length+1 if context is provided.
        """
        B, T, _ = seq.shape # B X T x (x_dim+y_dim)
        
        # If context length is not provided, use full sequence (original behavior)
        if ctx_length is None or ctx_length >= T-1:
            tokens = seq[:, :-1, :]  # All but last step as input
            next_values = seq[:, 1:, :]  # shape [B, T-1, input_dim]
        else:
            # Create sliding window inputs and targets
            tokens_list = []
            targets_list = []
            
            # Start from ctx_length+1 to have enough context
            for i in range(ctx_length, T-1): # T-1
                # Get context window (from i-ctx_length to i)
                context_window = seq[:, i-ctx_length:i, :]  # B x ctx x (x_dim+y_dim)
                print (f"{i-ctx_length}:{i}")
                tokens_list.append(context_window)
                
                # Target is the next step after context window
                target = seq[:, i+1:i+2, :]  # B x 1 x (x_dim+y_dim)
                targets_list.append(target)
            
            # Stack along sequence dimension
            tokens = torch.cat(tokens_list, dim=1)  # [B, (T-1-ctx_length)*ctx_length, D]
            next_values = torch.cat(targets_list, dim=1)  # [B, T-1-ctx_length, D]

        if self.loss_type == 'bar':        
            binned_tokens = torch.cat([self.bar_distribution_x.map_to_bucket_idx(tokens[..., :self.x_dim]),
                                    self.bar_distribution_y.map_to_bucket_idx(tokens[..., self.x_dim:])], dim=-1)
            x_embed = self.embed_tokens(binned_tokens)
        else: # self.loss_type in ['mse', "rank", "quantile"]:
            x_embed = self.embed_tokens(tokens, T, ctx_length)

        x_embed = x_embed.reshape(B,max(1, (T-1)-ctx_length),-1)
        # Feed to transformer layers
        for block in self.blocks:
            x_embed = block(x_embed)

        # Project to (input_dim * num_bins) then reshape
        logits_all = self.out_proj(x_embed)
        
        if self.loss_type == 'bar':
            logits_all = logits_all.view(B, -1, self.input_dim, self.num_bins)

        # Get targets based on distribution type
        if self.use_bar_distribution:
            targets = next_values  # raw
        else:
            if self.loss_type == "bar":
                target_bins_x = self.binner_x.bin_values(next_values[..., :self.x_dim])            
                target_bins_y = self.binner_y.bin_values(next_values[..., self.x_dim:-1])
                targets = torch.cat([target_bins_x, target_bins_y], dim=-1)
            elif self.loss_type in ["mse", "quantile", "rank"]:
                targets = next_values

        return logits_all, targets

    def _forward_nar(self, seq: torch.Tensor, last_x=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Non-autoregressive forward pass. We do teacher forcing on T-1 steps to predict T-1 next steps.
        """
        B, T, _ = seq.shape
    
        tokens = seq[:, :-1, :]  # All but last step as input  # Batch x seq-1 x (dimx+dimy)

        # Usage of ordinalRegressionHead
        # x_bucket = self.orh_x(tokens[..., :self.x_dim])
        # y_bucket = self.orh_y(tokens[..., self.x_dim:])
        # binned_tokens = torch.cat([x_bucket.argmax(dim=-1, keepdims=True), 
        #                            y_bucket.argmax(dim=-1, keepdims=True)], dim=-1)
        
        if self.loss_type == 'bar':        
            binned_tokens = torch.cat([self.bar_distribution_x.map_to_bucket_idx(tokens[..., :self.x_dim]),    # bar dist yields: B x T-1x1
                                    self.bar_distribution_y.map_to_bucket_idx(tokens[..., self.x_dim:])], dim=-1) # BxT-1xdimx+dimy
            if False:
                t = tokens[..., self.x_dim:]
                argmin_idx = torch.argmin(t, dim=1)
                min_vals = t[torch.arange(t.shape[0]),argmin_idx.squeeze()]
                bucket_min_vals = self.bar_distribution_x.map_to_bucket_idx(min_vals)
                bt = binned_tokens
                #print ("Before masking:", binned_tokens.shape)
                bt = bt[~((bucket_min_vals==0) | (bucket_min_vals==self.num_bins-1)).flatten()]
                seq = seq[~((bucket_min_vals==0) | (bucket_min_vals==self.num_bins-1)).flatten()]
                #bt = bt[~(bucket_min_vals==self.num_bins-1).flatten()]
                #seq = seq[~(bucket_min_vals==self.num_bins-1).flatten()]
                B = bt.shape[0]
                binned_tokens=bt
                #print("after masking;", binned_tokens.shape)
                #bt = binned_tokens
                #value_cnts = (bt[:,:,:self.x_dim]==0).sum(dim=1)
                #mask = value_cnts <=1
                #bt = bt[mask.flatten()]
                #seq = seq[mask.flatten()]
                #value_cnts = (bt[:,:,:self.x_dim]==self.num_bins-1).sum(dim=1)
                #mask = value_cnts <=1
                #bt = bt[mask.flatten()]
                #seq = seq[mask.flatten()]
                #B = bt.shape[0]
                #binned_tokens = bt
            x_embed = self.embed_tokens(binned_tokens, T)
        elif self.loss_type in ["rank", "mse"]:
            x_embed = self.embed_tokens(tokens, T) # tokens: B x T-1 x xdim + ydim; x_embed: B x T-1, hiddendim

        # Feed to transformer layers
        for block in self.blocks:
            x_embed = block(x_embed)

        # Project to (input_dim * num_bins) then reshape
        logits_all = self.out_proj(x_embed)  # [B, T-1, input_dim * num_bins]
        
        if self.loss_type == 'bar':
            logits_all = logits_all.view(B, -1, self.input_dim, self.num_bins)

        # Next-step ground-truth for each feature
        # print (seq.shape) # batch x seqLenght x (xdim + ydim)
        next_values = seq[:, 1:, :]  # shape [B, T-1, input_dim]

        # If using bar distribution, we return the raw continuous targets
        # otherwise, we bin them for cross-entropy
        if self.use_bar_distribution:
            targets = next_values  # raw
        else:
            if self.loss_type=="bar":
                target_bins_x = self.binner_x.bin_values(next_values[..., :self.x_dim])            
                target_bins_y = self.binner_y.bin_values(next_values[..., self.x_dim:-1])
                targets = torch.cat([target_bins_x, target_bins_y], dim=-1)
            elif self.loss_type in ["rank", "mse"]:
                targets = next_values

        return logits_all, targets #, target_bins_y_best

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

    def embed_tokens(self, tokens: torch.Tensor, T:int, ctx_length:int = None) -> torch.Tensor:
        """
        tokens: [B, T, input_dim].
        1) Project to hidden_dim.
        2) Optionally add positional embeddings.
        Returns: [B, T, hidden_dim].
        """
        B, _, d = tokens.shape # B x T x (x_dim + y_dim)
        # tokens_binned = self.binner_x.bin_values(tokens[..., :self.x_dim]) # B x T x xdim
        # x = self.input_embed(tokens)
        if self.loss_type == 'bar':
            x = self.input_embedding_x(tokens[:, :, 0]) 
            y = self.input_embedding_y(tokens[:, :, 1])
            x = torch.cat([x,y], dim=-1)
        else:
            x = self.input_embed(tokens)
        
        if self.pos_embed is not None:
            if ctx_length is not None:
                if T == ctx_length + 1:
                    T+=1
                positions = torch.stack([torch.arange(T)[i-ctx_length:i] for i in range(ctx_length,T-1,1)]).flatten().to(tokens.device)
                x += self.pos_embed(positions)
            else:
                positions = torch.arange(T-1, device=tokens.device).unsqueeze(0)
                x += self.pos_embed(positions)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

class BinningProcessor(nn.Module):
    """
    Dynamically create bins (uniform or quantile) for x,y each batch or periodically.
    For simplicity, we show uniform binning in [0, 1].
    """
    def __init__(self, num_bins=32, min_val=0.0, max_val=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.bin_width = (max_val - min_val) / num_bins

    def bin_values(self, values: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous values in [min_val, max_val] to bucket indices [0..num_bins-1].
        values shape: [B, T] or [B, T, 2].
        """
        # clamp for safety
        clipped = torch.clamp(values, self.min_val, self.max_val)
        indices = ((clipped - self.min_val) / self.bin_width).long()
        indices = torch.clamp(indices, 0, self.num_bins-1)
        return indices

    def unbin_values(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert bucket indices back to continuous by using midpoints.
        indices shape: [B, T].
        """
        centers = self.min_val + (indices.float() + 0.5) * self.bin_width
        return centers


class TransformerBlock(nn.Module):
    """Basic transformer block with multi-head attention + feed-forward."""
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask=None):
        # self-attention
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        # feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class PFNTransformer(nn.Module):
    """
    A basic transformer model that:
      - Embeds the (x, y) input tokens
      - Optionally runs autoregressively
      - Outputs bin probabilities for next-step x,y or multiple steps
    """
    def __init__(
        self,
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        num_bins=32,
        forecast_steps=1,
        num_steps=50,
        use_positional_encoding=False,
        use_autoregression=False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_bins = num_bins
        self.forecast_steps = forecast_steps
        self.use_autoregression = use_autoregression

        # Binning
        self.binner = BinningProcessor(num_bins=num_bins)

        # Input embedding
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        # Positional embedding
        if use_positional_encoding:
            self.pos_embed = nn.Embedding(num_steps, hidden_dim)
        else:
            self.pos_embed = None
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # Final classification head for x,y => we produce 2 * num_bins logits per token
        # But let's produce them in shape [B, T, 2, num_bins]
        # i.e. transform hidden_dim -> 2 * num_bins
        self.out_proj = nn.Linear(hidden_dim, input_dim * num_bins)

    def forward_with_binning(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq: [B, T, 2] in [0,1].
        Returns:
          logits: [B, T_out, 2, num_bins]
          target_bins: [B, T_out, 2]
        Where T_out depends on forecast_steps.
        """
        B, T, _ = seq.shape

        # We'll create an input embedding for all but last step (teacher forcing),
        # then predict next steps. For simplicity, let's do 1-step next or AR if flagged.
        if not self.use_autoregression:
            # teacher forcing: We embed the entire sequence
            # Then we predict the next step at each time index => T_out = T - 1, typically
            # Or we do forecasting up to forecast_steps in a single shot. 
            # For demonstration, let's keep it simpler: next-step at each position => T_out = T-1
            # Then we handle the target bins for that next step
            tokens = seq[:, :-1, :]  # [B, T-1, 2]
            x_embed = self.embed_tokens(tokens)  # [B, T-1, hidden_dim]
            # pass through transformer
            for block in self.blocks:
                x_embed = block(x_embed)
            # project to logits
            # shape => [B, T-1, 2*num_bins]
            logits_2d = self.out_proj(x_embed)
            # reshape => [B, T-1, 2, num_bins]
            logits = logits_2d.view(B, T-1, 2, self.num_bins)

            # target bins => discretize the next step
            # next step is seq[:, 1:, :]
            next_values = seq[:, 1:, :]  # [B, T-1, 2]
            # convert to bins
            target_bins = self.binner.bin_values(next_values)

            return logits, target_bins

        else:
            # Autoregressive mode:
            # We'll iterate step-by-step, always feeding the last predicted x,y as input for next step
            # This can be slow if T is large => we can optimize with caching, but let's keep it straightforward.

            all_logits = []
            all_targets = []
            current_input = seq[:, 0, :]  # [B, 2] first token
            # We'll store hidden states in a buffer if we want to do fancy partial forward, 
            # but let's do naive re-run each time for clarity.

            for t_idx in range(T - 1):
                # embed single token or the entire so-far sequence
                # Let's embed up to t_idx
                partial_seq = seq[:, :t_idx+1, :].clone()
                # but the last element might be model predictions if we had predicted it
                # for the previous step, see bigger architectural details below if you want 
                x_embed = self.embed_tokens(partial_seq)
                for block in self.blocks:
                    x_embed = block(x_embed)

                # last hidden state => next step prediction
                last_h = x_embed[:, -1, :]  # shape [B, hidden_dim]
                logits_2d = self.out_proj(last_h)  # [B, 2*num_bins]
                logits_2d = logits_2d.view(B, 2, self.num_bins)
                # store for training
                all_logits.append(logits_2d.unsqueeze(1))  # => [B,1,2,num_bins]

                # true next step is seq[:, t_idx+1, :]
                target_bins = self.binner.bin_values(seq[:, t_idx+1, :])  # [B,2]
                all_targets.append(target_bins.unsqueeze(1))  # => [B,1,2]

                # if we are actually rolling out predictions, we'd pick the bin with max prob 
                # or sample from the distribution and convert to continuous x,y
                # then append to partial_seq. But for training, let's keep it simple 
                # and use teacher forcing or a mixture.
                # If fully autoregressive, we replace seq[:, t_idx+1,:] with predicted bin
                # or combine. We'll keep it teacher forcing for now.

            # concat all
            logits = torch.cat(all_logits, dim=1)  # [B, T-1, 2, num_bins]
            target_bins = torch.cat(all_targets, dim=1)  # [B, T-1, 2]

            return logits, target_bins

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, T, 2], each in [0,1]
        returns embedded shape [B, T, hidden_dim]
        """
        B, T, _ = tokens.shape
        # flatten 2 features => embed
        x = self.input_embed(tokens)  # [B, T, hidden_dim]

        # Add a learned positional embedding
        if self.pos_embed is not None:
            positions = torch.arange(T, device=tokens.device).unsqueeze(0)
            pos_emb = self.pos_embed(positions)
            x = x + pos_emb

        return x

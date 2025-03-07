import torch

import torch.nn as nn

from torch.nn import functional as F

from xformers.components.attention import LocalAttention
from torchtune.modules import RotaryPositionalEmbeddings

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MultiAttentionHead(nn.Module):

    def __init__(
        self,
        embedding_size,
        num_heads,
        dropout=0.2,
        sliding_attention=False,
        casual=True,
    ):
        super(MultiAttentionHead, self).__init__()

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embedding_size, 3 * embedding_size, bias=False)

        # output projection
        self.c_proj = nn.Linear(embedding_size, embedding_size, bias=False)
        self.casual = casual
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.embedding_size = embedding_size
        self.n_head = num_heads
        self.query_embedding = RotaryPositionalEmbeddings(int(embedding_size / num_heads))
        self.key_embedding = RotaryPositionalEmbeddings(int(embedding_size / num_heads))

        self.sliding_attention = sliding_attention

        if sliding_attention:
            self.sliding_attn = LocalAttention(
                window_size=99,
                causal=self.casual,
                dropout=dropout,
        )

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer('tril_mask', torch.tril(torch.ones(embedding_size, embedding_size)))

    def forward(self, x):

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        q, k, v = self.c_attn(x).split(self.embedding_size, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)

        #q = self.query_embedding(q)
        #k = self.key_embedding(k)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)


        #k = k.reshape(B * self.n_head, T, C // self.n_head)

        if self.sliding_attention:
            k = k.reshape(B * self.n_head, T, C // self.n_head)
            q = q.reshape(B * self.n_head, T, C // self.n_head)
            v = v.reshape(B * self.n_head, T, C // self.n_head)

        if self.sliding_attention:
            output = self.sliding_attn(q, k, v)
        else:
            if self.flash:
                output = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout_rate if self.training else 0,
                    is_causal=self.casual,
                )

            else:
                out = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** -0.5))
                #att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                #out = F.softmax(out, dim=-1)
                out = F.sigmoid(out)
                output = self.attn_dropout(out)

        if self.sliding_attention:
            output = output.view(B, self.n_head, T, C // self.n_head).transpose(1, 2).reshape(B, T, C)
        else:
            output = output.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.resid_dropout(self.c_proj(output))
        return y


class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout=0.2):
        super(FeedForward, self).__init__()

        self.w1 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w2 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w3 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))
        #return self.dropout(self.w2(nn.functional.sigmoid(self.w1(x)) * self.w3(x)))


class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout):
        super(AttentionBlock, self).__init__()

        self.multi_head_attention = MultiAttentionHead(embedding_size, num_heads, dropout)
        self.feed_forward = FeedForward(embedding_size, dropout)
        self.norm1 = RMSNorm(embedding_size)
        self.norm2 = RMSNorm(embedding_size)

    def forward(self, x):

        x = x + self.multi_head_attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x


class TabFound(nn.Module):

    def __init__(
        self,
        input_features: int,
        mean_embedding_value: int = 64,
        nr_blocks:int = 3,
        nr_heads:int = 4,
        dropout: float = 0.2,
        nr_hyperparameters: int = 32,
        **kwargs,
    ):

        super(TabFound, self).__init__()

        self.input_features = input_features
        self.mean_embedding_value = mean_embedding_value
        self.input_layer = nn.Linear(input_features, self.mean_embedding_value)
        self.norm_f = RMSNorm(self.mean_embedding_value)
        self.ln_head = nn.Linear(self.mean_embedding_value, nr_hyperparameters)
        self.kwargs = kwargs

        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(self.mean_embedding_value, nr_heads, dropout)
                for _ in range(nr_blocks)
            ]
        )

    def forward(self, x):

        x = self.input_layer(x)
        for att_block in self.attention_blocks:
            x = att_block(x)

        x = self.norm_f(x)
        matrix = self.ln_head(x)

        return matrix

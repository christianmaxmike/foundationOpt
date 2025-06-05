import torch
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MultiAttentionHead(nn.Module):
    def __init__(self, feature_size, num_heads, dropout=0.0):
        super(MultiAttentionHead, self).__init__()

        self.new_size = 8
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(feature_size, 3 * feature_size, bias=False)
        # output projection
        self.c_proj = nn.Linear(feature_size, feature_size, bias=False)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.feature_size = feature_size
        self.n_head = num_heads

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, is_causal=False, attn_mask=None):
        B, T, E = x.size()
        x = x.float()
        q, k, v = self.c_attn(x).split(self.feature_size, dim=2)
        head_dim = self.feature_size // self.n_head

        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)

        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                  dropout_p=self.dropout_rate if self.training else 0,
                                                                  is_causal=is_causal)

        output = output.transpose(1, 2).contiguous().view(B, T, self.feature_size)
        y = self.resid_dropout(self.c_proj(output))
        return y


class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout=0.0):
        super(FeedForward, self).__init__()

        self.w1 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w2 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w3 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))


class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout):
        super(AttentionBlock, self).__init__()

        self.multi_head_attention = MultiAttentionHead(embedding_size, num_heads, dropout)
        self.feed_forward = FeedForward(embedding_size, dropout)
        self.norm1 = RMSNorm(embedding_size)
        self.norm2 = RMSNorm(embedding_size)

    def forward(self, x, is_causal=False, attn_mask=None):
        B, C, T, E = x.size()
        x = self.norm1(x)
        x_flatten = x.flatten(start_dim=0, end_dim=1)
        attention = self.multi_head_attention(x_flatten, is_causal=is_causal, attn_mask=attn_mask)
        attention = attention.reshape(B, C, T, E)
        x = x + attention

        x = x.float()
        x = x + self.feed_forward(self.norm2(x))

        return x


class PFTSN(nn.Module):
    def __init__(self, input_dim,output_dim, emb_dim, nr_blocks=4, nr_heads=4, dropout=0.0):
        super(PFTSN, self).__init__()
        self.new_emb_size = emb_dim

        self.up_proj_1 = nn.Linear(1, emb_dim)
        self.up_proj_2 = nn.Linear(1, emb_dim)
        self.up_proj_3 = nn.Linear(1, emb_dim)


        self.row_attention_blocks = nn.Sequential(
            *[AttentionBlock(self.new_emb_size, nr_heads, dropout) for _ in range(nr_blocks)])
        self.col_attention_blocks = nn.Sequential(
            *[AttentionBlock(self.new_emb_size, nr_heads, dropout) for _ in range(nr_blocks)])

        self.current_row_attention_blocks = nn.Sequential(
            *[AttentionBlock(self.new_emb_size, nr_heads, dropout) for _ in range(nr_blocks)])

        self.norm_f = RMSNorm(self.new_emb_size)

        self.ln_head = nn.Sequential(nn.Linear(self.new_emb_size*3, output_dim))

    def forward(self, context):
        proj_1 = self.up_proj_1(context[:, :, 0:1, :])
        proj_2 = self.up_proj_2(context[:, :, 1:2, :])
        proj_3 = self.up_proj_3(context[:, :, 2:3, :])
        context = torch.cat([proj_1, proj_2, proj_3], dim=2)
        for row_att_block, col_att_block in zip(self.row_attention_blocks, self.col_attention_blocks):
            context = row_att_block(context)
            context = context.transpose(1, 2)
            context = col_att_block(context)
            context = context.transpose(1, 2)
        context = self.norm_f(context)

        context= context.mean(dim=(1))
        context= context.flatten(1)


        output = self.ln_head(context)
        return output

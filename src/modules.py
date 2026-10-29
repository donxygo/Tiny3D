import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 6, include_input: bool = False):
        super().__init__()
        self.include_input = include_input
        freqs = (2.0 ** torch.arange(0, num_frequencies, dtype=torch.float32)) * torch.pi 
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional encoding for 3D coordinates.
        args:
            x: (batch_size, n_points, 3) 
        returns:
            pos_embedding: (batch_size, n_points, 6 * num_frequencies)
                           [sin(2⁰πx), cos(2⁰πx), sin(2¹πx), cos(2¹πx), ...]
            or if include_input=True:
                           (batch_size, n_points, 3 + 6 * num_frequencies)
                           [x, sin(2⁰πx), cos(2⁰πx), ...]
        """        
        x_proj = x.unsqueeze(-1) * self.freqs # (b, n, 3, 1) * (n_freq, ) -> (b, n, 3, n_freq)
        sin_emb =  torch.sin(x_proj) 
        cos_emb =  torch.cos(x_proj) 
        emb = torch.cat([sin_emb, cos_emb], dim=-1).flatten(-2) # (b, n, 6*n_freq)
        if self.include_input:
            emb = torch.cat([x, emb], dim=-1) # (b, n, 6*n_freq + 3) 
        return emb


class PosEmb(nn.Module):
    def __init__(self, 
                 n_freqs: int = 6, 
                 include_input: bool = False, 
                 vec_dim: int = 256, 
                 n_layers: int = 3):
        super().__init__()
        self.FE = FourierEncoding(n_freqs, include_input)
        input_dim = (n_freqs * 6 + 3) if include_input else (n_freqs * 6)
        self.proj = MLP([input_dim] + [vec_dim] * n_layers)
        self.vec_dim = vec_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute position embedding given input x.
        args:
            x: (batch_size, n_tokens, 3)
        returns:
            x: (batch_size, n_tokens, vec_dim)
        """
        x = self.FE(x)
        x = self.proj(x)
        return x


def scaled_dot_product_attention_(q, k, v):
    """
    Compute the scaled product attention given the query, key, and value.
    args:
        q: query, (B, N_q, D_qk)
        k: key, (B, N_kv, D_qk)
        v: value, (B, N_kv, D_v)
    return:
        (B, N_q, D_v)
    """
    D_qk = q.shape[-1]
    attention = torch.matmul(q,k.transpose(-1, -2))/math.sqrt(D_qk) # (B, N_q, N_kv)
    attention = F.softmax(attention, dim=-1)
    values = torch.matmul(attention, v) # (B, N_q, D_v)
    return values 


def scaled_dot_product_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, n_heads: int = 8):
        super().__init__()
        assert embedding_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.W_qkv = nn.Linear(input_dim, 3 * embedding_dim, bias=False)
        self.W_o = nn.Linear(embedding_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute self-attention given input x.
        args:
            x: (batch_size, n_tokens, input_dim)
        returns:
            values: (batch_size, n_tokens, input_dim)
        """
        batch_size, n_tokens, _ = x.shape

        qkv = self.W_qkv(x) # (batch_size, n_tokens, 3 * embedding_dim) 
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 3, 0, 1, 4) # (3, n_heads, batch_size, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (n_heads, batch_size, n_tokens, head_dim)

        values = scaled_dot_product_attention(q, k, v) # (n_heads, batch_size, n_tokens, head_dim)
        values = values.permute(1, 2, 0, 3) # (batch_size, n_tokens, n_heads, head_dim)
        values = values.reshape(batch_size, n_tokens, -1) # (batch_size, n_tokens, embedding_dim)
        values = self.W_o(values) # (batch_size, n_tokens, input_dim)

        return values
        

class MultiheadCrossAttention(nn.Module):
    def __init__(self, dim_q: int, dim_k: int, embedding_dim: int, n_heads: int = 8):
        super().__init__()
        assert embedding_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.W_q = nn.Linear(dim_q, embedding_dim, bias=False)
        self.W_kv = nn.Linear(dim_k, 2 * embedding_dim, bias=False)
        self.W_o = nn.Linear(embedding_dim, dim_q, bias=False)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention given input x_q and x_k.
        args:
            x_q: (batch_size, n_q, dim_q)
            x_k: (batch_size, n_k, dim_k)
        returns:
            values: (batch_size, n_q, dim_q)
        """
        batch_size, n_q, _ = x_q.shape
        _, n_k, _ = x_k.shape

        q = self.W_q(x_q) # (batch_size, n_q, embedding_dim)
        q = q.reshape(batch_size, n_q, self.n_heads, self.head_dim)
        q = q.permute(2, 0, 1, 3) # (n_heads, batch_size, n_q, head_dim)
        kv = self.W_kv(x_k) # (batch_size, n_k, 2 * embedding_dim)
        kv = kv.reshape(batch_size, n_k, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 3, 0, 1, 4) # (2, n_heads, batch_size, n_k, head_dim)
        k, v = kv[0], kv[1] # (n_heads, batch_size, n_k, head_dim)

        values = scaled_dot_product_attention(q, k, v) # (n_heads, batch_size, n_q, head_dim)
        values = values.permute(1, 2, 0, 3) # (batch_size, n_q, n_heads, head_dim)
        values = values.reshape(batch_size, n_q, -1) # (batch_size, n_q, embedding_dim)
        values = self.W_o(values) # (batch_size, n_q, dim_q)

        return values


class LayerNorm(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute layer normalization given input x.
        args:
            x: (batch_size, n_tokens, input_dim)
        returns:
            x: (batch_size, n_tokens, input_dim)
        """
        x = (x - x.mean(dim = -1, keepdim=True)) / (torch.sqrt(x.var(dim = -1, keepdim=True, unbiased=False)+self.eps))
        x = self.gamma * x + self.beta
        return x


class MLP(nn.Module):
    def __init__(self, dim_list: list[int], activation: type[nn.Module] = nn.GELU):
        super().__init__()
        layers = []
        for i in range(len(dim_list) -2):
            layers.append(nn.Linear(dim_list[i], dim_list[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(dim_list[-2], dim_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class CrossAttentionTransformerBlock(nn.Module):
    def __init__(self, dim: int, embedding_dim: int, n_heads: int = 8, expansion_ratio: int = 4):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.cross_attn = MultiheadCrossAttention(dim, dim, embedding_dim, n_heads)
        self.ln2 = LayerNorm(dim)
        self.ffn = MLP([dim, dim * expansion_ratio, dim])
        
    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention transformer block given input x_q and x_k.
        args:
            x_q: (batch_size, n_q, dim_q)
            x_k: (batch_size, n_k, dim_k)
        returns:
            x: (batch_size, n_q, dim_q)
        """
        x_q = x_q + self.cross_attn(self.ln1(x_q), self.ln1(x_k))
        x_q = x_q + self.ffn(self.ln2(x_q))
        return x_q
    

class SelfAttentionTransformerBlock(nn.Module):
    def __init__(self, dim: int, embedding_dim: int, n_heads: int = 8, expansion_ratio: int = 4):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.self_attn = MultiheadSelfAttention(dim, embedding_dim, n_heads)
        self.ln2 = LayerNorm(dim)
        self.ffn = MLP([dim, dim * expansion_ratio, dim])       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute self-attention transformer block given input x.
        args:
            x: (batch_size, n_tokens, dim)
        returns:
            x: (batch_size, n_tokens, dim)
        """
        x = x + self.self_attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


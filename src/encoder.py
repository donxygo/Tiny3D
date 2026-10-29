import torch
import torch.nn as nn
from typing import Optional
from config import ModelConfig as M
from modules import PosEmb, LayerNorm, CrossAttentionTransformerBlock, SelfAttentionTransformerBlock


class Shape2Vecset(nn.Module):
    def __init__(self,
                 pos_emb: Optional[PosEmb] = None,
                 self_attn_blocks: int = M.ENC_SA_BLOCKS,
                 self_attn_emb_dim: int = M.ENC_SA_EMB,
                 self_attn_heads: int = M.ENC_SA_HEADS,
                 self_attn_expansion_ratio: int = M.ENC_SA_EXPANSION,
                 cross_attn_blocks: int = M.ENC_CA_BLOCKS,
                 cross_attn_emb_dim: int = M.ENC_CA_EMB,
                 cross_attn_heads: int = M.ENC_CA_HEADS, 
                 cross_attn_expansion_ratio: int = M.ENC_CA_EXPANSION
    ):
        super().__init__()
        if pos_emb is None:
            pos_emb = PosEmb(
                 n_freqs = M.POS_EMB_FREQS, 
                 include_input = M.POS_EMB_INCLUDE_INPUT, 
                 vec_dim = M.DIM_VEC, 
                 n_layers = M.POS_EMB_LAYERS
            )
        self.pos_emb = pos_emb
        self.vec_dim = pos_emb.vec_dim
        self.Pu_blocks = nn.ModuleList([
            CrossAttentionTransformerBlock(pos_emb.vec_dim, cross_attn_emb_dim, cross_attn_heads, cross_attn_expansion_ratio)
            for _ in range(cross_attn_blocks)
        ])  
        self.Ps_blocks = nn.ModuleList([
            CrossAttentionTransformerBlock(pos_emb.vec_dim, cross_attn_emb_dim, cross_attn_heads, cross_attn_expansion_ratio)
            for _ in range(cross_attn_blocks)
        ])
        self.SA_blocks = nn.Sequential(*[
            SelfAttentionTransformerBlock(pos_emb.vec_dim, self_attn_emb_dim, self_attn_heads, self_attn_expansion_ratio)
            for _ in range(self_attn_blocks)
        ])

    def forward(self, Pq: torch.Tensor, Pu: torch.Tensor,Ps: torch.Tensor) -> torch.Tensor:
        """
        Encode the sampled points into a vector set.
        args:
            Pq: (batch_size, n_q, 3)
            Pu: (batch_size, n_u, 3)
            Ps: (batch_size, n_s, 3)
        returns:
            vecset: (batch_size, n_q, vec_dim)
        """
        Pq = self.pos_emb(Pq)
        Pu = self.pos_emb(Pu)
        Ps = self.pos_emb(Ps)
        Cu = Pq
        Cs = Pq
        for block in self.Pu_blocks:
            Cu = block(Cu, Pu)
        for block in self.Ps_blocks:
            Cs = block(Cs, Ps)
        vecset = Cu + Cs
        vecset = self.SA_blocks(vecset)
        return vecset
    

class Encoder(nn.Module):
    def __init__(self, shape2vecset: Optional[Shape2Vecset] = None, latent_dim: int = M.DIM_LATENT):
        super().__init__()
        if shape2vecset is None:
            shape2vecset = Shape2Vecset()
        self.shape2vecset = shape2vecset
        self.ln = LayerNorm(shape2vecset.vec_dim)
        self.mu = nn.Linear(shape2vecset.vec_dim, latent_dim)
        self.logvar = nn.Linear(shape2vecset.vec_dim, latent_dim)

    def forward(self, Pq: torch.Tensor, Pu: torch.Tensor,Ps: torch.Tensor) ->  tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the sampled points into a latent space.
        args:
            Pq: (batch_size, n_q, 3)
            Pu: (batch_size, n_u, 3)
            Ps: (batch_size, n_s, 3)
        returns:
            mu: (batch_size, n_q, latent_dim)
            logvar: (batch_size, n_q, latent_dim)
        """
        vecset = self.shape2vecset(Pq, Pu, Ps)
        vecset = self.ln(vecset)
        mu = self.mu(vecset)
        logvar = self.logvar(vecset)
        return mu, logvar

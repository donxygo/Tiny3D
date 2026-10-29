import torch
import torch.nn as nn
from typing import Optional
from config import ModelConfig as M
from modules import PosEmb, MLP, CrossAttentionTransformerBlock, SelfAttentionTransformerBlock


class Decoder(nn.Module):
    def __init__(self, 
                 pos_emb: Optional[PosEmb] = None,
                 latent_dim: int = M.DIM_LATENT, 
                 vec_dim: int = M.DIM_VEC,
                 self_attn_blocks: int = M.DEC_SA_BLOCKS,
                 self_attn_emb_dim: int = M.DEC_SA_EMB,
                 self_attn_heads: int = M.DEC_SA_HEADS,
                 self_attn_expansion_ratio: int = M.DEC_SA_EXPANSION,
                 cross_attn_blocks: int = M.DEC_CA_BLOCKS,
                 cross_attn_emb_dim: int = M.DEC_CA_EMB,
                 cross_attn_heads: int = M.DEC_CA_HEADS, 
                 cross_attn_expansion_ratio: int = M.DEC_CA_EXPANSION,
                 feature_field_dim: int = M.DIM_FEATURE_FIELD,
    ):
        super().__init__()
        if pos_emb is None:
            pos_emb = PosEmb(
                 n_freqs = M.POS_EMB_FREQS, 
                 include_input = M.POS_EMB_INCLUDE_INPUT, 
                 vec_dim = vec_dim, 
                 n_layers = M.POS_EMB_LAYERS
            )           
        self.pos_emb = pos_emb 
        self.latent2vec = nn.Linear(latent_dim, vec_dim)
        self.SA_blocks = nn.ModuleList([
            SelfAttentionTransformerBlock(vec_dim, self_attn_emb_dim, self_attn_heads, self_attn_expansion_ratio)
            for _ in range(self_attn_blocks)
        ])
        self.CA_blocks = nn.ModuleList([
            CrossAttentionTransformerBlock(vec_dim, cross_attn_emb_dim, cross_attn_heads, cross_attn_expansion_ratio)
            for _ in range(cross_attn_blocks)
        ])
        self.feature_field = MLP([vec_dim, vec_dim//2, feature_field_dim])
        self.feature_field_dim = feature_field_dim

    def forward(self, latents: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to FC codes given position coordinates.
        args:
            latents: (batch_size, n_ls, latent_dim)
            query: (batch_size, n_query, 3)
        returns:
            feature_field: (batch_size, n_query, feature_field_dim)
        """
        vec_set = self.latent2vec(latents)
        for block in self.SA_blocks:
            vec_set = block(vec_set)
        query = self.pos_emb(query)
        for block in self.CA_blocks:
            query = block(query, vec_set)
        feature_field = self.feature_field(query)
        return feature_field

    @torch.no_grad()
    def latents2vecset(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to vector set. This method is used for inference.
        args:
            latents: (batch_size, n_ls, latent_dim)
        returns:
            vec_set: (batch_size, n_ls, vec_dim)
        """
        vec_set = self.latent2vec(latents)
        for block in self.SA_blocks:
            vec_set = block(vec_set)
        return vec_set

    @torch.no_grad()
    def query_feature_field(self, vec_set: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Query feature field given vector set and query coordinates. This method is used for inference.
        args:
            vec_set: (batch_size, n_ls, vec_dim)
            query: (batch_size, n_query, 3)
        returns:
            feature_field: (batch_size, n_query, feature_field_dim)
        """
        query = self.pos_emb(query)
        for block in self.CA_blocks:
            query = block(query, vec_set)
        feature_field = self.feature_field(query)
        return feature_field
            
            
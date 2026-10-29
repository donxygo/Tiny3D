import torch
import torch.nn as nn
import math
from encoder import Encoder
from decoder import Decoder
from config import ModelConfig as M
import trimesh
from atom3d.grid import OctreeIndexer
from faithcontour import FCTResult

class VAE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, Pq: torch.Tensor, Pu: torch.Tensor, Ps: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        args:
            Pq: (batch_size, n_q, 3)
            Pu: (batch_size, n_u, 3)
            Ps: (batch_size, n_s, 3)
            query: (batch_size, n_query, 3)
        returns:
            mu: (batch_size, n_q, latent_dim)
            logvar: (batch_size, n_q, latent_dim)
            feature_field: (batch_size, n_query, feature_field_dim)
        """
        mu, logvar = self.encoder(Pq, Pu, Ps)
        std = torch.exp(0.5 * logvar)
        latents = mu + torch.randn_like(std) * std
        feature_field = self.decoder(latents, query)
        return mu, logvar, feature_field


class S1_Model(nn.Module):
    def __init__(self, vae: VAE, device: str):
        super().__init__()
        self.vae = vae
        ff_dim = vae.decoder.feature_field_dim
        self.active_head = nn.Linear(ff_dim, 1)
        self.anchor_head = nn.Linear(ff_dim, 3)
        self.normal_head = nn.Linear(ff_dim, 3)
        self.flux_head = nn.Linear(ff_dim, 3 * 3) 
        self.grid = OctreeIndexer(max_level=int(math.log2(M.RESOLUTION)), device=device)
        
    def forward(self, Pq, Pu, Ps, query):
        """
        Forward pass through the S1_Model.
        args:
            Pq: (batch_size, n_q, 3)
            Pu: (batch_size, n_u, 3)
            Ps: (batch_size, n_s, 3)
            query: (batch_size, n_query, 3)
        returns:
            mu: (batch_size, n_q, latent_dim)
            logvar: (batch_size, n_q, latent_dim)
            fc_field:
                active_logits: (batch_size, n_query, 1)
                anchor: (batch_size, n_query, 3)
                normal: (batch_size, n_query, 3)
                flux_3_logits: (batch_size, n_query, 9)
        """
        mu, logvar, feature_field = self.vae(Pq, Pu, Ps, query)
        fc_field = {
            'active_logits': self.active_head(feature_field),  # (B, N, 1)
            'anchor': self.anchor_head(feature_field),  # (B, N, 3)
            'normal': self.normal_head(feature_field),  # (B, N, 3)
            'flux_3_logits': self.flux_head(feature_field)  # (B, N, 9)            
        }

        return mu, logvar, fc_field

    @torch.no_grad()
    def extract_fct(self, Pq: torch.Tensor, Pu: torch.Tensor, Ps: torch.Tensor,
                    resolution: int = M.RESOLUTION, threshold: float = 0.5, chunk_size: int = 16384) -> FCTResult:
        device = next(self.parameters()).device

        mu, _ = self.vae.encoder(Pq, Pu, Ps)
        vec_set = self.vae.decoder.latents2vecset(mu)  # (1, 512, 256)

        total_voxels = resolution ** 3
        active_indices_list = []
        anchor_list = []
        normal_list = []
        flux_12_list = []

        for start in range(0, total_voxels, chunk_size):
            end = min(start + chunk_size, total_voxels)
            indices = torch.arange(start, end, device=device)
            centers = self._idx_to_center(indices, resolution)  # (chunk, 3)

            feature_field = self.vae.decoder.query_feature_field(vec_set, centers.unsqueeze(0))  # (1, chunk, 64)

            active_logits = self.active_head(feature_field)  # (1, chunk, 1)
            probs = torch.sigmoid(active_logits).squeeze(0).squeeze(-1)  # (chunk,)
            chunk_mask = probs > threshold  # (chunk,)
            
            if chunk_mask.any():
                active_indices = indices[chunk_mask]  # (n_active,)
                active_indices_list.append(active_indices)

                flux_12_list.append(self.build_flux_12(vec_set, active_indices))  # (n_active, 12)

                active_feature = feature_field.squeeze(0)[chunk_mask]  # (n_active, 64)
                anchor_list.append(self.anchor_head(active_feature))  # (n_active, 3)
                normal_list.append(self.normal_head(active_feature))  # (n_active, 3)

        active_indices = torch.cat(active_indices_list) if active_indices_list else torch.empty(0, dtype=torch.long, device=device)
        anchor = torch.cat(anchor_list) if anchor_list else torch.empty(0, 3, dtype=torch.float32, device=device)
        normal = torch.cat(normal_list) if normal_list else torch.empty(0, 3, dtype=torch.float32, device=device)
        flux_12 = torch.cat(flux_12_list) if flux_12_list else torch.empty(0, 12, dtype=torch.int8, device=device)

        return FCTResult(
            active_voxel_indices=active_indices,
            anchor=anchor,
            normal=normal,
            edge_flux_sign=flux_12,
            resolution=resolution,
            max_level=int(math.log2(resolution)) 
        )

    def build_flux_12(self, vec_set: torch.Tensor, active_indices: torch.Tensor) -> torch.Tensor:
        """
        Build 12-dimensional edge flux signs from flux_head predictions at edge midpoints.
        args:
            vec_set: (1, n_vec, dim_vec)
            active_indices: (n_active,)
        returns:
            flux_12: (n_active, 12)
        """
        if active_indices.numel() == 0:
            return torch.empty(0, 12, dtype=torch.int8, device=next(self.parameters()).device)
        voxel_edge_indices = self.grid.cube_edge_indices(active_indices) # (n_active, 12)
        edges_flat = voxel_edge_indices.reshape(-1) # (n_active * 12,)
        unique_edges, inverse = torch.unique(edges_flat, return_inverse=True)
        edge_endpoints = self.grid.edge_endpoints(unique_edges)  # (E, 2, 3)
        edge_midpoints = edge_endpoints.mean(dim=1)  # (E, 3)
        edge_dirs = torch.abs(edge_endpoints[:, 1, :] - edge_endpoints[:, 0, :]) # (E, 3)
        edge_axis = torch.argmax(edge_dirs, dim=1)  # (E,)

        feature_field = self.vae.decoder.query_feature_field(vec_set, edge_midpoints.unsqueeze(0))  # (1, E, feature_field_dim)
        flux_3_logits = self.flux_head(feature_field).squeeze(0).view(-1, 3, 3)  # (E, 3, 3) 
        flux_selected_logits = torch.gather(flux_3_logits, dim=1, index=edge_axis.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)  # (E, 3)
        flux_selected = (torch.argmax(flux_selected_logits, dim=1) - 1).to(torch.int8) # (E,) map to {-1, 0, 1}
        return flux_selected[inverse].view(-1, 12)  # (n_active, 12)

    def _idx_to_center(self, indices, resolution):
        bound = 1.0
        step = 2 * bound / resolution
        x = indices % resolution
        y = (indices // resolution) % resolution
        z = indices // (resolution * resolution)
        centers = torch.stack([
            -bound + step * (x + 0.5),
            -bound + step * (y + 0.5),
            -bound + step * (z + 0.5)
        ], dim=-1).float()
        return centers

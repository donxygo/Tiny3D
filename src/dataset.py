import os
import torch
from torch.utils.data import Dataset
import numpy as np
from config import ModelConfig as M, TrainConfig as T

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(CUR_DIR)
DATA_ROOT = os.path.join(PAR_DIR, 'datasets', 'ModelNet40_processed')
                            
class MyDataset(Dataset):
    REQUIRED_FILES = ('pu.npy', 'ps.npy', 'pv.npy', 'fct.npz', 'flux_3.npz')

    def __init__(self, data_root: str = DATA_ROOT, mode: str = 'train'):
        self.data_root = data_root
        self.mode = mode
        self.samples = []
        self.epoch = 0

        for category in sorted(os.listdir(data_root)):
            category_path = os.path.join(data_root, category, mode)
            if not os.path.isdir(category_path):
                print(f"Warning: {category_path} is not a directory. Skipping.")
                continue
            for model_id in sorted(os.listdir(category_path)):
                model_path = os.path.join(category_path, model_id)
                if not os.path.isdir(model_path):
                    print(f"Warning: {model_path} is not a directory. Skipping.")
                    continue
                if not all(os.path.exists(os.path.join(model_path, f)) for f in self.REQUIRED_FILES):
                    print(f"Warning: Missing required files in {model_path}. Skipping.")
                    continue             
                self.samples.append(model_path)

        print(f"Loaded {len(self.samples)} samples for mode '{mode}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, idx: int) -> dict:
        model_path = self.samples[idx]
        pu = torch.from_numpy(np.load(os.path.join(model_path, 'pu.npy'))).float() 
        ps = torch.from_numpy(np.load(os.path.join(model_path, 'ps.npy'))).float()
        pv = torch.from_numpy(np.load(os.path.join(model_path, 'pv.npy'))).float()

        fct = np.load(os.path.join(model_path, 'fct.npz'))
        anchor = torch.from_numpy(fct['anchor']).float()
        normal = torch.from_numpy(fct['normal']).float()
        edge_flux_sign = torch.from_numpy(fct['edge_flux_sign']).long()
        active_voxel_indices = torch.from_numpy(fct['active_voxel_indices']).long()

        flux_3 = np.load(os.path.join(model_path, 'flux_3.npz'))
        flux_3 = torch.from_numpy(flux_3['flux_3']).long()

        K = active_voxel_indices.shape[0]
        actual_sample_fct = min(T.SAMPLE_FCT, K)
        active_perm = torch.randperm(K)[:actual_sample_fct]  

        active_indices = active_voxel_indices[active_perm]
        active_mask = torch.ones(actual_sample_fct, dtype=torch.float32)
        active_anchor= anchor[active_perm]
        active_normal = normal[active_perm]
        active_flux_3 = flux_3[active_perm]

        actual_sample_hard_neg = 0
        if self.epoch > 0:
            if T.HARD_START_EPOCH <= self.epoch < T.HARD_END_EPOCH:
                actual_sample_hard_neg = (self.epoch - T.HARD_START_EPOCH + 1) * T.HARD_PER_EPOCH
            elif self.epoch >= T.HARD_END_EPOCH:
                actual_sample_hard_neg = T.HARD_TARGET

        hard_negative_indices = self._sample_hard_negatives(
            active_indices = active_voxel_indices,      
            num_samples = actual_sample_hard_neg,
            shell_radius = T.SHELL_RADIUS            
        )

        actual_sample_uni_neg = (T.SAMPLE_NEG - actual_sample_hard_neg) + (T.SAMPLE_FCT - actual_sample_fct)
        uni_negative_indices = self._sample_uni_negatives(
            sampled_indices = torch.cat([active_voxel_indices, hard_negative_indices]),
            num_samples = actual_sample_uni_neg
        )

        negative_indices = torch.cat([hard_negative_indices, uni_negative_indices])
        actual_sample_neg = len(negative_indices)
        negative_mask = torch.zeros(actual_sample_neg, dtype=torch.float32)
        negative_anchor = torch.zeros(actual_sample_neg, 3, dtype=torch.float32)
        negative_normal = torch.zeros(actual_sample_neg, 3, dtype=torch.float32)
        negative_flux_3 = torch.zeros(actual_sample_neg, 3, dtype=torch.long)

        all_mask = torch.cat([active_mask, negative_mask]).reshape(-1, 1)
        all_anchor = torch.cat([active_anchor, negative_anchor])
        all_normal = torch.cat([active_normal, negative_normal])
        all_flux_3 = torch.cat([active_flux_3, negative_flux_3])

        query_points = torch.zeros((actual_sample_fct + actual_sample_neg, 3), dtype=torch.float32)
        query_points[:actual_sample_fct] = self._idx_to_center(active_indices)
        query_points[actual_sample_fct:] = self._idx_to_center(negative_indices)

        """
        Faithful Contour Field (FCF): Continuous geometric field supervision.

        Extends discrete FCT features into continuous neural fields:
        - active  → Active Field (occupancy)
        - anchor  → Anchor Field (surface position)
        - normal  → Normal Field (surface orientation)
        - flux    → Flux Field (3D center-aligned directional field)
        
        """
        
        fc_field = {
            'active': all_mask,            
            'anchor': all_anchor,
            'normal': all_normal,
            'flux_3': all_flux_3,
        }

        if self.mode == 'test':
            raw_fct = {
                'active_voxel_indices': active_voxel_indices,                
                'anchor': anchor,
                'normal': normal,
                'edge_flux_sign': edge_flux_sign,
            }
            return {'model_id': os.path.basename(model_path), 'pu': pu, 'ps': ps, 'pv': pv, 'query_points': query_points, 'fc_field': fc_field, 'raw_fct': raw_fct}

        return {'model_id': os.path.basename(model_path), 'pu': pu, 'ps': ps, 'pv': pv, 'query_points': query_points, 'fc_field': fc_field}


    def _idx_to_center(self, indices: torch.Tensor) -> torch.Tensor:
        resolution = M.RESOLUTION
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


    def _sample_hard_negatives(self, active_indices: torch.Tensor, num_samples: int, shell_radius: int = 5) -> torch.Tensor:
        if num_samples == 0:
            return torch.empty(0, dtype=torch.long)
        
        K = active_indices.shape[0]

        resolution = M.RESOLUTION
        x = active_indices % resolution
        y = (active_indices // resolution) % resolution
        z = active_indices // (resolution * resolution)
        active_coords = torch.stack([x, y, z], dim=-1).float()  # (K, 3)

        hard_neg_indices = []
        used_set = set()        

        extra = num_samples 
        while extra > 0:
            rand_idx = torch.randint(0, K, (extra * 2,))
            ref_coords = active_coords[rand_idx]  # (extra*2, 3)
            offsets = torch.randint(-shell_radius, shell_radius + 1, (extra * 2, 3))
            cand_coords = torch.clamp(ref_coords + offsets, 0, resolution - 1)
            cand_indices = cand_coords[:, 0] + cand_coords[:, 1] * resolution + cand_coords[:, 2] * resolution * resolution
            mask = ~torch.isin(cand_indices, active_indices)  
            valid = list(set(cand_indices[mask].tolist()) - used_set)
            if not valid:
                continue
            taken = valid[:extra]
            hard_neg_indices.extend(taken)
            used_set.update(taken)
            extra -= len(taken)
        
        return torch.tensor(hard_neg_indices, dtype=torch.long)[:num_samples]


    def _sample_uni_negatives(self, sampled_indices: torch.Tensor, num_samples: int):
        if num_samples == 0:
            return torch.empty(0, dtype=torch.long)

        total_voxels = M.RESOLUTION**3
        uni_neg_indices = []
        used_set = set()

        extra = num_samples
        while extra > 0:
            cand_indices = torch.randint(0, total_voxels, (extra * 3,))
            mask = ~torch.isin(cand_indices, sampled_indices)
            valid = list(set(cand_indices[mask].tolist()) - used_set)
            if not valid:
                continue
            taken = valid[:extra]
            uni_neg_indices.extend(taken)
            used_set.update(taken)
            extra -= len(taken)

        return torch.tensor(uni_neg_indices, dtype=torch.long)[:num_samples]




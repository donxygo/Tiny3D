import torch
import torch.nn.functional as F
import math
from config import TrainConfig as T


def active_loss(preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    (batch_size, n_query, 1), (batch_size, n_query, 1) -> scalar
    """
    return F.binary_cross_entropy_with_logits(preds, gt, reduction='mean') 


def anchor_loss(preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    (batch_size, n_query, 3), (batch_size, n_query, 3) -> (batch_size, n_query, 1)
    """
    return F.mse_loss(preds, gt, reduction='none').mean(dim=-1, keepdim=True)


def normal_loss(preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    (batch_size, n_query, 3), (batch_size, n_query, 3) -> (batch_size, n_query, 1)
    """
    normal_loss = (1.0 - F.cosine_similarity(preds, gt, dim=-1, eps=1e-8)).unsqueeze(-1)
    return normal_loss


def flux_loss(preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    (batch_size, n_query, 9), (batch_size, n_query, 3) -> (batch_size, n_query, 3)
    """
    B, N, _ = preds.shape

    preds = preds.reshape(-1, 3) # (B * N * 3, 3)
    gt = (gt + 1).reshape(-1)  # (B * N * 3,)

    loss_per_elem = F.cross_entropy(preds, gt, reduction='none')  # (B * N * 3,)
    loss_per_axis = loss_per_elem.reshape(B, N, 3)

    return loss_per_axis


def KL_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    mu, logvar: (batch_size, n_vec, latent_dim) -> scalar
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def S1_loss(epoch: int, mu: torch.Tensor, logvar: torch.Tensor, fcf_preds: dict, fcf_gt: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Args:
        epoch: current epoch
        mu: (batch_size, n_vec, latent_dim)
        logvar: (batch_size, n_vec, latent_dim)      

        fcf_preds:
            'active_logits': (batch_size, n_query, 1),
            'anchor': (batch_size, n_query, 3),
            'normal': (batch_size, n_query, 3),
            'flux_3_logits': (batch_size, n_query, 9),
        
        fcf_gt:
            'active': (batch_size, n_query, 1),            
            'anchor': (batch_size, n_query, 3),
            'normal': (batch_size, n_query, 3),
            'flux_3': (batch_size, n_query, 3),

    Returns:
        dict: Scalar losses with keys:
            'total': combined loss
            'active', 'anchor', 'normal', 'flux': per-field regression/classification losses
            'kl': KL divergence
    """
    mask = fcf_gt['active']
    n_active = mask.sum() + 1e-8

    loss_active = active_loss(fcf_preds['active_logits'], fcf_gt['active']) 
    loss_anchor = (mask * anchor_loss(fcf_preds['anchor'], fcf_gt['anchor'])).sum() / n_active
    loss_normal = (mask * normal_loss(fcf_preds['normal'], fcf_gt['normal'])).sum() / n_active
    loss_flux = (mask * flux_loss(fcf_preds['flux_3_logits'], fcf_gt['flux_3'])).sum() / n_active
    loss_fcf = T.LAMBDA_ACTIVE * loss_active + T.LAMBDA_ANCHOR * loss_anchor + T.LAMBDA_NORMAL * loss_normal + T.LAMBDA_FLUX * loss_flux
  
    progress = min(1.0, (epoch-1) / T.KL_ANNEAL_EPOCHS)
    cos_factor = (1 - math.cos(progress * math.pi)) / 2
    loss_KL = cos_factor * T.KL_TARGET_WEIGHT * KL_loss(mu, logvar)

    loss_total = loss_fcf + loss_KL

    loss_dict = {
        'total': loss_total,
        'active': loss_active,
        'anchor': loss_anchor,
        'normal': loss_normal,
        'flux': loss_flux,
        'kl': loss_KL,
    }

    return loss_dict


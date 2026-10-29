import os
import math
import torch
import torch.nn.utils
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from dataset import MyDataset
from modules import PosEmb
from encoder import Shape2Vecset, Encoder 
from decoder import Decoder
from vae import VAE, S1_Model
from loss import S1_loss
from config import TrainConfig as T

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CUR_DIR)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def init_train_S1():
    full_train_dataset = MyDataset(mode='train') 
    train_ratio = T.TRAIN_RATIO
    train_len = int(len(full_train_dataset) * train_ratio)
    val_len = len(full_train_dataset) - train_len

    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_len, val_len],
        generator=torch.Generator().manual_seed(T.RANDOM_SEED) 
    )

    print(f"train_dataset: {train_len} samples, val_dataset: {val_len} samples")

    train_loader = DataLoader(
        train_dataset, 
        batch_size = T.TRAIN_BATCH_SIZE, 
        shuffle = T.TRAIN_SHUFFLE, 
        num_workers = T.TRAIN_NUM_WORKERS, 
        pin_memory = T.TRAIN_PIN_MEMORY,
        drop_last = T.TRAIN_DROP_LAST,
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size = T.VAL_BATCH_SIZE, 
        shuffle = T.VAL_SHUFFLE, 
        num_workers = T.VAL_NUM_WORKERS, 
        pin_memory = T.VAL_PIN_MEMORY,
        drop_last = T.VAL_DROP_LAST,
    )

    encoder = Encoder()
    decoder = Decoder()
    vae = VAE(encoder, decoder)
    model = S1_Model(vae, device=str(DEVICE))
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    warmup_scheduler = LinearLR(optimizer, start_factor=T.WARMUP_START_FACTOR, end_factor=T.WARMUP_END_FACTOR, total_iters=T.WARMUP_ITERS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T.COSINE_T_MAX, eta_min=T.COSINE_ETA_MIN)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=T.SCHEDULER_MILESTONES)
    loss_fn = S1_loss

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, 'checkpoints_S1')
    os.makedirs(checkpoint_dir, exist_ok=True)

    return model, train_loader, val_loader, optimizer, scheduler, loss_fn, checkpoint_dir


def train_S1(resume_from: str = None, load_weights_only: bool = False):
    model, train_loader, val_loader, optimizer, scheduler, loss_fn, checkpoint_dir = init_train_S1()
    model.train()

    optimizer.zero_grad()
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.isfile(best_model_path):
        best_val_loss = torch.load(best_model_path, map_location=DEVICE)['best_val_loss_dict']['total']
    start_epoch = 1

    training_session = 1
    if resume_from is not None:
        if os.path.isfile(resume_from):
            # Load pre-trained model
            if load_weights_only:
                training_session = 2
                pre_model = torch.load(resume_from, map_location=DEVICE)
                model.load_state_dict(pre_model['model_state_dict'])
                print(f"Loaded pre-trained weights from {resume_from}, current best val loss: {best_val_loss:.4f}")
            else:
                print(f"Loading checkpoint from {resume_from}")
                checkpoint = torch.load(resume_from, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1 
                print(f"Resuming from epoch {start_epoch}, current best val loss: {best_val_loss:.4f}")
        else:        
            print(f"Checkpoint file {resume_from} not found, starting from scratch, current best val loss: {best_val_loss:.4f}")
    else:
        print(f"Starting from scratch, current best val loss: {best_val_loss:.4f}")

    for epoch in range(start_epoch, T.EPOCHS + 1):
        epoch_loss_dict = {
            'total': torch.tensor(0.0, device=DEVICE, dtype=torch.float32),
            'active': torch.tensor(0.0, device=DEVICE, dtype=torch.float32), 
            'anchor': torch.tensor(0.0, device=DEVICE, dtype=torch.float32), 
            'normal': torch.tensor(0.0, device=DEVICE, dtype=torch.float32), 
            'flux': torch.tensor(0.0, device=DEVICE, dtype=torch.float32), 
            'kl': torch.tensor(0.0, device=DEVICE, dtype=torch.float32),
            }
        if training_session == 2:
            train_loader.dataset.dataset.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            pu = batch['pu'].to(DEVICE, non_blocking=True)
            ps = batch['ps'].to(DEVICE, non_blocking=True)
            pv = batch['pv'].to(DEVICE, non_blocking=True)
            query_points = batch['query_points'].to(DEVICE, non_blocking=True)
            fcf_gt = batch['fc_field']
            for key in fcf_gt:
                fcf_gt[key] = fcf_gt[key].to(DEVICE, non_blocking=True)

            mu, logvar, fcf_preds = model(pv, pu, ps, query_points)
            loss_dict = loss_fn(epoch, mu, logvar, fcf_preds, fcf_gt)   
            for k in epoch_loss_dict:
                epoch_loss_dict[k] += loss_dict[k].float()     
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), T.GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

            if (batch_idx + 1) % T.LOG_INTERVAL == 0:
                for name, param in model.named_parameters():
                    if torch.isinf(param).any() or torch.isnan(param).any():
                        print(f"!!! Parameter {name} contains Inf/ NaN after step at Epoch {epoch} | Batch {batch_idx + 1}")
                print(f"Epoch {epoch} | Batch {batch_idx + 1} | "
                      f"Total: {loss_dict['total'].item():.4f}, "
                      f"Active: {loss_dict['active'].item():.4f}, "
                      f"Anchor: {loss_dict['anchor'].item():.4f}, "
                      f"Normal: {loss_dict['normal'].item():.4f}, "
                      f"Flux: {loss_dict['flux'].item():.4f}, "
                      f"KL: {loss_dict['kl'].item():.4f}")   

        scheduler.step()
        
        num_batches = len(train_loader)
        avg_loss_dict = {k: v.item() / num_batches for k, v in epoch_loss_dict.items()}
        
        print(f"Epoch {epoch} Average Loss: "
              f"Total: {avg_loss_dict['total']:.4f}, "
              f"Active: {avg_loss_dict['active']:.4f}, "
              f"Anchor: {avg_loss_dict['anchor']:.4f}, "
              f"Normal: {avg_loss_dict['normal']:.4f}, "
              f"Flux: {avg_loss_dict['flux']:.4f}, "
              f"KL: {avg_loss_dict['kl']:.4f}")

        if epoch % T.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_dict': avg_loss_dict,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f"Checkpoint saved at epoch {epoch}")

        if epoch % T.VAL_INTERVAL == 0:
            val_loss_dict = validate_S1(model, val_loader, loss_fn, epoch)
            if val_loss_dict['total'] < best_val_loss:
                best_val_loss = val_loss_dict['total']
                best_val_loss_dict = {k: v for k, v in val_loss_dict.items()}
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss_dict': best_val_loss_dict,
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"New best model saved! Val Loss: {best_val_loss:.4f}")


def validate_S1(model, data_loader, loss_fn, epoch: int):
    torch.cuda.empty_cache()
    model.eval()
    val_loss_dict = {
        'total': torch.tensor(0.0, device=DEVICE, dtype=torch.float32),
        'active': torch.tensor(0.0, device=DEVICE, dtype=torch.float32), 
        'anchor': torch.tensor(0.0, device=DEVICE, dtype=torch.float32), 
        'normal': torch.tensor(0.0, device=DEVICE, dtype=torch.float32), 
        'flux': torch.tensor(0.0, device=DEVICE, dtype=torch.float32), 
        'kl': torch.tensor(0.0, device=DEVICE, dtype=torch.float32),
        }

    with torch.no_grad():
        for batch in data_loader:
            pu = batch['pu'].to(DEVICE, non_blocking=True)
            ps = batch['ps'].to(DEVICE, non_blocking=True)
            pv = batch['pv'].to(DEVICE, non_blocking=True)
            query_points = batch['query_points'].to(DEVICE, non_blocking=True)
            fcf_gt = batch['fc_field']
            for key in fcf_gt:
                fcf_gt[key] = fcf_gt[key].to(DEVICE, non_blocking=True)

            mu, logvar, fcf_preds = model(pv, pu, ps, query_points)
            loss_dict = loss_fn(epoch, mu, logvar, fcf_preds, fcf_gt) 
            for k in val_loss_dict:
                val_loss_dict[k] += loss_dict[k].float()

    num_batches = len(data_loader)
    val_loss_dict = {k: v.item() / num_batches for k, v in val_loss_dict.items()}

    print(f"Validation Average Loss: "
          f"Total: {val_loss_dict['total']:.4f}, "
          f"Active: {val_loss_dict['active']:.4f}, "
          f"Anchor: {val_loss_dict['anchor']:.4f}, "
          f"Normal: {val_loss_dict['normal']:.4f}, "
          f"Flux: {val_loss_dict['flux']:.4f}, "
          f"KL: {val_loss_dict['kl']:.4f}")

    model.train()
    torch.cuda.empty_cache()

    return val_loss_dict
    
    
if __name__ == '__main__':
    train_S1()

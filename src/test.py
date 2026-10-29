import os
import torch
import numpy as np
from dataset import MyDataset
from torch.utils.data import DataLoader
from encoder import Encoder 
from decoder import Decoder
from vae import VAE, S1_Model
from config import TestConfig as C

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOLUTION = 512

def test_vae():
    test_dataset = MyDataset(split='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size = C.TEST_BATCH_SIZE, 
        shuffle = C.TEST_SHUFFLE, 
        num_workers = C.TEST_NUM_WORKERS, 
        pin_memory = C.TEST_PIN_MEMORY,
        drop_last = C.TEST_DROP_LAST,
    )

    encoder = Encoder()
    decoder = Decoder()
    model = VAE(encoder, decoder)
    model.eval()

    for batch_idx, batch in enumerate(test_loader):
        pu = batch['pu'].to(DEVICE, non_blocking=True)
        ps = batch['ps'].to(DEVICE, non_blocking=True)
        pv = batch['pv'].to(DEVICE, non_blocking=True)

        total_voxels = RESOLUTION * RESOLUTION * RESOLUTION
        query_points = torch.arange(total_voxels, dtype=torch.float32)
        query_points = query_points.reshape(-1, 3)

        with torch.no_grad():
            mu, logvar, pred = model(pv, pu, ps, query_points)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ##test_vae()
    encoder = Encoder()
    decoder = Decoder()
    vae = VAE(encoder, decoder)
    model = S1_Model(vae).to(DEVICE)
    model.eval()

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CUR_DIR)
    DST_DIR = os.path.join(PROJECT_ROOT, "datasets/ModelNet40_processed")

    categories = [d for d in os.listdir(DST_DIR) if os.path.isdir(os.path.join(DST_DIR, d))]
    if not categories:
        raise FileNotFoundError("No categories found in processed dataset.")
    first_cat = categories[0]
    cat_path = os.path.join(DST_DIR, first_cat)

    sample_path = None
    for split in ['train', 'test']:
        split_path = os.path.join(cat_path, split)
        if os.path.isdir(split_path):
            model_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            if model_dirs:
                sample_path = os.path.join(split_path, model_dirs[0])
                break
    if sample_path is None:
        raise FileNotFoundError("No sample found in any split.")

    print(f"Loading sample from: {sample_path}")

    # 3. 加载点云数据 (添加 batch 维度)
    pu = torch.from_numpy(np.load(os.path.join(sample_path, 'pu.npy'))).float().unsqueeze(0).to(device)
    ps = torch.from_numpy(np.load(os.path.join(sample_path, 'ps.npy'))).float().unsqueeze(0).to(device)
    pv = torch.from_numpy(np.load(os.path.join(sample_path, 'pv.npy'))).float().unsqueeze(0).to(device)

    print(f"pu shape: {pu.shape}, ps shape: {ps.shape}, pv shape: {pv.shape}")

    # 4. 快速体素化测试 (低分辨率)
    resolution = 32      # 快速测试用
    threshold = 0.5
    chunk_size = 8192

    try:
        mask = model.coarse_voxel_reconstruction(pv, pu, ps, resolution=resolution, threshold=threshold, chunk_size=chunk_size)
        print(f"Mask shape: {mask.shape}")
        print(f"Occupied voxels: {mask.sum().item()} / {mask.numel()}")
        print(f"Occupied ratio: {mask.sum().item() / mask.numel():.4f}")
    except Exception as e:
        print(f"Error during voxelize: {e}")
        import traceback
        traceback.print_exc()
import os
import trimesh
import open3d as o3d
import numpy as np
import torch
from faithcontour import FCTEncoder
from faithcontour.segment_ops import compute_edge_flux_sign
from atom3d import MeshBVH
from atom3d.grid import OctreeIndexer
import signal
from contextlib import contextmanager
import sys
import time

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(CUR_DIR)

SRC_DIR = os.path.join(PAR_DIR, "datasets/ModelNet40")
DST_DIR = os.path.join(PAR_DIR, "datasets/ModelNet40_processed")

CATEGORIES = [
    "airplane",   "bathtub",    "bed",        "bench",
    "bookshelf",  "bottle",     "bowl",       "car",
    "chair",      "cup",        "curtain",    "desk",
    "door",       "dresser",    "flower_pot", "glass_box",
    "guitar",     "keyboard",   "lamp",       "laptop",
    "mantel",     "monitor",    "night_stand","piano",
    "plant",      "radio",      "range_hood", "sink",
    "sofa",       "stairs",     "stool",      "table",
    "tent",       "toilet",     "tv_stand",   "vase",
    "wardrobe",   "xbox",
]

NUM_PU = 8192
NUM_PS = 8192
DOWN_PU = 256
DOWN_PS = 256
ANGLE_THRESHOLD = (1/6) * np.pi  
EPS = 1e-8

FCT_MAX_LEVEL = 9
FCT_MIN_LEVEL = 4
FCT_RESOLUTION = 2**FCT_MAX_LEVEL


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()   # 实时写入，防止丢失

    def flush(self):
        for f in self.files:
            f.flush()

LOG_PATH = os.path.join(CUR_DIR, "build_dataset.log")
log_file = open(LOG_PATH, "a", buffering=1)  
sys.stdout = Tee(sys.stdout, log_file)
log_file.write(f"\n{'='*50}\nRun started at {time.ctime()}\n{'='*50}\n")
log_file.flush()

class EmptyMeshError(Exception):
    pass

class SamplingTimeoutError(Exception):
    pass

class NotEnoughPointsError(Exception):
    pass

class EncodeFCTError(Exception):
    pass


def process_duplicate_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    vertices = mesh.vertices
    faces = mesh.faces

    # Map faces to unique vertices
    _, index, inverse = np.unique(vertices, axis=0, return_index=True, return_inverse=True)
    faces = index[inverse[faces]]

    # Remove duplicate faces
    _, index = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces_unique = faces[index]

    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces_unique)
    new_mesh.remove_unreferenced_vertices()
    
    return new_mesh


def normalize_mesh(mesh: trimesh.Trimesh, half_target: float = 0.95) -> trimesh.Trimesh:
    V = mesh.vertices
    if V.size == 0:
        return mesh
    center = (V.min(axis=0) + V.max(axis=0)) / 2.0
    V = V - center
    if abs(V).max() < 1e-12:
        raise ValueError("normalize_mesh: mesh is too small to normalize")
    scale = half_target / abs(V).max()
    V = V * scale
    return trimesh.Trimesh(vertices=V, faces=mesh.faces, process=False)


def farthest_point_sample(points: np.ndarray, n_sample: int) -> np.ndarray:
    N, _ = points.shape

    if n_sample > N:
        raise ValueError(f"farthest_point_sample: n_desired ({n_sample}) must be less than or equal to n_total ({N})")
    
    centroids = np.zeros(n_sample, dtype=np.int64) 
    distance = np.ones(N) * 1e10      
    index = np.random.randint(0, N)
    centroids[0] = index
    
    for i in range(1, n_sample):
        centroid = points[index:index+1, :]         
        dist = np.sum((points - centroid) ** 2, axis=1)  
        mask = dist < distance
        distance[mask] = dist[mask]      
        index = np.argmax(distance)
        centroids[i] = index
    
    return centroids


def detect_salient_edges(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    salient_edges = []
    weights = []  
    face_pairs = mesh.face_adjacency
    adjacency_edges = mesh.face_adjacency_edges
    eps = EPS * np.linalg.norm(mesh.extents)

    for i, (face1, face2) in enumerate(face_pairs):
        verts_1 = set(mesh.faces[face1])
        verts_2 = set(mesh.faces[face2])
        v0, v1 = adjacency_edges[i]
        v_1 = (verts_1 - {v0, v1}).pop()
        v_2 = (verts_2 - {v0, v1}).pop()

        shared_edge = mesh.vertices[v1] - mesh.vertices[v0]
        len_shared = np.linalg.norm(shared_edge)
        if(len_shared < eps): continue
        shared_edge = shared_edge / len_shared  # 单位化
        edge1 = mesh.vertices[v_1] - mesh.vertices[v0]
        edge2 = mesh.vertices[v_2] - mesh.vertices[v0]

        perp1 = edge1 - np.dot(shared_edge, edge1) * shared_edge
        length = np.linalg.norm(perp1)
        if(length < eps): continue   
        perp1 = perp1 / length  # 单位化

        perp2 = edge2 - np.dot(shared_edge, edge2) * shared_edge
        length = np.linalg.norm(perp2)
        if(length < eps): continue   
        perp2 = perp2 / length  # 单位化

        dihedral_angle = np.arccos(np.clip(np.dot(perp1, perp2), -1.0, 1.0))
        if (np.pi - dihedral_angle) > ANGLE_THRESHOLD:
            salient_edges.append([v0, v1])
            weights.append(dihedral_angle * len_shared) # 用边缘曲率积分计算weight

    salient_edges = np.array(salient_edges) # (E, 2)
    weights = np.array(weights) # (E,)

    return salient_edges, weights


@contextmanager
def sampling_timeout(seconds: int):
    def handler(signum, frame):
        raise SamplingTimeoutError(f"Sampling timeout after {seconds} seconds")
    default_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, default_handler)
    

def sample_uniform_points(mesh, num_points: int = NUM_PU, max_retries: int = 3, timeout_sec: int = 60) -> np.ndarray:
    for cur_try in range(max_retries):
        try:
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
            mesh_o3d.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
            with sampling_timeout(timeout_sec):
                pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=num_points)
            points = np.asarray(pcd.points, dtype=np.float32)
            if(points.shape[0] < num_points):
                if cur_try == max_retries-1:
                    raise NotEnoughPointsError(f"(Sampling uniform points) Failed: Can't sample enough({num_points}) points after {max_retries} retries")
                continue
            if(points.shape[0] > num_points):
                points = points[farthest_point_sample(points, num_points)]
            return points
        except SamplingTimeoutError as e:
            print(f"(Sampling uniform points) Try {cur_try + 1} to sample {num_points} points failed: {e}")
            if cur_try == max_retries-1:
                raise RuntimeError(f"(Sampling uniform points) Failed to sample {num_points} points: Sampling timeout after {max_retries} retries")


def sample_salient_points(mesh: trimesh.Trimesh, num_points: int = NUM_PS) -> np.ndarray:
    if(num_points == 0):
        return np.empty((0, 3), dtype=np.float32)
    
    salient_edges, weights = detect_salient_edges(mesh)

    if len(salient_edges) == 0:
        raise NotEnoughPointsError(f"(Sampling salient points) No salient edges found")
    
    vertices_idx = np.unique(salient_edges) # (V,)
    salient_vertices = np.array(mesh.vertices[vertices_idx], dtype=np.float32) # (V, 3)

    if num_points <= len(vertices_idx):
        return salient_vertices[farthest_point_sample(salient_vertices, num_points)]

    total_weights = np.sum(weights)
    if total_weights < EPS:
        probs = np.ones_like(weights) / len(weights)
    else:
        probs = weights / total_weights

    extra_num = num_points - len(salient_vertices)
    counts = np.floor(extra_num * probs).astype(int)
    remainder = extra_num - np.sum(counts)
    indices = np.argsort(-(extra_num * probs - counts))[:remainder]
    counts[indices] += 1

    salient_points_list = [salient_vertices] # [(V, 3)]
    for i, (v_idx1, v_idx2) in enumerate(salient_edges): 
        n_cur_edge = counts[i]
        t = np.linspace(0, 1, n_cur_edge + 2)[1:-1, None] # (n_cur_edge, 1)
        v1 = mesh.vertices[v_idx1] # (3,)
        v2 = mesh.vertices[v_idx2] # (3,)
        extra_points = v1 + (v2 - v1) * t # (n_cur_edge, 3)
        salient_points_list.append(extra_points)
    salient_points = np.concatenate(salient_points_list, axis=0) # (N, 3)
    
    return salient_points[:num_points]


def downsample(Pu: np.ndarray, Ps: np.ndarray, down_Pu: int = DOWN_PU, down_Ps: int = DOWN_PS) -> np.ndarray:
    try:
        Pu = Pu[farthest_point_sample(Pu, down_Pu)] # (down_Pu, 3)
        Ps = Ps[farthest_point_sample(Ps, down_Ps)] # (down_Ps, 3)
        return np.concatenate((Pu, Ps), axis=0) # (down_Pu + down_Ps, 3)
    except ValueError as e:
        raise ValueError(f"(Downsample) error: {e}") from e


def sample_Pu_Ps_Pv(
        mesh: trimesh.Trimesh, 
        n_Pu: int = NUM_PU, 
        n_Ps: int = NUM_PS, 
        down_Pu: int = DOWN_PU, 
        down_Ps: int = DOWN_PS
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        Ps = sample_salient_points(mesh, n_Ps)
    except NotEnoughPointsError as e:
        print(f"{e}")
        Pu = sample_uniform_points(mesh, n_Pu)
        Ps = sample_uniform_points(mesh, n_Ps)
    else:
        Pu = sample_uniform_points(mesh, n_Pu)
    Pv = downsample(Pu, Ps, down_Pu, down_Ps)
    return Pu, Ps, Pv


def encode_fct(
        mesh: trimesh.Trimesh, 
        max_level: int = FCT_MAX_LEVEL, 
        min_level: int = FCT_MIN_LEVEL, 
        compute_flux: bool = True, 
        clamp_anchors: bool = True
    ) -> dict[str, np.ndarray]:
    # Load mesh
    V = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
    F = torch.tensor(mesh.faces, dtype=torch.long, device='cuda')

    # Build spatial structures
    bvh = MeshBVH(V, F)
    bounds = torch.tensor([[-1., -1., -1.], [1., 1., 1.]], device='cuda')
    octree = OctreeIndexer(max_level=max_level, bounds=bounds, device='cuda')  # 512^3
    
    try:
        # Encode
        encoder = FCTEncoder(bvh, octree, device='cuda')
        fct = encoder.encode(min_level=min_level, compute_flux=compute_flux, clamp_anchors=clamp_anchors)
        # fct.anchor:             [K, 3]  — surface anchor points
        # fct.normal:             [K, 3]  — surface normals
        # fct.edge_flux_sign:     [K, 12] — edge crossing signs {-1, 0, +1}
        # fct.active_voxel_indices: [K]   — linear voxel indices
        fct_data = {
        "anchor": fct.anchor.cpu().numpy(),
        "normal": fct.normal.cpu().numpy(),
        "edge_flux_sign": fct.edge_flux_sign.cpu().numpy(),
        "active_voxel_indices": fct.active_voxel_indices.cpu().numpy(),
        }
        del encoder, fct, bvh, octree, V, F
        torch.cuda.empty_cache()
        return fct_data
    except Exception as e:
        raise EncodeFCTError(f"(Encode FCT) error: {e}") from e
    finally:
        torch.cuda.empty_cache()


def compute_flux_3(mesh: trimesh.Trimesh, active_voxel_indices: np.ndarray) -> np.ndarray:
    V = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
    F = torch.tensor(mesh.faces, dtype=torch.long, device='cuda')
    bvh = MeshBVH(V, F)

    K = len(active_voxel_indices)
    indices = torch.tensor(active_voxel_indices, dtype=torch.long, device='cuda')
    resolution = FCT_RESOLUTION
    step = 2.0 / resolution
    x = indices % resolution
    y = (indices // resolution) % resolution
    z = indices // (resolution * resolution)
    centers = torch.stack([
        -1.0 + step * (x + 0.5),
        -1.0 + step * (y + 0.5),
        -1.0 + step * (z + 0.5)
    ], dim=-1).float()  # (K, 3)
    centers_rep = centers.repeat_interleave(3, dim=0)      # (3*K, 3)

    directions = torch.eye(3, dtype=torch.float32, device='cuda') 
    half_len = 1.0 / resolution
    offsets_rep = (half_len * directions).repeat(K, 1)    # (3*K, 3)  

    start = centers_rep - offsets_rep
    end = centers_rep + offsets_rep
    edge_endpoints = torch.stack([start, end], dim=1)      # (3*K, 2, 3)

    flux_flat = compute_edge_flux_sign(bvh, edge_endpoints)  # (3*K,) int8
    flux_3 = flux_flat.view(K, 3)                          # (K, 3)

    return flux_3.cpu().numpy()


def build_dataset(n_Pu: int = NUM_PU, n_Ps: int = NUM_PS, down_Pu: int = DOWN_PU, down_Ps: int = DOWN_PS):
    os.makedirs(DST_DIR, exist_ok=True)
    for category in CATEGORIES:
        for split in ["test", "train"]:
            print(f"=== {category}/{split} ===")
            src_dir = os.path.join(SRC_DIR, category, split)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(DST_DIR, category, split)
            os.makedirs(dst_dir, exist_ok=True)
            off_files = sorted(os.listdir(src_dir))
            for file in off_files:
                model_id = os.path.splitext(file)[0]
                dst_files = os.path.join(dst_dir, model_id)
                if (os.path.exists(os.path.join(dst_files, "pu.npy")) and
                    os.path.exists(os.path.join(dst_files, "ps.npy")) and
                    os.path.exists(os.path.join(dst_files, "pv.npy")) and
                    os.path.exists(os.path.join(dst_files, "fct.npz"))):
                    ##print(f"{file} already processed")
                    continue
                print(f"{file}", end=" ")
                try:
                    mesh = trimesh.load(os.path.join(src_dir, file), force=True)
                    mesh = process_duplicate_faces(mesh)
                    mesh = normalize_mesh(mesh)
                    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                        raise EmptyMeshError(f"Mesh is empty: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
                    pu, ps, pv = sample_Pu_Ps_Pv(mesh, n_Pu, n_Ps, down_Pu, down_Ps)
                    status = ""
                    if pu.shape[0] != n_Pu or ps.shape[0] != n_Ps or pv.shape[0] != down_Pu + down_Ps:
                        status = "  shape error!"
                    fct_data = encode_fct(mesh)                    
                    os.makedirs(dst_files, exist_ok=True)
                    np.save(os.path.join(dst_files, "pu.npy"), pu)
                    np.save(os.path.join(dst_files, "ps.npy"), ps)
                    np.save(os.path.join(dst_files, "pv.npy"), pv)
                    np.savez_compressed(os.path.join(dst_files, "fct.npz"), **fct_data)
                    print(f"pu={pu.shape}, ps={ps.shape}, pv={pv.shape}{status}, n_fct={fct_data['active_voxel_indices'].shape[0]}")
                except (EmptyMeshError, NotEnoughPointsError, RuntimeError, ValueError, EncodeFCTError) as e:
                    print(f"{e}")
                except Exception as e:
                    print(f"{e}")
 

def build_flux_3():
    for category in CATEGORIES:
        for split in ["test", "train"]:
            print(f"=== {category}/{split} ===")
            src_dir = os.path.join(SRC_DIR, category, split)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(DST_DIR, category, split)
            off_files = sorted(os.listdir(src_dir))
            for file in off_files:
                model_id = os.path.splitext(file)[0]
                dst_files = os.path.join(dst_dir, model_id)
                if (os.path.exists(os.path.join(dst_files, "pu.npy")) and
                    os.path.exists(os.path.join(dst_files, "ps.npy")) and
                    os.path.exists(os.path.join(dst_files, "pv.npy")) and
                    os.path.exists(os.path.join(dst_files, "fct.npz"))):
                    print(f"{file}", end=" ")
                    try:
                        mesh = trimesh.load(os.path.join(src_dir, file), force=True)
                        mesh = process_duplicate_faces(mesh)
                        mesh = normalize_mesh(mesh)
                        fct_data = np.load(os.path.join(dst_files, "fct.npz"))
                        flux_3 = compute_flux_3(mesh, fct_data["active_voxel_indices"])
                        np.savez_compressed(os.path.join(dst_files, "flux_3.npz"), flux_3=flux_3)
                        print(f"flux_3={flux_3.shape}")
                    except Exception as e:
                        print(f"{e}")

                               
if __name__ == "__main__":
    build_dataset()
    build_flux_3()

        
        
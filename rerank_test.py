"""
Re-ranking Test Script — Horse Re-ID
Zhong et al. CVPR 2017 — k-reciprocal encoding
Uses GPU for feature extraction + chunked distance computation to avoid OOM
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets.horse import HorseReID
from src.eval_metrics import evaluate

# ── ImageDataset ─────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, img_path

# ── Chunked distance matrix — GPU friendly ───────────────

def compute_distmat(qf, gf, chunk_size=256):
    """
    Computes Euclidean distance matrix in chunks.
    qf, gf: numpy arrays (num_query x feat_dim), (num_gallery x feat_dim)
    Returns: distance matrix (num_query x num_gallery) as numpy array
    """
    qf_t = torch.from_numpy(qf)
    gf_t = torch.from_numpy(gf)

    num_query = qf_t.shape[0]
    num_gallery = gf_t.shape[0]
    distmat = np.zeros((num_query, num_gallery), dtype=np.float32)

    for start in range(0, num_query, chunk_size):
        end = min(start + chunk_size, num_query)
        q_chunk = qf_t[start:end]  # (chunk x feat)
        # Euclidean distance: ||q - g||^2 = ||q||^2 + ||g||^2 - 2*q*g^T
        dist_chunk = (
            torch.sum(q_chunk ** 2, dim=1, keepdim=True) +
            torch.sum(gf_t ** 2, dim=1, keepdim=True).T -
            2 * torch.mm(q_chunk, gf_t.T)
        )
        dist_chunk = torch.clamp(dist_chunk, min=0).sqrt()
        distmat[start:end] = dist_chunk.numpy()

    return distmat

# ── Re-ranking (Zhong et al. CVPR 2017) ──────────────────

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)], axis=0)

    original_dist = 2. - 2 * original_dist
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(
        1. * original_dist / np.max(original_dist, axis=0))

    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)
    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(
                initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,
                                   k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index,
                    candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist, V, jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--arch', type=str, default='resnet50_fc512')
    parser.add_argument('--root', type=str, default=r'D:\horse4\dataset')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Transform
    transform_test = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    dataset = HorseReID(root=args.root)

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers)

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers)

    # Model
    if args.arch == 'resnet50_fc512':
        from src.models.resnet import resnet50_fc512
        model = resnet50_fc512(num_classes=dataset.num_train_pids, pretrained=False)
    elif args.arch == 'swin_base':
        from src_models_swin_reid import SwinReID
        model = SwinReID(num_classes=dataset.num_train_pids)
    elif args.arch == 'efficientnetv2_s':
        from src_models_efficientnet_reid import EfficientNetV2ReID
        model = EfficientNetV2ReID(num_classes=dataset.num_train_pids)

    # Load checkpoint — strip DataParallel 'module.' prefix
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f'Loaded: {args.checkpoint}')

    # Extract features — GPU accelerated
    def extract_features(loader):
        feats, pids, camids = [], [], []
        with torch.no_grad():
            for imgs, batch_pids, batch_camids, _ in loader:
                imgs = imgs.to(device)
                out = model(imgs)
                f = out[1] if isinstance(out, tuple) else out
                feats.append(f.cpu())
                pids.extend(batch_pids.numpy())
                camids.extend(batch_camids.numpy())
        return torch.cat(feats, 0).numpy(), np.array(pids), np.array(camids)

    print('Extracting query features...')
    qf, q_pids, q_camids = extract_features(queryloader)
    print(f'Query matrix: {qf.shape}')

    print('Extracting gallery features...')
    gf, g_pids, g_camids = extract_features(galleryloader)
    print(f'Gallery matrix: {gf.shape}')

    # Normalise
    qf = qf / np.linalg.norm(qf, axis=1, keepdims=True)
    gf = gf / np.linalg.norm(gf, axis=1, keepdims=True)

    # Standard evaluation — chunked distance matrix
    print('\n--- Standard Evaluation (no re-ranking) ---')
    distmat = compute_distmat(qf, gf, chunk_size=256)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print(f'Rank-1 : {cmc[0]*100:.1f}%')
    print(f'Rank-5 : {cmc[4]*100:.1f}%')
    print(f'Rank-10: {cmc[9]*100:.1f}%')
    print(f'mAP    : {mAP*100:.1f}%')

    # Re-ranking
    print('\n--- Re-ranking Evaluation (Zhong et al. CVPR 2017) ---')
    q_q_dist = np.dot(qf, qf.T)
    g_g_dist = np.dot(gf, gf.T)
    q_g_dist = np.dot(qf, gf.T)

    print('Computing re-ranking (1-2 mins)...')
    final_dist = re_ranking(
        q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)
    cmc_rr, mAP_rr = evaluate(
        final_dist, q_pids, g_pids, q_camids, g_camids)
    print(f'Rank-1 : {cmc_rr[0]*100:.1f}%')
    print(f'Rank-5 : {cmc_rr[4]*100:.1f}%')
    print(f'Rank-10: {cmc_rr[9]*100:.1f}%')
    print(f'mAP    : {mAP_rr*100:.1f}%')

    print('\n--- Improvement from Re-ranking ---')
    print(f'Rank-1: {cmc[0]*100:.1f}% -> {cmc_rr[0]*100:.1f}% '
          f'({(cmc_rr[0]-cmc[0])*100:+.1f}%)')
    print(f'mAP:    {mAP*100:.1f}% -> {mAP_rr*100:.1f}% '
          f'({(mAP_rr-mAP)*100:+.1f}%)')

if __name__ == '__main__':
    main()

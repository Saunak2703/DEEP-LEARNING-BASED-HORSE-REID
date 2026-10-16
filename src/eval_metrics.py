# Copyright (c) EEEM071, University of Surrey

import numpy as np


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Compute CMC and mAP for the horse Re-ID query/gallery split."""
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]

        # Remove only exact same-identity same-camera entries. The horse dataset
        # assigns query and gallery different pseudo camera IDs, so valid
        # single-camera matches remain evaluable.
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]

        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i + 1.0) for i, x in enumerate(tmp_cmc)])
        all_ap.append((tmp_cmc * raw_cmc).sum() / num_rel)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    m_ap = np.mean(all_ap)

    return all_cmc, m_ap

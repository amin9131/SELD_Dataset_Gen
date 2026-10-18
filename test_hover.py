# import numpy as np
# from trajectory import HoverJitterTrajectory

# j, T = 1.2, 12.0
# trajs = [HoverJitterTrajectory([0, 0, 0], T, jitter_std_m=j,
#                                rng=np.random.default_rng(s)) for s in range(500)]
# for t in (0.5, 3, 6, 12):
#     pts = np.array([tr.position_at(t) for tr in trajs])
#     print(f"t={t:5.1f}s  std per axis = {pts.std(axis=0).round(2)}   (target {j})")

# import numpy as np
# from feature_extraction import FeatureConfig
# from feature_normalization import FeatureNormalizer

# if __name__ == "__main__":
#     fc = FeatureConfig()
#     a = FeatureNormalizer().fit("output/train", fc, max_chunks=40, n_workers=1, verbose=False)
#     b = FeatureNormalizer().fit("output/train", fc, max_chunks=40, n_workers=4, verbose=False)
#     print(np.allclose(a.logmel_mean, b.logmel_mean), np.allclose(a.logmel_std, b.logmel_std))
#     print(np.allclose(a.gcc_inter_mean, b.gcc_inter_mean), np.allclose(a.gcc_intra_std, b.gcc_intra_std))

import torch
from feature_extraction import FeatureConfig
from feature_normalization import FeatureNormalizer
from torch_dataset import SELDDataset

if __name__ == "__main__":
    norm = FeatureNormalizer.load("output/train/feature_norm_stats.json")
    fc = FeatureConfig()
    a = SELDDataset("output/train", fc, 3, normalizer=norm, use_cache=True)[0]
    b = SELDDataset("output/train", fc, 3, normalizer=norm, use_cache=False)[0]
    for k in a:
        if k != "chunk_dir":
            print(k, torch.allclose(a[k].float(), b[k].float(), atol=1e-4))
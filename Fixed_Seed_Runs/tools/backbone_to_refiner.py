#!/usr/bin/env python
"""Put a frozen backbone's raw-count predictions into the layout V16TK_train.py / V16TK_test.py read.

Copy of 261105_DFHiC_comps/scripts/dfhic_to_refiner.py with one change: the backbone name in the file names
is the --name argument instead of the fixed word 'dfhic'. The normalization code is untouched.

Normalization (same form as the repo backbones): log1p, then (x - center) / scale, then clip to [-5, 5].
center, scale = median and IQR (no 1.35 factor) of the NONZERO log1p values of the GM12878 /16
training HR tiles. Fitted once and used for every cell line, as in the July runs.

Writes under --out:
  norm_stats.pt                              {'Y_mean': center, 'Y_std': scale} (names the test script reads)
  bb_train/norm/predictions.npy, ground_truth.npy          training tiles, normalized, (N,1,40,40)
  bb_test/chrN/predictions_norm.npy, ground_truth.npy      test tiles per chromosome, normalized
  <name>_raw/preds_lr_test_chrN_ratio<R>.npy               backbone raw counts (negatives set to 0), to score the backbone alone
Run in the 'chromdiffusion' env. Written by Tomoya Kanno, 2026-11-05; --name added 2026-11-09.
"""
import argparse, os
import numpy as np, torch

p = argparse.ArgumentParser()
p.add_argument('--tiles', required=True, help="the condition's tile folder")
p.add_argument('--ratio', required=True, help='ratio label used in the file names: 16, 7p5, 6p8')
p.add_argument('--name', required=True, help='backbone name used in the file names, e.g. capricorn')
p.add_argument('--pred', required=True, help='folder with <name>_train_raw.npy and <name>_test_raw.npy')
p.add_argument('--out', required=True)
p.add_argument('--stats_from', required=True, help='GM12878 train_data_raw_ratio16.npz')
a = p.parse_args()

hr = np.log1p(np.load(a.stats_from)['train_hr'])
nz = hr[hr > 0]
center, (q25, q75) = float(np.median(nz)), np.percentile(nz, [25, 75])
scale = float(q75 - q25)
assert scale > 0.1, f'scale collapsed: {scale}'
print(f'center {center:.4f}, scale {scale:.4f}, from {nz.size} nonzero HR values')

def norm(counts):  # (N,40,40,1) counts -> (N,1,40,40) normalized
    z = (np.log1p(np.clip(counts, 0, None)) - center) / scale
    return np.clip(z, -5, 5).transpose(0, 3, 1, 2).astype(np.float32)

def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

os.makedirs(a.out, exist_ok=True)
torch.save({'Y_mean': center, 'Y_std': scale}, os.path.join(a.out, 'norm_stats.pt'))

train = np.load(os.path.join(a.tiles, f'train_data_raw_ratio{a.ratio}.npz'))
pred = np.load(os.path.join(a.pred, f'{a.name}_train_raw.npy'))
assert pred.shape == train['train_hr'].shape
save(f'{a.out}/bb_train/norm/predictions.npy', norm(pred))
save(f'{a.out}/bb_train/norm/ground_truth.npy', norm(train['train_hr']))

test = np.load(os.path.join(a.tiles, f'test_data_raw_ratio{a.ratio}.npz'))
pred = np.load(os.path.join(a.pred, f'{a.name}_test_raw.npy'))
for c in [f'chr{n}' for n in range(18, 23)]:
    m = test['distance'][:, 1] == c
    # the scoring step rebuilds tile positions from the per-chromosome LR files; check they are the same tiles
    assert np.array_equal(test['test_lr'][m], np.load(os.path.join(a.tiles, f'lr_test_{c}_ratio{a.ratio}.npy')))
    save(f'{a.out}/bb_test/{c}/predictions_norm.npy', norm(pred[m]))
    save(f'{a.out}/bb_test/{c}/ground_truth.npy', norm(test['test_hr'][m]))
    save(f'{a.out}/{a.name}_raw/preds_lr_test_{c}_ratio{a.ratio}.npy', np.clip(pred[m], 0, None))
    print(f'{c}: {m.sum()} tiles')

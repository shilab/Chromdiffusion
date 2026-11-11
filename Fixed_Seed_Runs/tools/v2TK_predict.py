#!/usr/bin/env python
"""Run one trained v2TK backbone on a condition's tiles and write its predictions as raw counts.

Writes, in the same tile order and shape (N,40,40,1) as the input files (the layout backbone_to_refiner.py reads):
  <out>/<backbone>_train_raw.npy   from train_data_raw_ratio<R>.npz (train_lr)
  <out>/<backbone>_test_raw.npy    from test_data_raw_ratio<R>.npz  (test_lr)
The network class comes from the backbone's own v2TK training script, and the checkpoint is the one that trainer
picked on its validation tiles (best PCC; HiCARN: best SSIM). The normalization numbers saved with that checkpoint
are applied the way all four trainers apply them: log1p, (x - median) / IQR, clip to [-5, 5] on the way in;
clip to [-5, 5], x * IQR + median, expm1, negatives set to 0 on the way out.
Run in the 'chromdiffusion' env on one GPU. Written by Tomoya Kanno, 2026-11-10.
"""
import argparse, glob, importlib.util, os, sys
import numpy as np, torch


class RobustHiCPreprocessor:  # stand-in, so the preprocessor object pickled inside HiCARN's checkpoint can be read
    pass


p = argparse.ArgumentParser()
p.add_argument('--backbone', required=True, choices=['modelA_CNN', 'modelB_ResNet', 'modelC_GAN15', 'hicarn_robust'])
p.add_argument('--run', required=True, help="the output folder of the backbone's training run")
p.add_argument('--tiles', required=True, help='folder holding train_/test_data_raw_ratio<R>.npz')
p.add_argument('--ratio', required=True, help='ratio label used in the file names, e.g. 16')
p.add_argument('--out', required=True)
p.add_argument('--scripts', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backbones'))
a = p.parse_args()
sys.path.insert(0, a.scripts)


def trainer(name):  # the v2TK training script as a module; its training code sits under __main__ and does not run
    spec = importlib.util.spec_from_file_location(name, os.path.join(a.scripts, f'v2TK_train_{name}.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


if a.backbone == 'modelA_CNN':      # HiCNN
    ckpt = load(f'{a.run}/best_model.pt'); cfg = ckpt['config']
    assert not cfg.get('original_arch'), 'expected the 40x40-output HiCNN'
    net = trainer('modelA_CNN').HiCNN(num_res_blocks=cfg['num_res_blocks'], num_features=cfg['num_features'],
                                      output_full_size=True)
    weights, s = ckpt['model_state_dict'], ckpt['preprocessor']
    stats = s['X_mean'], s['X_std'], s['Y_mean'], s['Y_std']
elif a.backbone == 'modelB_ResNet':  # SRHiC; its checkpoints keep no settings, and run_backbone.sh used the defaults
    ckpt = load(f'{a.run}/best_pcc.pt')
    net = trainer('modelB_ResNet').SRHiC(in_channels=1, out_channels=1, num_features=64, num_residual_blocks=16)
    weights, s = ckpt['model_state_dict'], ckpt['preprocessor_stats']
    stats = s['X_median'], s['X_iqr'], s['Y_median'], s['Y_iqr']
elif a.backbone == 'modelC_GAN15':  # DeepHiC generator
    ckpt = load(f'{a.run}/best_model.pt'); cfg = ckpt['config']
    net = trainer('modelC_GAN15').Generator(in_channels=1, out_channels=1, num_features=cfg['num_features'],
                                            num_res_blocks=cfg['num_res_blocks'])
    weights, s = ckpt['generator_state_dict'], ckpt['preprocessor']
    stats = s['X_mean'], s['X_std'], s['Y_mean'], s['Y_std']
else:                                # HiCARN-1
    from Models.HiCARN_1 import Generator
    (f,) = glob.glob(f'{a.run}/checkpoints_robust/*_bestg_robust_HiCARN_1.pytorch')
    ckpt = load(f)
    net = Generator(num_channels=64)
    weights, s = ckpt['model_state_dict'], ckpt['preprocessor']
    stats = s.X_mean, s.X_std, s.Y_mean, s.Y_std
Xm, Xs, Ym, Ys = (float(v) for v in stats)
net.load_state_dict(weights)  # strict: stops if the network does not match the checkpoint
dev = torch.device('cuda:0')
net.to(dev).eval()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print(f'{a.backbone}: epoch {ckpt.get("epoch")}, LR {Xm:.4f}/{Xs:.4f}, HR {Ym:.4f}/{Ys:.4f}')


def predict(lr):  # (N,40,40,1) counts -> normalized output (N,1,40,40)
    assert lr.ndim == 4 and lr.shape[-1] == 1, lr.shape
    x = np.clip((np.log1p(lr.transpose(0, 3, 1, 2).astype(np.float64)) - Xm) / Xs, -5, 5).astype(np.float32)
    out = np.empty_like(x)
    with torch.no_grad():
        for i in range(0, len(x), 512):
            out[i:i + 512] = net(torch.from_numpy(x[i:i + 512]).to(dev)).float().cpu().numpy()
    return np.clip(out, -5, 5)


os.makedirs(a.out, exist_ok=True)
for split in ('train', 'test'):
    d = np.load(os.path.join(a.tiles, f'{split}_data_raw_ratio{a.ratio}.npz'))
    z = predict(d[f'{split}_lr'])
    counts = np.maximum(np.expm1(z.astype(np.float64) * Ys + Ym), 0).transpose(0, 2, 3, 1).astype(np.float32)
    np.save(os.path.join(a.out, f'{a.backbone}_{split}_raw.npy'), counts)
    hr = np.clip((np.log1p(d[f'{split}_hr'].transpose(0, 3, 1, 2).astype(np.float64)) - Ym) / Ys, -5, 5)
    r = np.corrcoef(z.ravel(), hr.ravel())[0, 1]  # a quick check: agreement with full depth, normalized scale
    print(f'{split}: {len(z)} tiles, counts {counts.min():.2f} to {counts.max():.1f}, PCC with full depth {r:.4f}')

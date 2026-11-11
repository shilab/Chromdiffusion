# Fixed_Seed_Runs: repeatable training and testing

This folder holds fixed-seed copies of the ChromDiffusion refiner scripts and of the four
backbone trainers. With the same seed, GPU type and software stack, a run gives the same
model and the same refined maps every time. The copies were written after the manuscript
was submitted. The manuscript's results come from the original scripts listed in the root
README under "Scripts Used in the Manuscript".

The originals in the backbone folders are not changed. Each copy says in its header which
file it was copied from, and every added or edited line ends with a tag (`# V16TK`,
`# V17TK` or `# v2TK`). To see the changes, grep for the tag or diff the copy against its
original, for example `diff srhic/train_v15.py Fixed_Seed_Runs/refiner/V16TK_train.py`.

## Contents

| File | Copied from | What changed |
|---|---|---|
| `refiner/V16TK_train.py` | `srhic/train_v15.py` (identical in all four backbone folders) | `--seed`; repeatable GPU settings; one line of training losses per epoch in `epoch_train_losses.jsonl`; `torch.load(..., weights_only=False)` for `--resume` on newer PyTorch |
| `refiner/V17TK_train.py` | `refiner/V16TK_train.py` | The validation check uses the run's seed instead of a fixed 42, and the training random state is saved before each check and restored after it (3 lines) |
| `refiner/V16TK_test.py` | `test_v15_strratio.py` (see below) | `--seed`, with each chromosome seeded as seed + chromosome number; repeatable GPU settings; `weights_only=False`; the seed is recorded in `evaluation_results.json` |
| `backbones/v2TK_train_modelA_CNN.py` | `HiCNN/train_hicnn.py` | see "Backbone trainers" |
| `backbones/v2TK_train_modelB_ResNet.py` | `srhic/train_srhic.py` | see "Backbone trainers" |
| `backbones/v2TK_train_modelC_GAN15.py` | `deephic/train_deephic.py` | see "Backbone trainers" |
| `backbones/v2TK_train_hicarn_robust.py` | `HiCARN/train_hicarn_robust.py` | see "Backbone trainers" |
| `tools/v2TK_predict.py` | new | Runs a trained backbone on the training and test tiles and writes its predictions as raw counts |
| `tools/backbone_to_refiner.py` | new | Normalizes a backbone's predictions and writes them in the layout the refiner scripts read |

`test_v15_strratio.py` is not in this repository. It is `srhic/test_v15.py` with `--ratio`
read as text instead of an integer, so fractional depth labels such as `7p5` can be used
in file names. That is its only difference.

## Seeds

- `--seed 17` is the default. Any whole number from 0 to 4,294,967,295 gives a repeatable run.
- `--seed none` turns off all seeding and the repeatable GPU settings, so the script behaves
  like its original.
- A seeded run sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.benchmark = False`,
  `cudnn.deterministic = True` and `torch.use_deterministic_algorithms(True)`.
- `V16TK_test.py` seeds each chromosome separately (seed + chromosome number). A chromosome
  therefore gives the same result whether it is refined alone or together with others.
  Use well-spaced seeds for repeats (for example 17, 117, 217), so that one run's chr19 seed
  is not the next run's chr18 seed.
- In the original `train_v15.py` (and in `V16TK_train.py`), every validation check resets
  torch's random generator to 42. Validation runs at epoch 1 and every 5 epochs, so after
  each check training replays the same tile order and noise. `V17TK_train.py` fixes this.
  As a result, `V16TK_train.py` and `V17TK_train.py` give different models from the same
  seed. Use `V17TK_train.py` for new runs. Testing needs no V17 version.
- Exact repeats need the same GPU type and software stack. The runs behind this folder used
  Python 3.11, PyTorch 2.9.1 with CUDA 12.8 and numpy 1.26 on an NVIDIA RTX PRO 6000
  Blackwell GPU.

## Backbone trainers

All four trainers were changed in the same ways:

- One shared normalization: log1p, then the median and IQR of the nonzero values (without
  the 1.35 factor), an IQR floor of 0.1, and a clip to ±5. The HiCARN trainer now fits these
  numbers on all tiles, like the other three.
- `--seed` (default 17; `none` gives the old unseeded behaviour). The seed fixes the
  starting weights, the shuffle order and the random train/validation split (HiCNN, SRHiC
  and DeepHiC draw one; HiCARN validates on the last 10% of the file), and turns on the
  repeatable GPU settings. `--resume` reloads the weights and optimizer but not the random
  state, so a resumed run differs from an uninterrupted one after the restart.
- One JSON line per epoch in `epoch_log.jsonl`. Two runs with the same settings should
  have the same first line.
- Fixes needed to run on PyTorch 2.9.

Things to know when running them:

- HiCNN (`modelA`) defaults to 200 epochs and DeepHiC (`modelC`) to 300. Pass `--epochs 100`
  to match our runs.
- SRHiC (`modelB`) needs `--device cuda:0`. Its default, `cuda:3`, comes from the original
  script, and plain `cuda` fails on PyTorch 2.9.
- HiCARN has 100 epochs and batch size 64 fixed in the script. It writes
  `./checkpoints_robust` in the working directory, and it imports `Models/` and `Utils/`
  from `../HiCARN`, so put that folder on `PYTHONPATH`. Its perceptual loss downloads the
  VGG16 weights through torchvision on first use (`TORCH_HOME` chooses where).
- The validation numbers in `epoch_log.jsonl` are not comparable across backbones, because
  each trainer scores a different set of validation tiles with a different measure. Compare
  backbones by scoring their test-chromosome predictions with `../benchmark`.

`tools/v2TK_predict.py` loads the checkpoint each trainer picked on its validation tiles:
`best_model.pt` for HiCNN and DeepHiC, `best_pcc.pt` for SRHiC, and
`checkpoints_robust/*_bestg_robust_HiCARN_1.pytorch` for HiCARN.

## Refiner inputs

`tools/backbone_to_refiner.py` puts a backbone's raw-count predictions into the refiner's
normalized space: log1p, then (x − median) / IQR, then a clip to ±5. The median and IQR
are those of the nonzero log1p values of the GM12878 ÷16 training HR tiles (`--stats_from`).
They are fitted once and used for every cell line. The script writes `norm_stats.pt`,
`bb_train/norm/` (training tiles), `bb_test/chrN/` (test tiles for chr18–22), and
`<name>_raw/` (the backbone's own predictions, ready for the benchmark).

The refiner is tested with the checkpoint that scored best on the training script's
built-in validation overlap score, `best_model_iou.pt`.

## Requirements

Python 3, PyTorch with CUDA, torchvision, numpy, scipy, tqdm, matplotlib and scikit-image.
Training tiles and test tiles come from `../preprocessing`. Scoring uses `../benchmark`.

## Examples

- `../benchmark/examples/run_fixed_seed_backbone.sh` trains one backbone with a fixed seed.
- `../benchmark/examples/run_fixed_seed_refiner.sh` takes a trained backbone and trains a
  refiner on its predictions once per seed. It refines chr18–22 with test seed 17 for
  every run, so the runs differ only in training, and it scores each run with stages 1–3
  of the benchmark.

## Provenance

Written by Tomoya Kanno, November 2026.

# Part 1 — Policy Gradient Training Guide

Everything needed to train or continue training the RL Connect-4 bot
using Policy Gradient (REINFORCE) on Google Colab.

## Folder contents

```
part1_pg/
  train_colab.ipynb        ← open this in Colab to train
  train_local.py           ← run locally: python part1_pg/train_local.py
  COLAB_GUIDE.md           ← you are here
  best_model/
    m1_pg_final.keras      ← best trained model (~450 iters, 87% vs Strong)
  checkpoints/
    m1_iter100.keras       ← 100 iters
    m1_iter200.keras       ← 200 iters (89% vs Strong — best checkpoint)
    m1_iter300.keras       ← ~500 total iters
    TRAINING_HISTORY.md    ← full results table
```

## How to run on Colab

### 1. Open the notebook
Go to [colab.research.google.com](https://colab.research.google.com) →
**File → Open notebook → GitHub** → paste the repo URL → select
`part1_pg/train_colab.ipynb`.

### 2. Set runtime to GPU
**Runtime → Change runtime type → T4 GPU** before running anything.

### 3. Upload these 5 files when prompted (Step 2 cell)
```
connect4_env.py                         ← repo root
models/loader.py                        → upload as "loader.py"
models/josh_cnn.h5                      → upload as "josh_cnn.h5"
part1_pg/checkpoints/m1_iter200.keras   → upload as "m1_iter200.keras"
part1_pg/best_model/m1_pg_final.keras   → upload as "m1_pg_final.keras"
```

### 4. Mount Google Drive (Step 3b)
Auto-saves every checkpoint to Drive so a session crash never wipes progress.

### 5. Run all cells
300 iterations (~45 min). Win rates print every 50 iters.
Checkpoints auto-save every 100 iters to Drive.

### 6. Download results (Step 12)
Downloads `m1_pg_final.keras` + training curve plot.
Replace `best_model/m1_pg_final.keras` with the new file and push to GitHub.

---

## Training details

| Setting | Value |
|---|---|
| Algorithm | REINFORCE (on-policy) |
| Batch size | 64 (fixed — do not change) |
| Learning rate | 1e-4 (Adam) |
| Discount γ | 0.99 |
| Entropy coef | 0.01 |
| Random init moves | 3 per game |

## Training history

| Model | Total Iters | vs StrongRule | vs Orig M1 |
|---|---|---|---|
| `josh_cnn.h5` | 0 (baseline) | ~50% | — |
| `checkpoints/m1_iter200.keras` | 200 | 89% | 95% |
| `best_model/m1_pg_final.keras` | ~450 | 87% | 95% |

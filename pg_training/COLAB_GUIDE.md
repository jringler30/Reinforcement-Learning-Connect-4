# Policy Gradient Training — Colab Guide

This folder contains everything needed to train or continue training the
RL Connect-4 bot using Policy Gradient (REINFORCE) on Google Colab.

## Contents

| File | Purpose |
|---|---|
| `m1_pg_final.keras` | Best trained model (~450 total PG iterations, 87%+ vs StrongRuleAgent) |
| `part1_policy_gradient_colab.ipynb` | Training notebook — open this in Colab |

## How to run on Colab

### 1. Open the notebook
Go to [colab.research.google.com](https://colab.research.google.com), click
**File → Open notebook → GitHub**, paste the repo URL and select
`pg_training/part1_policy_gradient_colab.ipynb`.

### 2. Set the runtime to GPU
**Runtime → Change runtime type → T4 GPU** — do this before running anything.

### 3. Upload these files when prompted (Step 2 cell)
From this repo, upload:
```
connect4_env.py                     ← from repo root
models/loader.py                    → upload as "loader.py"
models/josh_cnn.h5                  → upload as "josh_cnn.h5"
pg_checkpoints/m1_iter200.keras     → upload as "m1_iter200.keras"
pg_training/m1_pg_final.keras       → upload as "m1_pg_final.keras"
```

### 4. Mount Google Drive (Step 3b cell)
This auto-saves every checkpoint to your Drive so a session crash
doesn't wipe your progress. Highly recommended.

### 5. Run all cells
Training runs for 300 iterations (~45 min on T4 GPU), printing win rates
every 50 iterations. Checkpoints save every 100 iterations.

### 6. Download the trained model (Step 12 cell)
At the end, run the download cell to save `m1_pg_final.keras` and the
training curve plot to your machine.

## Training details

- **Algorithm:** REINFORCE (on-policy policy gradient)
- **M1:** `m1_pg_final.keras` — the model being improved
- **M2 opponent pool:** StrongRuleAgent, RandomAgent, original josh_cnn,
  iter200 snapshot, pg_final snapshot — randomly sampled each iteration
- **Batch size:** 64 (fixed — changing this will slow TF down)
- **Discount factor γ:** 0.99
- **Learning rate:** 1e-4 (Adam)

## Current training history

| Model | Total Iters | vs StrongRuleAgent |
|---|---|---|
| `josh_cnn.h5` (original) | 0 | ~50% |
| `m1_iter200.keras` | 200 | 89% |
| `m1_pg_final.keras` | ~450 | 87% |

## To continue training further

The notebook is already set up to resume from `m1_pg_final.keras`.
Just upload the files and run — it picks up where it left off.
After training, replace `m1_pg_final.keras` in this folder with the
new version and push to GitHub.

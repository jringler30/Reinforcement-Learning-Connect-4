# Reinforcement-Learning-Connect-4

Project 3 for RM294 — self-play reinforcement learning for Connect 4.
Trains a Policy Gradient agent, compares it against DQN and rule-based bots,
and culminates in a class tournament.

> **Teammates: read [`TEAMMATE_GUIDE.md`](TEAMMATE_GUIDE.md) before integrating your model.**
> It's 30 seconds and shows exactly how to wrap your trained network so it drops into the tournament system.
>
> Assignment spec: [`project3_overview.pdf`](project3_overview.pdf)

---

## Repo layout

```
Reinforcement-Learning-Connect-4/
│
├── connect4_env.py          ← game engine (board logic, agents, evaluator)
├── tournament.py            ← full 32-team tournament simulator
├── test_smoke.py            ← sanity checks — run first after any change
├── play_vs_bot.ipynb        ← play against the trained bot in Colab
│
├── models/                  ← all Project 1 baseline CNNs
│   ├── josh_cnn.h5          ← Josh's model (Keras)
│   ├── cnn_emily.pt         ← Emily's model (PyTorch)
│   ├── cnn_prisca.pt        ← Prisca's model (PyTorch)
│   ├── loader.py            ← load_model("name") — auto-detects framework
│   └── README.md            ← model I/O spec + naming convention
│
└── part1_pg/                ← Policy Gradient training (Steps 1–3)
    ├── train_colab.ipynb    ← open in Colab to train on GPU
    ├── train_local.py       ← run locally: python part1_pg/train_local.py
    ├── COLAB_GUIDE.md       ← full guide for Colab training + Drive save
    ├── best_model/
    │   └── m1_pg_final.keras  ← best trained model (~450 iters, 87% vs Strong)
    └── checkpoints/
        ├── m1_iter100.keras
        ├── m1_iter200.keras   ← 89% vs Strong — best checkpoint
        ├── m1_iter300.keras
        └── TRAINING_HISTORY.md
└── part4_dqn/                             ← DQN training (Step 4)
    ├── part4_dqn_colab.ipynb              ← original Colab notebook
    ├── part4_dqn_local.ipynb              ← original local notebook
    ├── best_model/                        ← original training run
    │   ├── dqn_best.keras
    │   ├── dqn_final.keras
    │   ├── dqn_training_curves.png
    │   └── dqn_vs_ddqn_comparison.png
    └── updated_dqn/                       ← updated run with bug fixes + continuation
        ├── part4_dqn_updated_colab.ipynb  ← fixed next_state bug, faster epsilon decay
        ├── dqn_best.keras                 ← run 1 best: 75% vs Strong (5000 episodes)
        ├── dqn_best_cont.keras            ← continuation best: use for tournament
        ├── dqn_training_curves.png        ← run 1 training curves (episodes 1-5000)
        ├── dqn_continuation_curves.png    ← continuation run (episodes 5001-8001)
        └── dqn_summary_bar.png            ← best vs final win rates across all opponents
        
        Note: episode checkpoints (dqn_ep*.keras) stored in Google Drive only
        due to file size — available on request
```

---

## Part split

| Part | Owner | Status |
|---|---|---|
| 1 — Policy Gradient | Josh | Done — see `part1_pg/` |
| 2 — DQN | Prisca (original) + Emily (updated) | Done — see `part4_dqn/` |
| 3 — Game Engine + Tournament | Josh | Done — `connect4_env.py` + `tournament.py` |
| 4 — Evaluation + Report | All | Pending |

---

## Quick start

```bash
python test_smoke.py            # run all sanity checks (~5 sec)
python tournament.py            # demo: match + round-robin pool + bracket
python part1_pg/train_local.py  # local PG training (slow without GPU)
```

To train on GPU: open `part1_pg/train_colab.ipynb` in Google Colab.
See `part1_pg/COLAB_GUIDE.md` for the full walkthrough.

---

## Agent interface

All agents subclass `Agent` and implement `select_move(board, player) -> col`:

```python
from connect4_env import RandomAgent, StrongRuleAgent, ModelAgent
from models.loader import load_model

RandomAgent()                                        # random legal move
StrongRuleAgent()                                    # wins/blocks if possible
ModelAgent(load_model("josh_cnn"), sample=False, strong=True)  # trained NN
```

Drop any agent into a match, pool, or bracket — the tournament code is framework-agnostic.

---

## Tournament rules (per assignment)

- 32 teams → 8 pools of 4 → round-robin (6 games/team)
- Every match = 2 games; each side goes first once
- Standings: wins desc → avg moves-to-win asc → coin flip
- 1st/2nd/3rd/4th from each pool → brackets A/B/C/D (single-elim)
- Bracket winners → semis → final

```python
from tournament import run_full_tournament
results = run_full_tournament(agents, n_pools=8, pool_size=4)
```

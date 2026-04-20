# Policy Gradient Checkpoints

These are snapshots of `josh_cnn` after being improved via REINFORCE
(Policy Gradient) training. Each file is a full Keras model with the
same architecture and I/O as the original `models/josh_cnn.h5`.

## Training results

| Checkpoint | Total Iters | vs StrongRuleAgent | vs Original M1 | Notes |
|---|---|---|---|---|
| `m1_iter100.keras` | 100 | 83% | 98% | Run 1, starting from josh_cnn |
| `m1_iter200.keras` | 200 | 89% | 95% | Run 1 |
| `m1_iter300.keras` | ~500 | 88% | 95% | Run 2, starting from iter200 |
| `m1_pg_final.keras` | ~450 | 87% | 95% | Best available final model |

**Use `m1_pg_final.keras` for the tournament** — it has the most total training.

## What these models are

- **Input:** `(batch, 6, 7, 2)` float32 — same as original CNN
- **Output:** `(batch, 7)` softmax — same as original CNN
- **Difference from original:** weights updated via REINFORCE against an
  evolving opponent pool (StrongRuleAgent + RandomAgent + self-play snapshots)

## How to load and use

```python
from models.loader import load_agent

# Load a checkpoint as a ready-to-play agent
agent = load_agent_from_path("pg_checkpoints/m1_iter200.keras")
```

Or load directly with Keras:

```python
from tensorflow import keras
from connect4_env import ModelAgent

model = keras.models.load_model("pg_checkpoints/m1_iter200.keras", compile=False)
agent = ModelAgent(model, sample=False, strong=True)
```

Then use anywhere:

```python
from connect4_env import evaluate_agents, StrongRuleAgent
results = evaluate_agents(agent, StrongRuleAgent(), n_games=100)
print(results)  # {'wins': 89, 'losses': 9, 'draws': 2, 'win_rate': 0.89}
```

## How to continue training from a checkpoint

Open `part1_policy_gradient_colab.ipynb` in Colab and change cell 6 from:

```python
m1_model = load_model("josh_cnn")   # starts from original
```

to:

```python
m1_model = keras.models.load_model("pg_checkpoints/m1_iter200.keras", compile=False)
```

This resumes training from where it left off instead of starting fresh.

## Training setup

- **Algorithm:** REINFORCE (Policy Gradient)
- **Batch size:** 64 (fixed)
- **Learning rate:** 1e-4 (Adam)
- **Discount factor γ:** 0.99
- **Opponent pool:** StrongRuleAgent, RandomAgent, original M1, + periodic
  self-play snapshots added every 100 iterations
- **M2 rule:** strong=True (wins/blocks) — M1 only sees legal moves
- **Random init moves per game:** 3 (board diversity)

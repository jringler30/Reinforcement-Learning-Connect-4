# Shared Project 1 Models

This folder holds each teammate's **Project 1 CNN** so we can use them as
opponents (M2 candidates) during PG/DQN training and in the final tournament.

## Naming convention

```
<firstname>_<arch>.h5
```

Examples:
- `josh_cnn.h5` ✅
- `taylor_cnn.h5`
- `jordan_transformer.h5`

Lowercase. Keep it short.

## What each file must be

- A **Keras** model (`.h5` or `.keras`)
- Input shape: `(batch, 6, 7, 2)` — 2-channel board (yours / opponent)
- Output shape: `(batch, 7)` — one score per column (softmax is fine)

If your Project 1 model used different I/O, wrap it so it matches before saving here.

## Loading any model in your code

```python
from models.loader import load_agent

josh = load_agent("josh_cnn", sample=False, strong=True)
taylor = load_agent("taylor_cnn", sample=False, strong=True)

# Use directly in any tournament / eval function:
from connect4_env import evaluate_agents
print(evaluate_agents(josh, taylor, n_games=100))
```

## Size note

Keep models under ~25MB each. GitHub's soft limit is 50MB per file
and the hard limit is 100MB. If your model is larger, consider Git LFS
or just pruning.

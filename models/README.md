# Shared Project 1 Models

This folder holds each teammate's **Project 1 model** so we can use them as
opponents (M2 candidates) during PG/DQN training and in the tournament.

Both **Keras** (`.h5`, `.keras`) and **PyTorch** (`.pt`, `.pth`) models are
supported — pick whichever framework you trained in.

## Naming convention

```
<firstname>_<arch>.<ext>
```

Examples:
- `josh_cnn.h5`          ← Keras
- `emily_cnn.pt`         ← PyTorch
- `prisca_transformer.pt`

Lowercase. Keep it short.

## Input / output contract

Your model must accept a batched 2-channel Connect-4 board and return one
score per column.

|                | Input shape | Output shape |
|---|---|---|
| **Keras**      | `(batch, 6, 7, 2)` (channels-last) | `(batch, 7)` |
| **PyTorch**    | `(batch, 2, 6, 7)` (channels-first, default) — or `(batch, 6, 7, 2)` if you set `channels_first=False` when loading | `(batch, 7)` |

Channel 0 = your pieces, channel 1 = opponent's.
Output can be logits, softmax, or tanh — the wrapper treats it as logits
and masks illegal columns.

## PyTorch: save the FULL model, not just the state_dict

```python
# ✅ Correct — saves the whole module so it can be loaded without arch code
torch.save(model, "emily_cnn.pt")

# ❌ Wrong — saves only weights, can't load without the class definition
torch.save(model.state_dict(), "emily_cnn.pt")
```

## Loading any model

```python
from models.loader import load_agent, list_available

print(list_available())           # ['josh_cnn', 'emily_cnn', ...]

josh  = load_agent("josh_cnn")                     # Keras, auto-detected
emily = load_agent("emily_cnn")                    # PyTorch, channels-first
# If your PyTorch model was trained with channels-last:
other = load_agent("other_cnn", channels_first=False)
```

Then use anywhere:

```python
from connect4_env import evaluate_agents
print(evaluate_agents(josh, emily, n_games=100))
```

## Size note

Keep models under ~25 MB each. GitHub's soft limit is 50 MB per file.
If yours is larger, consider pruning before upload.

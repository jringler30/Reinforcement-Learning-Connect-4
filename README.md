# Reinforcement-Learning-Connect-4

Project 3 for RM294 — self-play reinforcement learning for Connect 4.
Compares Policy Gradient, DQN, and rule-based agents; culminates in a
class tournament.

> **Teammates: read [`TEAMMATE_GUIDE.md`](TEAMMATE_GUIDE.md) before integrating your model.**
> It's 30 seconds and shows exactly how to wrap your trained network so it drops into the tournament system.

## Repo layout

| File | Part | Purpose |
|---|---|---|
| `connect4_env.py` | 3 | Board logic, win detection, tactical helpers, neural-net encoding, unified `Agent` interface, game runner. No external deps beyond numpy. |
| `tournament.py` | 3 | Match runner (alternates first player), round-robin pool, single-elim bracket, and full 32-team tournament simulator with standings + tiebreakers. |
| `test_smoke.py` | 3 | Fast sanity checks for everything above. Run: `python test_smoke.py`. |

## Part split

1. **Policy Gradient** — train M1 via self-play vs M2 (Steps 1–3 of the spec).
2. **DQN** — epsilon-greedy + replay buffer + target network (Step 4).
3. **Game Engine + Tournament System** — this repo's `connect4_env.py` + `tournament.py` (done).
4. **Evaluation + Report** — PG vs DQN vs MCTS comparison, plots, write-up.

## Agent interface

All agents subclass `Agent` and implement `select_move(board, player) -> col`:

```python
from connect4_env import Agent, RandomAgent, StrongRuleAgent, ModelAgent

# Rule-based stand-ins (great for testing before PG/DQN are ready):
RandomAgent()         # picks a uniformly random legal move
StrongRuleAgent()     # wins/blocks if possible, else avoids losing moves
ModelAgent(keras_model, sample=True, strong=False)  # wraps a trained NN
```

Drop any of these into a match/pool/bracket — the tournament code doesn't
care whether it's a PG network, a DQN, or a rule-based bot.

## Quick start

```bash
python test_smoke.py            # run all sanity checks
python tournament.py            # demo: match + pool + bracket
```

## Tournament rules (per assignment)

- 32 teams → 8 pools of 4 → round-robin (6 games / team)
- Every match = 2 games; each side goes first once
- Pool standings: wins desc, then avg moves-to-win asc (fewer = better)
- 1st/2nd/3rd/4th → brackets A/B/C/D (single-elim)
- Bracket winners → semis (A vs B, C vs D) → final

`run_full_tournament(agents, n_pools=8, pool_size=4)` simulates the whole
thing end-to-end and returns pool standings, bracket winners, semis, final,
and the champion.

# Teammate Guide — Plugging your model into the game engine

Your model plugs into the shared game engine in 3 lines.

```python
from connect4_env import ModelAgent
from tournament import play_match, run_pool, evaluate_agents

my_agent = ModelAgent(your_trained_model, sample=False, strong=True)
```

## Parameters

- `sample=False` → argmax (for tournament / evaluation).
  Use `sample=True` during PG training so moves are sampled from the policy.
- `strong=True` → auto-wins / blocks on top of the model's output (recommended for the tournament).

## Your model must

- Accept input shape `(batch, 6, 7, 2)` — channel 0 = your pieces, channel 1 = opponent.
- Output shape `(batch, 7)` — one score per column. Softmax (PG) or tanh (DQN) both work.

## Test your agent against anything

```python
from connect4_env import RandomAgent, StrongRuleAgent, evaluate_agents

results = evaluate_agents(my_agent, StrongRuleAgent(), n_games=100)
print(results)
# {'wins': 87, 'losses': 10, 'draws': 3, 'win_rate': 0.87}
```

`evaluate_agents` alternates who goes first, so the result is unbiased.

## Run a full tournament with everyone's models

```python
from tournament import run_full_tournament

agents = {
    "PG_v1":  pg_agent,
    "DQN_v1": dqn_agent,
    "MCTS":   mcts_agent,
    # ...
}
result = run_full_tournament(agents, n_pools=8, pool_size=4)
print(result['champion'])
```

Returns a dict with `pool_standings`, `bracket_winners`, `semi_finals`, `final`, and `champion`.

## Before you start

Run `python test_smoke.py` to confirm the engine works on your machine.
All 9 checks should pass.

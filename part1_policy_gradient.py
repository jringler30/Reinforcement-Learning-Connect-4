# =============================================================================
# Part 1 — Policy Gradient (REINFORCE) Training
# =============================================================================
# Strategy:
#   M1 = josh_cnn (our Project 1 CNN, the model being improved)
#   M2 = randomly sampled from an evolving opponent pool
#
# Each iteration:
#   1. Sample an opponent M2 from the pool
#   2. Play GAMES_PER_ITER games, randomizing who goes first each game
#   3. For each game, collect M1's (board, move, discounted_reward) triplets
#   4. Sample exactly BATCH_SIZE triplets (fixed size — TF slows with variable!)
#   5. Take ONE gradient step via REINFORCE
#   6. Every ADD_TO_POOL_EVERY iterations, freeze a copy of M1 into the pool
#
# Key adversarial training detail (from the spec):
#   M2 uses win/block rules  (strong=True)  — tough opponent
#   M1 only sees legal moves (strong=False) — learns strength via gradient
# =============================================================================

import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from connect4_env import (
    encode_board, legal_moves, game_over, make_board, step,
    ModelAgent, RandomAgent, StrongRuleAgent, evaluate_agents, play_game
)
from models.loader import load_model

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA            = 0.99   # discount factor for future rewards
LR               = 1e-4   # Adam learning rate
GAMES_PER_ITER   = 10     # games to play before each gradient step
BATCH_SIZE       = 64     # MUST stay fixed — TF traces the graph at this size
RANDOM_INIT_MOVES = 3     # random moves at game start (board diversity)
ENTROPY_COEF     = 0.01   # entropy bonus to encourage exploration
GRAD_CLIP        = 1.0    # global gradient norm clip

ADD_TO_POOL_EVERY = 100   # freeze an M1 snapshot into opponent pool every N iters
MAX_OWN_SNAPSHOTS = 6     # cap on how many of our own snapshots stay in pool
N_ITERATIONS     = 2000   # total training iterations (increase for tournament)
EVAL_EVERY       = 50     # evaluate M1 win rate every N iterations
EVAL_GAMES       = 100    # games per evaluation (alternating first player)

CHECKPOINT_DIR   = "checkpoints/pg/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =============================================================================
# 1. Load M1
# =============================================================================
print("Loading M1 (josh_cnn)...")
m1_model = load_model("josh_cnn")   # Keras model, output = softmax over 7 cols

# Optimizer — defined outside train_step so state persists across iterations
optimizer = keras.optimizers.Adam(learning_rate=LR)

# =============================================================================
# 2. Build initial opponent pool
# =============================================================================
# The spec says the original M1 copy must always stay in the pool.
# We also include StrongRuleAgent as a baseline hard opponent with no model cost.

def _make_m2_agent(model):
    """Wrap a Keras model as a strong (win/block) M2 opponent."""
    return ModelAgent(model, sample=True, strong=True)

# Store opponent pool as list of Agents (not raw models, so we can mix types)
opponent_pool = [
    StrongRuleAgent(),                    # rule-based baseline (always in pool)
    RandomAgent(),                        # easy warm-up opponent
    _make_m2_agent(load_model("josh_cnn")),  # original M1 — never removed
]

# Snapshot counter (we track how many of our own PG snapshots are in the pool)
own_snapshot_count = 0

# =============================================================================
# 3. Core helpers
# =============================================================================

def collect_episode(m1_player_id, m2_agent):
    """
    Play one game. M1 is assigned player_id (1 or -1).
    Returns list of (encoded_board, col, legal_mask) for M1's turns,
    and the game outcome from M1's perspective (+1 win, -1 loss, 0 draw).

    M1 uses sample=True, strong=False so it learns purely from gradients.
    M2 uses strong=True so it's a tough training partner.
    """
    # Build a lightweight M1 inference agent for this episode (no strong rules)
    m1_agent = ModelAgent(m1_model, sample=True, strong=False)

    # Assign agents to player slots based on who goes first this game
    if m1_player_id == 1:
        agent1, agent2 = m1_agent, m2_agent
    else:
        agent1, agent2 = m2_agent, m1_agent

    board = make_board()
    current = 1
    m1_transitions = []   # (encoded_board, col, legal_cols) for M1 moves only

    # Random init moves — not recorded, just diversify starting positions
    for _ in range(RANDOM_INIT_MOVES):
        cols = legal_moves(board)
        if not cols:
            break
        col = np.random.choice(cols)
        board, _ = step(board, col, current)
        done, _ = game_over(board)
        if done:
            return [], 0   # game ended during random init — skip
        current = -current

    # Main game loop
    while True:
        cols = legal_moves(board)
        if not cols:
            outcome = 0
            break

        if current == m1_player_id:
            # Record board state before M1 moves
            enc = encode_board(board, current)   # (6, 7, 2)
            legal_mask = np.zeros(7, dtype=np.float32)
            legal_mask[cols] = 1.0
            col = m1_agent.select_move(board, current)
            m1_transitions.append((enc, col, legal_mask))
        else:
            col = m2_agent.select_move(board, current)

        board, _ = step(board, col, current)
        done, winner = game_over(board)
        if done:
            if winner == m1_player_id:
                outcome = 1
            elif winner == 0:
                outcome = 0
            else:
                outcome = -1
            break
        current = -current

    return m1_transitions, outcome


def compute_discounted_returns(transitions, outcome, gamma=GAMMA):
    """
    Apply discounted returns to M1's move sequence.
    G_t = outcome * gamma^(T-1-t)  (earlier moves discounted more)
    Returns list of (encoded_board, col, return, legal_mask).
    """
    T = len(transitions)
    result = []
    for t, (enc, col, legal_mask) in enumerate(transitions):
        G_t = outcome * (gamma ** (T - 1 - t))
        result.append((enc, col, G_t, legal_mask))
    return result


# ── TF training step (traced once at BATCH_SIZE — do NOT change batch size) ──

@tf.function
def train_step(boards, cols, returns):
    """
    Single REINFORCE gradient step.
    boards  : (BATCH_SIZE, 6, 7, 2) float32
    cols    : (BATCH_SIZE,)          int32
    returns : (BATCH_SIZE,)          float32
    """
    with tf.GradientTape() as tape:
        probs = m1_model(boards, training=True)           # (B, 7)  softmax out
        log_probs = tf.math.log(probs + 1e-8)             # (B, 7)

        # Log prob of the action actually taken
        indices = tf.stack(
            [tf.range(BATCH_SIZE, dtype=tf.int32), cols], axis=1
        )
        chosen_log_probs = tf.gather_nd(log_probs, indices)  # (B,)

        # Entropy bonus — keeps policy from collapsing too early
        entropy = -tf.reduce_mean(
            tf.reduce_sum(probs * log_probs, axis=1)
        )

        # REINFORCE loss (negative because we maximise expected return)
        pg_loss = -tf.reduce_mean(chosen_log_probs * returns)
        loss    = pg_loss - ENTROPY_COEF * entropy

    grads = tape.gradient(loss, m1_model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP)
    optimizer.apply_gradients(zip(grads, m1_model.trainable_variables))
    return loss, pg_loss, entropy


def sample_batch(all_triplets):
    """
    Randomly sample exactly BATCH_SIZE triplets from all_triplets.
    Samples with replacement if we have fewer than BATCH_SIZE (rare early on).
    Returns TF-ready tensors.
    """
    indices = np.random.choice(len(all_triplets), size=BATCH_SIZE, replace=True)
    boards  = np.stack([all_triplets[i][0] for i in indices]).astype(np.float32)
    cols    = np.array([all_triplets[i][1] for i in indices], dtype=np.int32)
    rets    = np.array([all_triplets[i][2] for i in indices], dtype=np.float32)

    # Normalise returns within the batch to reduce variance
    if rets.std() > 1e-8:
        rets = (rets - rets.mean()) / (rets.std() + 1e-8)

    return (
        tf.constant(boards),
        tf.constant(cols),
        tf.constant(rets),
    )


# =============================================================================
# 4. Training loop
# =============================================================================

# Logging
log_loss       = []
log_pg_loss    = []
log_entropy    = []
log_win_rate_strong = []   # vs StrongRuleAgent
log_win_rate_orig   = []   # vs original M1 (gauges how much we improved)
log_iters_eval = []

# Held-out eval agents — never in the training pool
eval_strong  = StrongRuleAgent()
eval_orig_m1 = _make_m2_agent(load_model("josh_cnn"))

print(f"\nStarting PG training for {N_ITERATIONS} iterations...")
print(f"  BATCH_SIZE={BATCH_SIZE}, GAMES_PER_ITER={GAMES_PER_ITER}, "
      f"GAMMA={GAMMA}, LR={LR}\n")

for iteration in range(1, N_ITERATIONS + 1):

    # ── Step 1: pick a random M2 from the pool ────────────────────────────────
    m2_agent = random.choice(opponent_pool)

    # ── Step 2: play GAMES_PER_ITER games, collect M1 triplets ───────────────
    all_triplets = []
    for _ in range(GAMES_PER_ITER):
        # Randomly assign who goes first each game (spec requirement)
        m1_player_id = random.choice([1, -1])
        transitions, outcome = collect_episode(m1_player_id, m2_agent)
        if not transitions:
            continue
        triplets = compute_discounted_returns(transitions, outcome)
        all_triplets.extend(triplets)

    if not all_triplets:
        continue   # safety — shouldn't happen in practice

    # ── Step 3: one gradient step on a fixed-size random minibatch ───────────
    boards_t, cols_t, rets_t = sample_batch(all_triplets)
    loss, pg_loss, entropy = train_step(boards_t, cols_t, rets_t)

    log_loss.append(float(loss))
    log_pg_loss.append(float(pg_loss))
    log_entropy.append(float(entropy))

    # ── Step 4: evaluate every EVAL_EVERY iterations ─────────────────────────
    if iteration % EVAL_EVERY == 0:
        m1_eval_agent = ModelAgent(m1_model, sample=False, strong=True)

        res_strong = evaluate_agents(m1_eval_agent, eval_strong,
                                     n_games=EVAL_GAMES, alternate_first=True)
        res_orig   = evaluate_agents(m1_eval_agent, eval_orig_m1,
                                     n_games=EVAL_GAMES, alternate_first=True)

        log_win_rate_strong.append(res_strong['win_rate'])
        log_win_rate_orig.append(res_orig['win_rate'])
        log_iters_eval.append(iteration)

        print(f"[{iteration:>5}] loss={float(loss):.4f} | "
              f"vs Strong: {res_strong['win_rate']:.1%} | "
              f"vs OrigM1: {res_orig['win_rate']:.1%}")

    # ── Step 5: periodically freeze a snapshot into the opponent pool ─────────
    if iteration % ADD_TO_POOL_EVERY == 0:
        # Clone current M1 weights into a new model
        snapshot = keras.models.clone_model(m1_model)
        snapshot.set_weights(m1_model.get_weights())
        opponent_pool.append(_make_m2_agent(snapshot))
        own_snapshot_count += 1

        # Evict oldest own snapshot if pool gets too large
        # (never remove indices 0, 1, 2 — StrongRule, Random, original M1)
        if own_snapshot_count > MAX_OWN_SNAPSHOTS:
            # Find and remove the oldest own snapshot (index 3 onwards)
            opponent_pool.pop(3)
            own_snapshot_count -= 1

        print(f"  → Snapshot added. Pool size: {len(opponent_pool)}")

        # Save checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"m1_iter{iteration}.keras")
        m1_model.save(ckpt_path)


# =============================================================================
# 5. Save final trained model
# =============================================================================
final_path = os.path.join(CHECKPOINT_DIR, "m1_pg_final.keras")
m1_model.save(final_path)
print(f"\nFinal model saved to {final_path}")

# =============================================================================
# 6. Plots
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Policy Gradient Training — M1 (josh_cnn)", fontsize=13)

# Loss curve
axes[0].plot(log_loss, alpha=0.4, label="Total loss")
axes[0].plot(log_pg_loss, alpha=0.4, label="PG loss")
# Rolling average for readability
window = 20
if len(log_loss) >= window:
    roll = np.convolve(log_loss, np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, len(log_loss)), roll, label=f"Loss (MA{window})")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss"); axes[0].legend()

# Win rate vs StrongRuleAgent
axes[1].plot(log_iters_eval, log_win_rate_strong, marker='o')
axes[1].axhline(0.5, color='gray', linestyle='--', label='50% baseline')
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Win Rate")
axes[1].set_title("Win Rate vs StrongRuleAgent"); axes[1].legend()
axes[1].set_ylim(0, 1)

# Win rate vs original M1 (shows how much PG improved us)
axes[2].plot(log_iters_eval, log_win_rate_orig, marker='o', color='orange')
axes[2].axhline(0.5, color='gray', linestyle='--', label='50% = no improvement')
axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Win Rate")
axes[2].set_title("Win Rate vs Original M1 (Improvement Gauge)")
axes[2].legend(); axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "pg_training_curves.png"), dpi=150)
plt.show()
print("Training curves saved.")

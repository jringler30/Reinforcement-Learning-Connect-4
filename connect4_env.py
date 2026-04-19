"""
connect4_env.py
---------------
Shared Connect-4 game engine for Project 3.
No dependencies beyond numpy.

Board convention:
    0  = empty
    1  = Player 1
   -1  = Player 2

Board shape: (6 rows, 7 cols), row 0 is the TOP, row 5 is the BOTTOM.
Pieces fall to the lowest empty row in a column (gravity).
"""

import numpy as np


# ─── Board ────────────────────────────────────────────────────────────────────

def make_board():
    """Return a fresh 6×7 board of zeros."""
    return np.zeros((6, 7), dtype=np.int8)


def copy_board(board):
    return board.copy()


# ─── Moves ────────────────────────────────────────────────────────────────────

def legal_moves(board):
    """Return list of columns (0-6) that are not full."""
    return [c for c in range(7) if board[0, c] == 0]


def drop_piece(board, col, player):
    """
    Drop player's piece into col (in-place).
    Returns row where piece landed, or -1 if column is full.
    """
    for row in range(5, -1, -1):
        if board[row, col] == 0:
            board[row, col] = player
            return row
    return -1  # column full — caller should avoid this


def step(board, col, player):
    """
    Apply move and return (new_board, row) without mutating original.
    Raises ValueError if move is illegal.
    """
    if board[0, col] != 0:
        raise ValueError(f"Column {col} is full.")
    new_board = copy_board(board)
    row = drop_piece(new_board, col, player)
    return new_board, row


# ─── Win / Draw detection ─────────────────────────────────────────────────────

def _check_window(window, player):
    return all(cell == player for cell in window)


def check_win(board, player):
    """Return True if `player` has four in a row anywhere on the board."""
    # horizontal
    for r in range(6):
        for c in range(4):
            if _check_window(board[r, c:c+4], player):
                return True
    # vertical
    for c in range(7):
        for r in range(3):
            if _check_window(board[r:r+4, c], player):
                return True
    # diagonal ↘
    for r in range(3):
        for c in range(4):
            if _check_window([board[r+i, c+i] for i in range(4)], player):
                return True
    # diagonal ↙
    for r in range(3):
        for c in range(3, 7):
            if _check_window([board[r+i, c-i] for i in range(4)], player):
                return True
    return False


def is_draw(board):
    """Return True if board is full and nobody won."""
    return len(legal_moves(board)) == 0


def game_over(board):
    """Return (done, winner): winner is 1, -1, 0 (draw), or None (not done)."""
    for player in [1, -1]:
        if check_win(board, player):
            return True, player
    if is_draw(board):
        return True, 0
    return False, None


# ─── Tactical helpers (used by strong opponents / M2) ─────────────────────────

def find_winning_move(board, player):
    """Return a column that wins immediately, or None."""
    for col in legal_moves(board):
        b, _ = step(board, col, player)
        if check_win(b, player):
            return col
    return None


def find_non_losing_moves(board, player):
    """
    Return legal moves that don't immediately hand the opponent a win.
    Falls back to all legal moves if every move loses.
    """
    opponent = -player
    safe = []
    for col in legal_moves(board):
        b, _ = step(board, col, player)
        # after we move, can opponent win immediately?
        if find_winning_move(b, opponent) is None:
            safe.append(col)
    return safe if safe else legal_moves(board)


# ─── Neural-network board encoding ────────────────────────────────────────────

def encode_board(board, player):
    """
    Return a (6, 7, 2) float32 array for the neural network.
      channel 0: squares occupied by `player`
      channel 1: squares occupied by opponent
    This matches the input format from Project 1.
    """
    enc = np.zeros((6, 7, 2), dtype=np.float32)
    enc[:, :, 0] = (board == player).astype(np.float32)
    enc[:, :, 1] = (board == -player).astype(np.float32)
    return enc


# ─── Agent interface ──────────────────────────────────────────────────────────

class Agent:
    """
    Base class. Subclass and override `select_move(board, player) -> col`.
    All agents in this project should follow this interface so they
    can be swapped into any training loop or tournament runner.
    """
    def select_move(self, board, player):
        raise NotImplementedError


class RandomAgent(Agent):
    """Plays a uniformly random legal move."""
    def select_move(self, board, player):
        return np.random.choice(legal_moves(board))


class StrongRuleAgent(Agent):
    """
    Win if possible, block opponent win, else play a non-losing move at random.
    Useful as a strong M2 opponent during PG training.
    """
    def select_move(self, board, player):
        # 1. win immediately
        col = find_winning_move(board, player)
        if col is not None:
            return col
        # 2. block opponent
        col = find_winning_move(board, -player)
        if col is not None:
            return col
        # 3. avoid gifting the win
        return np.random.choice(find_non_losing_moves(board, player))


def _infer_framework(model):
    """Detect whether `model` is a Keras or PyTorch model."""
    cls = type(model).__module__
    if cls.startswith("torch") or cls.startswith("__main__"):
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                return "torch"
        except ImportError:
            pass
    # default: assume Keras (it has .predict)
    return "keras"


class ModelAgent(Agent):
    """
    Wraps a trained policy/value network and exposes it as an Agent.

    Supports BOTH Keras and PyTorch models:
      • Keras  model:  expects input (batch, 6, 7, 2), returns (batch, 7)
      • PyTorch model: expects input (batch, 2, 6, 7) by default
                       (channels-first). If your PyTorch model was trained
                       with channels-last, pass `channels_first=False`.

    The model's output (7 scores per board) is treated as logits. They're
    masked for illegal columns, then either sampled or argmaxed.

    Parameters
    ----------
    model          : keras.Model | torch.nn.Module
    sample         : True = stochastic (PG training), False = argmax (eval)
    strong         : apply win/block rules on top of the model
    channels_first : for PyTorch models only; default True matches the
                     typical PyTorch convention (N, C, H, W)
    framework      : "auto" (default), "keras", or "torch" — override if
                     autodetection gets it wrong
    """
    def __init__(self, model, sample=True, strong=False,
                 channels_first=True, framework="auto"):
        self.model = model
        self.sample = sample
        self.strong = strong
        self.channels_first = channels_first
        self.framework = (_infer_framework(model)
                          if framework == "auto" else framework)

        if self.framework == "torch":
            import torch
            self._torch = torch
            try:
                self.model.eval()
            except AttributeError:
                pass

    def _predict_logits(self, enc_hwc):
        """
        Run one forward pass and return the 7-dim score vector (numpy).
        `enc_hwc` is shape (6, 7, 2), channels-last.
        """
        if self.framework == "keras":
            x = enc_hwc[np.newaxis]                    # (1, 6, 7, 2)
            return self.model.predict(x, verbose=0)[0] # (7,)

        # PyTorch
        torch = self._torch
        if self.channels_first:
            x = np.transpose(enc_hwc, (2, 0, 1))       # (2, 6, 7)
        else:
            x = enc_hwc
        x = x[np.newaxis].astype(np.float32)           # add batch dim
        with torch.no_grad():
            out = self.model(torch.from_numpy(x))
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        return np.asarray(out).reshape(-1)[:7]         # flatten to (7,)

    def select_move(self, board, player):
        if self.strong:
            col = find_winning_move(board, player)
            if col is not None:
                return col
            col = find_winning_move(board, -player)
            if col is not None:
                return col

        legal = legal_moves(board)
        enc = encode_board(board, player)              # (6, 7, 2)
        logits = self._predict_logits(enc)             # (7,)

        # mask illegal columns
        mask = np.full(7, -1e9)
        mask[legal] = 0.0
        logits = logits + mask

        if self.sample:
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            return int(np.random.choice(7, p=probs))
        else:
            return int(np.argmax(logits))


# ─── Game runner ──────────────────────────────────────────────────────────────

def play_game(agent1, agent2, random_init_moves=0, verbose=False):
    """
    Play one full game.

    Parameters
    ----------
    agent1 : Agent   plays as player  1
    agent2 : Agent   plays as player -1
    random_init_moves : int
        Number of moves to play randomly at the start (for both sides,
        alternating). Useful for DQN warm-starting as described in Step 4b.
    verbose : bool

    Returns
    -------
    history : list of (board_before_move, col, player) for non-random moves
    winner  : 1, -1, or 0 (draw)
    """
    board = make_board()
    history = []
    agents = {1: agent1, -1: agent2}
    current = 1  # player 1 always goes first

    # random initialization moves (not recorded in history)
    for _ in range(random_init_moves):
        moves = legal_moves(board)
        if not moves:
            break
        col = np.random.choice(moves)
        board, _ = step(board, col, current)
        done, winner = game_over(board)
        if done:
            return history, winner
        current = -current

    # main game loop
    while True:
        moves = legal_moves(board)
        if not moves:
            return history, 0

        board_snapshot = copy_board(board)
        col = agents[current].select_move(board, current)
        board, _ = step(board, col, current)
        history.append((board_snapshot, col, current))

        if verbose:
            print(f"Player {current} plays col {col}")
            print_board(board)

        done, winner = game_over(board)
        if done:
            return history, winner

        current = -current


# ─── Utility ──────────────────────────────────────────────────────────────────

def print_board(board):
    """Pretty-print the board. 'X'=player1, 'O'=player2, '.'=empty."""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    print(' '.join(str(c) for c in range(7)))
    for row in board:
        print(' '.join(symbols[cell] for cell in row))
    print()


def evaluate_agents(agent1, agent2, n_games=100, random_init_moves=0,
                    alternate_first=True):
    """
    Play n_games between agent1 and agent2. When alternate_first=True,
    they take turns going first — this removes the first-move bias that
    otherwise skews Connect-4 results heavily.

    Returns dict with wins/losses/draws/win_rate for agent1.
    """
    wins = losses = draws = 0
    for i in range(n_games):
        swap = alternate_first and (i % 2 == 1)
        if swap:
            # agent2 plays as player 1 this game
            _, winner = play_game(agent2, agent1,
                                  random_init_moves=random_init_moves)
            winner = -winner  # flip so +1 still means agent1 won
        else:
            _, winner = play_game(agent1, agent2,
                                  random_init_moves=random_init_moves)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
    return {
        'wins':     wins,
        'losses':   losses,
        'draws':    draws,
        'win_rate': wins / n_games,
    }

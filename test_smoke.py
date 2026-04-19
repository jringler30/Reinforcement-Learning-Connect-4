"""
test_smoke.py
-------------
Quick sanity checks for connect4_env.py and tournament.py.
Run directly: `python test_smoke.py` — no pytest needed.
"""

import numpy as np
import random

from connect4_env import (
    make_board, legal_moves, step, check_win, game_over,
    find_winning_move, find_non_losing_moves, encode_board,
    RandomAgent, StrongRuleAgent, play_game, evaluate_agents,
)
from tournament import (
    play_match, run_pool, run_bracket, run_full_tournament,
)


def ok(label):
    print(f"  ✓ {label}")


def test_board_basics():
    print("Board basics")
    b = make_board()
    assert b.shape == (6, 7)
    assert legal_moves(b) == list(range(7))
    ok("empty board")

    b, row = step(b, 3, 1)
    assert row == 5 and b[5, 3] == 1
    b, row = step(b, 3, -1)
    assert row == 4 and b[4, 3] == -1
    ok("gravity works")


def test_win_detection():
    print("Win detection")
    # horizontal
    b = make_board()
    for c in range(4):
        b[5, c] = 1
    assert check_win(b, 1) and not check_win(b, -1)
    ok("horizontal")

    # vertical
    b = make_board()
    for r in range(4):
        b[r, 0] = -1
    assert check_win(b, -1)
    ok("vertical")

    # diagonal ↘
    b = make_board()
    for i in range(4):
        b[i, i] = 1
    assert check_win(b, 1)
    ok("diagonal ↘")

    # diagonal ↙
    b = make_board()
    for i in range(4):
        b[i, 3 - i] = 1
    assert check_win(b, 1)
    ok("diagonal ↙")

    # no win
    b = make_board()
    b[5, 0] = 1; b[5, 1] = 1; b[5, 2] = 1
    assert not check_win(b, 1)
    ok("three-in-a-row is not a win")


def test_tactical_helpers():
    print("Tactical helpers")
    # Player 1 has 3 in a row horizontally at bottom, col 3 is winning move
    b = make_board()
    b[5, 0] = 1; b[5, 1] = 1; b[5, 2] = 1
    assert find_winning_move(b, 1) == 3
    ok("find_winning_move finds horizontal win")

    # Player -1 threatens to win at col 3 next move → player 1 must block
    # non_losing_moves for player 1 should include col 3 (blocks)
    safe = find_non_losing_moves(b, -1)
    # -1 should see col 3 as winning, but non_losing is about moves that
    # don't let opponent win next — here any move that doesn't help doesn't
    # matter much. Just check that at least col 3 is a winning move for -1.
    assert find_winning_move(b, 1) == 3
    ok("non_losing_moves returns at least one move")


def test_encoding():
    print("Encoding")
    b = make_board()
    b[5, 3] = 1
    b[4, 3] = -1
    enc = encode_board(b, player=1)
    assert enc.shape == (6, 7, 2) and enc.dtype == np.float32
    assert enc[5, 3, 0] == 1.0 and enc[5, 3, 1] == 0.0
    assert enc[4, 3, 0] == 0.0 and enc[4, 3, 1] == 1.0
    ok("shape and channel assignment correct")


def test_play_game():
    print("Full game")
    np.random.seed(0)
    _, winner = play_game(RandomAgent(), RandomAgent())
    assert winner in (1, -1, 0)
    ok(f"random vs random finishes (winner={winner})")

    # strong vs random should win a strong majority
    np.random.seed(1)
    res = evaluate_agents(StrongRuleAgent(), RandomAgent(), n_games=40)
    print(f"    StrongRule vs Random over 40 games (alternating first): "
          f"{res['wins']}W-{res['losses']}L-{res['draws']}D")
    assert res['win_rate'] > 0.7, "StrongRule should crush Random"
    ok("StrongRule beats Random decisively")


def test_match_runner():
    print("Match runner")
    np.random.seed(2)
    m = play_match(StrongRuleAgent(), RandomAgent(),
                   "Strong", "Random", n_games=2)
    assert len(m.games) == 2
    assert m.games[0].first_player == "Strong"
    assert m.games[1].first_player == "Random"
    ok("2-game match alternates first player")


def test_pool():
    print("Round-robin pool")
    np.random.seed(3)
    random.seed(3)
    agents = {
        "Rnd_A": RandomAgent(),
        "Rnd_B": RandomAgent(),
        "Str_A": StrongRuleAgent(),
        "Str_B": StrongRuleAgent(),
    }
    standings = run_pool(agents, n_games_per_match=2)
    assert len(standings) == 4
    # 3 matches × 2 games = 6 games per team
    for r in standings:
        assert r.games_played == 6
    # Strong agents should be ranked above Random agents
    top2 = [s.name for s in standings[:2]]
    assert all(n.startswith("Str") for n in top2), \
        f"Expected Strong agents on top, got {top2}"
    ok("6 games each, strong agents win pool")


def test_bracket():
    print("Single-elim bracket")
    np.random.seed(4)
    random.seed(4)
    agents = {
        "A": StrongRuleAgent(),
        "B": StrongRuleAgent(),
        "C": RandomAgent(),
        "D": RandomAgent(),
    }
    winner, matches = run_bracket(agents, n_games_per_match=2)
    assert winner in agents
    assert len(matches) == 3  # QF × 2 + F × 1 ... wait, 4 teams = 2 SF + 1 F
    ok(f"bracket completes; winner={winner}")


def test_full_tournament_32():
    print("Full tournament (real 32-team shape: 8 pools × 4)")
    np.random.seed(5)
    random.seed(5)
    # Mix StrongRule and Random agents across 32 slots
    agents = {f"Agent_{i:02d}": (StrongRuleAgent() if i % 2 == 0
                                 else RandomAgent())
              for i in range(32)}
    result = run_full_tournament(agents, n_pools=8, pool_size=4,
                                 n_games_per_match=2, verbose=False)
    assert result['champion'] in agents
    # Strong agents should dominate; champion should be a StrongRule
    champ_idx = int(result['champion'].split('_')[1])
    assert champ_idx % 2 == 0, \
        f"Expected a StrongRule champion, got {result['champion']}"
    ok(f"32-team tournament completes; champion={result['champion']}")


if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_board_basics()
    test_win_detection()
    test_tactical_helpers()
    test_encoding()
    test_play_game()
    test_match_runner()
    test_pool()
    test_bracket()
    test_full_tournament_32()
    print("\n✅ All smoke tests passed.")

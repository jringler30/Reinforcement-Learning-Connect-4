"""
tournament.py
-------------
Match, pool, bracket, and full tournament runners for Project 3.

Everything here is built on top of the `Agent` interface in connect4_env.py,
so any mix of PG / DQN / MCTS / rule-based agents can be dropped in.

Tournament format (per the assignment):
  • 32 teams → 8 pools of 4 → round-robin within pool (6 games each)
  • Match = 2 games, each agent goes first once
  • Pool standings: wins desc, then avg moves-to-win asc (fewer = better)
  • 1st → bracket A, 2nd → B, 3rd → C, 4th → D
  • Single-elim brackets, then semis (A vs B, C vs D), then finals
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import itertools
import random
import numpy as np

from connect4_env import Agent, play_game, RandomAgent, StrongRuleAgent


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class GameResult:
    """Outcome of a single game."""
    first_player: str     # name of the agent who moved first
    second_player: str    # name of the agent who moved second
    winner_name: Optional[str]   # None if draw
    n_moves: int          # moves played (excludes random init)


@dataclass
class MatchResult:
    """Outcome of a match (default: 2 games, each side goes first once)."""
    agent_a: str
    agent_b: str
    games: List[GameResult] = field(default_factory=list)

    @property
    def wins_a(self) -> int:
        return sum(1 for g in self.games if g.winner_name == self.agent_a)

    @property
    def wins_b(self) -> int:
        return sum(1 for g in self.games if g.winner_name == self.agent_b)

    @property
    def draws(self) -> int:
        return sum(1 for g in self.games if g.winner_name is None)

    @property
    def winner(self) -> Optional[str]:
        """Match winner, or None if tied."""
        if self.wins_a > self.wins_b:
            return self.agent_a
        if self.wins_b > self.wins_a:
            return self.agent_b
        return None


@dataclass
class TeamRecord:
    """Aggregated stats for one team in a pool."""
    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    moves_in_wins: int = 0     # sum of n_moves across games this team won
    moves_in_losses: int = 0   # sum of n_moves across games this team lost
    games_played: int = 0

    @property
    def avg_moves_to_win(self) -> float:
        """Primary tiebreaker — lower is better (faster wins)."""
        return self.moves_in_wins / self.wins if self.wins else float('inf')


# ─── Core match runner ────────────────────────────────────────────────────────

def _play_one_game(
    first: Agent, first_name: str,
    second: Agent, second_name: str,
    random_init_moves: int = 0,
) -> GameResult:
    """Play a single game. `first` is player 1, `second` is player -1."""
    history, winner_flag = play_game(first, second,
                                     random_init_moves=random_init_moves)
    if winner_flag == 1:
        winner_name = first_name
    elif winner_flag == -1:
        winner_name = second_name
    else:
        winner_name = None
    return GameResult(
        first_player=first_name,
        second_player=second_name,
        winner_name=winner_name,
        n_moves=len(history),
    )


def play_match(
    agent_a: Agent, agent_b: Agent,
    name_a: str = "A", name_b: str = "B",
    n_games: int = 2,
    random_init_moves: int = 0,
) -> MatchResult:
    """
    Play a match. First player alternates every game so the who-goes-first
    advantage is split evenly. For n_games=2 (tournament default), each
    agent goes first once.
    """
    result = MatchResult(agent_a=name_a, agent_b=name_b)
    for i in range(n_games):
        if i % 2 == 0:
            game = _play_one_game(agent_a, name_a, agent_b, name_b,
                                  random_init_moves)
        else:
            game = _play_one_game(agent_b, name_b, agent_a, name_a,
                                  random_init_moves)
        result.games.append(game)
    return result


# ─── Pool play (round-robin) ──────────────────────────────────────────────────

def run_pool(
    agents: Dict[str, Agent],
    n_games_per_match: int = 2,
    verbose: bool = False,
) -> List[TeamRecord]:
    """
    Round-robin within a pool. Every agent plays every other agent.
    Returns standings sorted by:
       1. wins (desc)
       2. avg moves per win (asc — faster wins rank higher)
       3. total losses (asc)
       4. random (coin flip)
    """
    records: Dict[str, TeamRecord] = {n: TeamRecord(name=n) for n in agents}

    for name_a, name_b in itertools.combinations(agents.keys(), 2):
        match = play_match(agents[name_a], agents[name_b],
                           name_a, name_b, n_games=n_games_per_match)
        if verbose:
            print(f"  {name_a} vs {name_b}: "
                  f"{match.wins_a}-{match.wins_b} (draws {match.draws})")

        for g in match.games:
            records[name_a].games_played += 1
            records[name_b].games_played += 1
            if g.winner_name == name_a:
                records[name_a].wins += 1
                records[name_a].moves_in_wins += g.n_moves
                records[name_b].losses += 1
                records[name_b].moves_in_losses += g.n_moves
            elif g.winner_name == name_b:
                records[name_b].wins += 1
                records[name_b].moves_in_wins += g.n_moves
                records[name_a].losses += 1
                records[name_a].moves_in_losses += g.n_moves
            else:
                records[name_a].draws += 1
                records[name_b].draws += 1

    standings = list(records.values())
    standings.sort(key=lambda r: (-r.wins, r.avg_moves_to_win, r.losses,
                                  random.random()))
    return standings


# ─── Single-elimination bracket ───────────────────────────────────────────────

@dataclass
class BracketNode:
    """One match in a bracket tree."""
    agent_a: str
    agent_b: str
    match: Optional[MatchResult] = None
    next_round: Optional['BracketNode'] = None


def run_bracket(
    agents: Dict[str, Agent],
    n_games_per_match: int = 2,
    verbose: bool = False,
) -> Tuple[str, List[MatchResult]]:
    """
    Single-elimination. If an even number of entries: normal bracket.
    If a match ties, the winner is decided by fewer moves-in-wins
    (with random fallback).

    Returns: (winner_name, list_of_match_results_in_play_order)
    """
    if len(agents) < 2:
        raise ValueError("Need at least 2 agents for a bracket.")

    remaining = list(agents.keys())
    all_matches: List[MatchResult] = []

    round_num = 1
    while len(remaining) > 1:
        if verbose:
            print(f"\n  ── Round {round_num} ──")
        next_round: List[str] = []

        # bye for odd number of entrants
        if len(remaining) % 2 == 1:
            bye = remaining.pop()
            next_round.append(bye)
            if verbose:
                print(f"  {bye} gets a bye")

        for i in range(0, len(remaining), 2):
            a_name, b_name = remaining[i], remaining[i + 1]
            match = play_match(agents[a_name], agents[b_name],
                               a_name, b_name, n_games=n_games_per_match)
            all_matches.append(match)

            winner = match.winner
            if winner is None:
                # tie → pick the one with fewer moves-to-win, then coin flip
                moves_a = sum(g.n_moves for g in match.games
                              if g.winner_name == a_name)
                moves_b = sum(g.n_moves for g in match.games
                              if g.winner_name == b_name)
                if moves_a < moves_b and match.wins_a > 0:
                    winner = a_name
                elif moves_b < moves_a and match.wins_b > 0:
                    winner = b_name
                else:
                    winner = random.choice([a_name, b_name])

            next_round.append(winner)
            if verbose:
                print(f"  {a_name} vs {b_name}: "
                      f"{match.wins_a}-{match.wins_b} → {winner}")

        remaining = next_round
        round_num += 1

    return remaining[0], all_matches


# ─── Full 32-team tournament simulation ───────────────────────────────────────

def run_full_tournament(
    agents: Dict[str, Agent],
    n_pools: int = 8,
    pool_size: int = 4,
    n_games_per_match: int = 2,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Simulate the full class tournament format.

    Splits `agents` into `n_pools` pools of `pool_size` teams each.
    Pool winners → bracket A, 2nds → B, 3rds → C, 4ths → D.
    Bracket winners play semis (A vs B, C vs D), then the final.

    Returns a dict with pool_standings, bracket_winners, and final champion.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    names = list(agents.keys())
    expected = n_pools * pool_size
    if len(names) != expected:
        raise ValueError(
            f"Expected {expected} agents ({n_pools} pools × {pool_size}), "
            f"got {len(names)}."
        )

    random.shuffle(names)

    # ── pool play ──
    pool_standings: List[List[TeamRecord]] = []
    for p in range(n_pools):
        pool_names = names[p * pool_size:(p + 1) * pool_size]
        pool_agents = {n: agents[n] for n in pool_names}
        if verbose:
            print(f"\nPool {p + 1}: {pool_names}")
        standings = run_pool(pool_agents, n_games_per_match, verbose=verbose)
        pool_standings.append(standings)
        if verbose:
            print(f"  Pool {p + 1} standings:")
            for i, rec in enumerate(standings):
                print(f"    {i + 1}. {rec.name:20s} "
                      f"{rec.wins}-{rec.losses}-{rec.draws} "
                      f"(avg moves/win: {rec.avg_moves_to_win:.1f})")

    # ── brackets A, B, C, D ──
    brackets = {'A': [], 'B': [], 'C': [], 'D': []}
    for standings in pool_standings:
        brackets['A'].append(standings[0].name)
        brackets['B'].append(standings[1].name)
        brackets['C'].append(standings[2].name)
        brackets['D'].append(standings[3].name)

    bracket_winners = {}
    for letter, team_names in brackets.items():
        if verbose:
            print(f"\n═══ Bracket {letter} ═══")
        sub = {n: agents[n] for n in team_names}
        winner, _ = run_bracket(sub, n_games_per_match, verbose=verbose)
        bracket_winners[letter] = winner
        if verbose:
            print(f"  Bracket {letter} winner: {winner}")

    # ── semi-finals: A vs B, C vs D ──
    if verbose:
        print("\n═══ Semi-finals ═══")
    semi1 = play_match(agents[bracket_winners['A']],
                       agents[bracket_winners['B']],
                       bracket_winners['A'], bracket_winners['B'],
                       n_games=n_games_per_match)
    semi2 = play_match(agents[bracket_winners['C']],
                       agents[bracket_winners['D']],
                       bracket_winners['C'], bracket_winners['D'],
                       n_games=n_games_per_match)
    finalist1 = semi1.winner or random.choice([semi1.agent_a, semi1.agent_b])
    finalist2 = semi2.winner or random.choice([semi2.agent_a, semi2.agent_b])
    if verbose:
        print(f"  {semi1.agent_a} vs {semi1.agent_b}: "
              f"{semi1.wins_a}-{semi1.wins_b} → {finalist1}")
        print(f"  {semi2.agent_a} vs {semi2.agent_b}: "
              f"{semi2.wins_a}-{semi2.wins_b} → {finalist2}")

    # ── final ──
    final = play_match(agents[finalist1], agents[finalist2],
                       finalist1, finalist2, n_games=n_games_per_match)
    champion = final.winner or random.choice([finalist1, finalist2])
    if verbose:
        print("\n═══ FINAL ═══")
        print(f"  {finalist1} vs {finalist2}: "
              f"{final.wins_a}-{final.wins_b} → {champion}")
        print(f"\n🏆 Champion: {champion}")

    return {
        'pool_standings': pool_standings,
        'bracket_winners': bracket_winners,
        'semi_finals': [semi1, semi2],
        'final': final,
        'champion': champion,
    }


# ─── Pretty-printers ──────────────────────────────────────────────────────────

def print_standings(standings: List[TeamRecord], title: str = "Standings"):
    """Pretty-print a list of TeamRecord."""
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Rank':<5}{'Name':<22}{'W':>3}{'L':>4}{'D':>4}"
          f"{'AvgMoves/Win':>14}")
    for i, r in enumerate(standings):
        avg = f"{r.avg_moves_to_win:.1f}" if r.wins else "—"
        print(f"{i + 1:<5}{r.name:<22}{r.wins:>3}{r.losses:>4}{r.draws:>4}"
              f"{avg:>14}")


# ─── Demo / smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build a toy pool of 4 agents from rule-based stand-ins.
    np.random.seed(0)
    random.seed(0)

    agents = {
        "Random_1":   RandomAgent(),
        "Random_2":   RandomAgent(),
        "Strong_1":   StrongRuleAgent(),
        "Strong_2":   StrongRuleAgent(),
    }

    print("=== Single match ===")
    m = play_match(agents["Strong_1"], agents["Random_1"],
                   "Strong_1", "Random_1", n_games=4)
    print(f"{m.agent_a} vs {m.agent_b}: "
          f"{m.wins_a}-{m.wins_b} (draws {m.draws})")

    print("\n=== Round-robin pool ===")
    standings = run_pool(agents, n_games_per_match=2, verbose=True)
    print_standings(standings, "Pool final standings")

    print("\n=== Single-elim bracket ===")
    winner, _ = run_bracket(agents, n_games_per_match=2, verbose=True)
    print(f"\nBracket winner: {winner}")

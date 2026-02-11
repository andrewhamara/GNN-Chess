"""Sequential game loop using python-chess for mixed agent type matches."""

import datetime
from dataclasses import dataclass
from typing import List

import chess
import chess.pgn

from agents import Agent


@dataclass
class GameResult:
    outcome: int  # +1 white wins, 0 draw, -1 black wins
    pgn_string: str
    move_count: int
    white_name: str
    black_name: str


def play_game(
    white: Agent,
    black: Agent,
    max_plies: int = 512,
) -> GameResult:
    """Play a single game between two agents.

    Args:
        white: Agent playing white.
        black: Agent playing black.
        max_plies: Maximum number of half-moves before declaring a draw.

    Returns:
        GameResult with outcome, PGN, and metadata.
    """
    board = chess.Board()
    white.reset()
    black.reset()

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        agent = white if board.turn == chess.WHITE else black
        move = agent.select_move(board)
        board.push(move)

    # Determine outcome
    result_obj = board.outcome(claim_draw=True)
    if result_obj is None:
        # max_plies reached without conclusion
        outcome = 0
        result_str = "1/2-1/2"
    elif result_obj.winner is None:
        outcome = 0
        result_str = "1/2-1/2"
    elif result_obj.winner == chess.WHITE:
        outcome = 1
        result_str = "1-0"
    else:
        outcome = -1
        result_str = "0-1"

    # Build PGN
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Tournament"
    game.headers["Site"] = "GNN-Chess"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = white.name
    game.headers["Black"] = black.name
    game.headers["Result"] = result_str

    return GameResult(
        outcome=outcome,
        pgn_string=str(game),
        move_count=board.ply(),
        white_name=white.name,
        black_name=black.name,
    )


@dataclass
class MatchResult:
    games: List[GameResult]
    wins: int  # from agent1's perspective
    draws: int
    losses: int

    @property
    def score(self) -> float:
        total = self.wins + self.draws + self.losses
        if total == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / total


def play_match(
    agent1: Agent,
    agent2: Agent,
    n_games: int = 20,
    max_plies: int = 512,
) -> MatchResult:
    """Play a match of n_games between two agents, alternating colors.

    Args:
        agent1: First agent.
        agent2: Second agent.
        n_games: Number of games to play.
        max_plies: Maximum plies per game.

    Returns:
        MatchResult with individual game results and aggregate score
        from agent1's perspective.
    """
    results: List[GameResult] = []
    wins, draws, losses = 0, 0, 0

    for i in range(n_games):
        if i % 2 == 0:
            # agent1 plays white
            result = play_game(agent1, agent2, max_plies)
            if result.outcome == 1:
                wins += 1
            elif result.outcome == 0:
                draws += 1
            else:
                losses += 1
        else:
            # agent1 plays black
            result = play_game(agent2, agent1, max_plies)
            if result.outcome == -1:
                wins += 1
            elif result.outcome == 0:
                draws += 1
            else:
                losses += 1
        results.append(result)

    return MatchResult(games=results, wins=wins, draws=draws, losses=losses)

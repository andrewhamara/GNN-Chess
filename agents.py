"""Agent abstraction and implementations for round-robin tournament.

Provides a unified select_move(board) interface for GNN models,
Stockfish, and Leela Chess Zero.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional

import chess
import chess.engine
import jax
import jax.numpy as jnp
import mctx
import pgx

from action_conversion import chess_move_to_pgx_action, pgx_action_to_chess_move
from mcts import recurrent_fn
from models import ModelManager, load_model


@dataclass
class AgentConfig:
    name: str
    agent_type: str  # "gnn", "stockfish", "leela"
    params: Dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Base class for all tournament agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    def select_move(self, board: chess.Board) -> chess.Move:
        """Select a move given the current board position."""

    def reset(self):
        """Prepare for a new game. Override if needed."""

    def close(self):
        """Release resources. Override if needed."""

    @property
    def is_gnn(self) -> bool:
        return self.config.agent_type == "gnn"


class GNNAgent(Agent):
    """Agent wrapping a GNN model with MCTS search.

    Maintains a pgx State in sync with the python-chess Board
    by replaying moves through pgx.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        checkpoint_path: str = config.params["checkpoint_path"]
        model_name: str = config.params.get("model_name", config.name)
        self.n_sim: int = config.params.get("n_sim", 128)

        self.env = pgx.make("chess")
        self.model: ModelManager
        self.model_params: dict
        self.model, self.model_params = load_model(
            self.env, checkpoint_path, model_name
        )

        self._rng_key = jax.random.PRNGKey(
            config.params.get("seed", 42)
        )
        self._pgx_state: Optional[pgx.State] = None
        self._move_count = 0

    def reset(self):
        key, self._rng_key = jax.random.split(self._rng_key)
        self._pgx_state = self.env.init(key)
        self._move_count = 0

    def _sync_pgx_state(self, board: chess.Board):
        """Ensure pgx state is in sync with the python-chess board.

        If our internal move count doesn't match the board's ply count,
        we need to apply the opponent's last move to the pgx state.
        """
        if self._pgx_state is None:
            raise RuntimeError("Must call reset() before select_move()")

        board_ply = board.ply()

        # If we're behind, replay the missing moves
        while self._move_count < board_ply:
            # Get the move that was played at this ply from the board's move stack
            move = board.move_stack[self._move_count]
            is_black = self._move_count % 2 == 1
            pgx_action = chess_move_to_pgx_action(move, board, is_black)
            self._pgx_state = self.env.step(
                self._pgx_state, jnp.int32(pgx_action)
            )
            self._move_count += 1

    def select_move(self, board: chess.Board) -> chess.Move:
        self._sync_pgx_state(board)
        assert self._pgx_state is not None

        is_black = board.turn == chess.BLACK

        # Add batch dimension for model inference
        state_batch = jax.tree.map(
            lambda x: jnp.expand_dims(x, 0), self._pgx_state
        )

        logits, value = self.model(
            self.model.format_data(state=state_batch),
            legal_action_mask=state_batch.legal_action_mask,
            params=self.model_params,
        )

        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state_batch,
        )

        key, self._rng_key = jax.random.split(self._rng_key)
        policy_output = mctx.gumbel_muzero_policy(
            params=self.model_params,
            rng_key=key,
            root=root,
            recurrent_fn=partial(recurrent_fn, env=self.env, model=self.model),
            num_simulations=self.n_sim,
            invalid_actions=~state_batch.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )

        pgx_action = int(policy_output.action[0])

        # Step the pgx state forward with our chosen action
        self._pgx_state = self.env.step(
            self._pgx_state, jnp.int32(pgx_action)
        )
        self._move_count += 1

        return pgx_action_to_chess_move(pgx_action, board, is_black)


class StockfishAgent(Agent):
    """Agent wrapping Stockfish via UCI protocol."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        engine_path: str = config.params["engine_path"]
        self.threads: int = config.params.get("threads", 1)
        self.hash_mb: int = config.params.get("hash", 16)
        self.move_time: Optional[float] = config.params.get("movetime", None)
        self.total_time: Optional[float] = config.params.get("total_time", None)

        self._engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self._engine.configure({
            "Threads": self.threads,
            "Hash": self.hash_mb,
        })

        self._remaining_time: float = 0.0
        self._est_moves_remaining: int = 40

    def reset(self):
        self._remaining_time = self.total_time or 0.0
        self._est_moves_remaining = 40

    def select_move(self, board: chess.Board) -> chess.Move:
        if self.move_time is not None:
            limit = chess.engine.Limit(time=self.move_time)
        elif self.total_time is not None:
            # Divide remaining time by estimated moves remaining
            moves_left = max(self._est_moves_remaining, 1)
            time_for_move = self._remaining_time / moves_left
            limit = chess.engine.Limit(time=max(time_for_move, 0.001))
            self._remaining_time -= time_for_move
            self._est_moves_remaining = max(self._est_moves_remaining - 1, 1)
        else:
            limit = chess.engine.Limit(time=0.1)

        result = self._engine.play(board, limit)
        assert result.move is not None
        return result.move

    def close(self):
        self._engine.quit()


class LeelaAgent(Agent):
    """Agent wrapping Leela Chess Zero via UCI protocol."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        engine_path: str = config.params["engine_path"]
        self.nodes: int = config.params.get("nodes", 400)
        self.temperature: Optional[float] = config.params.get("temperature", None)
        weights_path: Optional[str] = config.params.get("weights_path", None)
        backend: str = config.params.get("backend", "cpu")
        threads: int = config.params.get("threads", 1)

        self._engine = chess.engine.SimpleEngine.popen_uci(engine_path)

        uci_options: Dict[str, Any] = {
            "Threads": threads,
            "Backend": backend,
        }
        if weights_path:
            uci_options["WeightsFile"] = weights_path
        if self.temperature is not None:
            uci_options["Temperature"] = self.temperature

        self._engine.configure(uci_options)

    def reset(self):
        pass

    def select_move(self, board: chess.Board) -> chess.Move:
        limit = chess.engine.Limit(nodes=self.nodes)
        result = self._engine.play(board, limit)
        assert result.move is not None
        return result.move

    def close(self):
        self._engine.quit()


def create_agent(config: AgentConfig) -> Agent:
    """Factory function to create an agent from config."""
    if config.agent_type == "gnn":
        return GNNAgent(config)
    elif config.agent_type == "stockfish":
        return StockfishAgent(config)
    elif config.agent_type == "leela":
        return LeelaAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")

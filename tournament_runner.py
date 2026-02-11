"""Tournament orchestrator for round-robin matches between mixed agent types.

Dispatches GNN-vs-GNN matches to the batched JAX full_pit() path and
all other matches to the sequential python-chess game driver.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import numpy as np
import pgx
import pgx.chess as pgc
import rich.progress as rp

from agents import Agent, AgentConfig, GNNAgent, create_agent
from elo import compute_elo
from game_driver import MatchResult, play_match
from mcts import full_pit
from utils import to_pgn


@dataclass
class TournamentConfig:
    agents: List[AgentConfig]
    n_games: int = 20
    n_games_batch: int = 64  # for GNN-vs-GNN batched path
    max_plies: int = 512
    n_sim: int = 128
    output_dir: str = "./tournament_results"
    seed: int = 42


class TournamentRunner:
    def __init__(self, config: TournamentConfig):
        self.config = config
        self.agents: Dict[str, Agent] = {}
        self.results: Dict[str, Dict[str, List[int]]] = {}
        self.pgn_games: List[str] = []

        self._env: Optional[pgx.Env] = None
        self._devices = jax.local_devices()
        self._n_devices = len(self._devices)
        self._rng_key = jax.random.PRNGKey(config.seed)

    def setup_agents(self):
        """Instantiate all agents from config."""
        for agent_config in self.config.agents:
            agent = create_agent(agent_config)
            self.agents[agent.name] = agent

    def _is_gnn_vs_gnn(self, name1: str, name2: str) -> bool:
        return self.agents[name1].is_gnn and self.agents[name2].is_gnn

    def _get_env(self) -> pgx.Env:
        if self._env is None:
            self._env = pgx.make("chess")
        return self._env

    def _run_gnn_match(
        self, name1: str, name2: str
    ) -> Tuple[int, int, int, List[str]]:
        """Run a GNN-vs-GNN match using the batched full_pit() path.

        Returns (wins, draws, losses) from agent1's perspective and PGN list.
        """
        agent1 = self.agents[name1]
        agent2 = self.agents[name2]
        assert isinstance(agent1, GNNAgent) and isinstance(agent2, GNNAgent)

        env = self._get_env()
        self._rng_key, subkey = jax.random.split(self._rng_key)

        n_games = self.config.n_games_batch
        # Ensure n_games is divisible by 2*n_devices
        divisor = 2 * self._n_devices
        n_games = max(divisor, (n_games // divisor) * divisor)

        R, games = full_pit(
            env,
            agent1.model,
            jax.device_put_replicated(agent1.model_params, self._devices),
            agent2.model,
            jax.device_put_replicated(agent2.model_params, self._devices),
            subkey,
            n_games=n_games,
            max_plies=self.config.max_plies,
            n_sim=self.config.n_sim,
            n_devices=self._n_devices,
        )

        wins = int((R == 1).sum())
        draws = int((R == 0).sum())
        losses = int((R == -1).sum())

        # Generate PGN for a sample of games
        pgns = []
        count = [min(3, n_games)] * 3  # up to 3 games per outcome type
        for r, g in zip(R, games):
            r_i = int(np.round(r))
            if count[r_i + 1] > 0:
                count[r_i + 1] -= 1
                pgns.append(to_pgn(
                    g,
                    round="Tournament",
                    player0=name1,
                    player1=name2,
                    result=r_i,
                    pgc=pgc,
                ))

        return wins, draws, losses, pgns

    def _run_sequential_match(
        self, name1: str, name2: str
    ) -> Tuple[int, int, int, List[str]]:
        """Run a match using the sequential python-chess game driver.

        Returns (wins, draws, losses) from agent1's perspective and PGN list.
        """
        agent1 = self.agents[name1]
        agent2 = self.agents[name2]

        match_result: MatchResult = play_match(
            agent1, agent2,
            n_games=self.config.n_games,
            max_plies=self.config.max_plies,
        )

        pgns = [g.pgn_string for g in match_result.games]
        return match_result.wins, match_result.draws, match_result.losses, pgns

    def _record_result(
        self, name1: str, name2: str, wins: int, draws: int, losses: int
    ):
        """Record match results symmetrically."""
        if name1 not in self.results:
            self.results[name1] = {}
        if name2 not in self.results:
            self.results[name2] = {}
        self.results[name1][name2] = [wins, draws, losses]
        self.results[name2][name1] = [losses, draws, wins]

    def run_tournament(self) -> Dict[str, int]:
        """Run the full round-robin tournament.

        Returns:
            Dictionary of agent name -> ELO rating.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)

        agent_names = list(self.agents.keys())
        pairings = [
            (agent_names[i], agent_names[j])
            for i in range(len(agent_names))
            for j in range(i + 1, len(agent_names))
        ]

        with rp.Progress(
            *rp.Progress.get_default_columns(),
            rp.TimeElapsedColumn(),
            rp.MofNCompleteColumn(),
            rp.TextColumn("{task.fields[logs]}"),
        ) as progress:
            task = progress.add_task(
                "[green]Tournament",
                total=len(pairings),
                logs="Starting...",
            )

            for name1, name2 in pairings:
                progress.update(
                    task, logs=f"{name1} vs {name2}"
                )

                if self._is_gnn_vs_gnn(name1, name2):
                    wins, draws, losses, pgns = self._run_gnn_match(
                        name1, name2
                    )
                else:
                    wins, draws, losses, pgns = self._run_sequential_match(
                        name1, name2
                    )

                self._record_result(name1, name2, wins, draws, losses)
                self.pgn_games.extend(pgns)

                # Save PGN for this match
                pgn_path = os.path.join(
                    self.config.output_dir,
                    f"{name1} vs {name2}.pgn",
                )
                with open(pgn_path, "w") as f:
                    f.write("\n\n".join(pgns))

                progress.update(
                    task,
                    advance=1,
                    logs=f"{name1} vs {name2}: {wins}W/{draws}D/{losses}L",
                )
                print(
                    f"{name1} vs {name2}: "
                    f"{wins}W / {draws}D / {losses}L"
                )

        # Compute ELO ratings
        elos = compute_elo(self.results)

        # Save results
        results_path = os.path.join(self.config.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "results": self.results,
                "elos": elos,
            }, f, indent=2)

        return elos

    def cleanup(self):
        """Close all engine processes."""
        for agent in self.agents.values():
            agent.close()

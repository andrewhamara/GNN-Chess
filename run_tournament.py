"""CLI entry point for the round-robin tournament system."""

import argparse
import json
import sys

from agents import AgentConfig
from tournament_runner import TournamentConfig, TournamentRunner


def build_default_agents(
    stockfish_path: str = "stockfish",
    leela_path: str = "lc0",
    gnn_base_checkpoint: str = "./models/base/checkpoint.ckpt",
    gnn_large_checkpoint: str = "./models/large/checkpoint.ckpt",
    edgenet_checkpoint: str = "./models/chess_2024-02-05:14h08/000050.ckpt",
    edgenet2_checkpoint: str = "./models/chess_2024-08-20:00h13/000499.ckpt",
    n_sim: int = 128,
) -> list:
    """Build the default set of 10 agents for the tournament."""
    return [
        # Heterogeneous GNN models
        AgentConfig(
            name="HeteroEdgeNet-base",
            agent_type="gnn",
            params={
                "checkpoint_path": gnn_base_checkpoint,
                "n_sim": n_sim,
            },
        ),
        AgentConfig(
            name="HeteroEdgeNet-large",
            agent_type="gnn",
            params={
                "checkpoint_path": gnn_large_checkpoint,
                "n_sim": n_sim,
            },
        ),
        # Homogeneous GNN models (auto-detected from checkpoint config)
        AgentConfig(
            name="EdgeNet",
            agent_type="gnn",
            params={
                "checkpoint_path": edgenet_checkpoint,
                "n_sim": n_sim,
            },
        ),
        AgentConfig(
            name="EdgeNet2",
            agent_type="gnn",
            params={
                "checkpoint_path": edgenet2_checkpoint,
                "n_sim": n_sim,
            },
        ),
        # Leela Chess Zero variants
        AgentConfig(
            name="Leela-policy-only",
            agent_type="leela",
            params={
                "engine_path": leela_path,
                "nodes": 1,
            },
        ),
        AgentConfig(
            name="Leela-value-only",
            agent_type="leela",
            params={
                "engine_path": leela_path,
                "nodes": 400,
                "temperature": 10.0,
            },
        ),
        AgentConfig(
            name="Leela-400-MCTS",
            agent_type="leela",
            params={
                "engine_path": leela_path,
                "nodes": 400,
            },
        ),
        # Stockfish variants
        AgentConfig(
            name="Stockfish-50ms",
            agent_type="stockfish",
            params={
                "engine_path": stockfish_path,
                "movetime": 0.05,
            },
        ),
        AgentConfig(
            name="Stockfish-1.5s",
            agent_type="stockfish",
            params={
                "engine_path": stockfish_path,
                "total_time": 1.5,
            },
        ),
    ]


def load_agents_from_config(config_path: str) -> list:
    """Load agent configs from a JSON file."""
    with open(config_path) as f:
        data = json.load(f)
    return [AgentConfig(**agent) for agent in data["agents"]]


def main():
    parser = argparse.ArgumentParser(
        prog="run_tournament",
        description="Run a round-robin chess tournament between GNN, Stockfish, and Leela agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file with custom agent definitions",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=20,
        help="Number of games per match (sequential path)",
    )
    parser.add_argument(
        "--n-games-batch",
        type=int,
        default=64,
        help="Number of games per match (GNN-vs-GNN batched path)",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=128,
        help="Number of MCTS simulations per move (GNN agents)",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=512,
        help="Maximum half-moves per game",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tournament_results",
        help="Directory for output files (PGN, results JSON)",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="stockfish",
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--leela-path",
        type=str,
        default="lc0",
        help="Path to Leela Chess Zero binary",
    )
    parser.add_argument(
        "--gnn-base-checkpoint",
        type=str,
        default="./models/base/checkpoint.ckpt",
        help="Path to HeteroEdgeNet base checkpoint",
    )
    parser.add_argument(
        "--gnn-large-checkpoint",
        type=str,
        default="./models/large/checkpoint.ckpt",
        help="Path to HeteroEdgeNet large checkpoint",
    )
    parser.add_argument(
        "--edgenet-checkpoint",
        type=str,
        default="./models/chess_2024-02-05:14h08/000050.ckpt",
        help="Path to EdgeNet (original homogeneous GNN) checkpoint",
    )
    parser.add_argument(
        "--edgenet2-checkpoint",
        type=str,
        default="./models/chess_2024-08-20:00h13/000499.ckpt",
        help="Path to EdgeNet2 (improved homogeneous GNN) checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Build agent configs
    if args.config:
        agents = load_agents_from_config(args.config)
    else:
        agents = build_default_agents(
            stockfish_path=args.stockfish_path,
            leela_path=args.leela_path,
            gnn_base_checkpoint=args.gnn_base_checkpoint,
            gnn_large_checkpoint=args.gnn_large_checkpoint,
            edgenet_checkpoint=args.edgenet_checkpoint,
            edgenet2_checkpoint=args.edgenet2_checkpoint,
            n_sim=args.n_sim,
        )

    config = TournamentConfig(
        agents=agents,
        n_games=args.n_games,
        n_games_batch=args.n_games_batch,
        max_plies=args.max_plies,
        n_sim=args.n_sim,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    runner = TournamentRunner(config)

    try:
        print(f"Setting up {len(agents)} agents...")
        runner.setup_agents()

        print(f"Running round-robin tournament ({len(agents)} agents)...")
        elos = runner.run_tournament()

        print("\n" + "=" * 50)
        print("Final ELO Ratings")
        print("=" * 50)
        for name, elo in sorted(elos.items(), key=lambda x: -x[1]):
            print(f"  {name:<30s} {elo:>5d}")
        print("=" * 50)
        print(f"\nResults saved to {args.output_dir}/")

    except KeyboardInterrupt:
        print("\nTournament interrupted.")
        sys.exit(1)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()

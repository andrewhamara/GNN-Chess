"""Play interactively against a trained GNN chess model."""

# LLVM workaround (must precede JAX imports)
import os
os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '') +
    ' --xla_gpu_enable_triton_gemm=false'
    ' --xla_gpu_triton_gemm_any=false'
    ' --xla_gpu_force_compilation_parallelism=1'
)

import jax
jax.config.update('jax_default_matmul_precision', 'float32')

import argparse
import glob
import sys

import chess

from agents import AgentConfig, GNNAgent


def find_latest_checkpoint() -> str:
    """Auto-discover the most recent checkpoint in models/.

    Model dirs follow models/chess_YYYY-MM-DD:HHhMM/ which is
    lexicographically sortable. Picks the last chess_* dir, then the
    highest-numbered .ckpt file inside it.
    """
    model_dirs = sorted(glob.glob("models/chess_*"))
    if not model_dirs:
        print("Error: No model directories found in models/chess_*")
        sys.exit(1)

    latest_dir = model_dirs[-1]
    checkpoints = sorted(glob.glob(os.path.join(latest_dir, "*.ckpt")))
    if not checkpoints:
        print(f"Error: No .ckpt files found in {latest_dir}")
        sys.exit(1)

    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Play interactively against a trained GNN chess model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-discovers latest if omitted)",
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
        help="Side for the human player",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=128,
        help="MCTS simulations per model move",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Resolve checkpoint
    checkpoint = args.checkpoint or find_latest_checkpoint()
    print(f"Loading model from: {checkpoint}")

    human_is_white = args.color == "white"

    config = AgentConfig(
        name="GNN",
        agent_type="gnn",
        params={
            "checkpoint_path": checkpoint,
            "n_sim": args.n_sim,
            "seed": args.seed,
        },
    )
    agent = GNNAgent(config)
    agent.reset()

    board = chess.Board()
    print()
    print(board.unicode(invert_color=not human_is_white))
    print()

    while not board.is_game_over():
        human_turn = (board.turn == chess.WHITE) == human_is_white

        if human_turn:
            # Human move
            while True:
                try:
                    raw = input("Your move (UCI, e.g. e2e4): ").strip()
                except EOFError:
                    print("\nGoodbye!")
                    return
                if raw.lower() in ("quit", "resign"):
                    print("You resigned.")
                    return
                try:
                    move = chess.Move.from_uci(raw)
                except ValueError:
                    print(f"Invalid UCI format: {raw}")
                    continue
                if move not in board.legal_moves:
                    print(f"Illegal move: {move}")
                    continue
                break
            board.push(move)
        else:
            # Model move
            print("Model is thinking...")
            move = agent.select_move(board)
            print(f"Model plays: {move.uci()}")
            board.push(move)

        print()
        print(board.unicode(invert_color=not human_is_white))
        print()

    # Game over
    result = board.result()
    print(f"Game over: {result}")
    outcome = board.outcome()
    if outcome and outcome.winner is not None:
        winner = "White" if outcome.winner else "Black"
        side = "You" if (outcome.winner == human_is_white) else "Model"
        print(f"{winner} ({side}) wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    main()

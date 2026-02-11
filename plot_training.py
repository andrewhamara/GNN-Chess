#!/usr/bin/env python3
"""Plot training progress from CSV log file."""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_log(csv_path):
    """Plot training metrics from CSV log."""
    df = pd.read_csv(csv_path)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Progress - {Path(csv_path).parent.name}', fontsize=16)

    # Plot 1: Loss over time
    ax = axes[0, 0]
    ax.plot(df['iteration'], df['loss'], label='Total Loss', linewidth=2)
    ax.plot(df['iteration'], df['policy_loss'], label='Policy Loss', alpha=0.7)
    ax.plot(df['iteration'], df['value_loss'], label='Value Loss', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Win/Draw/Loss rates
    ax = axes[0, 1]
    # Filter out NaN values (iterations without eval)
    eval_df = df.dropna(subset=['win_rate'])
    if len(eval_df) > 0:
        ax.plot(eval_df['iteration'], eval_df['win_rate'], 'g-', label='Win Rate', linewidth=2)
        ax.plot(eval_df['iteration'], eval_df['draw_rate'], 'y-', label='Draw Rate', linewidth=2)
        ax.plot(eval_df['iteration'], eval_df['lose_rate'], 'r-', label='Lose Rate', linewidth=2)
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='50%')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Rate')
        ax.set_title('Win/Draw/Loss vs Baseline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'No evaluation data yet',
                ha='center', va='center', transform=ax.transAxes)

    # Plot 3: ELO rating
    ax = axes[1, 0]
    if len(eval_df) > 0:
        ax.plot(eval_df['iteration'], eval_df['elo_rating'], 'b-', linewidth=2)
        ax.axhline(y=1000, color='k', linestyle='--', alpha=0.3, label='Baseline (1000)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELO Rating')
        ax.set_title('Estimated ELO vs Baseline')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No evaluation data yet',
                ha='center', va='center', transform=ax.transAxes)

    # Plot 4: Gradient magnitude
    ax = axes[1, 1]
    ax.semilogy(df['iteration'], df['max_grad'], 'purple', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Gradient (log scale)')
    ax.set_title('Gradient Magnitude')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save plot
    plot_path = Path(csv_path).parent / 'training_progress.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # Also save to current directory for easy viewing
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print(f"Plot also saved to: ./training_progress.png")

    plt.show()

def find_latest_log():
    """Find the most recent training log."""
    models_dir = Path('models')
    if not models_dir.exists():
        return None

    log_files = list(models_dir.glob('*/training_log.csv'))
    if not log_files:
        return None

    # Sort by modification time
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return log_files[0]

if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = find_latest_log()
        if csv_path is None:
            print("No training logs found in models/*/training_log.csv")
            print("\nUsage:")
            print("  python plot_training.py [path/to/training_log.csv]")
            sys.exit(1)
        print(f"Using latest log: {csv_path}")

    plot_training_log(csv_path)

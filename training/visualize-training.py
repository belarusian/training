"""
Visualize GRPO training metrics - proper chart with history.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd


def get_latest_grpo_run():
    """Find the most recent GRPO run by looking at summary file mtime."""
    wandb_dir = "wandb"
    
    runs = []
    for summary_file in glob.glob(os.path.join(wandb_dir, "run-*", "files", "wandb-summary.json")):
        run_dir = os.path.dirname(os.path.dirname(summary_file))
        mtime = os.path.getmtime(summary_file)
        runs.append((run_dir, mtime))
    
    if not runs:
        print("No wandb runs with summary files found!")
        return None
    
    runs.sort(key=lambda x: x[1], reverse=True)
    return runs[0][0]


def get_wandb_history(run_id):
    """Get full training history from wandb API."""
    import wandb
    api = wandb.Api()
    run = api.run(f'kodep-sasha-cs-boutique/effect-codegen-grpo/{run_id}')
    history = run.history()
    
    # Filter out NaN rows
    df = history
    if 'train/reward' in df.columns:
        df = df[df['train/reward'].notna()]
    
    return df


def plot_training(run_dir, run_id):
    """Create training metrics chart."""
    print(f"Analyzing run: {run_dir}")
    print(f"Run ID: {run_id}")
    
    # Get history from wandb
    df = get_wandb_history(run_id)
    
    if df.empty:
        print("No training data found!")
        return
    
    print(f"Found {len(df)} training steps with data")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"GRPO Training - Run {run_id}", fontsize=14, fontweight='bold')
    
    # 1. Reward over time
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['train/reward'], label='Reward', color='green', linewidth=2, marker='o', markersize=3)
    if 'train/reward_std' in df.columns:
        ax1.fill_between(df.index, 
                         df['train/reward'] - df['train/reward_std'],
                         df['train/reward'] + df['train/reward_std'],
                         alpha=0.2, color='green', label='± Std')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. KL Divergence
    ax2 = axes[0, 1]
    ax2.plot(df.index, df['train/kl'], color='purple', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('KL Divergence (Policy Change)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss and Grad Norm
    ax3 = axes[1, 0]
    if 'train/loss' in df.columns:
        ax3.plot(df.index, df['train/loss'], label='Loss', color='red', linewidth=2, marker='^', markersize=3)
    if 'train/grad_norm' in df.columns:
        ax3.plot(df.index, df['train/grad_norm'], label='Grad Norm', color='orange', linewidth=2, marker='v', markersize=3)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Value')
    ax3.set_title('Loss and Gradient Norm')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward components and completion length
    ax4 = axes[1, 1]
    if 'train/rewards/reward_fn/mean' in df.columns:
        ax4.plot(df.index, df['train/rewards/reward_fn/mean'], label='Reward (fn mean)', color='blue', linewidth=2)
    if 'train/completion_length' in df.columns:
        ax4.plot(df.index, df['train/completion_length'], label='Completion Length', color='brown', linewidth=2)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Value')
    ax4.set_title('Reward Components & Length')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(run_dir, "training_chart.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved chart to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total steps: {len(df)}")
    print(f"Final reward: {df['train/reward'].iloc[-1]:.4f} ± {df['train/reward_std'].iloc[-1]:.4f}")
    print(f"Initial reward: {df['train/reward'].iloc[0]:.4f} ± {df['train/reward_std'].iloc[0]:.4f}")
    print(f"Reward improvement: {(df['train/reward'].iloc[-1] - df['train/reward'].iloc[0]):.4f}")
    print(f"Final KL: {df['train/kl'].iloc[-1]:.4f}")
    print(f"Final loss: {df['train/loss'].iloc[-1]:.6f}")
    print(f"Final grad norm: {df['train/grad_norm'].iloc[-1]:.4f}")
    
    plt.show()


if __name__ == "__main__":
    run_dir = get_latest_grpo_run()
    if run_dir:
        run_id = os.path.basename(run_dir).split('-')[-1]
        plot_training(run_dir, run_id)

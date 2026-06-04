import wandb

api = wandb.Api()
runs = api.runs('effect-codegen-grpo')

print(f'Found {len(runs)} runs')

for r in runs[:10]:
    print(f'\nRun: {r.name}')
    print(f'  State: {r.state}')
    print(f'  Reward: {r.summary.get("reward", "N/A")}')
    print(f'  KL: {r.summary.get("train/kl", "N/A")}')
    print(f'  Loss: {r.summary.get("train/loss", "N/A")}')

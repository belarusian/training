# Training scripts for Qwen3-4B Effect TypeScript generation

This directory contains training scripts and utilities:

## Scripts

### test_setup.py
Test GPU setup and Unsloth installation:
```bash
python scripts/test_setup.py
```

### fix_grpo.py
Fix GRPO training notebook (deprecated - now in notebook directly):
```bash
python scripts/fix_grpo.py training/main-training.ipynb
```

### insert_reward.py
Insert reward function into GRPO training notebook (deprecated - now in notebook directly):
```bash
python scripts/insert_reward.py training/main-training.ipynb
```

### train_qwen3_grpo_effect_codegen.py
Standalone Python training script (alternative to notebook):
```bash
python train_qwen3_grpo_effect_codegen.py
```

## Batch Files

### train.bat
Windows batch file to run training via JupyterLab:
```cmd
train.bat
```

## Data Extraction

### extract-code.ts (in root directory)
Extract training data from Effect repositories:
```bash
bun run extract-code.ts
```

## Notes

- The main training workflow is now in `training/main-training.ipynb`
- Scripts in this directory are legacy/alternative options
- Training data is extracted via `bun run extract-code.ts` and stored in `extracted-code/`

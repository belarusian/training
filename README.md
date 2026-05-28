# Training

Effect-style TypeScript fine-tuning with GRPO on Qwen3-4B.

## Overview

This project trains a language model to generate high-quality Effect-style TypeScript code using Reinforcement Learning (GRPO).

## What's Inside

- **examples/** - Learning resources (4 notebooks studying Effect patterns)
- **training/** - Your training notebooks and configurations
- **data/extracted-code/** - Training data directory
- **extracted-code/** - 428 extracted Effect TypeScript code samples
- **scripts/** - Utilities: `fix_grpo`, `insert_reward`, `test_setup`
- **train_qwen3_grpo_effect_codegen.py** - Main GRPO training script

## Setup

```bash
uv venv unsloth_env --python 3.13
.\unsloth_env\Scripts\activate
uv pip install unsloth --torch-backend=auto
```

## Usage

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

## Hardware

- GPU: NVIDIA GeForce RTX 4090 (24GB VRAM)
- CUDA: 13.0
- PyTorch: 2.10.0+cu130

## Topics

#effect #typescript #reinforcement-learning #grpo #fine-tuning #qwen #unsloth

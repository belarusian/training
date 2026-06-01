# Training Playbook

## Quick Start

### Stage 1: SFT (Supervised Fine-Tuning)
```powershell
.\unsloth_env\Scripts\python.exe training\run-sft-training-1.py
```
- Trains on 428 Effect code samples
- Output: `training/output-v2/qwen3-4b-effect-codegen-sft/`
- Time: ~7-10 minutes

### Stage 2: GRPO (Reinforcement Fine-Tuning)
```powershell
.\unsloth_env\Scripts\python.exe training\run-grpo-training-2.py
```
- Requires SFT output from Stage 1
- Trains with reward signals
- Output: `training/output-v2/qwen3-4b-effect-codegen-v2/`
- Time: ~60-90 minutes

---

## Testing Your Training

### After Stage 1 (SFT)
```powershell
.\unsloth_env\Scripts\python.exe training\test-training.py --stage sft
```

### After Stage 2 (GRPO)
```powershell
.\unsloth_env\Scripts\python.exe training\test-training.py --stage grpo
```

## Convert to GGUF (for llama.cpp inference)

After training is complete, convert to GGUF format:

```powershell
# Default (use GRPO model)
.\unsloth_env\Scripts\python.exe training\convert-to-gguf.py

# Specify a different model
.\unsloth_env\Scripts\python.exe training\convert-to-gguf.py --model training/output-v2/qwen3-4b-effect-codegen-sft

# Different quantization methods
.\unsloth_env\Scripts\python.exe training\convert-to-gguf.py --quant q8_0
.\unsloth_env\Scripts\python.exe training\convert-to-gguf.py --quant q5_k_m
```

Output: Same directory with GGUF files (e.g., `qwen3-4b-effect-codegen-v2.Q4_K_M.gguf`)

---

## Reward Function (GRPO)

The GRPO phase uses these reward signals:
- **+1.0** for `<CODE>` tags present
- **+0.5** for Effect imports (`from effect`)
- **+0.3** for Schema usage
- **+0.2** for exports
- **+0.3 to +0.7** based on response length
- **-0.5** if response < 100 chars

---

## File Structure

```
training/
├── run-sft-training-1.py    # Stage 1: SFT
├── run-grpo-training-2.py   # Stage 2: GRPO (requires SFT first)
├── test-training.py         # Test script (--stage sft | grpo)
├── convert-to-gguf.py       # Convert to GGUF for llama.cpp
├── run-training.py          # Original GRPO-only (keep for reference)
├── run-training-v1.py       # Original GRPO-only (keep for reference)
├── main-training.ipynb      # Notebook reference
└── output-v2/               # Training outputs (gitignored)
    ├── qwen3-4b-effect-codegen-sft/
    └── qwen3-4b-effect-codegen-v2/
```

---

## Notes

- SFT loads model in **full precision** (float16, not quantized)
- GRPO requires SFT to complete first (enforced by check)
- Both scripts auto-convert to GGUF Q4_K_M format after training
- WandB logging is optional (enabled if `wandb` package installed)

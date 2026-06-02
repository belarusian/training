# Post-Training Summary: Qwen3-4B Effect TypeScript Fine-Tuning

**Training Date:** June 1, 2026  
**GPU:** NVIDIA GeForce RTX 4090 (24GB VRAM)  
**CUDA:** 13.0  
**Framework:** Unsloth 2026.5.8 + TRL GRPO

---

## Training Pipeline Overview

We completed a **two-stage training pipeline**:

### Stage 1: SFT (Supervised Fine-Tuning)
- **Duration:** ~7.5 minutes
- **Learning rate:** 2e-5
- **Batch size:** 1 (gradient accumulation x4)
- **Final loss:** 0.834
- **Output:** `training/output-v2/qwen3-4b-effect-codegen-sft/`

### Stage 2: GRPO (Reinforcement Learning)
- **Duration:** ~55.5 minutes
- **Learning rate:** 2e-6
- **Batch size:** 4 (gradient accumulation x4)
- **Final reward:** 0.9775
- **Reward std:** 0.212
- **KL divergence:** 0.19
- **Output:** `training/output-v2/qwen3-4b-effect-codegen-v2/`

---

## Reward Function Details

The GRPO phase used these reward signals:

| Component | Reward | Description |
|-----------|--------|-------------|
| `<CODE>` tags | +1.0 | Response contains Effect code |
| Effect imports | +0.5 | Uses `from effect` or `import Effect` |
| Schema usage | +0.3 | Uses Effect Schema |
| Exports | +0.2 | Includes export statements |
| Length (200-1000 chars) | +0.5 | Optimal response length |
| Length (<100 chars) | -0.5 | Too short |

---

## Training Metrics

### Loss Curve
- **SFT stage:** Loss decreased from ~1.5 → 0.834
- **GRPO stage:** Loss remained low (~0.0002) with small gradient updates

### Reward Progression
- **Initial:** 0.7763 ± 0.1431
- **Final:** 0.9775 ± 0.2134
- **Improvement:** +0.2012 (+26% relative increase)

### KL Divergence
- **Final value:** 0.19
- **Interpretation:** Healthy - policy is learning without catastrophic forgetting

### Gradient Norm
- **Final value:** 0.282
- **Interpretation:** Stable - not too small (learning) or too large (unstable)

---

## Model Information

**Model ID:** `Kodep/qwen3-4b-effect-codegen-v2`  
**Base Model:** `unsloth/Qwen3-4B`  
**Adapter Type:** LoRA (rank 64)  
**Training Data:** 428 Effect TypeScript code samples  
**Total Trainable Parameters:** 132M (3.18% of base model)

---

## How to Use

### Load with Unsloth
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Kodep/qwen3-4b-effect-codegen-v2",
    max_seq_length=4096,
    load_in_4bit=True,  # For lower VRAM
)

FastLanguageModel.for_inference(model)

inputs = tokenizer([
    "You are an expert TypeScript developer specializing in the Effect framework.\n"
    "Analyze the requirement and generate high-quality Effect code.\n"
    "Requirement: Create a simple HTTP server that responds with JSON",
], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

### Load with Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Kodep/qwen3-4b-effect-codegen-v2",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("Kodep/qwen3-4b-effect-codegen-v2")
```

---

## Training Scripts

| Script | Purpose |
|--------|---------|
| `training/run-sft-training-1.py` | Stage 1: SFT training |
| `training/run-grpo-training-2.py` | Stage 2: GRPO training |
| `training/test-training.py` | Test trained model |
| `training/visualize-training.py` | Visualize wandb metrics |
| `training/evaluate-model.py` | Standalone evaluation |
| `training/compare-models.py` | Compare local vs remote models |

---

## WandB Dashboard

View training metrics at: https://wandb.ai/kodep-sasha-cs-boutique/effect-codegen-grpo

Key charts:
- Reward over time (showing +0.20 improvement)
- KL divergence (stable at 0.19)
- Loss and gradient norm

---

## Known Limitations

1. **GGUF Conversion:** Currently failing due to llama.cpp subprocess issues on Windows. Working on a fix.
2. **WandB History:** Only 15 data points synced (wandb's default sampling). Full 214 steps completed.
3. **Batch Size:** GRPO uses batch size 4 which limits gradient diversity. Consider increasing for better results.

---

## Next Steps

1. **Test model** with `training/test-training.py --stage grpo`
2. **Fix GGUF conversion** for llama.cpp inference
3. **Experiment with learning rate** - current grad_norm is small, might benefit from higher LR
4. **More training data** - 428 samples is limited, consider extracting more from Effect repos

---

## Git Commit

All code committed to: `859313f`  
Branch: `master`  
Remote: `https://github.com/belarusian/training.git`

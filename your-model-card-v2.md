---
base_model: unsloth/Qwen3-4B
tags:
- effect
- typescript
- reinforcement-learning
- grpo
- fine-tuning
- qwen
- unsloth
- lora
- text-generation-inference
- transformers
- trl
license: apache-2.0
language:
- en
pipeline_tag: text-generation
---

# Qwen3-4B Effect TypeScript Code Generation (v2)

Fine-tuned Qwen3-4B model specialized in generating high-quality **Effect-style TypeScript code** using a two-stage training pipeline: Supervised Fine-Tuning (SFT) + Group Relative Policy Optimization (GRPO).

## Model Overview

**Qwen3-4B Effect Codegen v2** has the following features:

- **Base Model**: Qwen3-4B (Qwen3ForCausalLM)
- **Number of Layers**: 36
- **Hidden Size**: 2560
- **Number of Attention Heads**: 32 (Q), 8 (KV)
- **Vocabulary Size**: 151936
- **Max Position Embeddings**: 40960
- **LoRA Rank**: 64
- **Trainable Parameters**: 132M (3.18% of base model)
- **Precision**: float16
- **License**: Apache 2.0

## Training Details

### Training Data
- **428** TypeScript code samples extracted from:
  - `effect-smol` — 185 samples
  - `effect` — 208 samples
  - `opencode` — 28 samples
  - `effect-examples` — 7 samples
- Sources: Real Effect.js library code, OpenCode LLM integrations, and Effect examples

### Training Pipeline (Two-Stage)

**Stage 1: Supervised Fine-Tuning (SFT)**
- Duration: ~7.5 minutes on RTX 4090
- Learning rate: 2e-5
- Epochs: 2
- Final loss: 0.834
- Teaches code format and structure

**Stage 2: Group Relative Policy Optimization (GRPO)**
- Duration: ~63 minutes on RTX 4090
- Learning rate: 2e-6
- Batch size: 4
- Final reward: 0.9775 ± 0.2134
- Reward improvement: +0.2012 (+26% relative increase)
- KL divergence: 0.19 (healthy - policy learning without catastrophic forgetting)

### Reward Function (GRPO)
| Component | Reward |
|-----------|--------|
| `<CODE>` tags | +1.0 |
| Effect imports | +0.5 |
| Schema usage | +0.3 |
| Exports | +0.2 |
| Length 200-1000 chars | +0.5 |
| Length <100 chars | -0.5 |

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-4B |
| LoRA rank | 64 |
| Max sequence | 4096 |
| SFT learning rate | 2e-5 |
| GRPO learning rate | 2e-6 |
| SFT epochs | 2 |
| GRPO epochs | 1 |
| Optimizer | adamw_8bit |
| Gradient accumulation | 4x |

### Hardware
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **CUDA**: 13.0
- **PyTorch**: 2.10.0+cu130
- **Unsloth**: 2026.5.8

## How to Use

### With Unsloth (recommended for faster inference)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Kodep/qwen3-4b-effect-codegen-v2",
    max_seq_length=4096,
    load_in_4bit=True,
)

messages = [
    {"role": "system", "content": "You are an expert TypeScript developer specializing in the Effect framework."},
    {"role": "user", "content": "Generate an Effect service pattern for a user repository"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### With Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Kodep/qwen3-4b-effect-codegen-v2")
model = AutoModelForCausalLM.from_pretrained(
    "Kodep/qwen3-4b-effect-codegen-v2",
    torch_dtype=torch.float16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are an expert TypeScript developer specializing in the Effect framework."},
    {"role": "user", "content": "Generate an Effect service pattern for a user repository"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=1024, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## What Makes This Model Unique

This is the first model fine-tuned specifically for **Effect TypeScript code generation**. Unlike general-purpose code models, it learns:

- Effect imports and core patterns (`Effect.succeed`, `Effect.flatMap`, `Effect.gen`, etc.)
- Effect Schema definitions (`Schema`, `decodeSync`, `make`, etc.)
- Effect service patterns (`Layer`, `Context`, `provide`, etc.)
- Proper TypeScript exports and types
- Functional programming patterns in TypeScript

## Evaluation

### Training Metrics
| Metric | Value |
|--------|-------|
| SFT Final Loss | 0.834 |
| GRPO Final Reward | 0.9775 ± 0.2134 |
| GRPO Reward Improvement | +0.2012 (+26%) |
| KL Divergence | 0.19 |
| Gradient Norm | 0.282 |

### Limitations
- Fine-tuned on a small dataset (428 samples) — may not cover all Effect patterns
- May generate syntactically valid but logically incorrect code
- Not suitable for production use without evaluation
- Training focused on code format and Effect patterns, not correctness verification
- Does not aim to compete with general-purpose models on math, reasoning, or multi-modal tasks

## Related Resources

- [Effect TypeScript Library](https://effect.website/)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Training Repository](https://github.com/belarusian/training)
- [Model Card v1](https://huggingface.co/Kodep/qwen3-4b-effect-codegen)

## Citation

```bibtex
@misc{qwen3-4b-effect-codegen-v2,
  author = {Kodep},
  title = {Qwen3-4B Effect TypeScript Code Generation (v2)},
  year = {2026},
  url = {https://huggingface.co/Kodep/qwen3-4b-effect-codegen-v2}
}
```

## License

Released under the Apache 2.0 license.

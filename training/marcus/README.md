# Marcus Aurelius Training Pipeline

This directory contains training scripts for fine-tuning Qwen3-4B to become an expert in Stoic philosophy and Marcus Aurelius' teachings.

## Architecture

```
marcus/
├── run-sft-training.py      # Stage 1: Supervised fine-tuning on Meditations
├── run-grpo-training.py     # Stage 2: GRPO with citation-aware rewards
├── data-generator.py        # Generate Q&A pairs from Meditations
├── reward-functions.py      # Reward for citations, consistency, brevity
└── README.md                # This file
```

## Approach: Knowledge Distillation

### Teacher Model
- **Gemma 4** (16GB VRAM) - The "expert"
- **RAG** - Retrieves relevant Meditations passages
- **Output**: High-quality, citation-rich responses

### Student Model
- **Qwen3-4B** (or smaller) - The "student"
- **Fine-tuned** on teacher's outputs
- **Goal**: Achieve similar quality with minimal footprint

## Training Pipeline

### Stage 1: SFT (Supervised Fine-Tuning)

**Goal**: Teach the model Marcus' style and reasoning patterns.

**Data Generation**:
```powershell
.\unsloth_env\Scripts\python.exe training\marcus\data-generator.py --mode synthetic
```

This creates Q&A pairs where:
- **Question**: A modern philosophical/practical dilemma
- **Answer**: Gemma 4 + RAG response citing Meditations

**Training**:
```powershell
.\unsloth_env\Scripts\python.exe training\marcus\run-sft-training.py
```

**Expected Output**:
- Loss < 1.5 after 2 epochs
- Model generates coherent Stoic-style responses

### Stage 2: GRPO (Reinforcement Learning)

**Goal**: Improve citation accuracy and philosophical consistency.

**Reward Function**:
| Component | Reward | Description |
|-----------|--------|-------------|
| Citation present | +1.0 | Response includes Book X, Section Y |
| Citation accurate | +0.5 | Citation matches Meditations text |
| Philosophical consistency | +0.3 | Response aligns with Stoic principles |
| Brevity (200-500 chars) | +0.5 | Concise but complete |
| Response < 100 chars | -0.5 | Too short |

**Training**:
```powershell
.\unsloth_env\Scripts\python.exe training\marcus\run-grpo-training.py
```

**Expected Output**:
- Reward > 0.8
- KL divergence < 0.2 (healthy learning)

## Data Sources

### Primary Source
- **Meditations** by Marcus Aurelius (`../meditations.mb.txt`)
- Parsed into passages (Book X, Section Y format)

### Synthetic Data Generation
The `data-generator.py` script:
1. Extracts Meditations passages
2. Generates questions that each passage answers
3. Uses Gemma 4 + RAG to generate high-quality answers
4. Creates training dataset in chat format

```python
# Example output
{
  "messages": [
    {"role": "user", "content": "How should I respond when someone insults me?"},
    {"role": "assistant", "content": "As Marcus writes in Book 4, Section 1: 'When another blames you or hates you...'\n\nThe Stoic approach is to pause, recognize that others act based on their own perceptions, and choose your response deliberately."}
  ]
}
```

## Hardware Requirements

| Stage | VRAM | Time | Notes |
|-------|------|------|-------|
| SFT | ~6GB | ~15 min | 4-bit quantization |
| GRPO | ~8GB | ~45 min | 8-bit for stability |

## Output Artifacts

```
training/output-marcus/
├── qwen3-4b-marcus-sft/      # SFT LoRA adapter (~500MB)
├── qwen3-4b-marcus-v2/       # GRPO LoRA adapter (~500MB)
├── eval-metrics.json         # Post-training evaluation
└── qwen3-4b-marcus-q8_0.gguf  # Quantized model for inference (~4GB)
```

## Evaluation

```powershell
# Run post-training evaluation
.\unsloth_env\Scripts\python.exe training\marcus\run-grpo-training.py --eval-only

# Compare with teacher model
.\unsloth_env\Scripts\python.exe training\compare-models.py --model training\output-marcus\qwen3-4b-marcus-v2
```

## Deployment

### With Unsloth (fastest)
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Kodep/qwen3-4b-marcus-v2",
    max_seq_length=4096,
    load_in_4bit=True,
)
```

### With GGUF (lowest VRAM)
```bash
# Convert to GGUF
.\unsloth_env\Scripts\python.exe training\convert-to-gguf.py --model training\output-marcus\qwen3-4b-marcus-v2 --quant q8_0

# Run with llama.cpp
./llama.cpp/main -m qwen3-4b-marcus-q8_0.gguf -p "How should I respond when someone insults me?" -n 512
```

## Iteration Loop

1. **Generate synthetic data** → `data-generator.py`
2. **SFT training** → `run-sft-training.py`
3. **Evaluate** → `evaluate-model.py`
4. **GRPO training** → `run-grpo-training.py`
5. **Quantize** → `convert-to-gguf.py`
6. **Deploy** → Test with llama.cpp or Unsloth

## Known Challenges

1. **Citation accuracy**: The model may invent citations. GRPO reward function helps.
2. **Philosophical consistency**: May contradict Stoic principles. SFT on high-quality data helps.
3. **Response length**: May be too verbose or too brief. Length reward helps.

## Future Enhancements

- [ ] Multi-turn dialogue training (chatbot format)
- [ ] Contrastive learning (correct vs. incorrect citations)
- [ ] MoE architecture for better parameter efficiency
- [ ] Knowledge distillation from larger teacher models

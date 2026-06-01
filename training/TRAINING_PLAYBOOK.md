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

### 1. Verify Training Output

After SFT completes, check:
```
training/output-v2/qwen3-4b-effect-codegen-sft/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── training_args.bin
└── vocab.json
```

### 2. Test Inference

```powershell
.\unsloth_env\Scripts\python.exe -c "
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='training/output-v2/qwen3-4b-effect-codegen-sft',
    max_seq_length=4096,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

prompt = 'Generate an Effect service pattern with Schema validation'
messages = [
    {'role': 'system', 'content': 'You are an expert TypeScript developer specializing in the Effect framework.'},
    {'role': 'user', 'content': prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0]))
"
```

### 3. Check WandB Dashboard

If wandb is installed, go to https://wandb.ai and check:
- SFT: `effect-codegen-sft` project
- GRPO: `effect-codegen-grpo` project

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
├── run-sft-training-1.py    # Stage 1: SFT (new)
├── run-grpo-training-2.py   # Stage 2: GRPO (new)
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

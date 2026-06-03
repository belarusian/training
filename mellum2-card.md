---
library_name: transformers
language:
- en
pipeline_tag: text-generation
model-index:
- name: Mellum2 Thinking
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: livecodebench
      name: LiveCodeBench v6
    metrics:
    - name: pass@1
      type: pass@1
      value: 69.9
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: bfcl
      name: BFCL v3
    metrics:
    - name: accuracy
      type: acc
      value: 69.4
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: bfcl
      name: BFCL v4 (macro-avg of 5 subtasks)
    metrics:
    - name: accuracy
      type: acc
      value: 45.6
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: aime
      name: "AIME 2025+2026 (mean, 30 questions each)"
    metrics:
    - name: exact match
      type: exact_match
      value: 58.4
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: gsm-plus
      name: GSM-Plus
    metrics:
    - name: exact match
      type: exact_match
      value: 87.0
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: mmlu-redux
      name: MMLU-Redux
    metrics:
    - name: accuracy
      type: acc
      value: 86.2
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: gpqa
      name: GPQA Diamond
    metrics:
    - name: accuracy
      type: acc
      value: 57.6
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: ifeval
      name: IFEval (prompt-level strict accuracy)
    metrics:
    - name: accuracy
      type: acc
      value: 76.5
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: mixeval
      name: MixEval
    metrics:
    - name: accuracy
      type: acc
      value: 66.9
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: bs-bench
      name: BS-Bench (detection rate)
    metrics:
    - name: detection rate
      type: detection_rate
      value: 15.0
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: harmbench
      name: "HarmBench (harmful rate, lower is better)"
    metrics:
    - name: harmful rate
      type: harmful_rate
      value: 20.6
      verified: false
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: xstest
      name: XSTest (safe compliance)
    metrics:
    - name: safe compliance
      type: safe_compliance
      value: 89.6
      verified: false
license: apache-2.0
---

<img alt="Mellum" src="mellum-logo-dark.svg" width="320">

# Mellum2 Thinking

> [!Note]
> Use this model when you want explicit chain-of-thought before the final answer — complex debugging, multi-step planning, agentic workflows, and math- or reasoning-heavy tasks. For direct, low-latency answers without reasoning traces, use [Instruct](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct) instead.

## Mellum2 Thinking Highlights

Mellum 2 Thinking is a post-trained reasoning-augmented assistant model trained by JetBrains.

The model uses a Mixture-of-Experts architecture with 64 experts and activates 8 experts per token. It uses a combination of sliding-window and full attention layers, with a context length of 131,072 tokens.

It is produced from [`Mellum2-12B-A2.5B-Base`](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base) by supervised fine-tuning (loss computed only on the final assistant turn) followed by reinforcement learning with verifiable rewards (RLVR) on a harder data mix that includes a long-form math subset. The model emits its reasoning inside `<think>...</think>` blocks before the final answer.

## Mellum2 Model Family

This repository contains one checkpoint from the Mellum 2 family.

| Checkpoint | Description |
|---|---|
| [Base Pretrain](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base-Pretrain) | Base checkpoint before long-context extension |
| [Base](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base) | Final base model |
| [Instruct SFT](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct-SFT) | Supervised instruction-tuned checkpoint |
| [Thinking SFT](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking-SFT) | Supervised thinking checkpoint |
| [Instruct](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct) | RL-tuned instruction model |
| Thinking | RL-tuned thinking model |

## Model Overview

**Mellum2 Thinking** has the following features:

- Number of Layers: 28
- Hidden Size: 2304
- Intermediate Size: 7168
- MoE Intermediate Size: 896
- Number of Experts: 64
- Number of Activated Experts: 8
- Number of Attention Heads (GQA): 32 for Q and 4 for KV
- Context Length: 131,072
- Sliding Window: 1,024
- Vocabulary Size: 98,304
- Precision: bfloat16
- License: Apache 2.0

## Serving with vLLM

```sh
# Without tool calling
vllm serve JetBrains/Mellum2-12B-A2.5B-Thinking \
  --max-model-len 131072 \
  --reasoning-parser qwen3

# With tool calling
vllm serve JetBrains/Mellum2-12B-A2.5B-Thinking \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

## Quickstart

Text-Only Input

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [
    {"role": "user", "content": "Is 1024 a power of 2? Explain your reasoning."},
]

chat_response = client.chat.completions.create(
    model="JetBrains/Mellum2-12B-A2.5B-Thinking",
    messages=messages,
    max_tokens=81920,
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "top_k": 20,
    },
)
print("Chat response:", chat_response)
```

## Evaluation

Post-training evaluation for the thinking/reasoning variants. All values are percentages; higher is better except HarmBench, where lower is better. All values self-reported by JetBrains.

| Benchmark          | Mellum2 Thinking SFT | Mellum2 Thinking | Qwen3.5 (4B) | Qwen3.5 (9B) | OLMo-3 (7B) | Ministral 3 (14B) |
| :----------------- | --------------------: | ----------------: | -----------: | -----------: | ----------: | ----------------: |
| **Coding**         |                       |                   |              |              |             |                   |
| LiveCodeBench v6   |                  75.1 |              69.9 |         59.4 |         68.3 |        59.8 |              42.7 |
| **Tool Use**       |                       |                   |              |              |             |                   |
| BFCL v4            |                  38.8 |              45.6 |         42.9 |         42.7 |           — |              35.9 |
| BFCL v3            |                  60.5 |              69.4 |         73.9 |         68.5 |           — |              52.2 |
| **Math**           |                       |                   |              |              |             |                   |
| AIME               |                  20.0 |              58.4 |         68.3 |         73.4 |        61.7 |              38.3 |
| GSM-Plus           |                  62.6 |              87.0 |         89.3 |         90.7 |        88.1 |              86.5 |
| **Knowledge**      |                       |                   |              |              |             |                   |
| MMLU-Redux         |                  84.8 |              86.2 |         88.3 |         91.7 |        71.3 |              84.4 |
| GPQA Diamond       |                  39.9 |              57.6 |         76.8 |         81.3 |        29.3 |              46.0 |
| **Conversational** |                       |                   |              |              |             |                   |
| IFEval             |                  69.1 |              76.5 |         87.1 |         89.8 |        84.7 |              59.7 |
| JetBrains pairwise |                  64.4 |              69.5 |         40.5 |         56.7 |        32.2 |              63.8 |
| MixEval            |                  63.4 |              66.9 |         71.9 |         76.0 |        67.0 |              70.8 |
| BS-Bench           |                  14.0 |              15.0 |         63.0 |         70.0 |        23.0 |               9.0 |
| **Safety**         |                       |                   |              |              |             |                   |
| HarmBench (↓)      |                  12.2 |              20.6 |         15.9 |          6.6 |        48.7 |              70.0 |
| XSTest             |                  90.8 |              89.6 |         96.8 |         97.6 |        93.2 |              96.8 |

Notes:
- **AIME** is the mean of AIME 2025 and AIME 2026 (30 questions each).
- **BFCL v4** is the macro-average of five subtasks: v1, v2, v3, web search, memory.
- **JetBrains pairwise** is win rate against `Qwen2.5-7B-Instruct` on an internal benchmark.
- `—` indicates the model lacks native tool calling (OLMo-3-7B-Thinking).

For more details, see the [Mellum2 Technical Report](https://arxiv.org/abs/2605.31268).

## License

Released under the Apache 2.0 license.

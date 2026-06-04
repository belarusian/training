# RLHF & PPO

A visual guide to the four moving parts inside the RLHF agent.

## Legend

- 🔵 Generating Policy
- 🟢 Reference Policy  
- 🟠 Value Model
- 🔴 Reward Model
- 🟣 Training Data

---

## 0. Terminology: Policy vs Model

Before looking at the diagrams, here's the most important distinction:

> **A "model" is a _thing_. A "policy" is a _role_.**

**Model** = a neural network. It has weights, a forward pass, inputs and outputs. It's a concrete object in code.

**Policy** = a job. In RL, a policy is "the thing that decides actions." It's not a different object — it's the same network but wearing a different hat.

**So: every policy is a model, but not every model is called a policy.** The word "policy" just signals that this network's job is to make decisions (generate text).

With that out of the way, here's the full picture:

You start with **one language model** (e.g. Qwen3 4B). During PPO-based RLHF you build **four components** around it. Two of those components are literally the **same neural network**:

### The core LM (appears twice)

| Component | Description |
|-----------|-------------|
| 🟢 Generating Policy | The LM as it is during training. Gets updated every step. |
| 🔵 Reference Policy | An identical copy of the LM's weights at the start of training. Frozen forever. They share the same architecture and start with the same weights; after training begins their weights diverge. |

### Helper networks (two separate networks)

| Component | Description |
|-----------|-------------|
| 🟠 Value Model | A separate neural network that estimates "how much reward do I expect from this prompt?" (a baseline). |
| 🔴 Reward Model | A separate neural network that scores a (prompt, completion) pair. Trained beforehand on human preferences. |

---

## 1. RLHF at a Glance

The goal of **Reinforcement Learning from Human Feedback** is to take a pre-trained language model and teach it to produce outputs that humans prefer. It works by treating the model as an **agent** in a reinforcement-learning loop:

```
Training Data Prompts (s) → Agent π_θ(·) → Completions (a) → Reward Model → Reward (r) → Final Output
                                                                         ↓
                                                                    θ ← θ + α∇J(π_θ)
```

A **prompt** enters the agent, the agent generates a **completion**, a **reward model** scores it, and that score flows back as feedback to update the agent's weights. That's the RLHF loop in its simplest form.

![RLHF Diagram](https://belarusian.github.io/training/rlhf-ppo-visual-guide.html#rlhf-diagram)

---

## 2. PPO Architecture — The Full Agent

PPO (Proximal Policy Optimization) is the algorithm used inside the loop above. The "agent" is composed of **four components**. Two of these are the **same neural network** (Generating Policy and Reference Policy — same architecture, same starting weights, but one is updated and one is frozen). The other two are separate networks (Value Model and Reward Model).

![PPO Architecture Diagram](https://belarusian.github.io/training/rlhf-ppo-visual-guide.html#ppo-diagram)

---

## 3. Component Deep Dives

### 🟢 π_θ Generating Policy — "The Student"

The model **actively being trained**. Given a prompt (state `s`), it generates completions (actions `a`). Its parameters `θ` are updated every training step via the PPO loss to maximize expected reward. In the PPO formula this is `π_θ(a|s)` — the probability of taking action `a` given state `s`.

---

### 🟢 π_ref Reference Policy — "The Anchor"

An **identical copy** of the Generating Policy's weights at the start of training. **Same network, same architecture, same starting weights.** They diverge during training — the Generating Policy's weights update, the Reference Policy's stay frozen.

1. **KL regularization** — the objective penalizes `KL(π_θ || π_ref)`, preventing the model from drifting into gibberish just because it found a reward-hacking shortcut.
2. **Importance sampling reference** — the PPO clip ratio `π_θ / π_θ_old` is bounded relative to `π_ref`, keeping updates stable.

Think of it as a **guardrail**: the model can learn new behavior, but can't forget its pre-training.

---

### 🟠 V(s) Value Model (Critic) — "The Baseline"

A separate neural network that **estimates expected future reward** for a given state. It takes only the state `s` (the prompt, not the completion) and predicts `V(s)`.

```
A(s, a) = r + γ · V(s') − V(s)     ← advantage
```

The advantage tells the agent: *"was this action better or worse than what we expected?"* Positive advantage → reinforce. Negative → discourage. GRPO removes this model entirely and estimates advantage from **group statistics** (mean + std of rewards across multiple sampled completions).

---

### 🔴 R Reward Model — "The Judge"

In PPO, a neural network trained on human preference data (thumbs up / thumbs down) scores `(prompt, completion)` pairs. It is the external judge in the loop.

```
r = RewardModel(prompt, completion)
```

GRPO / RLVR replaces this with a **verifiable reward function** — a deterministic function that checks if the output satisfies certain criteria. No network, no training, just code.

#### Real example: Effect TypeScript reward function

This is the actual reward function used to train `qwen3-4b-effect-codegen` in this repo:

```python
def reward_fn(completions):
    score = 0.0
    if "<CODE>" in text and "</CODE>" in text:
        score += 1.0
    if "from effect" in text or "import Effect" in text:
        score += 0.5
    if "Schema" in text:
        score += 0.3
    if "export " in text:
        score += 0.2
    if len(text) < 100:
        score -= 0.5
```

It checks for structural markers of good Effect TypeScript code — code tags, imports, Schema usage, exports — and penalizes too-short responses. GRPO then computes advantage from the spread of these scores across N sampled completions per prompt (Z-score standardization).

> **Trained model:** This reward function trains `Qwen3-4B` to generate Effect-style TypeScript code. The resulting LoRA adapter is available at [huggingface.co/Kodep/qwen3-4b-effect-codegen](https://huggingface.co/Kodep/qwen3-4b-effect-codegen).

---

## 4. The PPO Objective

The loss function drives the gradient updates. Here's the clipped surrogate objective with the KL penalty:

```
J(θ) = E [ min( (π_θ / π_θ_old) · A , clip( (π_θ / π_θ_old), 1−ε, 1+ε ) · A ) ] − β · KL(π_θ || π_ref)
```

| Symbol | Meaning |
|--------|---------|
| π_θ | Generating policy |
| π_ref | Reference policy (frozen) |
| A | Advantage (r − V(s)) |
| clip(…, 1−ε, 1+ε) | Stability: no large policy jumps |
| β · KL | Keeps π_θ close to π_ref |

> **Key insight:** The `min()` picks the smaller of the unclipped ratio and the clipped ratio. When the advantage is positive, clipping prevents the policy from increasing too far. When negative, it prevents decreasing too far. `ε` is typically `0.2` (±20%).

---

## 5. PPO vs GRPO — What Changes

GRPO (Group Relative Policy Optimization, used in DeepSeek-R1) is a simplified variant.

| Component | PPO | GRPO |
|-----------|-----|------|
| **Generating Policy** | ✅ Trained, updated every step | ✅ Trained, updated every step |
| **Reference Policy** | ✅ Frozen copy, KL anchor | ✅ Frozen copy, KL anchor |
| **Value Model** | ✅ Separate network — estimates `V(s)` | ❌ Removed — replaced by **group statistics** (mean + std of rewards across N sampled completions) |
| **Reward Model** | ✅ Neural network trained on human preferences | ✅ Replaced — with **verifiable reward functions** (RLVR) — regex, exec, exact match |
| **VRAM savings** | — | ~8x less — no value network + no reward network to load |
| **Example in this repo** | — | Structural checks: code tags (+1.0), Effect imports (+0.5), Schema (+0.3), exports (+0.2), length penalty (−0.5) |

![PPO vs GRPO](https://belarusian.github.io/training/rlhf-ppo-visual-guide.html#ppo-grpo-comparison)

> **TL;DR:** PPO has **4 components** (generating + reference + value + reward), but only **3 separate networks** — generating and reference are the same LM. GRPO has just **2 components** (generating + reference, both the same network). The advantage is computed from **statistics across N sampled completions** instead — hence "Group Relative."

---

## 6. One-Sentence Summary

> The Generating Policy produces completions for prompts, the Reward Model scores them → the Value Model estimates baselines → the Reference Policy prevents drift → gradients flow back to the Generating Policy

---

*Original HTML version: [literature/rlhf-ppo-visual-guide.html](../literature/rlhf-ppo-visual-guide.html)*

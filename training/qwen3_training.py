"""
Qwen3-14B Reasoning Conversational Fine-Tuning Script
Converted from: Qwen3_(14B)_Reasoning_Conversational.ipynb
"""

import os
import re
import sys

# Check if running in Colab
IS_COLAB = "COLAB_" in "".join(os.environ.keys())

if not IS_COLAB:
    # Standard local/cloud installation
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "unsloth"], check=True)
else:
    # Colab-specific installation
    import torch
    v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    
    subprocess.run([sys.executable, "-m", "pip", "install", "sentencepiece", "protobuf", 
                    "datasets==4.3.0", "huggingface_hub>=0.34.0", "hf_transfer"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "unsloth_zoo", 
                    "bitsandbytes", "accelerate", xformers, "peft", "trl", "triton", "unsloth"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade", "torchao>=0.16.0"], check=True)

# Install specific versions
subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.56.2"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "trl==0.22.2"], check=True)

# Now import after installation
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
import pandas as pd
from datasets import Dataset


def main():
    print("=== Loading Qwen3-14B Model ===")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-14B",
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    print("=== Adding LoRA Adapters ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("=== Loading Datasets ===")
    reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    
    print(f"Reasoning dataset size: {len(reasoning_dataset)}")
    print(f"Non-reasoning dataset size: {len(non_reasoning_dataset)}")
    
    print("=== Converting Reasoning Dataset to Conversational Format ===")
    def generate_conversation(examples):
        problems = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ])
        return {"conversations": conversations}
    
    reasoning_conversations = tokenizer.apply_chat_template(
        list(reasoning_dataset.map(generate_conversation, batched=True)["conversations"]),
        tokenize=False,
    )
    print(f"First reasoning example:\n{reasoning_conversations[0][:500]}...")
    
    print("=== Converting Non-Reasoning Dataset to Conversational Format ===")
    from unsloth.chat_templates import standardize_sharegpt
    dataset = standardize_sharegpt(non_reasoning_dataset)
    
    non_reasoning_conversations = tokenizer.apply_chat_template(
        list(dataset["conversations"]),
        tokenize=False,
    )
    print(f"First non-reasoning example:\n{non_reasoning_conversations[0][:500]}...")
    
    print("=== Combining Datasets ===")
    chat_percentage = 0.25
    
    non_reasoning_subset = pd.Series(non_reasoning_conversations)
    non_reasoning_subset = non_reasoning_subset.sample(
        int(len(reasoning_conversations) * (chat_percentage / (1 - chat_percentage))),
        random_state=2407,
    )
    
    print(f"Reasoning conversations: {len(reasoning_conversations)}")
    print(f"Non-reasoning subset: {len(non_reasoning_subset)}")
    print(f"Chat percentage: {len(non_reasoning_subset) / (len(non_reasoning_subset) + len(reasoning_conversations)):.2%}")
    
    data = pd.concat([
        pd.Series(reasoning_conversations),
        pd.Series(non_reasoning_subset)
    ])
    data.name = "text"
    
    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    combined_dataset = combined_dataset.shuffle(seed=3407)
    
    print(f"Combined dataset size: {len(combined_dataset)}")
    
    print("=== Training Model ===")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined_dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            padding_free=False,
        ),
    )
    
    # Show memory stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer_stats = trainer.train()
    
    # Show final memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    print("\n=== Saving Model ===")
    model.save_pretrained("qwen_lora")
    tokenizer.save_pretrained("qwen_lora")
    print("Model saved to 'qwen_lora' directory")
    
    print("\n=== Inference Test (Non-Thinking Mode) ===")
    FastLanguageModel.for_inference(model)
    
    messages = [{"role": "user", "content": "Solve (x + 2)^2 = 0."}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    
    print("\n=== Inference Test (Thinking Mode) ===")
    messages = [{"role": "user", "content": "Solve (x + 2)^2 = 0."}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    
    print("\n=== Training Complete ===")
    print("Model saved to 'qwen_lora' directory")
    print("You can load it with: FastLanguageModel.from_pretrained('qwen_lora', ...)")


if __name__ == "__main__":
    main()

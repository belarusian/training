from unsloth import FastLanguageModel

# Load SFT model
model_path = "training/output-v2/qwen3-4b-effect-codegen-sft"
print(f"Loading SFT model from {model_path}...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=4096,
    load_in_4bit=False,
    load_in_8bit=False,
    dtype="float16",
    fast_inference=False,
    max_lora_rank=64,
)

print("Model loaded.\n")

# Test prompts
test_prompts = [
    "Generate an Effect service pattern for a database connection pool",
    "Create an Effect stream that processes HTTP requests with retries",
]

SYSTEM_PROMPT = """You are an expert TypeScript developer specializing in the Effect framework.
Analyze the requirements and generate high-quality Effect code.
Provide your reasoning first, then provide your complete TypeScript code implementation."""

for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_encoded = tokenizer([inputs], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs_encoded,
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract code
    import re
    code_match = re.search(r'<CODE>(.*?)</CODE>', response, re.DOTALL)
    code = code_match.group(1) if code_match else "NO CODE TAGS"
    
    print(f"Response length: {len(response)}")
    print(f"Code length: {len(code)}")
    print(f"Has <CODE> tags: {'<CODE>' in response}")
    print(f"Code (first 200 chars): {repr(code[:200])}")
    print()

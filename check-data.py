import json
with open(r'C:\Users\kodep\Training\extracted-code\effect-code-samples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Show first sample
sample = data[0]
print('First sample:')
print(f'Prompt: {sample["prompt"][:100]}...')
print(f'Completion (first 200 chars): {repr(sample["completion"][:200])}')
print(f'Completion length: {len(sample["completion"])}')

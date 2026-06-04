import json
with open(r'C:\Users\kodep\Training\training\output-v2\qwen3-4b-effect-codegen-v2\adapter_config.json', 'r') as f:
    config = json.load(f)
print(f'Base model: {config["base_model_name_or_path"]}')
print(f'Trainable: {config.get("trainable", "N/A")}')

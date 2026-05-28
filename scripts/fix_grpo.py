"""
Fix GRPO training notebook to add reward function
"""

import json
import sys

def fix_grpo_notebook(notebook_path):
    """Add reward function to GRPO training notebook"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Reward function code cell
    reward_code = '''import re
import torch

def reward_fn(completions):
    """Reward function to score generated code quality.
    Rewards: code structure, Effect imports, proper formatting
    """
    rewards = []
    for text in completions:
        score = 0.0
        
        # Reward: Has <CODE> tags
        if "<CODE>" in text and "</CODE>" in text:
            score += 1.0
        
        # Reward: Has Effect imports
        if "from effect" in text or "import Effect" in text:
            score += 0.5
        
        # Reward: Has Schema usage
        if "Schema" in text:
            score += 0.3
        
        # Reward: Proper TypeScript syntax indicators
        if "export " in text:
            score += 0.2
        
        # Penalize: Too short responses
        if len(text) < 100:
            score -= 0.5
        
        rewards.append(score)
    
    return torch.tensor(rewards)

print('Reward function defined!')
'''
    
    # Find the cell with GRPOConfig
    grpo_cell_idx = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'from trl import GRPOConfig' in source:
                grpo_cell_idx = i
                break
    
    if grpo_cell_idx is None:
        print("Error: Could not find GRPO trainer cell")
        return False
    
    # Insert reward function cell before GRPO cell
    reward_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': reward_code.split('\n')
    }
    
    notebook['cells'].insert(grpo_cell_idx, reward_cell)
    
    # Update GRPO cell to include reward function
    grpo_cell = notebook['cells'][grpo_cell_idx + 1]
    source_lines = grpo_cell['source']
    
    # Find the GRPOConfig line and add reward_functions
    for i, line in enumerate(source_lines):
        if 'report_to = "none",' in line:
            source_lines.insert(i + 1, '        reward_functions = [reward_fn],\n')
            break
    
    # Save
    output_path = notebook_path.replace('.ipynb', '_fixed.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Fixed notebook saved to: {output_path}")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_grpo.py <notebook_path>")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    fix_grpo_notebook(notebook_path)

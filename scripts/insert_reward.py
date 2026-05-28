"""
Insert reward function into GRPO training notebook
"""

import json

def insert_reward_function(notebook_path):
    """Insert reward function cell before GRPO trainer cell"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Reward function cell
    reward_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import re\n",
            "import torch\n",
            "\n",
            "def reward_fn(completions):\n",
            '    """Reward function to score generated code quality.\n',
            '    Rewards: code structure, Effect imports, proper formatting\n',
            '    """\n',
            "    rewards = []\n",
            "    for text in completions:\n",
            "        score = 0.0\n",
            "        \n",
            '        # Reward: Has <CODE> tags\n',
            '        if "<CODE>" in text and "</CODE>" in text:\n',
            "            score += 1.0\n",
            "        \n",
            '        # Reward: Has Effect imports\n',
            '        if "from effect" in text or "import Effect" in text:\n',
            "            score += 0.5\n",
            "        \n",
            '        # Reward: Has Schema usage\n',
            '        if "Schema" in text:\n',
            "            score += 0.3\n",
            "        \n",
            '        # Reward: Proper TypeScript syntax indicators\n',
            '        if "export " in text:\n',
            "            score += 0.2\n",
            "        \n",
            '        # Penalize: Too short responses\n',
            '        if len(text) < 100:\n',
            "            score -= 0.5\n",
            "        \n",
            "        rewards.append(score)\n",
            "    \n",
            "    return torch.tensor(rewards)\n",
            "\n",
            "print('Reward function defined!')"
        ]
    }
    
    # Find GRPO trainer cell
    grpo_idx = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'from trl import GRPOConfig, GRPOTrainer' in source:
                grpo_idx = i
                break
    
    if grpo_idx is None:
        print("Error: Could not find GRPO trainer cell")
        return False
    
    # Insert reward cell before GRPO cell
    notebook['cells'].insert(grpo_idx, reward_cell)
    
    # Update GRPO cell to include reward function
    grpo_cell = notebook['cells'][grpo_idx + 1]
    source = grpo_cell['source']
    
    for i, line in enumerate(source):
        if 'report_to = "none",' in line:
            source.insert(i + 1, '        reward_functions = [reward_fn],\n')
            break
    
    # Save
    output_path = notebook_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Reward function inserted into: {output_path}")
    return True

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python insert_reward.py <notebook_path>")
        sys.exit(1)
    
    insert_reward_function(sys.argv[1])

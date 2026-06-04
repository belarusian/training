import re

def compute_reward_details(text):
    components = {
        'code_tags': 0.0,
        'effect_imports': 0.0,
        'schema_usage': 0.0,
        'exports': 0.0,
        'length': 0.0,
    }
    
    if '<CODE>' in text and '</CODE>' in text:
        components['code_tags'] = 1.0
    
    if 'from effect' in text.lower() or 'import Effect' in text:
        components['effect_imports'] = 0.5
    
    if 'Schema' in text:
        components['schema_usage'] = 0.3
    
    if 'export ' in text:
        components['exports'] = 0.2
    
    code_match = re.search(r'<CODE>(.*?)</CODE>', text, re.DOTALL)
    code_content = code_match.group(1) if code_match else ''
    code_len = len(code_content)
    
    if code_len < 50:
        components['length'] = -1.0
    elif code_len < 100:
        components['length'] = -0.5
    elif code_len < 200:
        components['length'] = 0.2
    elif code_len < 500:
        components['length'] = 0.4
    elif code_len < 1000:
        components['length'] = 0.6
    elif code_len < 2000:
        components['length'] = 0.8
    else:
        components['length'] = 0.5
    
    total = sum(components.values())
    return total, components, code_len

# Test with the bad output
bad = '<start_working_out>and</start_working_out><CODE> and </CODE>'
reward, components, code_len = compute_reward_details(bad)
print(f'Bad output: {repr(bad)}')
print(f'Code length: {code_len}')
print(f'Reward: {reward}')
print(f'Components: {components}')
print()

# Test with good output
good = '<start_working_out>Here is my analysis</start_working_out><CODE>import * as Effect from "effect"\nexport const service = Effect.succeed(42)</CODE>'
reward, components, code_len = compute_reward_details(good)
print(f'Good output: {repr(good[:100])}...')
print(f'Code length: {code_len}')
print(f'Reward: {reward}')
print(f'Components: {components}')

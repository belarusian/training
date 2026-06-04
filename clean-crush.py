#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import tempfile
import os
import sys

# Force UTF-8 output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Python script to clean commit messages
clean_script = r'''
import sys
import re

# Read message from stdin
message = sys.stdin.read()

# Pattern: Generated with Crush + Assisted-by (handles various spacing)
# Matches: "Generated with Crush\n\n Assisted-by: Crush:qwen-coder"
pattern = r'Generated with Crush\s*\n\s*Assisted-by: Crush.*?(?=\n\n|\Z)'
message = re.sub(pattern, '', message, flags=re.DOTALL)

# Clean up excessive newlines
message = re.sub(r'\n{3,}', '\n\n', message)

# Print cleaned message
print(message.strip(), end='')
'''

# Create temp script file with UTF-8 encoding
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
    f.write(clean_script)
    script_path = f.name

try:
    # Run filter-branch with the Python script
    cmd = [
        'git', '-C', r'C:\Users\kodep\Training',
        'filter-branch', '-f',
        '--msg-filter', f'python "{script_path}"',
        '--', '--all'
    ]
    
    print(f"Running filter-branch...")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("\n✓ Success! Verifying...")
        
        # Verify
        cmd_verify = ['git', '-C', r'C:\Users\kodep\Training', 'log', '--pretty=format:%B']
        result_verify = subprocess.run(cmd_verify, capture_output=True, text=True, encoding='utf-8')
        
        if 'Crush' in result_verify.stdout:
            print("✗ Still found Crush in some commits")
            # Count how many
            count = result_verify.stdout.count('Crush')
            print(f"  Found {count} remaining 'Crush' occurrences")
        else:
            print("✓ All Crush attributions removed!")
            
        print("\nNext steps:")
        print("1. git log --oneline")
        print("2. git push --force")
        print("3. git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin")
    else:
        print(f"✗ Failed with code {result.returncode}")
        
finally:
    os.unlink(script_path)

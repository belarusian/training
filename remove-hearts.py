#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import tempfile
import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Python script to remove heart emoji from commit messages
clean_script = r'''
import sys
import re

message = sys.stdin.read()

# Remove the 💘 and ♥ emojis from commit messages
message = message.replace('💘', '').replace('♥', '')

# Clean up excessive newlines
message = re.sub(r'\n{3,}', '\n\n', message)

print(message.strip(), end='')
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
    f.write(clean_script)
    script_path = f.name

try:
    for repo_name, repo_path in [("Training", r"C:\Users\kodep\Training"), ("Practice", r"C:\Users\kodep\Practice")]:
        print(f"Processing {repo_name}...")
        
        cmd = [
            'git', '-C', repo_path,
            'filter-branch', '-f',
            '--msg-filter', f'python "{script_path}"',
            '--', '--all'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        # Verify by checking raw output
        cmd_verify = ['git', '-C', repo_path, 'log', '--pretty=format:%B']
        result_verify = subprocess.run(cmd_verify, capture_output=True, text=True, encoding='utf-8')
        
        heart_count = result_verify.stdout.count('💘') + result_verify.stdout.count('♥')
        
        if heart_count == 0:
            print(f"  ✓ No heart emojis found in {repo_name}")
        else:
            print(f"  ✗ Found {heart_count} heart emojis in {repo_name}")
            # Show sample
            import re
            matches = re.findall(r'.{0,50}[💘♥].{0,50}', result_verify.stdout)
            for m in matches[:3]:
                print(f"    {repr(m)}")
        
        # Clean up backup
        subprocess.run(['git', '-C', repo_path, 'update-ref', '-d', 'refs/original/refs/heads/master'], 
                      capture_output=True, encoding='utf-8')
        subprocess.run(['git', '-C', repo_path, 'update-ref', '-d', 'refs/original/refs/remotes/origin/master'],
                      capture_output=True, encoding='utf-8')
        subprocess.run(['git', '-C', repo_path, 'update-ref', '-d', 'refs/original/refs/heads/main'],
                      capture_output=True, encoding='utf-8')
        subprocess.run(['git', '-C', repo_path, 'update-ref', '-d', 'refs/original/refs/remotes/origin/main'],
                      capture_output=True, encoding='utf-8')
        
    print("\nDone! Heart emojis removed from both repos.")
    print("\nTo push: git push --force")
    
finally:
    os.unlink(script_path)

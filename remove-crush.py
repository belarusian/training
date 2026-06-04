#!/usr/bin/env python3
"""
Remove Crush attribution from git commit messages using Python.
More reliable than sed for multi-line patterns.
"""

import subprocess
import re
import sys

def run_git(command):
    """Run a git command and return output."""
    result = subprocess.run(
        ["git"] + command,
        capture_output=True,
        text=True,
        cwd=r"C:\Users\kodep\Training"
    )
    return result.returncode, result.stdout, result.stderr

def get_commits():
    """Get all commits with full message."""
    returncode, stdout, stderr = run_git([
        "log", "--pretty=format:%H|%B", "--no-merges"
    ])
    if returncode != 0:
        print(f"Error getting commits: {stderr}")
        sys.exit(1)
    
    commits = []
    for line in stdout.strip().split('\n'):
        if '|' in line:
            parts = line.split('|', 1)
            commits.append({
                'hash': parts[0],
                'message': parts[1] if len(parts) > 1 else ''
            })
    return commits

def clean_message(message):
    """Remove Crush attribution from a commit message."""
    # Pattern to match Crush lines (handles various spacing)
    pattern = r'\s*💘 Generated with Crush\s*Assisted-by: Crush.*?(?=\n\n|\n[a-z]|\Z)'
    cleaned = re.sub(pattern, '', message, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up extra newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()

def rewrite_commits(commits):
    """Rewrite commits with cleaned messages using filter-branch."""
    print(f"Found {len(commits)} commits to process...")
    
    # Build the filter-branch command with Python script
    python_script = r"""
import sys
import re

message = sys.stdin.read()

# Pattern to match Crush lines
pattern = r'\s*💘 Generated with Crush\s*Assisted-by: Crush.*?(?=\n\n|\n[a-z]|\Z)'
cleaned = re.sub(pattern, '', message, flags=re.IGNORECASE | re.DOTALL)

# Clean up extra newlines
cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

print(cleaned.strip(), end='')
"""
    
    # Write the Python script to a temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(python_script)
        script_path = f.name
    
    try:
        filter_cmd = ["python", script_path]
        
        print(f"Running filter-branch with Python filter...")
        returncode, stdout, stderr = run_git([
            "filter-branch", "-f",
            "--msg-filter", "python " + script_path,
            "--", "--all"
        ])
        
        print(stdout)
        if stderr:
            print(stderr)
        
        return returncode == 0
    finally:
        import os
        os.unlink(script_path)

def main():
    print("=" * 60)
    print("Removing Crush Attribution from Git Commits")
    print("=" * 60)
    print()
    
    # Get commits first
    commits = get_commits()
    print(f"Found {len(commits)} commits")
    
    # Show sample of commits with Crush
    crush_commits = [c for c in commits if 'Crush' in c['message']]
    print(f"Commits with Crush: {len(crush_commits)}")
    
    if len(crush_commits) > 0:
        print("\nSample commits with Crush:")
        for c in crush_commits[:3]:
            print(f"\n{c['hash']}")
            # Show just the Crush part
            lines = c['message'].split('\n')
            for i, line in enumerate(lines):
                if 'Crush' in line:
                    print("  " + "\n  ".join(lines[max(0,i-1):min(len(lines),i+3)]))
                    break
    
    # Ask for confirmation
    print("\n" + "=" * 60)
    response = input("Proceed with rewriting commits? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Rewrite commits
    success = rewrite_commits(commits)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Commit messages rewritten!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Verify: git log --oneline")
        print("2. Push: git push --force")
        print("3. Clean up backup: git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin")
    else:
        print("\n✗ Failed to rewrite commits")

if __name__ == "__main__":
    main()

# Remove Crush attribution from commit messages
# This script rewrites git history to remove the Crush lines

$repoPath = "C:\Users\kodep\Training"
$backupRef = "refs/original/refs/heads/main"

Write-Host "=== Removing Crush Attribution from Commit Messages ===" -ForegroundColor Cyan
Write-Host ""

# Create backup
Write-Host "1. Creating backup of current branch..." -ForegroundColor Yellow
git -C $repoPath update-ref "$backupRef" "refs/heads/main" | Out-Null
Write-Host "✓ Backup created at $backupRef" -ForegroundColor Green
Write-Host ""

# Check for existing filter-branch backup
if (git -C $repoPath rev-parse "refs/original/main" 2>$null) {
    Write-Host "Warning: Previous filter-branch backup exists!" -ForegroundColor Red
    Write-Host "Run: git -C $repoPath filter-branch --force --unpack-refs" -ForegroundColor Yellow
    Write-Host "Or manually remove: git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin" -ForegroundColor Yellow
    Write-Host ""
}

# Define the sed filter to remove Crush lines
# This removes lines between "💘 Generated with Crush" and "Assisted-by: Crush"
$filterScript = "sed '/💘 Generated with Crush/,/Assisted-by: Crush/d'"

Write-Host "2. Rewriting commit messages..." -ForegroundColor Yellow
Write-Host "   Using filter: $filterScript" -ForegroundColor Gray
Write-Host ""

# Run filter-branch
# --msg-filter applies the filter to commit messages only
# --force overwrites any existing backup
$env:GIT_FILTER_BRANCH_DISABLE_SCRIPT_VALIDATION = "1"

git -C $repoPath filter-branch -f --msg-filter $filterScript -- --all 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Successfully rewrote commit messages!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Review changes: git log --oneline" -ForegroundColor White
    Write-Host "2. Push with force: git push --force" -ForegroundColor Yellow
    Write-Host "3. Clean up backup: git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "✗ filter-branch failed!" -ForegroundColor Red
    Write-Host "Restoring from backup..." -ForegroundColor Yellow
    git -C $repoPath reset --hard "$backupRef" 2>$null
}

# Auto-Retrain System - Branch Protection Configuration

## Branch Strategy

```
main           <- Production-ready code (protected)
├── develop    <- Integration branch (protected)  
│   └── feature/*    <- Feature branches
├── release/v*  <- Release branches
└── hotfix/*    <- Emergency fixes
```

## Protection Rules (Local Git)

### For Repository Hosting (if using GitHub, GitLab, etc.):

1. **main** branch:
   - Require pull request reviews (min 2)
   - Require status checks to pass (lint + test)
   - Require signed commits
   - No force push allowed
   - Admin cannot bypass

2. **develop** branch:
   - Require pull request reviews (min 1)
   - Require status checks to pass
   - No force push

3. **release/*** branches:
   - Require pull request reviews (min 1)
   - Direct push allowed for hotfixes

## Local Protection (Git Hooks)

This repository includes pre-configured hooks in `.git/hooks/`.

### To enable local protection:

```bash
# Make hooks executable
chmod +x .git/hooks/pre-push

# Or use Git config
git config core.hooksPath .git/hooks
```

## CI/CD Status Checks Required

Before any merge to main:

- [x] Lint passes (`ruff check src/`)
- [x] Tests pass (`pytest tests/`)
- [x] Type check passes (optional)
- [x] Config validation (`python scripts/validate_compliance.py`)
- [x] Security scan (Checkov)

## Workflow

1. Create feature branch from `develop`:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

3. Push and create PR:
   ```bash
   git push -u origin feature/your-feature
   ```

4. After CI passes and PR reviewed:
   ```bash
   git checkout develop
   git merge feature/your-feature
   git push origin develop
   ```

5. Release to main (from develop):
   ```bash
   git checkout main
   git merge develop
   git push origin main
   ```

## Emergency Hotfixes

For critical fixes directly to main:

```bash
git checkout -b hotfix/critical-fix main
# Make fix, test, commit
git checkout main
git merge hotfix/critical-fix
git push origin main
git checkout develop
git merge main  # Bring fix to develop
```
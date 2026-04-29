# GitHub Branch Protection Configuration

## For Repository Settings (github.com)

### Protect main branch:
1. Go to: Settings → Branches → Branch protection rules
2. Add rule for `main`:
   - ✅ Require pull request reviews before merging (2 reviewers)
   - ✅ Require status checks to pass
   - ✅ Require signed commits
   - ✅ Include administrators
   - ✅ Allow force pushes: OFF

### Protect develop branch:
1. Add rule for `develop`:
   - ✅ Require pull request reviews before merging (1 reviewer)
   - ✅ Require status checks to pass
   - ✅ Include administrators
   - ✅ Allow force pushes: OFF

---

## GitHub Secrets Required

Go to: Settings → Secrets and variables → Actions

Add these secrets:
- `DATA_PATH`: Path to training data
- `MODEL_PATH`: Path to save models

---

## Environment Variables

Go to: Settings → Environment

Create `production` environment:
- ✅ Required reviewers (1)
- ✅ Wait timer: OFF

---

## CODEOWNERS File

Create `.github/CODEOWNERS`:
```
# Default codeowner
*           @your-username

# ML codeowners
src/*.py    @your-username
main.py     @your-username
```

---

## Status Checks Required

In `.github/workflows/train.yml`, ensure these pass:
1. Lint (`ruff check src/`)
2. Tests (`pytest tests/`)
3. Config validation
4. Security scan (Checkov)

---

## To Enable This Repository on GitHub

```bash
# 1. Create repository on github.com

# 2. Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/auto-retrain-system.git
git push -u origin main
git push -u origin develop
git push origin v1.0

# 3. Configure protection rules on github.com
```

---

## After Push - Verification Checklist

- [ ] Repository appears on github.com
- [ ] Branch protection rules configured
- [ ] CODEOWNERS file created
- [ ] Secrets configured
- [ ] CI/CD runs on push
- [ ] All tests pass
# GitHub Setup Instructions

Your repository is ready to push to GitHub! Follow these steps:

## 1. Create a GitHub Repository

1. Go to https://github.com/new
2. Repository name: `ofc-poker-rl` (or any name you prefer)
3. Description: "Open Face Chinese Poker solver using reinforcement learning with PyTorch"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## 2. Push to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add the remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ofc-poker-rl.git

# Push to GitHub
git branch -M main
git push -u origin main
```

Or if you prefer SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/ofc-poker-rl.git
git branch -M main
git push -u origin main
```

## 3. Verify

Check your repository at: `https://github.com/YOUR_USERNAME/ofc-poker-rl`

## Notes

- Model checkpoints (`.pth` files) are excluded via `.gitignore` to keep the repo small
- Test files are excluded but you can add them if needed
- The `include/` folder with C++ headers is included


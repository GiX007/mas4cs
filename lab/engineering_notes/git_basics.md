# Git Basics

## 1. What is Git and Why Use It?

**Git** is a Version Control System. It tracks every change you make to your code.

- **Why it’s useful**:  
  - Acts like a “save game” for code. You can return to any working version.
  - Shows what changed, when, and by whom.
  - Allows safe parallel work using branches.
  - Enables collaboration without overwriting others’ work.
  
- **When to use it**:
  - For any real software or research project.
  - When collaborating (even with your future self).
  - When publishing code to GitHub/GitLab/Bitbucket.
  - When you want clean releases and reproducible results.

Key terms:
- Repository (repo): a project folder tracked by Git, containing a hidden `.git` folder.
- Commit: a saved snapshot with a message.
- Branch: an independent line of development.
- Remote: a server copy of the repo (for example, GitHub).

---

## 2. Checking and Managing Git

Check if Git is installed (PowerShell):
`git --version`

If not installed:
- Download from https://git-scm.com
- Install and verify again with `git --version`.

List all Git repositories on your PC:
`Get-ChildItem -Path "C:\Users\giorg\Projects" -Recurse -Directory -Hidden -Filter ".git" -ErrorAction SilentlyContinue`

Deleting a Git repository:

- Option A — delete entire project (repo + files): `Remove-Item -LiteralPath "C:\path\to\project" -Recurse -Force`. Verify: `Test-Path "C:\path\to\project"`   → should return False
- Option B — remove Git only (keep project files): `Remove-Item -LiteralPath "C:\path\to\project\.git" -Recurse -Force`. Verify: `Test-Path "C:\path\to\project\.git"`   → should return False

Confirm it is no longer a Git repo:
```bash
cd "C:\path\to\project"
git status
```
→ should return: fatal: not a git repository

---

## 3. Creating and Uploading Git Repositories

Professional basic workflow:

- **Step 1** — create `.gitignore` at the start (in project root).
         
  Purpose: Prevent uploading virtual environments, caches, IDE files, secrets, or very large data/outputs.
  
  Typical Python `.gitignore`:
  ```bash
  # Python cache 
  __pycache__/
  *.pyc
  *.pyo
  *.pyd
  
  # PyCharm
  .idea/
  *.iml
  
  # Virtual environments
  venv/
  env/
  .venv/
  
  # OS files
  .DS_Store
  Thumbs.db
  
  # Environment variables
  .env
  
  # Data and outputs
  data/
  outputs/
  logs/
  *.log
  
  # Python specific
  *.pkl
  *.joblib
  *.h5
  ```

- **Step 2** — initialize **Git** (PowerShell or IDE terminal):
    ```bash
    cd "C:\path\to\my_project"
    git init
    ```

    Remove `.idea` files (Pycharm):
    ```bash
    git rm -r --cached .idea
    git rm -r --cached -f .idea
    ```
    
    First Commit:
    ```bash
    git add .
    git commit -m "Initial commit"
    ```
  
    Verify: 
    ```git status```
    → working tree clean (no uncommitted changes)

  Note (during initial repository setup):

  Before running `git add .`, verify `.gitignore` exists and is correct.
  Otherwise, you might accidentally commit venv/, .idea/, or secrets.

  Safe workflow:
  1. Create `.gitignore` FIRST
  2. Run: `git status --ignored`
  3. Verify secrets/venv appear under "Ignored files"
  4. Then run: `git add .`
  5. Verify with: `git status` or `git status --short` (check what will be committed)
  6. Finally: `git commit -m "Initial commit"`

  If you see:
  - **"Changes not staged for commit:"**  → files already tracked by Git but modified during setup → Action: `git add <file>` # or: `git add .` and commit again.
  - **"Untracked files:"**  → new project files not yet tracked by Git → Action: `git add <file>` # or: `git add .` and commit again.
  
  View Commit History: 
  `git log --oneline`

- **Step 3** — upload to GitHub.

  Why: 
  - Off-device backup 
  - Collaboration 
  - Public portfolio
  - CI/CD automation
   
  **CLI method**:
  
  1. Create empty repo on GitHub (do NOT initialize with README/license).
  2. Copy the repo URL (e.g., https://github.com/GiX007/mas4cs.git).
  3. In your local terminal (inside project root):
  ```bash
  # Link local repo to GitHub
  git remote add origin https://github.com/GiX007/mas4cs.git
  
  # Rename branch to 'main' (GitHub expects it)
  git branch -M main
  
  # Push and set upstream tracking
  git push -u origin main
  ```
  
  **Note**: Local `git init` creates a branch called `master` by default. GitHub creates `main` by default. The above commands rename your local branch to `main` to match GitHub's convention. If you prefer to keep `master`, use `git push -u origin master` instead (then set `master` as default in GitHub repo Settings → Branches).

  Verify: 
  - `git remote -v` → shows origin URL
  - GitHub page shows your files

  Explanation:
  - `git remote add origin <URL>`: Links local repo to GitHub
  - `-u origin main`: Sets "upstream" so future `git push` works without arguments
  - After this, just use: `git push` (no need for `-u origin main` again)

  **IDE UI** is equivalent:
    - PyCharm: VCS → Enable Version Control → Git → Commit → Push
    - VS Code: Source Control → Initialize → Git → Commit → Push

---

## 4. Best Practices and Core Workflow

Recommended project order:

→ Create project folder → open in IDE → create virtual environment (`python -m venv venv`venv) → set this `venv` as the project interpreter in the IDE → create `.gitignore` → initialize Git (`git init` → `git add .`) → commit (`git commit -m "Initial commit"`) → push (`git push`)

Repository structure rule:
- For most personal, academic, and small-team projects, use ONE Git repository per project.
- Large companies may use monorepos, but this is not typical for small work.

### Daily Git workflow

  - `git status` → see current changes.
  - `git add <files>` → choose what goes into the commit.
  - `git commit -m "message"` → create a history snapshot.
  - `git pull` → download and merge latest changes from GitHub. `pull` before pushing to avoid conflicts.
  - `git push` → upload commits to GitHub.
   
    After push: backup exists, others can see code, CI/CD may run.

### Branches

A branch is an independent line of development. It allows safe work on features without breaking the main code.

Default branch: `main` (should stay stable and working).

Why branches matter:
- Safe experimentation
- Parallel teamwork
- Clean history
- **Required** in real team workflows

Basic branch workflow:

- Create and switch to a branch: 
  
  `git checkout -b feature/my-change`

- Work and commit (in branch): 
  ```bash
  git add . 
  git commit -m "Implement feature"
  ``` 

- Return to main and update: 
  ```bash
  git checkout main
  git pull
  ```
- Merge feature into main: 

  `git merge feature/my-change`

- Push updated main: 

  `git push` 

Concept: **Branch** = safe workspace for development without breaking main.

---

## 5. Common Mistakes to Avoid

- Running `git add .` before creating `.gitignore` → Fix: Always create `.gitignore` FIRST
- Committing venv/, .idea/, or .env files → Fix: Use `git rm -r --cached <folder>`, add to `.gitignore`
- Committing directly to main without testing → Fix: Use feature branches, test before merging
- Writing vague commit messages ("fix", "update", "changes") → Fix: Be specific: "Add user authentication", "Fix login bug"
- Not pulling before pushing → Fix: Always `git pull` before `git push` to avoid conflicts
- Forgetting to activate `venv` before installing packages → Fix: Check for (venv) in terminal before `pip install`

---

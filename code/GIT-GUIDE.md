# Git Workflow Guide — SEUPD2526-DSPy

This guide defines how the team uses Git to collaborate cleanly and avoid conflicts.
Read it once carefully. Then use it as a reference every time you work on the project.

-----

## Branch Structure

```
main
├── pair1-baseline      ← Pair 1 (data loading, parsing, indexing)
└── pair2-baseline      ← Pair 2 (topic parsing, search, evaluation)
```

### Rules — non negotiable

1. **Never commit directly to `main`** — always work on your pair branch
1. **Always pull before starting work** — every single time, no exceptions
1. **Communicate before merging to main** — announce it in the team chat first
1. **One pair merges at a time** — never two people merging simultaneously
1. **Write meaningful commit messages** — “fix bug” is not acceptable, “fix BM25 score rounding in search.py” is

-----

## First Time Setup (do this once)

If you have already cloned the repository, just create and switch to your pair branch:

**Pair 1:**

```bash
git checkout -b pair1-baseline
git push origin pair1-baseline
```

**Pair 2:**

```bash
git checkout -b pair2-baseline
git push origin pair2-baseline
```

Verify you are on the correct branch — you should see your branch name:

```bash
git branch
# output:
# * pair1-baseline      ← the * means you are here
#   main
```

-----

## Daily Workflow — follow this every single day

### Step 1 — Before you start working (every time, no exceptions)

Pull the latest changes from your pair branch:

```bash
git pull origin pair1-baseline    # Pair 1
git pull origin pair2-baseline    # Pair 2
```

Also sync with main regularly to stay updated with the other pair’s stable work:

```bash
git pull origin main
```

> ⚠️ Skipping this step is the number one cause of conflicts. Do not skip it.

### Step 2 — Work normally

Edit your files, run your scripts, test your code.

### Step 3 — Save your work (commit and push)

Check what you changed:

```bash
git status
```

Add your changes:

```bash
git add .                          # add everything
git add src/parse_documents.py     # or add a specific file
```

Commit with a meaningful message:

```bash
git commit -m "add document parser for CLEF corpus"
```

Push to your pair branch:

```bash
git push origin pair1-baseline    # Pair 1
git push origin pair2-baseline    # Pair 2
```

### Step 4 — Tell your pair partner

After pushing, send a message in the team chat so your partner knows to pull before continuing.

-----

## Paired Coding — how A and B share work

When two people work on the same branch (e.g. `pair1-baseline`):

**Person A** commits and pushes:

```bash
git add .
git commit -m "implement document parser"
git push origin pair1-baseline
```

**Person B** pulls before continuing:

```bash
git pull origin pair1-baseline
# now B has A's latest changes and can continue
```

> 💡 Always communicate in chat before pulling — “I just pushed, you can pull now”

-----

## Getting the Other Pair’s Stable Work

When Pair 1 merges something stable to `main`, Pair 2 should update their branch:

```bash
# Make sure you are on your branch
git checkout pair2-baseline

# Pull the latest main into your branch
git pull origin main
```

This keeps your branch up to date without touching main directly.

-----

## Merging to Main — the handoff procedure

Only merge to main when your code is:

- Tested and working
- Agreed upon by both people in your pair

**Procedure:**

1. Announce in team chat: *“Pair 1 is merging to main in 10 minutes”*
1. Wait for confirmation that nobody else is merging
1. Switch to main and merge:

```bash
git checkout main
git pull origin main              # sync first
git merge pair1-baseline          # merge your branch
git push origin main              # push to remote
```

1. Announce in chat: *“Pair 1 merge done, you can pull main now”*

-----

## Handling Conflicts

A conflict happens when two people edited the same line of the same file.
Git will mark the conflict like this inside the file:

```
<<<<<<< your branch
    your version of the code
=======
    the other person's version
>>>>>>> main
```

**How to resolve it:**

1. Open the file in VS Code
1. VS Code will highlight the conflict with buttons: `Accept Current`, `Accept Incoming`, `Accept Both`
1. Choose the correct version (talk to your teammate if unsure)
1. Save the file
1. Commit the resolution:

```bash
git add .
git commit -m "resolve merge conflict in parse_documents.py"
```

> 💡 The best way to avoid conflicts is to never edit the same file as someone else.
> Each pair owns specific files — stick to your files.

-----

## File Ownership — who edits what

To avoid conflicts, each pair owns specific files and only edits those:

|File                    |Owner                           |
|------------------------|--------------------------------|
|`src/load_data.py`      |Pair 1                          |
|`src/parse_documents.py`|Pair 1                          |
|`src/index.py`          |Pair 1                          |
|`src/parse_topics.py`   |Pair 2                          |
|`src/search.py`         |Pair 2                          |
|`src/evaluate.py`       |Pair 2                          |
|`src/pipeline.py`       |Both — coordinate before editing|
|`docker-compose.yml`    |Coordinate before editing       |
|`README-DOCKER.md`      |Coordinate before editing       |


> ⚠️ If you need to edit a file owned by the other pair, communicate first in the team chat.

-----

## Quick Reference

|Situation                    |Command                                            |
|-----------------------------|---------------------------------------------------|
|Check which branch you are on|`git branch`                                       |
|Switch to your branch        |`git checkout pair1-baseline`                      |
|Pull latest changes          |`git pull origin pair1-baseline`                   |
|Check what changed           |`git status`                                       |
|Add all changes              |`git add .`                                        |
|Commit                       |`git commit -m "your message"`                     |
|Push                         |`git push origin pair1-baseline`                   |
|Pull main into your branch   |`git pull origin main`                             |
|Merge your branch to main    |`git checkout main` then `git merge pair1-baseline`|

-----

## Common Mistakes and Fixes

### “I committed on main by accident”

```bash
# Undo the last commit but keep your changes
git reset HEAD~1
# Now switch to your branch and commit there
git checkout pair1-baseline
git add .
git commit -m "your message"
```

### “I forgot to pull before working and now there are conflicts”

```bash
git stash                        # temporarily save your work
git pull origin pair1-baseline   # pull latest
git stash pop                    # bring your work back
# resolve any conflicts if they appear
```

### “I don’t know what branch I am on”

```bash
git branch        # shows all branches, * marks current one
git status        # also shows current branch at the top
```

### “I want to see what changed before pulling”

```bash
git fetch origin
git diff pair1-baseline origin/pair1-baseline
```

-----

## Commit Message Convention

Use this format for all commit messages:

```
<type>: <short description>

Examples:
feat: add document parser for CLEF corpus
fix: resolve BM25 score rounding error
refactor: split search.py into smaller functions
docs: update README with new docker commands
test: add verification script for pyserini index
```

Types: `feat` (new feature), `fix` (bug fix), `refactor` (restructure), `docs` (documentation), `test` (testing)

-----

## Communication Rules

- Always announce in team chat **before** merging to main
- Always announce in team chat **after** pushing to your branch so your partner can pull
- If you are stuck on a conflict for more than 15 minutes, ask the team — do not guess
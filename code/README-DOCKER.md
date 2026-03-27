# Docker Environment Guide

This guide explains how to set up, use, and troubleshoot the Docker environment for the project.
There are two independent containers: one for **PyLucene** and one for **Pyserini**.
Both produce results in the shared `experiment/` folder.

-----

## Project Structure

```
code/
├── docker-compose.yml          ← controls both containers
├── README.md                   ← this file
├── experiment/                 ← shared output folder (visible on your PC)
│   ├── docs/                   ← raw documents used by Pyserini
│   ├── index/
│   │   ├── pylucene/           ← PyLucene index files (auto-generated)
│   │   └── pyserini/           ← Pyserini index files (auto-generated)
│   └── runs/
│       ├── runs-pylucene.txt   ← PyLucene results
│       └── run-pyserini.txt    ← Pyserini results
├── pylucene/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       └── hello-ir.py
└── pyserini/
    ├── Dockerfile
    ├── requirements.txt
    └── src/
        └── hello-ir-pyserini.py
```

-----

## Prerequisites

Before starting, make sure you have:

- **Docker Desktop** installed and running
- **WSL2** enabled (Windows users)
- Docker Desktop shows a green icon in the system tray

To verify Docker is working, open a terminal and run:

```bash
docker run hello-world
```

If it prints `Hello from Docker!` you are ready to go.

-----

## First Time Setup (build the images)

> ⚠️ This step is only needed **once**, or when the Dockerfile or requirements.txt change.
> It will take **10–20 minutes** the first time because it downloads Java, Python packages, and dependencies.
> **Do not interrupt it** — it may look frozen but it is downloading in the background.

To build **both** containers at once:

```bash
docker compose build
```

To build **only one** container (faster):

```bash
docker compose build pylucene
docker compose build pyserini
```

-----

## Daily Usage

### Working with PyLucene

**Step 1 — Enter the container:**

```bash
docker compose run pylucene
```

You will see a prompt like:

```
root@xxxxxxxxx:/app#
```

This means you are inside the container.

**Step 2 — Run the script:**

```bash
python src/hello-ir.py
```

**Step 3 — Exit the container when done:**

```bash
exit
```

Or press `Ctrl+D`.

-----

### Working with Pyserini

**Step 1 — Enter the container:**

```bash
docker compose run pyserini
```

**Step 2 — Run the script:**

```bash
python src/hello-ir-pyserini.py
```

**Step 3 — Exit the container when done:**

```bash
exit
```

Or press `Ctrl+D`.

-----

### Working on both independently

You can run each container completely independently — they do not interfere with each other.
Open **two separate terminal windows** and run one container in each:

Terminal 1:

```bash
docker compose run pylucene
```

Terminal 2:

```bash
docker compose run pyserini
```

-----

## Expected Output

After running both scripts, your `experiment/` folder on your PC should look like this:

```
experiment/
├── docs/
│   └── docs.jsonl              ← documents indexed by Pyserini
├── index/
│   ├── pylucene/               ← Lucene index files (do not edit)
│   └── pyserini/               ← Lucene index files (do not edit)
└── runs/
    ├── runs-pylucene.txt        ← PyLucene results
    └── run-pyserini.txt         ← Pyserini results
```

The run files follow the standard **TREC eval format**:

```
query_id   Q0   doc_id   rank   score   run_name
001        Q0   d1       0      0.4298  hello-IR-pylucene
001        Q0   d3       1      0.1974  hello-IR-pylucene
```

Each line means: for `query_id`, document `doc_id` was retrieved at position `rank` with score `score`.

-----

## Stopping and Cleaning

### Stop a running container

If a container is running in the background:

```bash
docker compose down
```

### Check disk space used by Docker

```bash
docker system df
```

### Clean unused images and containers (safe)

```bash
docker system prune
```

### Full clean — removes everything including cache (use when builds fail strangely)

```bash
docker system prune -a
```

> ⚠️ After a full clean you will need to rebuild with `docker compose build`.

-----

## Command Reference

|Command                        |What it does                       |
|-------------------------------|-----------------------------------|
|`docker compose build`         |Build both containers              |
|`docker compose build pylucene`|Build only PyLucene                |
|`docker compose run --rm pylucene`  |Enter PyLucene container           |
|`docker compose down`          |Stop all running containers        |
|`docker system df`             |Check Docker disk usage            |
|`docker system prune`          |Remove unused containers and images|
|`docker system prune -a`       |Full clean (requires rebuild after)|
|`exit` or `Ctrl+D`             |Exit a container terminal          |

-----

## Troubleshooting

### Problem: `python: can't open file 'src/hello-ir.py': No such file or directory`

**Cause:** You are running the command from outside the container, or in the wrong directory.  
**Fix:** Make sure you entered the container first with `docker compose run pylucene` and your prompt shows `root@xxxxx:/app#`. Then run `python src/hello-ir.py`.

-----

### Problem: `FileNotFoundError: experiment/runs/runs-pylucene.txt`

**Cause:** The `experiment/runs/` folder does not exist on your PC.  
**Fix:** Create it manually on your PC, or add `os.makedirs()` to the script (already done in the latest version).

-----

### Problem: `SyntaxError` when typing `python src/hello-ir.py`

**Cause:** You typed the command inside a Python shell (`>>>`) instead of the Linux terminal (`#`).  
**Fix:** Type `exit()` to leave Python, then run the command again.

-----

### Problem: Build is very slow or appears frozen

**Cause:** Docker is downloading large packages (PyTorch is ~500MB, PyLucene image is ~1.5GB).  
**Fix:** Wait. Do not interrupt the build. It can take 10–20 minutes on the first run.

-----

### Problem: `Package openjdk-21-jre-headless has no installation candidate`

**Cause:** Wrong base image or wrong Debian version.  
**Fix:** Make sure your `pyserini/Dockerfile` uses `FROM eclipse-temurin:21-jre-jammy` as the base image.

-----

### Problem: Container opens a Python shell `>>>` instead of a bash terminal `#`

**Cause:** The base image has Python as its default command.  
**Fix:** Make sure your Dockerfile ends with `CMD ["/bin/bash"]` and rebuild with `docker compose build pylucene`.

-----

### Problem: Results disappear after closing the container

**Cause:** The `experiment/` folder is not mapped as a volume in `docker-compose.yml`.  
**Fix:** Make sure your `docker-compose.yml` has these volume lines for both services:

```yaml
volumes:
  - ./pylucene/src:/app/src
  - ./experiment:/app/experiment
```

-----

### Problem: `docker compose` command not found

**Cause:** Old version of Docker uses `docker-compose` (with a hyphen) instead of `docker compose`.  
**Fix:** Either update Docker Desktop, or replace `docker compose` with `docker-compose` in all commands.

-----

## Reporting Issues

If you encounter a problem not listed above, please share:

1. **The exact command you ran**
1. **The full error message** (copy-paste, do not take a screenshot)
1. **Which container** (pylucene or pyserini)
1. **Your operating system** (Windows/Mac/Linux)

Post the issue in the team chat with all four pieces of information.
This will make it much faster to diagnose and fix.

# Upggrading Guide

This guide was initialized with AI tools that was used during the tests in order to create the right enviroment. All members can change it or add new information that can be useful for future needs.
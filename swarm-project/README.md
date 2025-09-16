# Hedge Meet Attention

Adaptive late-fusion of heterogeneous, multimodal agents for robust decision making.

This repository accompanies Daniel Huang’s MSc thesis, **“Hedge Meet Attention: Adaptive Late‑Fusion Heterogeneous Multimodal Agents.”** The README walks you through setup, data preparation, running simulations, and plotting results.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Setup](#project-setup)
3. [Data Loading](#data-loading)
4. [Preprocessing](#preprocessing)
5. [Configuration](#configuration)
6. [Run a Simulation](#run-a-simulation)
7. [Plotting & Analysis](#plotting--analysis)
8. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Prerequisites
- **Python**: 3.10.9 (recommended)
- **OS**: macOS or Linux (Windows should work with minor adjustments)
- **Dependencies**: listed in `requirements.txt`

> **Note:** All commands below assume you are running them **from the repository root**.

```bash
cd swarm-project
```

---

## Project Setup
Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# On Windows (PowerShell): .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Data Loading
Download the **AVE** dataset from the official repository:

- GitHub: https://github.com/YapengTian/AVE-ECCV18

Place it under `data/` so your directory tree looks like this:

```
swarm-project/
├── README.md
├── data/
│   └── AVE_Dataset/
│       ├── AVE/
│       ├── annotations.txt
│       ├── ReadMe.txt
│       ├── testSet.txt
│       ├── trainSet.txt
│       └── valSet.txt
```

Then load and split the dataset:

```bash
python3 python_sims/load_AVE.py
```

This will create CSV splits under:

```
 data/AVE_Dataset/splits/
```

---

## Preprocessing
Convert the dataset into TFRecords used by the project:

```bash
python3 python_sims/prepare_dataset.py
```

This will create preprocessed files under:

```
 data/AVE_Dataset/processed/
```

---

## Configuration
Simulation behavior is controlled by `configs/sim_config.yaml`. Adjust hyperparameters and options there (e.g., run name, number of rounds, agent settings, plotting windows, etc.).

> **Tip:** Keep a copy of each config used for experiments to ensure reproducibility.

---

## Run a Simulation
From the repository root:

```bash
python3 python_sims/sim2.py
```

On a typical laptop, a single run may take **~40 minutes** (your hardware and configuration may vary).

---

## Plotting & Analysis
After a run completes, generate the main figures/metrics by providing the same run name/date you used:

```bash
python3 python_sims/metric/plotting.py
```

### Corruption‑specific plots
If you ran experiments with a corrupted agent, use the dedicated script (arguments are provided via CLI, not the YAML config):

```bash
python3 python_sims/metric/plot_corruption.py \
  --date 2025-08-21-7 \
  --runs AVE_TEST_final_eval \
  --window 25 \
  --corrupted_mod audio
```

---

## Tips & Troubleshooting
- **Run from root**: All paths assume commands are executed at the repository root.
- **Python version**: If you encounter environment issues, verify you are on Python 3.10.x.
- **Missing data**: Ensure `data/AVE_Dataset/` matches the structure shown above before preprocessing.
- **Long runs**: Consider using `screen`/`tmux` or saving logs to track progress during long simulations.

---

If you use this repository or build upon it, please cite the AVE dataset and acknowledge this project accordingly.

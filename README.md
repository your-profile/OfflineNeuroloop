# Offline Neuro-Loop: Brain-guided Human-in-the-Loop Reinforcement Learning

Code for offline RL guided by fNIRS-derived human feedback. Neural signals steer experience replay via prioritization, Q-augmentation, and/or reward augmentation across Flappy Bird, Lunar Lander, and Robot domains.

## Overview

Participant fNIRS and task data are aligned, used to train a feedback model, then injected into DQN/DDPG training under one of several integration modes (`finetune`, `interleave`, `pretrain`/`baseline`). Experiments sweep:

- **Conditions:** Baseline, Prioritization, Q-Augmentation, Reward Augmentation, All (ER or PER)
- **Feedback granularity:** binary, ternary, continuous
- **Tasks:** Passive, Active, Pooled
- **Ablations:** `mlp.model_noise`, `neural.beta`, `experiment.finetune_threshold`

Full grids are defined in `configs/sweep_hpc_*.yaml`, expanded to CSVs under `manifests/`, and run as SLURM job arrays.

## Setup

```bash
conda env create -f environment.yml
conda activate offline-neuroloop
```

Point `DATA_PATH` / sweep `paths.data_path` at your fNIRS participant root (expects `fNIRS/LabeledData/`, `fNIRS/FilteredData/`, `TaskData/`).

## File Tree

```text
configs/
  base.yaml                 # shared defaults
  domains/                  # Flappy, Lunar, Robot domain configs
  sweep_hpc_*.yaml          # HPC experiment grids (ER / PER)
  test_*.yaml               # local smoke / test configs
manifests/                  # trial CSVs for SLURM arrays
docs/HPC.md                 # cluster workflow details
src/
  envs/                     # Flappy Bird, Lunar Lander (+ robot via gymnasium)
  neural/                   # fNIRS load / align / buffers
  networks/                 # DQN, DDPG
  models/                   # feedback model training
  training_loop_*.py        # baseline / finetune / interleave / surrogate
  results/                  # notebooks + aggregated CSVs
trial.py                    # single-trial runner
run_trial.py                # CLI / SLURM entry (one manifest row)
run_test.py                 # local combinatorial test loop
generate_manifest.py        # expand sweep YAML → CSV
experiment_sweep.py         # grid / config / naming helpers
submit_hpc.sh               # sbatch wrapper (one array job per manifest)
run_bash.sh / run_batch.sh  # SLURM worker scripts
merge_results.py            # merge trial CSVs
```

## Running Tests

Quick local checks (edit paths / episode counts in `run_test.py` or domain test YAMLs first):

```bash
python run_test.py
```

Or a single trial from CLI / a manifest row:

```bash
python run_trial.py \
  --domain-key Flappy \
  --integration finetune \
  --condition Baseline-PER \
  --granularity binary \
  --task Pooled \
  --seed 1 \
  --n-episodes 10
```

```bash
python run_trial.py --manifest manifests/test.csv --trial-id 1
```

Analysis notebooks live under `src/results/`.

## Creating Manifests

```bash
# Example: Robot, finetune, binary, PER grid
python generate_manifest.py \
  -s configs/sweep_hpc_PER.yaml \
  --filter-domain Robot \
  --filter-integration finetune \
  --filter-granularity binary \
  -o manifests/robot_finetune_binary_PER.csv
```

Useful flags: `--filter-condition`, `--filter-task`, `--filter-ablation-key`, `--shard-by`, `--dry-run`, `-p <profile>`.

Update `paths.data_path` / `paths.results_path` in the sweep YAML (or override via env vars at submit time) before generating large manifests.

## Running Manifests (HPC Cluster)

See [docs/HPC.md](docs/HPC.md) for full details. Short version:

```bash
module purge && module load miniforge/25.3.0
conda env create -f environment.yml   # once
export NEUROLOOP_CONDA_ENV=offline-neuroloop
export SCRATCH=/cluster/scratch/$USER
export NEUROLOOP_RESULTS_ROOT=$SCRATCH/OfflineNeuroloop_results
export NEUROLOOP_LOG_DIR=$SCRATCH/neuroloop_logs

cd ~/OfflineNeuroloop

./submit_hpc.sh --dry-run manifests/flappy_finetune_binary_PER.csv
./submit_hpc.sh manifests/flappy_finetune_binary_PER.csv
# or
./submit_hpc.sh --all
```

Default: one SLURM array per manifest CSV, up to 50 concurrent tasks (`ARRAY_CAP=50`). Set `SKIP_COMPLETED=1` (default) to resume. Per-domain wall times and env vars are documented in `docs/HPC.md`.

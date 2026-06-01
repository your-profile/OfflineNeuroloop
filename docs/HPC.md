# HPC parallel runs

## How jobs are structured

- **One SLURM array task = one trial** (`run_trial.py --trial-id N`).
- **`run_bash.sh` does not run the full grid** in a single task; it only works with `SLURM_ARRAY_TASK_ID`.
- Large studies are split into **manifest shards** (CSV files), each submitted as its own array job with a **domain-appropriate time limit**.

## Trial count

For `configs/sweep_hpc.yaml`:

```
trials = |integrations| × Σ|ablation vals| × |seeds| × |granularities| × |conditions| × |domains| × |tasks|
```

With all dimensions enabled (3 integrations, 2 ablation studies × 5 values, 15 seeds, 3 granularities, 5 conditions, 3 domains, 3 tasks):

**60,750 trials** total.

Default sharding (`domain_integration_ablation_granularity`) yields **81 manifests** (~750 trials each):

- 3 domains × 3 integrations × 2 ablation studies × 3 granularities

Use coarser sharding if you prefer: `SHARD_BY=domain_integration_ablation ./submit_hpc.sh --generate-shards` (27 manifests).

## Quick start

```bash
# 1) Generate shard manifests (manifests/*.csv)
./submit_hpc.sh --generate-shards

# 2) Preview submissions
./submit_hpc.sh --dry-run --all

# 3) Submit all shards (one sbatch per CSV)
./submit_hpc.sh --all

# Or submit a single shard
./submit_hpc.sh manifests/lunar__pretrain__mlp_model_noise.csv
```

## Time limits (per array task)

| Domain | `--time`   | ~trial duration |
|--------|------------|-----------------|
| Flappy | `1:30:00`  | ~40 min         |
| Lunar  | `3:00:00`  | ~2 h            |
| Robot  | `8:00:00`  | ~5+ h           |

Each shard job can use up to **3 days of wall clock** on the cluster while many tasks run in parallel (`ARRAY_CAP`, default 80).

## Single shard or custom filter

```bash
python generate_manifest.py -s configs/sweep_hpc.yaml \
  --filter-domain Flappy \
  --filter-integration pretrain \
  --filter-ablation-key mlp.model_noise \
  --filter-granularity binary \
  -o manifests/flappy_pretrain_noise_binary.csv

# Or generate all shards for one granularity
FILTER_GRANULARITY=binary ./submit_hpc.sh --generate-shards

export MANIFEST=manifests/flappy_pretrain_binary.csv
sbatch --time=3:00:00 --array=1-$(($(wc -l < "$MANIFEST") - 1))%50 \
  --export=ALL,MANIFEST="$(pwd)/$MANIFEST" run_bash.sh
```

## Smoke test on cluster

```bash
python generate_manifest.py -s configs/sweep_hpc.yaml -p smoke -o manifests/smoke.csv
./submit_hpc.sh manifests/smoke.csv
```

## Resuming

Completed trials write `src/results/runs/trial_XXXXX_<integration>.csv`.  
Arrays default to `SKIP_COMPLETED=1` so reruns skip existing outputs.

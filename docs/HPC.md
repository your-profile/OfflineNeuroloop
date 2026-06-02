# HPC parallel job arrays

Non-interactive **batch** jobs via `sbatch` (not `salloc` / interactive). Default: **one job array per manifest CSV**, with up to **50 trials running in parallel** (`ARRAY_CAP=50`).

## Recommended workflow

```bash
# On the cluster — point results at scratch (edit sweep_hpc.yaml, then regenerate manifests)
export SCRATCH=/cluster/scratch/$USER   # your site may use $WORK instead
export NEUROLOOP_RESULTS_ROOT=$SCRATCH/OfflineNeuroloop_results
export NEUROLOOP_LOG_DIR=$SCRATCH/neuroloop_logs

cd $SCRATCH/OfflineNeuroloop    # writable clone of the repo

# 1) Generate one CSV per shard (domain × integration × ablation × granularity)
./submit_hpc.sh --generate-shards

# 2) Preview (~wall time per manifest at ARRAY_CAP=50)
./submit_hpc.sh --dry-run --all

# 3) Submit one array job per CSV (9+ jobs for a large study)
./submit_hpc.sh --all

# Or a single filtered manifest (e.g. 1665 Flappy binary pretrain trials)
./submit_hpc.sh manifests/flappy_pretrain_binary.csv
```

Each submission is:

```text
sbatch --array=1-N%50 --time=<per-trial> run_bash.sh
```

So **1665 trials** → `1-1665%50` → about **34 waves × ~40 min ≈ 23 h** wall clock for Flappy (if the queue keeps 50 slots full).

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SUBMIT_MODE` | `array` | `array` = parallel; `batch` = sequential one job |
| `ARRAY_CAP` | `50` | Max concurrent array tasks (`%50`) |
| `NEUROLOOP_RESULTS_ROOT` | (from manifest) | Override where `trial_*.csv` are written |
| `NEUROLOOP_DATA_ROOT` | (from manifest) | Override fNIRS data root |
| `NEUROLOOP_LOG_DIR` | `$SCRATCH/neuroloop_logs` | SLURM log files |
| `NEUROLOOP_REPO` | (set by `submit_hpc.sh`) | Project root; avoids `/var/spool/slurm/...` path bug |
| `SCRATCH` | — | Used for logs/work dirs if set |

Always submit from the repo directory: `cd /path/to/OfflineNeuroloop && ./submit_hpc.sh ...`

Set `paths.results_path` in `configs/sweep_hpc.yaml` to scratch **before** `generate_manifest.py`, or use `NEUROLOOP_RESULTS_ROOT` at submit time.

## Per-domain SLURM time (per array task)

| Domain | `--time` / task | ~min / trial |
|--------|-----------------|--------------|
| Flappy | 1:30:00 | 40 |
| Lunar  | 3:00:00 | 120 |
| Robot  | 8:00:00 | 300 |

## Many manifests (domains × granularities × integrations)

Default shard name pattern: `manifests/<domain>__<integration>__<ablation>__<granularity>.csv`

Submitting `--all` launches **one array job per CSV**. Total cluster load ≈ `ARRAY_CAP × (number of concurrent array jobs)`. If jobs stay pending, lower `ARRAY_CAP` or submit in waves:

```bash
ARRAY_CAP=30 ./submit_hpc.sh manifests/flappy_*.csv
# later
ARRAY_CAP=30 ./submit_hpc.sh manifests/lunar_*.csv
```

## Sequential mode (not recommended for 1000+ trials)

```bash
SUBMIT_MODE=batch ./submit_hpc.sh manifests/foo.csv
```

## Logs and results

- SLURM logs: `${NEUROLOOP_LOG_DIR}/neuroloop_<jobid>_<taskid>.log`
- Trial CSVs: `${results_path}/src/results/runs/trial_XXXXX_<integration>.csv`
- `SKIP_COMPLETED=1` skips trials that already produced a CSV

## Smoke test

```bash
python generate_manifest.py -s configs/sweep_hpc.yaml -p smoke -o manifests/smoke.csv
./submit_hpc.sh manifests/smoke.csv
```

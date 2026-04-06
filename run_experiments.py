import itertools, copy, yaml
from train import run

with open("configs/base.yaml") as f:
    base = yaml.safe_load(f)

# Experimental Variables
DOMAINS_TASKS = {
    "robot":        ["passive", "active", "pooled"],
    "lunar_lander": ["passive", "active", "pooled"],
    "flappy_bird":  ["passive", "active", "pooled"],
}

NEURAL_CONDITIONS = [
    "baseline",
    "reward_shaping",
    "lr_modulation",
    "prioritization",
]

GRANULARITIES = ["binary", "ternary", "continuous"]

# Ablation Studies

ABLATIONS = [
    {"key": ["neural", "model_noise"], "vals": [0.0, 0.1, 0.3, 0.5]},
    {"key": ["neural", "smoothing_method"], "vals": ["none", "majority_vote"]},
    {"key": ["neural", "credit_assignment"], "vals": ["window_based", "temporal_shift"]},
    {"key": ["neural", "temporal_shift"], "vals": [1.0, 2.0, 3.0, 4.0, 5.0]},
    {"key": ["neural", "smoothing_window_size"], "vals": [3, 5, 7]},
]

def set_nested(cfg, keys, val):
    cfg[keys[0]][keys[1]] = val

def make_run_name(cfg):
    e = cfg["experiment"]
    n = cfg["neural"]
    return (
        f"{e['domain']}__{e['task']}__{e['neural_condition']}"
        f"__{e['model_granularity']}"
        f"__noise{n['model_noise']}__{n['smoothing_type']}"
        f"__{n['credit_assignment']}"
    )

# 1. Full condition grid (baseline ablation settings)
for (domain, tasks), condition, granularity in itertools.product(
    DOMAINS_TASKS.items(), NEURAL_CONDITIONS, GRANULARITIES
):
    for task in tasks:
        cfg = copy.deepcopy(base)
        cfg["experiment"].update({
            "domain": domain,
            "task": task,
            "neural_condition": condition,
            "model_granularity": granularity,
        })
        run(cfg, run_name=make_run_name(cfg))

# 2. Ablation sweeps across the full condition grid
for ablation, (domain, tasks), condition, granularity in itertools.product(
    ABLATIONS, DOMAINS_TASKS.items(), NEURAL_CONDITIONS, GRANULARITIES
):
    for task in tasks:
        for val in ablation["vals"]:
            cfg = copy.deepcopy(base)
            cfg["experiment"].update({
                "domain": domain,
                "task": task,
                "neural_condition": condition,
                "model_granularity": granularity,
            })
            set_nested(cfg, ablation["key"], val)
            run(cfg, run_name=make_run_name(cfg))
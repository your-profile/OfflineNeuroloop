"""
Script for running tests give the config file and data path. Real experiments were run through the HPC cluster
"""

import itertools
from pathlib import Path
import yaml
from experiment_sweep import build_cfg, make_run_name
from trial import run

REPO_ROOT = Path(__file__).resolve().parent

with open(REPO_ROOT / "configs/base.yaml") as f:
    base = yaml.safe_load(f)

# Ablations
ABLATIONS = [
    {"key": ["experiment", "finetune_threshold"], "vals": [0.0, 0.2, 0.5, 0.7]},
    {"key": ["mlp", "model_noise"], "vals": [0.0, 0.5, 1.0]},
    {"key": ["neural", "beta"], "vals": [0.5, 1.0, 5.0]},
]

INTEGRATION = "finetune"

NEURAL_CONDITIONS = [
    "Baseline-PER",
    "Prioritization-PER",
    "Q-Augmentation-PER",
    "Reward Augmentation-PER",
    "All-PER",
]

GRANULARITIES = ["binary", "ternary", "continuous"]

SEEDS = [1,2,3,4,5,6,7,8,9,10]

# Domain configs: Flappy, Lunar, Robot
DOMAIN_CONFIGS = {
    "Flappy": REPO_ROOT / "configs/test_flappy.yaml",
    "Lunar": REPO_ROOT / "configs/test_lunar.yaml",
    "Robot": REPO_ROOT / "configs/test_robot.yaml",
}

# Task types: Passive, Active, Pooled
TASKS_BY_DOMAIN = {
    "Flappy": ["Passive", "Active","Pooled"],
    "Lunar": ["Passive", "Active", "Pooled"],
    "Robot": ["Passive", "Active", "Pooled"],
}

RESULTS_FILE_NAME = "test_results.csv"

DATA_PATH = "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/"
RESULTS_PATH = "/Users/juliasantaniello/Desktop/OfflineNeuroloop/"
# DATA_PATH = '/Users/maddiebrower/workspace/tufts/fNIRS2RL/Experiment/ParticipantData/'
# RESULTS_PATH = '/Users/maddiebrower/workspace/tufts/OfflineNeuroloop/'
# DATA_PATH = '/cluster/home/mbrowe02/fNIRS2RL/Experiment/ParticipantData/'
# RESULTS_PATH = '/cluster/home/mbrowe02/OfflineNeuroloop/'

# print configurations
def print_cfg(cfg):
    for k, v in cfg.items():
        print(f"{k}: {v}")
        if isinstance(v, dict):
            print_cfg(v)

# run tests
for ablation, (domain, domain_config), seed, granularity, condition in itertools.product(
    ABLATIONS,
    DOMAIN_CONFIGS.items(),
    SEEDS,
    GRANULARITIES,
    NEURAL_CONDITIONS,
):
    tasks = TASKS_BY_DOMAIN[domain]
    for task in tasks:
        for val in ablation["vals"]:
            cfg = build_cfg(
                integration=INTEGRATION,
                condition=condition,
                seed=seed,
                granularity=granularity,
                task=task,
                ablation=ablation,
                ablation_val=val,
                base_config=REPO_ROOT / "configs/base.yaml",
                domain_config=domain_config,
            )
            if cfg is None:
                continue

            if base.get("rl", {}).get("n_episodes") == "test":
                cfg["rl"]["n_episodes"] = 10

            print_cfg(cfg)

            run(
                cfg,
                run_name=make_run_name(cfg),
                DATA_PATH=DATA_PATH,
                RESULTS_PATH=RESULTS_PATH,
                RESULTS_FILE_NAME=RESULTS_FILE_NAME,
                verbose=cfg["experiment"]["verbose"],
                inverse=False,
            )

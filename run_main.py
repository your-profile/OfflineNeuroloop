import itertools, copy, yaml
from trial import run

with open("configs/base.yaml") as f:
    base = yaml.safe_load(f)

#TODO: configure experiment variables for: smoothing method

# Experimental Variables
# DOMAINS_TASKS = {
#     # "robot":        ["Passive", "Active", "Pooled"],
#     "lunar": ["Passive", "Active", "Pooled"],
#     # "flappy":  ["Passive", "Active", "Pooled"],
# }

# NEURAL_CONDITIONS = [
#     "Baseline",
#     "Reward Augmentation",
#     "Prioritization",
#     "Epsilon Modulation",
#     "LR Modulation",
# ]
# 

# Ablation Studies

ABLATIONS = [
    # {"key": ["neural", "model_noise"], "vals": [0.1, 0.2, 0.3]},
    # {"key": ["neural", "temporal_shift"], "vals": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
#     {"key": ["neural", "smoothing_window_size"], "vals": [1, 3, 5, 7]},
]

# testing: single condition, binary granularity, no ablation sweeps
NEURAL_CONDITIONS = [
    "Baseline",
]

#GRANULARITIES = ["binary", "ternary", "continuous"]
GRANULARITIES = ["binary"]
SEEDS = [42] #, 43, 44, 45, 46]

DOMAINS_TASKS = {
    "Flappy": ["Passive"],
    # "Lunar": ["Passive"],
    # "Robot": ["Passive"],
}

DATA_PATH = '/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/' 
RESULTS_PATH = '/Users/juliasantaniello/Desktop/OfflineNeuroloop/' 

# DATA_PATH = '/Users/maddiebrower/workspace/tufts/fNIRS2RL/Experiment/ParticipantData/' 
# RESULTS_PATH = '/Users/maddiebrower/workspace/tufts/OfflineNeuroloop/' 

def set_nested(cfg, keys, val):
    cfg[keys[0]][keys[1]] = val

def make_run_name(cfg):
    e = cfg["experiment"]
    n = cfg["neural"]

    return (
        f"{e['domain']}__{e['task']}__{e['condition']}"
        f"__{e['model_granularity']}"
        f"__noise{n['model_noise']}__{n['smoothing_window_size']}"
        f"__{n['temporal_shift']}"
        f"__{n['temporal_shift']}"
        f"__{n['smoothing_window_size']}"
    )

# Full condition grid (baseline settings)
for (domain, tasks), condition, granularity, seed in itertools.product(
    DOMAINS_TASKS.items(), NEURAL_CONDITIONS, GRANULARITIES, SEEDS
):
    with open(f"configs/domains/{domain}.yaml") as f:
        domain_base = yaml.safe_load(f)
        domain_cfg = copy.deepcopy(domain_base)


    for task in tasks:
        cfg = copy.deepcopy(base)
        cfg["experiment"].update({
            "domain": domain_cfg["experiment"]["domain"],
            "task": task,
            "condition": condition,
            "experiment_list": [NEURAL_CONDITIONS.index(condition)],
            "model_granularity": granularity,
            "random_state": seed,
        })

        cfg["mlp"].update({
            "binary_hidden_layer_sizes": domain_cfg["mlp"]["binary_hidden_layer_sizes"],
            "ternary_hidden_layer_sizes": domain_cfg["mlp"]["ternary_hidden_layer_sizes"],
            "regressor_hidden_layer_sizes": domain_cfg["mlp"]["regressor_hidden_layer_sizes"],
            "clf_activation": domain_cfg["mlp"]["clf_activation"],
            "reg_activation": domain_cfg["mlp"]["reg_activation"],
        })

        cfg["rl"].update({
            "n_episodes": domain_cfg["rl"]["n_episodes"],
            "algorithm": domain_cfg["rl"]["algorithm"],
            "steps": domain_cfg["rl"]["steps"],
            "action_space": domain_cfg["rl"]["action_space"],
            "observation_space": domain_cfg["rl"]["observation_space"]

        })

        if condition == "Prioritization":
            cfg['buffer_type'] = "PER"

        run(cfg, run_name=make_run_name(cfg), DATA_PATH=DATA_PATH, RESULTS_PATH=RESULTS_PATH)

# Ablation sweeps across the full condition grid
for ablation, (domain, tasks), condition, granularity, seed in itertools.product(
    ABLATIONS, DOMAINS_TASKS.items(), NEURAL_CONDITIONS, GRANULARITIES, SEEDS
):
    with open(f"configs/domains/{domain}.yaml") as f:
        domain_base = yaml.safe_load(f)
        domain_cfg = copy.deepcopy(domain_base)

    for task in tasks:
        for val in ablation["vals"]:
            
            cfg = copy.deepcopy(base)

            cfg["experiment"].update({
               "domain": domain_cfg["experiment"]["domain"],
                "task": task,
                "condition": condition,
                "experiment_list": [NEURAL_CONDITIONS.index(condition)],
                "model_granularity": granularity,
                "random_state": seed,
            })

            cfg["mlp"].update({
                "binary_hidden_layer_sizes": domain_cfg["mlp"]["binary_hidden_layer_sizes"],
                "ternary_hidden_layer_sizes": domain_cfg["mlp"]["ternary_hidden_layer_sizes"],
                "regressor_hidden_layer_sizes": domain_cfg["mlp"]["regressor_hidden_layer_sizes"],
                "clf_activation": domain_cfg["mlp"]["clf_activation"],
                "reg_activation": domain_cfg["mlp"]["reg_activation"],
            })

            cfg["rl"].update({
                "n_episodes": domain_cfg["rl"]["n_episodes"],
                "algorithm": domain_cfg["rl"]["algorithm"],
                "steps": domain_cfg["rl"]["steps"],
                "action_space": domain_cfg["rl"]["action_space"],
                "observation_space": domain_cfg["rl"]["observation_space"]
            })

            if condition == "Prioritization":
                cfg['rl']['buffer_type'] = "PER"

            set_nested(cfg, ablation["key"], val)
            run(cfg, run_name=make_run_name(cfg), DATA_PATH=DATA_PATH, RESULTS_PATH=RESULTS_PATH)
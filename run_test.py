import itertools, copy, yaml
from trial import run

with open("configs/test.yaml") as f:
    base = yaml.safe_load(f)

NEURAL_CONDITION_MAP = {
    "Baseline-ER": [0],
    "Baseline-PER": [0],
    "Reward Augmentation-ER": [1],
    "Reward Augmentation-PER": [1],
    "Prioritization-ER": [2],
    "Prioritization-PER": [2],
    "Q-Augmentation-ER": [3],
    "Q-Augmentation-PER": [3],
    "All-ER": [0, 1, 3],
    "All-PER": [0, 1, 2, 3],

}
# Ablation Studies

ABLATIONS = [
    {"key": ["mlp", "model_noise"], "vals": [0.0]},#, 0.2, 0.5, 1.0]},
    # {"key": ["experiment", "finetune_threshold"], "vals": [0.0]},#, 0.2, 0.5, 1.0]},
    # {"key": ["neural", "temporal_shift"], "vals": [0.0, 1.0, 2.0, 3.0]},
    {"key": ["neural", "beta"], "vals": [1.0]},
    # {"key": ["neural", "window_size_s"], "vals": [4.0, 5.0]},
]

# testing: single condition, binary granularity, no ablation sweeps
NEURAL_CONDITIONS = [
    # "Baseline-PER",
    "Prioritization-PER",
    "Q-Augmentation-PER",
    "Reward Augmentation-PER",
    "All-PER",
]

GRANULARITIES = ["binary", "ternary", "continuous"]
GRANULARITIES = ["binary"]

SEEDS = [42] #, 43, 44, 45, 46, 47, 48, 49, 50, 51] 

DOMAINS_TASKS = {
    # "Lunar": ["Active"], #, "Active", "Pooled"],
    "Flappy": ["Passive"]#, "Active", "Pooled"],
    # "Robot": ["Passive"]#, "Active", "Pooled"],
}


DATA_PATH = '/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/' 
RESULTS_PATH = '/Users/juliasantaniello/Desktop/OfflineNeuroloop/' 
RESULTS_FILE_NAME = 'trial_results_finetuning.csv'
#DATA_PATH = '/Users/maddiebrower/workspace/tufts/fNIRS2RL/Experiment/ParticipantData/' 
#RESULTS_PATH = '/Users/maddiebrower/workspace/tufts/OfflineNeuroloop/' 

# DATA_PATH = '/cluster/home/mbrowe02/fNIRS2RL/Experiment/ParticipantData/'
# RESULTS_PATH = '/cluster/home/mbrowe02/OfflineNeuroloop/'

def set_nested(cfg, keys, val):
    cfg[keys[0]][keys[1]] = val

def make_run_name(cfg):
    e = cfg["experiment"]
    n = cfg["neural"]
    m = cfg["mlp"]
    return (
        f"{e['domain']}__{e['task']}__{e['condition']}"
        f"__{e['model_granularity']}"
        f"__{e['pretrained_success_rate']}"
        f"__noise{m['model_noise']}__{n['smoothing_window_size']}"
        f"__{n['temporal_shift']}"
    )

def print_cfg(cfg):
    #print the cfg in a readable format
    for k, v in cfg.items():
        print(f"{k}: {v}")
        if isinstance(v, dict):
            print_cfg(v)
        else:
            print(f"{k}: {v}")

# Ablation sweeps across the full condition grid
for ablation, (domain, tasks), seed, granularity, condition in itertools.product(
    ABLATIONS, DOMAINS_TASKS.items(), SEEDS, GRANULARITIES, NEURAL_CONDITIONS
):
    # if condition == "Baseline-PER" and granularity != "binary":
    #     continue

    # if condition == "Baseline" and granularity != "binary":
    #     continue

    with open(f"configs/test.yaml") as f:
        domain_base = yaml.safe_load(f)
        domain_cfg = copy.deepcopy(domain_base)

    if base["rl"]["n_episodes"] == "test":
        domain_cfg["rl"]["n_episodes"] = 10


    for task in tasks:
        for val in ablation["vals"]:
            
            cfg = copy.deepcopy(base)

            cfg["experiment"].update({
               "domain": domain_cfg["experiment"]["domain"],
                "task": task,
                "condition": condition,
                "experiment_list": NEURAL_CONDITION_MAP[condition],
                "model_granularity": granularity,
                "random_state": seed,
                "pretrained_success_rate": domain_cfg["experiment"]["pretrained_success_rate"],
            })

            cfg["mlp"].update({
                "binary_hidden_layer_sizes": domain_cfg["mlp"]["binary_hidden_layer_sizes"],
                "ternary_hidden_layer_sizes": domain_cfg["mlp"]["ternary_hidden_layer_sizes"],
                "regressor_hidden_layer_sizes": domain_cfg["mlp"]["regressor_hidden_layer_sizes"],
                "reg_activation": domain_cfg["mlp"]["reg_activation"],
                "early_stopping": domain_cfg["mlp"]["early_stopping"],
                "binary_alpha": domain_cfg["mlp"]["binary_alpha"],
                "ternary_alpha": domain_cfg["mlp"]["ternary_alpha"],
                "reg_alpha": domain_cfg["mlp"]["reg_alpha"],
                "binary_activation": domain_cfg["mlp"]["binary_activation"],
                "ternary_activation": domain_cfg["mlp"]["ternary_activation"],
                "reg_activation": domain_cfg["mlp"]["reg_activation"],
            })

            cfg["rl"].update({
                "n_episodes": domain_cfg["rl"]["n_episodes"],
                "algorithm": domain_cfg["rl"]["algorithm"],
                "steps": domain_cfg["rl"]["steps"],
                "action_space": domain_cfg["rl"]["action_space"],
                "observation_space": domain_cfg["rl"]["observation_space"],
                "buffer_type": cfg["rl"]["buffer_type"]

            })
            
            if condition == "Baseline" and ((ablation["key"][1] == "model_noise" and val != 0.0) or (ablation["key"][1] == "temporal_shift" and val != 0.0)):
                continue
            set_nested(cfg, ablation["key"], val)
            print(cfg)
            # input("Press Enter to continue... \n")
            run(cfg, run_name=make_run_name(cfg), DATA_PATH=DATA_PATH, RESULTS_PATH=RESULTS_PATH,RESULTS_FILE_NAME=RESULTS_FILE_NAME, verbose = cfg["experiment"]["verbose"], inverse = False)

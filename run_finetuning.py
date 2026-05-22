import itertools, yaml
from trial import run
import argparse

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

def main(cfg, data_path, results_path):
    results_file_name = "trial_results_finetuning.csv"

    NEURAL_CONDITION_MAP = {
        "Baseline-ER": [0],
        "Baseline-PER": [0],
        "Reward Augmentation": [1],
        "Prioritization": [2],
        "Q-Augmentation": [3],
        "All-ER": [0, 1, 3],
        "All-PER": [0, 1, 2, 3],

    }

    # testing: single condition, binary granularity, no ablation sweeps
    NEURAL_CONDITIONS_PER = [
        "Baseline-PER",
        "Prioritization",
        "Q-Augmentation",
        "Reward Augmentation",
        "All-PER",
    ]

    NEURAL_CONDITIONS_ER = [
        "Baseline-ER",
        "Q-Augmentation",
        "Reward Augmentation",
        "All-ER",
    ]

    SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]


    if cfg["rl"]["buffer_type"] == "PER":
        NEURAL_CONDITIONS = NEURAL_CONDITIONS_PER
    else:
        NEURAL_CONDITIONS = NEURAL_CONDITIONS_ER

    # Ablation sweeps across the full condition grid
    for seed, condition in itertools.product(SEEDS, NEURAL_CONDITIONS):

        cfg["experiment"]["seed"] = seed
        cfg["experiment"]["condition"] = condition
        cfg["experiment_list"] = NEURAL_CONDITION_MAP[condition]

        print(cfg)
        run(cfg, run_name=make_run_name(cfg), DATA_PATH=data_path, RESULTS_PATH=results_path, RESULTS_FILE_NAME=results_file_name, verbose = cfg["experiment"]["verbose"])

if __name__ == "__main__":
    #add config file as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config", "-c", type=str, help="Path to the config file")
    parser.add_argument("domain", "-d", type=str, help="Domain")
    parser.add_argument("data_path", "-dp", type=str, help="Data path", default="/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/")
    parser.add_argument("results_path", "-rp", type=str, help="Results path", default="/Users/juliasantaniello/Desktop/OfflineNeuroloop/")

    args = parser.parse_args()
    domain = args.domain
    if domain == "flappy":
        cfg = f"configs/domains/flappy/{args.config}.yaml"

    main(cfg)
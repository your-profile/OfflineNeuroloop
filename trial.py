"""
Script for running trials given the config file and data paths.
"""

from src.neural.loader import DataLoader
from src.neural.preprocessing import DatasetProcessor
from src.models.model_training import ModelTrainer
from src.seed_utils import set_global_seed
from sklearn.model_selection import train_test_split
import src.utils as utils
import pandas as pd
import os
import csv

def run(cfg, run_name = "test", verbose = False, DATA_PATH = '.', RESULTS_PATH='.', RESULTS_FILE_NAME = 'trial_results.csv', inverse = False):

    trial_seed = int(cfg["experiment"]["random_state"])
    set_global_seed(trial_seed)

    env = utils.load_domain(cfg["experiment"]["domain"], cfg["rl"]["steps"])
    agent = utils.load_agent(
        cfg["rl"]["algorithm"],
        cfg["rl"]["buffer_type"],
        filename=RESULTS_PATH,
        space=(cfg["rl"]["observation_space"], cfg["rl"]["action_space"]),
        pretrained_success_rate=cfg["experiment"]["pretrained_success_rate"],
        seed=trial_seed,
        verbose=verbose,
    )
    means = utils.get_expected_reward(cfg["experiment"]["domain"].lower())

    if cfg["experiment"]["integration_type"] == "irl":
        print("Inverse RL")
        from src.training_loop_surrogate import train, train_robot
    elif cfg["experiment"]["integration_type"] == "interleave":
        print("Interleave")
        from src.training_loop_interleaving import train, train_robot
    elif cfg["experiment"]["integration_type"] == "finetune":
        print("Finetune")
        from src.training_loop_finetuning import train, train_robot
    elif cfg["experiment"]["integration_type"] == "baseline":
        print("Baseline")
        from src.training_loop_baseline import train, train_robot
    else:
        raise ValueError(f"Invalid integration type: {cfg['experiment']['integration_type']}")

    if not os.path.exists(os.path.join(DATA_PATH, 'fNIRS/LabeledData/')):
        try:
            DATA_PATH = os.path.join(os.environ.get("HOME", ""),
                        '/Users/USERNAME/Desktop/fNIRSDataset/Experiment/ParticipantData/')
            assert os.path.exists(os.path.join(DATA_PATH, 'fNIRS/LabeledData/'))
        except AssertionError:
            print("Please store path to participant date in DATA_PATH")

    labeled_data_source_folder = os.path.join(DATA_PATH, 'fNIRS/LabeledData/')
    rl_taskstats_source_folder = os.path.join(DATA_PATH, 'TaskData/')
    filtered_data_source_folder = os.path.join(DATA_PATH, 'fNIRS/FilteredData/')

    # get conditions
    condition_list = utils.get_conditions(cfg["experiment"]["domain"], cfg["experiment"]["task"], verbose = verbose)
    
    # load neural and rl data
    loader = DataLoader(
        fnirs_data_source_path=filtered_data_source_folder,
        task_data_source_path=rl_taskstats_source_folder,
        labeled_data_source_path=labeled_data_source_folder,
        participant_list = cfg["experiment"]["participant_list"],
        conditions_list = condition_list,
    )

    fnirs_df = loader.load_fnirs()     
    task_df  = loader.load_task()
    labels_df = loader.load_labels()

    # align timestamps
    processor = DatasetProcessor(verbose = verbose)
    aligned_df, fnirs_channels = processor.align_streams(
        fnirs_df,
        task_df,
        labels_df,
        resample_rate_hz = cfg["neural"]["fnirs_rate_hz"],
    )

    shifted_df = processor.shift_labels_for_delay(aligned_df, delay_s = cfg["neural"]["temporal_shift"], verbose = verbose)

    # Split unique episodes: train MLP on one random subset; agent learns from the rest.
    episode_cols = ["participantKey", "episode"]
    episode_ids = (
        task_df[episode_cols]
        .assign(
            participantKey=lambda d: d["participantKey"].astype(str),
            episode=lambda d: pd.to_numeric(d["episode"], errors="coerce").astype("Int64"),
        )
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    mlp_episode_fraction = float(cfg.get("experiment", {}).get("mlp_episode_fraction", 0.5))
    mlp_episodes, agent_episodes = train_test_split(episode_ids, train_size=mlp_episode_fraction,random_state=trial_seed, shuffle=True)
    print(f"Episode split: {len(mlp_episodes)} for MLP "f"({mlp_episode_fraction:.0%}), {len(agent_episodes)} for agent")

    shifted_mlp = shifted_df.copy()
    shifted_mlp["participantKey"] = shifted_mlp["participantKey"].astype(str)
    shifted_mlp["episode"] = pd.to_numeric(shifted_mlp["episode"], errors="coerce").astype("Int64")
    mlp_keys = set(map(tuple, mlp_episodes.to_numpy()))
    
    shifted_mlp = shifted_mlp[
        shifted_mlp[episode_cols].apply(tuple, axis=1).isin(mlp_keys)
    ]

    X, y = processor.build_balanced_dataset(
        shifted_mlp,
        fnirs_channels = fnirs_channels,
        label_col = "label_shifted",
        granularity = cfg["experiment"]["model_granularity"],
        window_duration_s = cfg["neural"]["window_size_s"],
        resample_rate_hz = cfg["neural"]["fnirs_rate_hz"],
        random_state = trial_seed,
    )

    # Agent offline learning uses only held-out episodes (not used to fit the MLP).
    task_df = task_df.copy()
    task_df["participantKey"] = task_df["participantKey"].astype(str)
    task_df["episode"] = pd.to_numeric(task_df["episode"], errors="coerce").astype("Int64")
    task_df = task_df[
        ~task_df[episode_cols].apply(tuple, axis=1).isin(mlp_keys)
    ].reset_index(drop=True)

    modelTrainer = ModelTrainer(cfg=cfg["mlp"], seed=trial_seed, verbose=verbose)
    classifier, report = modelTrainer.train_classifier(
        X, y, granularity=cfg["experiment"]["model_granularity"], random_state=trial_seed
    )
    
    print("MLP Report: \n", report)
    print(cfg)

    if cfg["experiment"]["domain"][0].lower() == "l" or cfg["experiment"]["domain"][0].lower() == "f":        
        results_dictionary = train(env=env, 
            processor = processor,
            task_df = task_df, 
            agent = agent, 
            flags = cfg["experiment"]["experiment_list"], 
            granularity = cfg["experiment"]["model_granularity"],
            means = means,
            episodes_num = cfg["rl"]["n_episodes"],
            clf = classifier, 
            ml = modelTrainer, 
            fnirs_channel_names=fnirs_channels, 
            smoothing_window_size =  cfg["neural"]["smoothing_window_size"], 
            window_duration_s = cfg["neural"]["window_size_s"], 
            shift = cfg["neural"]["temporal_shift"], 
            fnirs_rate_hz = cfg["neural"]["fnirs_rate_hz"],
            beta = cfg["neural"]["beta"],
            noise = cfg["mlp"]["model_noise"],
            seed=trial_seed,
            buffer_type = cfg["rl"]["buffer_type"],
            steps = cfg["rl"]["steps"], 
            save_results = True,
            save_to_csv = False,
            verbose = verbose,
            finetune_threshold = cfg["experiment"]["finetune_threshold"],
            success_save_threshold = 0.0,
            save_agent = False,
            eval_update = cfg["experiment"]["eval_update"],
        )
    else:
        results_dictionary = train_robot(env=env, 
            processor = processor,
            task_df = task_df, 
            agent = agent, 
            flags = cfg["experiment"]["experiment_list"], 
            granularity = cfg["experiment"]["model_granularity"],
            means = means,
            episodes_num = cfg["rl"]["n_episodes"],
            clf = classifier, 
            ml = modelTrainer, 
            fnirs_channel_names=fnirs_channels, 
            smoothing_window_size =  cfg["neural"]["smoothing_window_size"], 
            window_duration_s = cfg["neural"]["window_size_s"], 
            shift = cfg["neural"]["temporal_shift"], 
            fnirs_rate_hz = cfg["neural"]["fnirs_rate_hz"],
            beta = cfg["neural"]["beta"],
            noise = cfg["mlp"]["model_noise"],
            seed=trial_seed,
            buffer_type = cfg["rl"]["buffer_type"],
            steps = cfg["rl"]["steps"], 
            save_results = True,
            save_to_csv = False,
            verbose = verbose,
            finetune_threshold = cfg["experiment"]["finetune_threshold"],
            success_save_threshold = 0.5,
            save_agent = False,
            eval_update = cfg["experiment"]["eval_update"],
        )

    trial_dict = {}
    trial_dict = {"parameters": cfg, "results": results_dictionary}

    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_trial = flatten_dict(trial_dict)

    csv_path = os.path.join(RESULTS_PATH, 'src/results/', RESULTS_FILE_NAME)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=flat_trial.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(flat_trial)
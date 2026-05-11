from src.neural.loader import DataLoader
from src.neural.preprocessing import DatasetProcessor
from src.models.model_training import ModelTrainer
from src.models.model_neural_predictor import FnirsFeaturePredictor
from src.training_loop import train, train_robot
# from src.training_inverse import train_inverse, train_inverse_robot
import src.utils as utils
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Optional


def train_fnirs_predictor(
    states: np.ndarray,
    fnirs_targets: np.ndarray,
    state_dim: int,
    hidden_sizes=(256, 256),
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 256,
    test_size: float = 0.1,
    random_state: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> FnirsFeaturePredictor:
    """
    Supervised MSE: observation -> raw fNIRS channel vector.

    ``states`` and ``fnirs_targets`` should use the same class-balanced index set as
    the sklearn decoder's ``X`` and ``y`` from ``build_balanced_dataset`` (e.g.
    ``S_full[sel]``, ``Y_fn_full[sel]``).
    """
    S = np.asarray(states, dtype=np.float32)
    Y = np.asarray(fnirs_targets, dtype=np.float32)
    if S.shape[0] != Y.shape[0]:
        raise ValueError(
            f"states and fnirs_targets must have the same length; got {S.shape[0]} vs {Y.shape[0]}"
        )
    if S.shape[1] != state_dim:
        raise ValueError(f"states second dim {S.shape[1]} != state_dim {state_dim}")
    mask = np.isfinite(S).all(axis=1) & np.isfinite(Y).all(axis=1)
    S, Y = S[mask], Y[mask]
    if len(S) < 10:
        raise ValueError(f"Too few rows for fNIRS predictor training: {len(S)}")

    S_tr, S_va, Y_tr, Y_va = train_test_split(
        S, Y, test_size=test_size, random_state=random_state
    )

    dev = device or torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    fnirs_dim = Y.shape[1]
    model = FnirsFeaturePredictor(
        state_dim=state_dim,
        fnirs_dim=fnirs_dim,
        hidden_sizes=hidden_sizes,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tr_ds = TensorDataset(
        torch.from_numpy(S_tr).float(), torch.from_numpy(Y_tr).float()
    )
    tr_loader = TorchDataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    S_va_t = torch.from_numpy(S_va).float().to(dev)
    Y_va_t = torch.from_numpy(Y_va).float().to(dev)

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(S_va_t), Y_va_t).item()
        if verbose and (ep == 0 or ep == epochs - 1 or (ep + 1) % 20 == 0):
            print(
                f"fNIRS predictor epoch {ep + 1}/{epochs} — val MSE: {va_loss:.6f}"
            )

    return model


def run(cfg, run_name = "test", verbose = False, DATA_PATH = '.', RESULTS_PATH='.', RESULTS_FILE_NAME = 'trial_results.csv', inverse = False):
    if cfg["experiment"]["domain"][0].lower() == "l" or cfg["experiment"]["domain"][0].lower() == "f":
        run_lunar(cfg, run_name, verbose, DATA_PATH, RESULTS_PATH, RESULTS_FILE_NAME, inverse)
    elif cfg["experiment"]["domain"][0].lower() == "r":
        run_robot(cfg, run_name, verbose, DATA_PATH, RESULTS_PATH, RESULTS_FILE_NAME)
    else:
        raise ValueError(f"Invalid domain: {cfg['experiment']['domain']}")

def run_lunar(cfg, run_name = "test", verbose = False, DATA_PATH = '.', RESULTS_PATH='.', RESULTS_FILE_NAME = 'trial_results.csv', inverse = False):

    env = utils.load_domain(cfg["experiment"]["domain"], cfg["rl"]["steps"])
    agent = utils.load_agent(cfg["rl"]["algorithm"], cfg["rl"]["buffer_type"], filename=filename, space = (cfg["rl"]["observation_space"], cfg["rl"]["action_space"]), pretrained_success_rate = cfg["experiment"]["pretrained_success_rate"], verbose = verbose)
    
    #TODO: Make anonymous/internal
    if not os.path.exists(os.path.join(DATA_PATH, 'fNIRS/LabeledData/')):
        try:
            DATA_PATH = os.path.join(os.environ.get("HOME", ""),
                        '/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/')
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
    # likelihood_of_events = processor.compute_poisson_likelihood(cfg["experiment"]["model_granularity"])

    X, y, sel = processor.build_balanced_dataset(
        shifted_df,
        fnirs_channels = fnirs_channels,
        label_col = "label_shifted",
        granularity = cfg["experiment"]["model_granularity"],
        window_duration_s = cfg["neural"]["window_size_s"],
        resample_rate_hz = cfg["neural"]["fnirs_rate_hz"],
        random_state = cfg["experiment"]["random_state"],
        return_indices=True,
    )

    modelTrainer = ModelTrainer(cfg = cfg["mlp"], seed = cfg["experiment"]["random_state"], verbose = verbose)
    classifier, report = modelTrainer.train_classifier(X, y, granularity = cfg["experiment"]["model_granularity"], random_state =  cfg["experiment"]["random_state"])
    
    print("MLP Report: \n", report)

    pred_cfg = cfg.get("predictor", {})
    state_dim = int(cfg["rl"]["observation_space"])
    S_full, Y_fn_full = processor.build_supervised_state_raw_fnirs(
        shifted_df,
        fnirs_channels=fnirs_channels,
        label_col="label_shifted",
        window_duration_s=cfg["neural"]["window_size_s"],
        resample_rate_hz=cfg["neural"]["fnirs_rate_hz"],
        state_dim=state_dim,
    )
    if S_full.shape[0] == 0 or int(np.max(sel)) >= S_full.shape[0]:
        raise ValueError(
            "fNIRS predictor: supervised state/fNIRS rows do not align with balance indices."
        )
    S_bal = S_full[sel]
    Y_fn_bal = Y_fn_full[sel]
    if X.shape[0] != S_bal.shape[0] or X.shape[0] != Y_fn_bal.shape[0]:
        raise ValueError(
            f"Decoder X ({X.shape[0]}) vs predictor batches ({S_bal.shape[0]}, {Y_fn_bal.shape[0]})"
        )

    fnirs_predictor = train_fnirs_predictor(
        S_bal,
        Y_fn_bal,
        state_dim=state_dim,
        hidden_sizes=tuple(pred_cfg.get("hidden_sizes", [256, 256])),
        epochs=int(pred_cfg.get("epochs", 80)),
        lr=float(pred_cfg.get("lr", 1e-3)),
        batch_size=int(pred_cfg.get("batch_size", 256)),
        test_size=float(pred_cfg.get("test_size", 0.1)),
        random_state=int(cfg["experiment"]["random_state"]),
        verbose=verbose,
    )
    ckpt_dir = os.path.join(RESULTS_PATH, "src", "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    pred_path = os.path.join(ckpt_dir, f"fnirs_predictor_{run_name}.pt")
    fnirs_predictor.save(pred_path)
    if verbose:
        print(f"Saved fNIRS state predictor to {pred_path}")

    results_dictionary = train(env=env, 
            processor = processor,
            task_df = task_df, 
            agent = agent, 
            flags = cfg["experiment"]["experiment_list"], 
            granularity = cfg["experiment"]["model_granularity"],
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
            seed = cfg["experiment"]["random_state"],
            buffer_type = cfg["rl"]["buffer_type"],
            steps = cfg["rl"]["steps"], 
            save_results = True,
            save_to_csv = False,
            verbose = verbose,
            fnirs_predictor=None,
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

    csv_path = os.path.join(RESULTS_PATH,'src/results/', RESULTS_FILE_NAME)
    write_header = not os.path.exists(os.path.join(RESULTS_PATH,'src/results/', RESULTS_FILE_NAME))
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=flat_trial.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(flat_trial)
    


def run_robot(cfg, run_name = "test", verbose = False, DATA_PATH = '.', RESULTS_PATH='.', RESULTS_FILE_NAME = 'trial_results.csv'):

    steps = cfg["rl"].get("steps", 50)
    env = utils.make_fetch_env(max_episode_steps=steps, mujoco_version=4, verbose = verbose)
    agent = utils.load_ddpg_agent(env, cfg["rl"]["buffer_type"], pretrained_success_rate = cfg["experiment"]["pretrained_success_rate"], verbose = verbose)

    if not os.path.exists(os.path.join(DATA_PATH, 'fNIRS/LabeledData/')):
        try:
            DATA_PATH = os.path.join(os.environ.get("HOME", ""),
                        '/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/')
            assert os.path.exists(os.path.join(DATA_PATH, 'fNIRS/LabeledData/'))
        except AssertionError:
                    print("Please store path to participant date in DATA_PATH")

    labeled_data_source_folder = os.path.join(DATA_PATH, 'fNIRS/LabeledData/')
    rl_taskstats_source_folder = os.path.join(DATA_PATH, 'TaskData/')
    filtered_data_source_folder = os.path.join(DATA_PATH, 'fNIRS/FilteredData/')

    condition_list = utils.get_conditions(cfg["experiment"]["domain"], cfg["experiment"]["task"], verbose=verbose)

    loader = DataLoader(
        fnirs_data_source_path=filtered_data_source_folder,
        task_data_source_path=rl_taskstats_source_folder,
        labeled_data_source_path=labeled_data_source_folder,
        participant_list=cfg["experiment"]["participant_list"],
        conditions_list=condition_list,
    )

    fnirs_df = loader.load_fnirs()
    task_df = loader.load_task()
    labels_df = loader.load_labels()

    processor = DatasetProcessor(verbose = verbose)
    aligned_df, fnirs_channels = processor.align_streams(
        fnirs_df,
        task_df,
        labels_df,
        resample_rate_hz=cfg["neural"]["fnirs_rate_hz"],
    )

    shifted_df = processor.shift_labels_for_delay(
        aligned_df, delay_s=cfg["neural"]["temporal_shift"], verbose = verbose
    )

    X, y = processor.build_balanced_dataset(
        shifted_df,
        fnirs_channels=fnirs_channels,
        label_col="label_shifted",
        granularity=cfg["experiment"]["model_granularity"],
        window_duration_s=cfg["neural"]["window_size_s"],
        resample_rate_hz=cfg["neural"]["fnirs_rate_hz"],
        random_state=cfg["experiment"]["random_state"],
    )

    modelTrainer = ModelTrainer(cfg=cfg["mlp"], seed=cfg["experiment"]["random_state"], verbose = verbose)

    classifier, report = modelTrainer.train_classifier(X, y, granularity=cfg["experiment"]["model_granularity"], random_state=cfg["experiment"]["random_state"])

    print("MLP Report: \n", report)

    results_dictionary = train_robot(
        env=env,
        processor=processor,
        task_df=task_df,
        agent=agent,
        experiment_list=cfg["experiment"]["experiment_list"],
        granularity=cfg["experiment"]["model_granularity"],
        episodes_num=cfg["rl"]["n_episodes"],
        clf=classifier,
        ml=modelTrainer,
        fnirs_channel_names=fnirs_channels,
        smoothing_window_size=cfg["neural"]["smoothing_window_size"],
        window_duration_s=cfg["neural"]["window_size_s"],
        shift=cfg["neural"]["temporal_shift"],
        fnirs_rate_hz=cfg["neural"]["fnirs_rate_hz"],
        beta=cfg["neural"]["beta"],
        noise=cfg["mlp"]["model_noise"],
        seed = cfg["experiment"]["random_state"],
        buffer_type=cfg["rl"]["buffer_type"],
        steps=cfg["rl"]["steps"],
        save_results=True,
        save_to_csv=False,
        verbose=verbose
    )

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

    csv_path = os.path.join(RESULTS_PATH,'src/results/', RESULTS_FILE_NAME)
    write_header = not os.path.exists(os.path.join(RESULTS_PATH,'src/results/', RESULTS_FILE_NAME))
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=flat_trial.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(flat_trial)


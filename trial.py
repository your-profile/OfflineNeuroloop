from src.neural.loader import DataLoader
from src.neural.preprocessing import DatasetProcessor
from src.models.model_training import ModelTrainer
import src.utils as utils
import os
import csv

def run(cfg, run_name = "test", verbose = False, DATA_PATH = '.', RESULTS_PATH='.', RESULTS_FILE_NAME = 'trial_results.csv', inverse = False):

    env = utils.load_domain(cfg["experiment"]["domain"], cfg["rl"]["steps"])
    agent = utils.load_agent(cfg["rl"]["algorithm"], cfg["rl"]["buffer_type"], filename=RESULTS_PATH, space = (cfg["rl"]["observation_space"], cfg["rl"]["action_space"]), pretrained_success_rate = cfg["experiment"]["pretrained_success_rate"], verbose = verbose)
    
    if cfg["experiment"]["integration_type"] == "early":
        print("Early Integration")
        from src.training_loop_early import train, train_robot

    if cfg["experiment"]["integration_type"] == "interleaved":
        from src.training_loop import train, train_robot

    if cfg["experiment"]["integration_type"] == "finetune":
        from src.training_loop_finetuning import train, train_robot
        agent.lr = 1e-5

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

    X, y = processor.build_balanced_dataset(
        shifted_df,
        fnirs_channels = fnirs_channels,
        label_col = "label_shifted",
        granularity = cfg["experiment"]["model_granularity"],
        window_duration_s = cfg["neural"]["window_size_s"],
        resample_rate_hz = cfg["neural"]["fnirs_rate_hz"],
        random_state = cfg["experiment"]["random_state"],
    )

    modelTrainer = ModelTrainer(cfg = cfg["mlp"], seed = cfg["experiment"]["random_state"], verbose = verbose)
    classifier, report = modelTrainer.train_classifier(X, y, granularity = cfg["experiment"]["model_granularity"], random_state =  cfg["experiment"]["random_state"])
    
    print("MLP Report: \n", report)

    if cfg["experiment"]["domain"][0].lower() == "l" or cfg["experiment"]["domain"][0].lower() == "f":        
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
            verbose = verbose)
    else:
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
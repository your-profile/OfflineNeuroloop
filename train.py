from pandas._libs.tslibs.offsets import MonthEnd
from src.neural.loader import DataLoader
from src.neural.preprocessing import DatasetProcessor
from src.models.training import ModelTrainer
from src.training import train
import src.utils as utils

def run(cfg = None, run_name = "test"):

    # if cfg is None:

        # cfg = {"domain": "lunar_lander",
        #        "participant_list": [2, 3, 4],
        #        "condition_list": ["LW"],
        #        "experiment"
        #        "resample_rate": 10.0,
        #        "fnirs_channels": ["L_O_DSI", "L_D_DSI", "L_O_DSphi", "L_D_DSphi", "R_O_DSI", "R_D_DSI", "R_O_DSphi", "R_D_DSphi"],
        #        "label_shifted": "label_shifted",
        #        "window_size_s": 4.0,
        #        "fnirs_rate_hz": 5.2,
        #        "temporal_shift": 3.0,
        #        "buffer_type": "ER",
        #        "random_state": 42,
        # }

    env = utils.load_domain(cfg["experiment"]["domain"])
    agent = utils.load_agent(cfg["rl"]["algorithm"], cfg["rl"]["buffer_type"], space = (env.observation_space.shape[0], env.action_space.n))
    print(env.observation_space.shape[0], env.action_space.n)

    labeled_data_source_folder = "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/fNIRS/LabeledData/"
    rl_taskstats_source_folder = "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/TaskData/"
    filtered_data_source_folder = "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/fNIRS/FilteredData/"

    loader = DataLoader(
        fnirs_data_source_path=filtered_data_source_folder,
        task_data_source_path=rl_taskstats_source_folder,
        labeled_data_source_path=labeled_data_source_folder,
        participant_list = cfg["experiment"]["participant_list"],
        conditions_list = cfg["experiment"]["condition_list"],
    )

    fnirs_df = loader.load_fnirs()     
    task_df  = loader.load_task()
    labels_df = loader.load_labels()


    processor = DatasetProcessor()

    aligned_df, fnirs_channels = processor.align_streams(
        fnirs_df,
        task_df,
        labels_df,
        resample_rate_hz = cfg["neural"]["fnirs_rate_hz"],
    )

    shifted_df = processor.shift_labels_for_delay(aligned_df, delay_s = cfg["neural"]["temporal_shift"])

    X_bin, y_bin = processor.build_balanced_binary_dataset(
        shifted_df,
        fnirs_channels = fnirs_channels,
        label_col = "label_shifted",
        window_duration_s = cfg["neural"]["window_size_s"],
        resample_rate_hz = cfg["neural"]["fnirs_rate_hz"],
        random_state= cfg["experiment"]["random_state"],
    )

    # TODO: differentiate between binary, ternary and regressor models
    modelTrainer = ModelTrainer()
    classifier, report = modelTrainer.train_classifier(X_bin, y_bin, random_state =  cfg["experiment"]["random_state"],)
    print(report)
    
    results_dictionary, parameters_dictionary = train(env=env, 
            processor = processor,
            task_df = task_df, 
            agent = agent, 
            experiment_name = cfg["experiment"]["experiment_list"], 
            clf = classifier, 
            fnirs_channel_names=fnirs_channels, 
            window_duration_s = cfg["neural"]["window_size_s"], 
            shift = cfg["neural"]["temporal_shift"], 
            fnirs_rate_hz = cfg["neural"]["fnirs_rate_hz"],
            save_results = True,
            save_to_csv = False,
            verbose = False)
    
    print("Classifier Report: \n", report)


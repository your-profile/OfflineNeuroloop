"""
Script for running trials given the config file and data paths.

Decoder modes (experiment.decoder_mode):
  pooled          — one model on a random episode subset (legacy)
  single_subject  — one LDA/Ridge per participant; RL uses that subject's
                    holdout episodes with matched routing
  ensemble        — one model per participant; predictions soft/hard-voted
"""

from src.neural.loader import DataLoader
from src.neural.preprocessing import DatasetProcessor
from src.models.model_training import ModelTrainer
from src.models.decoder_bank import SubjectDecoderBank, normalize_pid
from src.seed_utils import set_global_seed
from sklearn.model_selection import train_test_split
import src.utils as utils
import pandas as pd
import os
import csv


def _episode_table(task_df: pd.DataFrame) -> pd.DataFrame:
    episode_cols = ["participantKey", "episode"]
    return (
        task_df[episode_cols]
        .assign(
            participantKey=lambda d: d["participantKey"].astype(str),
            episode=lambda d: pd.to_numeric(d["episode"], errors="coerce").astype("Int64"),
            pid=lambda d: d["participantKey"].map(normalize_pid),
        )
        .dropna(subset=["episode"])
        .drop_duplicates(subset=episode_cols)
        .reset_index(drop=True)
    )


def _split_pooled(episode_ids: pd.DataFrame, fraction: float, seed: int):
    decoder_eps, agent_eps = train_test_split(
        episode_ids, train_size=fraction, random_state=seed, shuffle=True
    )
    return decoder_eps, agent_eps


def _split_per_subject(episode_ids: pd.DataFrame, fraction: float, seed: int, task_df: pd.DataFrame | None = None):
    """Within each participant, hold out later episodes for RL (time-ordered).

    Episodes are sorted by start time so the decoder train block is contiguous
    and an embargo gap between train and holdout is meaningful (random episode
    interleaving makes ``embargo_s`` wipe almost all train windows).
    """
    decoder_parts, agent_parts = [], []
    for pid, grp in episode_ids.groupby("pid", sort=True):
        g = grp.copy()
        if task_df is not None and len(g) >= 2:
            starts = []
            for _, row in g.iterrows():
                m = (
                    (task_df["participantKey"].astype(str) == str(row["participantKey"]))
                    & (pd.to_numeric(task_df["episode"], errors="coerce") == int(row["episode"]))
                )
                t = pd.to_datetime(task_df.loc[m, "time"], utc=True)
                starts.append(t.min() if len(t) else pd.NaT)
            g = g.assign(_start=starts).sort_values("_start").drop(columns="_start")
        else:
            g = g.sample(frac=1.0, random_state=seed) if len(g) else g

        if len(g) < 2:
            if fraction >= 1.0:
                decoder_parts.append(g)
            else:
                agent_parts.append(g)
            continue
        n_dec = max(1, int(round(len(g) * fraction)))
        n_dec = min(n_dec, len(g) - 1)
        decoder_parts.append(g.iloc[:n_dec])
        agent_parts.append(g.iloc[n_dec:])
    decoder_eps = pd.concat(decoder_parts, ignore_index=True) if decoder_parts else episode_ids.iloc[0:0]
    agent_eps = pd.concat(agent_parts, ignore_index=True) if agent_parts else episode_ids.iloc[0:0]
    return decoder_eps, agent_eps


def _keys(eps: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, eps[["participantKey", "episode"]].to_numpy()))


def _train_one(
    processor,
    shifted_df,
    fnirs_channels,
    decoder_keys,
    cfg,
    mlp_cfg,
    seed,
    verbose,
):
    episode_cols = ["participantKey", "episode"]
    sub = shifted_df.copy()
    sub["participantKey"] = sub["participantKey"].astype(str)
    sub["episode"] = pd.to_numeric(sub["episode"], errors="coerce").astype("Int64")
    sub = sub[sub[episode_cols].apply(tuple, axis=1).isin(decoder_keys)]
    if len(sub) < 50:
        return None, None, "too few aligned rows"

    try:
        X, y = processor.build_balanced_dataset(
            sub,
            fnirs_channels=fnirs_channels,
            label_col="label_shifted",
            granularity=cfg["experiment"]["model_granularity"],
            window_duration_s=cfg["neural"]["window_size_s"],
            resample_rate_hz=cfg["neural"]["fnirs_rate_hz"],
            step_size_s=cfg["neural"].get("step_size_s", 1.0),
            random_state=seed,
        )
    except Exception as exc:
        return None, None, str(exc)

    trainer = ModelTrainer(cfg=mlp_cfg, seed=seed, verbose=verbose)
    clf, report = trainer.train_classifier(
        X, y, granularity=cfg["experiment"]["model_granularity"], random_state=seed
    )
    return clf, report, trainer


def run(cfg, run_name="test", verbose=False, DATA_PATH=".", RESULTS_PATH=".", RESULTS_FILE_NAME="trial_results.csv", inverse=False):

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

    if not os.path.exists(os.path.join(DATA_PATH, "fNIRS/LabeledData/")):
        try:
            DATA_PATH = os.path.join(
                os.environ.get("HOME", ""),
                "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/",
            )
            assert os.path.exists(os.path.join(DATA_PATH, "fNIRS/LabeledData/"))
        except AssertionError:
            print("Please store path to participant date in DATA_PATH")

    labeled_data_source_folder = os.path.join(DATA_PATH, "fNIRS/LabeledData/")
    rl_taskstats_source_folder = os.path.join(DATA_PATH, "TaskData/")
    filtered_data_source_folder = os.path.join(DATA_PATH, "fNIRS/FilteredData/")

    condition_list = utils.get_conditions(
        cfg["experiment"]["domain"], cfg["experiment"]["task"], verbose=verbose
    )

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

    processor = DatasetProcessor(verbose=verbose)
    aligned_df, fnirs_channels = processor.align_streams(
        fnirs_df,
        task_df,
        labels_df,
        resample_rate_hz=cfg["neural"]["fnirs_rate_hz"],
    )
    shifted_df = processor.shift_labels_for_delay(
        aligned_df, delay_s=cfg["neural"]["temporal_shift"], verbose=verbose
    )
    shifted_df = shifted_df.copy()
    shifted_df["participantKey"] = shifted_df["participantKey"].astype(str)
    shifted_df["episode"] = pd.to_numeric(shifted_df["episode"], errors="coerce").astype("Int64")

    decoder_mode = str(cfg["experiment"].get("decoder_mode", "pooled")).lower().strip()
    decoder_type = str(cfg.get("mlp", {}).get("type", "lda")).lower()
    decoder_fraction = float(
        cfg.get("experiment", {}).get(
            "decoder_episode_fraction",
            cfg.get("experiment", {}).get("mlp_episode_fraction", 0.5),
        )
    )

    episode_ids = _episode_table(task_df)
    if decoder_mode in ("single_subject", "ensemble", "per_subject", "matched"):
        decoder_eps, agent_eps = _split_per_subject(
            episode_ids, decoder_fraction, trial_seed, task_df=task_df
        )
    else:
        decoder_eps, agent_eps = _split_pooled(episode_ids, decoder_fraction, trial_seed)

    decoder_keys = _keys(decoder_eps)
    agent_keys = _keys(agent_eps)
    print(
        f"Decoder mode={decoder_mode} type={decoder_type} | "
        f"{len(decoder_eps)} decoder episodes / {len(agent_eps)} RL holdout episodes "
        f"({decoder_fraction:.0%} decoder)"
    )

    mlp_cfg = dict(cfg.get("mlp", {}))
    mlp_cfg.setdefault("window_s", cfg["neural"]["window_size_s"])
    mlp_cfg.setdefault("type", "lda")
    granularity = cfg["experiment"]["model_granularity"]
    flags = cfg["experiment"].get("experiment_list")
    if flags is None:
        # default: reward + prioritization + Q (All-PER) when running a yaml directly
        flags = [1, 2, 3]
        cfg["experiment"]["experiment_list"] = flags

    modelTrainer = ModelTrainer(cfg=mlp_cfg, seed=trial_seed, verbose=verbose)

    processed_dir = os.path.join(DATA_PATH, "fNIRS/FilteredData/")
    labeled_dir = os.path.join(DATA_PATH, "fNIRS/LabeledData/")
    use_eval_decoder = decoder_type in ("lda", "shrinkage_lda", "ridge_lda", "ridge")

    if decoder_mode in ("single_subject", "ensemble", "per_subject", "matched"):
        models, reports = {}, {}
        for pid, grp in decoder_eps.groupby("pid", sort=True):
            agent_grp = agent_eps[agent_eps["pid"] == pid]
            if use_eval_decoder:
                from src.models.eval_compatible import train_subject_eval_decoder

                clf_i, report_i = train_subject_eval_decoder(
                    pid,
                    condition_list,
                    processed_dir,
                    labeled_dir,
                    task_df,
                    decoder_keys=_keys(grp),
                    agent_keys=_keys(agent_grp) if len(agent_grp) else set(),
                    granularity=granularity,
                    window_s=float(cfg["neural"]["window_size_s"]),
                    step_s=float(cfg["neural"].get("step_size_s", 1.0)),
                    embargo_s=float(mlp_cfg.get("embargo_s", 4.0)),
                    rate_hz=float(cfg["neural"]["fnirs_rate_hz"]),
                    seed=trial_seed,
                )
                if clf_i is None:
                    err = report_i.get("error", report_i)
                    detail = report_i.get("detail")
                    msg = f"  skip pid={pid}: {err}"
                    if detail:
                        msg += f" | {detail}"
                    print(msg)
                    continue
                metric = report_i.get("holdout_metric")
                mname = report_i.get("metric_name", "metric")
                print(
                    f"  fitted pid={pid} eval-LDA | train={report_i['n_train']} "
                    f"holdout={report_i['n_holdout']} embargo_dropped={report_i['n_embargo_dropped']} "
                    f"gap={report_i['gap_s']}s | holdout {mname}={metric}"
                )
            else:
                clf_i, report_i, err = _train_one(
                    processor,
                    shifted_df,
                    fnirs_channels,
                    _keys(grp),
                    cfg,
                    mlp_cfg,
                    trial_seed,
                    verbose,
                )
                if clf_i is None:
                    print(f"  skip pid={pid}: {err}")
                    continue
                print(f"  fitted pid={pid} on {len(grp)} episodes (legacy features)")

            models[pid] = clf_i
            reports[pid] = report_i

        if not models:
            from pathlib import Path

            proc = Path(processed_dir)
            lab = Path(labeled_dir)
            sample = sorted(proc.glob("*processed*.csv"))[:5] if proc.is_dir() else []
            sample_lab = sorted(lab.glob("*LabeledData*.csv"))[:5] if lab.is_dir() else []
            raise RuntimeError(
                "No per-subject decoders could be trained. "
                "LDA needs CSV files:\n"
                f"  {processed_dir}/{{pid}}_processed_{{COND}}.csv\n"
                f"  {labeled_dir}/{{pid}}_{{COND}}_LabeledData.csv\n"
                f"processed_dir exists={proc.is_dir()} "
                f"({len(list(proc.glob('*.csv'))) if proc.is_dir() else 0} csvs); "
                f"labeled_dir exists={lab.is_dir()} "
                f"({len(list(lab.glob('*.csv'))) if lab.is_dir() else 0} csvs).\n"
                f"sample processed: {[p.name for p in sample]}\n"
                f"sample labeled: {[p.name for p in sample_lab]}\n"
                "TaskData pickles alone are enough for episode counts but not for the LDA decoder. "
                "Check NEUROLOOP_DATA_ROOT / paths.data_path on the cluster."
            )

        bank_mode = "ensemble" if decoder_mode == "ensemble" else "single_subject"
        classifier = SubjectDecoderBank(
            models, mode=bank_mode, granularity=granularity, reports=reports
        )
        print(f"Decoder bank: {classifier}")
        for pid, rep in reports.items():
            print(f"--- pid {pid} ---\n{rep}")
    else:
        clf, report, err = _train_one(
            processor,
            shifted_df,
            fnirs_channels,
            decoder_keys,
            cfg,
            mlp_cfg,
            trial_seed,
            verbose,
        )
        if clf is None:
            raise RuntimeError(f"Pooled decoder failed: {err}")
        classifier = clf
        print(f"Decoder ({decoder_type}) report:\n", report)

    # RL uses only held-out episodes (never used to fit the decoder).
    task_df = task_df.copy()
    task_df["participantKey"] = task_df["participantKey"].astype(str)
    task_df["episode"] = pd.to_numeric(task_df["episode"], errors="coerce").astype("Int64")
    # Drop holdouts whose subject has no fitted decoder (single_subject/ensemble).
    if decoder_mode in ("single_subject", "ensemble", "per_subject", "matched"):
        ok_pids = set(classifier.models)
        task_df = task_df[
            task_df["participantKey"].map(normalize_pid).isin(ok_pids)
            & task_df[["participantKey", "episode"]].apply(tuple, axis=1).isin(agent_keys)
        ].reset_index(drop=True)
    else:
        task_df = task_df[
            task_df[["participantKey", "episode"]].apply(tuple, axis=1).isin(agent_keys)
        ].reset_index(drop=True)

    print(f"RL task rows after holdout filter: {len(task_df)}")
    print(cfg)

    train_kwargs = dict(
        env=env,
        processor=processor,
        task_df=task_df,
        agent=agent,
        flags=cfg["experiment"]["experiment_list"],
        granularity=granularity,
        means=means,
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
        seed=trial_seed,
        buffer_type=cfg["rl"]["buffer_type"],
        steps=cfg["rl"]["steps"],
        save_results=True,
        save_to_csv=False,
        verbose=verbose,
        finetune_threshold=cfg["experiment"]["finetune_threshold"],
        save_agent=False,
        eval_update=cfg["experiment"]["eval_update"],
    )

    if cfg["experiment"]["domain"][0].lower() in ("l", "f"):
        results_dictionary = train(**train_kwargs, success_save_threshold=0.0)
    else:
        results_dictionary = train_robot(**train_kwargs, success_save_threshold=0.5)

    trial_dict = {"parameters": cfg, "results": results_dictionary}

    def flatten_dict(d, parent_key="", sep="_"):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_trial = flatten_dict(trial_dict)

    csv_path = os.path.join(RESULTS_PATH, "src/results/", RESULTS_FILE_NAME)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_trial.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(flat_trial)

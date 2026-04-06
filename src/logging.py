import mlflow, yaml, hashlib, json

def log_run(config: dict):
    exp = config["experiment"]
    neural = config["neural"]
    mlp = config["mlp"]
    rl = config["rl"]

    mlflow.log_params({
        # Experimental factors
        "exp.domain":            exp["domain"],
        "exp.task":              exp["task"],
        "exp.neural_condition":  exp["neural_condition"],
        "exp.granularity":       exp["model_granularity"],

        # Ablation parameters
        "abl.model_noise":       neural["model_noise"],
        "abl.smoothing_method":  neural["smoothing_method"],
        "abl.credit_assignment": neural["credit_assignment"],
        "abl.smoothing_window_size":  neural["smoothing_window_size"],
        "abl.temporal_shift":     neural["temporal_shift"],        

        # Model + RL
        "mlp.hidden_layers":     str(mlp["hidden_layer_sizes"]),
        "mlp.lr":                mlp["learning_rate_init"],
        "rl.window_size":        rl["window_size"],
        "rl.temporal_shift":     rl["temporal_shift"],
        "rl.n_episodes":         rl["n_episodes"],
        "rl.window_size_s":      rl["window_size_s"],
        

        "config_hash": hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:8],
    })

    mlflow.set_tags({
        "domain":           exp["domain"],
        "task":             exp["task"],
        "neural_condition": exp["neural_condition"],
        "granularity":      exp["model_granularity"],
        "is_ablation":      any([
            neural["model_noise"] != 0.0,
            neural["smoothing_method"] != "none",
            neural["credit_assignment"] != "window_based",
            neural["temporal_shift"] != 0.0,
        ]),
    })
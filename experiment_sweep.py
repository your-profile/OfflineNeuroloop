"""Shared experiment grid: build configs, manifest, and run names."""
from __future__ import annotations

import copy
import csv
import itertools
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent

NEURAL_CONDITION_MAP = {
    "Baseline-ER": [0],
    "Baseline-PER": [0],
    "Reward Augmentation": [1],
    "Prioritization": [2],
    "Q-Augmentation": [3],
    "All-ER": [0, 1, 3],
    "All-PER": [0, 1, 2, 3],
}

INTEGRATION_RESULTS_SUFFIX = {
    "finetune": "finetuning",
    "interleaved": "interleaving",
    "pretrain": "pretraining",
}


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_nested(cfg: dict, keys: list[str], val: Any) -> None:
    cfg[keys[0]][keys[1]] = val


def domain_yaml_path(domain_key: str) -> Path:
    return REPO_ROOT / "configs" / "domains" / f"{domain_key.lower()}.yaml"


def make_run_name(cfg: dict) -> str:
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


def apply_profile(sweep: dict, profile: str | None) -> dict:
    if not profile:
        return sweep
    profiles = sweep.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(profiles)}")
    merged = copy.deepcopy(sweep)
    for key, val in profiles[profile].items():
        merged[key] = val
    return merged


def build_cfg(
    *,
    integration: str,
    condition: str,
    seed: int,
    granularity: str,
    task: str,
    ablation: dict,
    ablation_val: Any,
    base_config: Path,
    domain_key: str | None = None,
    domain_config: Path | None = None,
    n_episodes_override: int | None = None,
) -> dict:
    if domain_config is not None:
        cfg = copy.deepcopy(load_yaml(domain_config))
        domain_cfg = cfg
    else:
        if domain_key is None:
            raise ValueError("Either domain_config or domain_key is required")
        base = load_yaml(base_config)
        domain_cfg = load_yaml(domain_yaml_path(domain_key))
        cfg = copy.deepcopy(base)
        if base.get("rl", {}).get("n_episodes") == "test":
            domain_cfg.setdefault("rl", {})["n_episodes"] = 10

    cfg.setdefault("experiment", {})
    cfg.setdefault("neural", {})
    cfg.setdefault("mlp", {})
    cfg.setdefault("rl", {})

    if domain_config is None:
        cfg["experiment"].update(
            {
                "domain": domain_cfg["experiment"]["domain"],
                "task": task,
                "condition": condition,
                "experiment_list": NEURAL_CONDITION_MAP[condition],
                "model_granularity": granularity,
                "random_state": seed,
                "pretrained_success_rate": domain_cfg["experiment"]["pretrained_success_rate"],
            }
        )
        cfg["mlp"].update(
            {
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
            }
        )
        cfg["rl"].update(
            {
                "n_episodes": domain_cfg["rl"]["n_episodes"],
                "algorithm": domain_cfg["rl"]["algorithm"],
                "steps": domain_cfg["rl"]["steps"],
                "action_space": domain_cfg["rl"]["action_space"],
                "observation_space": domain_cfg["rl"]["observation_space"],
            }
        )
        if "buffer_type" in cfg["rl"]:
            pass
        elif "buffer_type" in domain_cfg.get("rl", {}):
            cfg["rl"]["buffer_type"] = domain_cfg["rl"]["buffer_type"]
    else:
        cfg["experiment"].update(
            {
                "task": task,
                "condition": condition,
                "experiment_list": NEURAL_CONDITION_MAP[condition],
                "model_granularity": granularity,
                "random_state": seed,
            }
        )

    cfg["experiment"]["integration_type"] = integration

    if condition == "Baseline" and (
        (ablation["key"][1] == "model_noise" and ablation_val != 0.0)
        or (ablation["key"][1] == "temporal_shift" and ablation_val != 0.0)
    ):
        return None

    set_nested(cfg, ablation["key"], ablation_val)

    if n_episodes_override is not None:
        cfg["rl"]["n_episodes"] = n_episodes_override

    return cfg


def iter_trial_specs(sweep: dict) -> list[dict]:
    """Expand sweep config into a list of trial specification dicts."""
    base_config = REPO_ROOT / sweep.get("base_config", "configs/test.yaml")
    paths = sweep.get("paths", {})
    data_path = paths.get("data_path", "")
    results_path = paths.get("results_path", str(REPO_ROOT))
    domains = sweep.get("domains", [])
    integrations = sweep["integrations"]
    seeds = sweep["seeds"]
    conditions = sweep["conditions"]
    granularities = sweep["granularities"]
    ablations = sweep.get("ablations", [{"key": ["experiment", "eval_success_threshold"], "vals": [0.0]}])
    n_episodes_override = sweep.get("n_episodes_override")

    specs: list[dict] = []
    trial_id = 0

    for integration in integrations:
        for ablation in ablations:
            for seed in seeds:
                for granularity in granularities:
                    for condition in conditions:
                        for domain_entry in domains:
                            domain_key = domain_entry.get("domain_key")
                            domain_config = domain_entry.get("domain_config")
                            tasks = domain_entry.get("tasks", ["Pooled"])
                            if domain_config:
                                domain_config = REPO_ROOT / domain_config
                            for task in tasks:
                                for val in ablation["vals"]:
                                    trial_id += 1
                                    specs.append(
                                        {
                                            "trial_id": trial_id,
                                            "integration": integration,
                                            "domain_key": domain_key or "",
                                            "domain_config": str(domain_config) if domain_config else "",
                                            "task": task,
                                            "condition": condition,
                                            "seed": seed,
                                            "granularity": granularity,
                                            "ablation_key": ".".join(ablation["key"]),
                                            "ablation_val": val,
                                            "base_config": str(base_config),
                                            "n_episodes_override": n_episodes_override or "",
                                            "data_path": data_path,
                                            "results_path": results_path,
                                        }
                                    )
    return specs


def should_skip_spec(spec: dict) -> bool:
    ablation_key = spec["ablation_key"].split(".")
    ablation = {"key": ablation_key}
    cfg = build_cfg(
        integration=spec["integration"],
        condition=spec["condition"],
        seed=int(spec["seed"]),
        granularity=spec["granularity"],
        task=spec["task"],
        ablation=ablation,
        ablation_val=_parse_ablation_val(spec["ablation_val"]),
        base_config=Path(spec["base_config"]),
        domain_key=spec["domain_key"] or None,
        domain_config=Path(spec["domain_config"]) if spec["domain_config"] else None,
        n_episodes_override=_optional_int(spec.get("n_episodes_override")),
    )
    return cfg is None


def _parse_ablation_val(raw: Any) -> Any:
    if raw == "" or raw is None:
        return 0.0
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    return raw


def _optional_int(raw: Any) -> int | None:
    if raw in ("", None):
        return None
    return int(raw)


def cfg_from_spec(spec: dict) -> dict:
    ablation_key = spec["ablation_key"].split(".")
    return build_cfg(
        integration=spec["integration"],
        condition=spec["condition"],
        seed=int(spec["seed"]),
        granularity=spec["granularity"],
        task=spec["task"],
        ablation={"key": ablation_key},
        ablation_val=_parse_ablation_val(spec["ablation_val"]),
        base_config=Path(spec["base_config"]),
        domain_key=spec["domain_key"] or None,
        domain_config=Path(spec["domain_config"]) if spec["domain_config"] else None,
        n_episodes_override=_optional_int(spec.get("n_episodes_override")),
    )


def trial_results_filename(trial_id: int, integration: str) -> str:
    suffix = INTEGRATION_RESULTS_SUFFIX.get(integration, integration)
    return f"runs/trial_{trial_id:05d}_{suffix}.csv"


def write_manifest(specs: list[dict], path: Path) -> None:
    if not specs:
        raise ValueError("No trials to write (empty sweep?)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(specs[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(specs)


def read_manifest_row(path: Path, trial_id: int) -> dict:
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["trial_id"]) == trial_id:
                return row
    raise ValueError(f"trial_id {trial_id} not found in {path}")


def load_sweep(path: Path, profile: str | None = None) -> dict:
    sweep = load_yaml(path)
    return apply_profile(sweep, profile)


def generate_manifest(sweep_path: Path, output_path: Path, profile: str | None = None) -> int:
    sweep = load_sweep(sweep_path, profile=profile)
    specs = iter_trial_specs(sweep)
    valid = []
    for spec in specs:
        if should_skip_spec(spec):
            continue
        spec["run_name"] = make_run_name(cfg_from_spec(spec))
        valid.append(spec)
    for i, spec in enumerate(valid, start=1):
        spec["trial_id"] = i
    write_manifest(valid, output_path)
    return len(valid)

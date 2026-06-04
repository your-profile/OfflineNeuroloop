"""Shared experiment grid: build configs, manifest, and run names."""
from __future__ import annotations

import copy
import csv
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent
REPO_DIRNAME = REPO_ROOT.name
MANIFESTS_DIR = REPO_ROOT / "manifests"


def path_for_manifest(path: Path) -> str:
    """Store paths relative to the repo so manifests work on HPC after local generation."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_repo_path(path: str | Path | None) -> Path | None:
    """Map manifest paths to files under this checkout (handles Mac → cluster manifests)."""
    if path is None or path == "":
        return None
    p = Path(path)
    if p.is_file():
        return p.resolve()
    if REPO_DIRNAME in p.parts:
        idx = p.parts.index(REPO_DIRNAME)
        candidate = REPO_ROOT.joinpath(*p.parts[idx + 1 :])
        if candidate.is_file():
            return candidate.resolve()
    candidate = REPO_ROOT / p
    if candidate.is_file():
        return candidate.resolve()
    return p

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

INTEGRATION_RESULTS_SUFFIX = {
    "finetune": "finetuning",
    "interleaved": "interleaving",
    "pretrain": "pretraining",
}

# Wall-clock hints for SLURM --time (one trial per array task).
DOMAIN_SLURM_TIME = {
    "flappy": "1:30:00",
    "lunar": "3:00:00",
    "robot": "8:00:00",
}


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def expand_list(val: Any) -> list[Any]:
    if val is None:
        return [None]
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def set_nested(cfg: dict, keys: list[str], val: Any) -> None:
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = val


def domain_yaml_path(domain_key: str) -> Path:
    return REPO_ROOT / "configs" / "domains" / f"{domain_key.lower()}.yaml"


def domain_label_from_cfg(cfg: dict, domain_config: Path | None = None) -> str:
    """Short label for sharding (Flappy, Lunar, Robot)."""
    if domain_config is not None:
        stem = domain_config.stem.lower()
        if stem in ("flappy", "lunar", "robot"):
            return stem.capitalize()
        parent = domain_config.parent.name.lower()
        if parent in ("flappy", "lunar", "robot"):
            return parent.capitalize()
    domain = str(cfg.get("experiment", {}).get("domain", ""))
    if domain:
        return domain.split()[0]
    return "Unknown"


def normalize_domain_filter(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def domain_config_path(domain_key: str) -> Path:
    return domain_yaml_path(domain_key)


def make_run_name(cfg: dict) -> str:
    e = cfg["experiment"]
    n = cfg["neural"]
    m = cfg["mlp"]
    return (
        f"{e['domain']}__{e['task']}__{e['condition']}"
        f"__{e['integration_type']}"
        f"__{e['model_granularity']}"
        f"__seed{e['random_state']}"
        f"__noise{m['model_noise']}"
        f"__beta{n['beta']}"
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
        if "buffer_type" not in cfg["rl"] and "buffer_type" in domain_cfg.get("rl", {}):
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

    if condition.startswith("Baseline") and (
        (ablation["key"] == ["mlp", "model_noise"] and ablation_val != 0.0)
        or (ablation["key"] == ["neural", "temporal_shift"] and ablation_val != 0.0)
    ):
        return None

    set_nested(cfg, ablation["key"], ablation_val)

    if n_episodes_override is not None:
        cfg["rl"]["n_episodes"] = n_episodes_override

    return cfg


def _resolve_domain_entry(domain_entry: dict) -> list[tuple[str | None, Path | None]]:
    """Return list of (domain_key label, domain_config path) pairs."""
    pairs: list[tuple[str | None, Path | None]] = []
    domain_configs = expand_list(domain_entry.get("domain_config"))
    domain_keys = expand_list(domain_entry.get("domain_key"))

    if domain_configs != [None]:
        for dc in domain_configs:
            path = REPO_ROOT / dc if not Path(str(dc)).is_absolute() else Path(dc)
            label = domain_entry.get("domain_key")
            if isinstance(label, list):
                label = None
            if label is None:
                label = domain_label_from_cfg(load_yaml(path), path)
            pairs.append((str(label), path))
        return pairs

    for dk in domain_keys:
        if dk is None:
            continue
        pairs.append((str(dk), domain_config_path(str(dk))))
    return pairs


def iter_trial_specs(sweep: dict) -> list[dict]:
    """Expand sweep config into a list of trial specification dicts."""
    base_config = REPO_ROOT / sweep.get("base_config", "configs/base.yaml")
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
            ablation_key_str = ".".join(ablation["key"])
            for seed in seeds:
                for granularity in granularities:
                    for condition in conditions:
                        for domain_entry in domains:
                            tasks = expand_list(domain_entry.get("tasks", ["Pooled"]))
                            for domain_key, domain_config in _resolve_domain_entry(domain_entry):
                                for task in tasks:
                                    for val in ablation["vals"]:
                                        trial_id += 1
                                        specs.append(
                                            {
                                                "trial_id": trial_id,
                                                "integration": integration,
                                                "domain_key": domain_key or "",
                                                "domain_config": path_for_manifest(domain_config)
                                                if domain_config
                                                else "",
                                                "task": task,
                                                "condition": condition,
                                                "seed": seed,
                                                "granularity": granularity,
                                                "ablation_key": ablation_key_str,
                                                "ablation_val": val,
                                                "base_config": path_for_manifest(base_config),
                                                "n_episodes_override": n_episodes_override or "",
                                                "data_path": data_path,
                                                "results_path": results_path,
                                            }
                                        )
    return specs


def filter_specs(
    specs: list[dict],
    *,
    domains: Iterable[str] | None = None,
    integrations: Iterable[str] | None = None,
    ablation_keys: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    granularities: Iterable[str] | None = None,
) -> list[dict]:
    domain_norm = {normalize_domain_filter(d) for d in domains} if domains else None
    integ_set = set(integrations) if integrations else None
    abl_set = set(ablation_keys) if ablation_keys else None
    cond_set = set(conditions) if conditions else None
    task_set = set(tasks) if tasks else None
    gran_set = {g.strip().lower() for g in granularities} if granularities else None

    out: list[dict] = []
    for spec in specs:
        if domain_norm is not None:
            if normalize_domain_filter(spec["domain_key"]) not in domain_norm:
                continue
        if integ_set is not None and spec["integration"] not in integ_set:
            continue
        if abl_set is not None and spec["ablation_key"] not in abl_set:
            continue
        if cond_set is not None and spec["condition"] not in cond_set:
            continue
        if task_set is not None and spec["task"] not in task_set:
            continue
        if gran_set is not None and spec["granularity"].strip().lower() not in gran_set:
            continue
        out.append(spec)
    return out


def shard_specs(specs: list[dict], by: str) -> dict[str, list[dict]]:
    """Group specs by domain, integration, or ablation_key for separate manifests."""
    groups: dict[str, list[dict]] = {}
    for spec in specs:
        if by == "domain":
            key = normalize_domain_filter(spec["domain_key"]) or "unknown"
        elif by == "integration":
            key = spec["integration"]
        elif by == "ablation":
            key = spec["ablation_key"].replace(".", "_")
        elif by == "domain_integration":
            key = f"{normalize_domain_filter(spec['domain_key'])}__{spec['integration']}"
        elif by == "domain_integration_ablation":
            key = (
                f"{normalize_domain_filter(spec['domain_key'])}__"
                f"{spec['integration']}__{spec['ablation_key'].replace('.', '_')}"
            )
        elif by == "domain_integration_ablation_granularity":
            key = (
                f"{normalize_domain_filter(spec['domain_key'])}__"
                f"{spec['integration']}__{spec['ablation_key'].replace('.', '_')}__"
                f"{spec['granularity']}"
            )
        else:
            raise ValueError(f"Unknown shard mode: {by}")
        groups.setdefault(key, []).append(spec)
    return groups


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
        base_config=resolve_repo_path(spec["base_config"]),
        domain_key=spec["domain_key"] or None,
        domain_config=resolve_repo_path(spec["domain_config"]),
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
        base_config=resolve_repo_path(spec["base_config"]),
        domain_key=spec["domain_key"] or None,
        domain_config=resolve_repo_path(spec["domain_config"]),
        n_episodes_override=_optional_int(spec.get("n_episodes_override")),
    )


def trial_results_filename(trial_id: int, integration: str) -> str:
    suffix = INTEGRATION_RESULTS_SUFFIX.get(integration, integration)
    return f"runs/trial_{trial_id:05d}_{suffix}.csv"


def trial_results_path(spec: dict) -> Path:
    results_root = Path(spec["results_path"] or REPO_ROOT)
    return results_root / "src/results" / trial_results_filename(int(spec["trial_id"]), spec["integration"])


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


def finalize_specs(specs: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for spec in specs:
        if should_skip_spec(spec):
            continue
        spec["run_name"] = make_run_name(cfg_from_spec(spec))
        valid.append(spec)
    for i, spec in enumerate(valid, start=1):
        spec["trial_id"] = i
    return valid


def generate_manifest(
    sweep_path: Path,
    output_path: Path,
    profile: str | None = None,
    *,
    domains: list[str] | None = None,
    integrations: list[str] | None = None,
    ablation_keys: list[str] | None = None,
    conditions: list[str] | None = None,
    tasks: list[str] | None = None,
    granularities: list[str] | None = None,
) -> int:
    sweep = load_sweep(sweep_path, profile=profile)
    specs = iter_trial_specs(sweep)
    specs = filter_specs(
        specs,
        domains=domains,
        integrations=integrations,
        ablation_keys=ablation_keys,
        conditions=conditions,
        tasks=tasks,
        granularities=granularities,
    )
    valid = finalize_specs(specs)
    write_manifest(valid, output_path)
    return len(valid)


def generate_manifest_shards(
    sweep_path: Path,
    output_dir: Path,
    shard_by: str = "domain_integration_ablation_granularity",
    profile: str | None = None,
    *,
    domains: list[str] | None = None,
    integrations: list[str] | None = None,
    ablation_keys: list[str] | None = None,
    conditions: list[str] | None = None,
    tasks: list[str] | None = None,
    granularities: list[str] | None = None,
) -> dict[str, int]:
    """Write one manifest CSV per shard; return {shard_name: n_trials}."""
    sweep = load_sweep(sweep_path, profile=profile)
    specs = filter_specs(
        iter_trial_specs(sweep),
        domains=domains,
        integrations=integrations,
        ablation_keys=ablation_keys,
        conditions=conditions,
        tasks=tasks,
        granularities=granularities,
    )
    specs = finalize_specs(specs)
    groups = shard_specs(specs, shard_by)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, group in sorted(groups.items()):
        path = output_dir / f"{name}.csv"
        write_manifest(group, path)
        counts[name] = len(group)
    return counts


def slurm_time_for_domain(domain_key: str) -> str:
    norm = normalize_domain_filter(domain_key)
    return DOMAIN_SLURM_TIME.get(norm, "4:00:00")

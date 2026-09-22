"""Shared eval settings: label granularity + passive/active/pooled tasks."""

from __future__ import annotations

from .channels import CHANNELS_8, CHANNELS_6, PAIRS_8, PAIRS_6, CONDS, DOMAINS

LABEL_COLS = {
    "binary": "binary_optimal",
    "ternary": "discrete_optimal",
    "discrete": "discrete_optimal",
    "continuous": "continuous_optimal",
}

TASK_MODE = {
    "passive": "W",
    "watch": "W",
    "w": "W",
    "active": "P",
    "play": "P",
    "p": "P",
    "pooled": "pooled",
    "all": "all",
}


def normalize_granularity(g: str | None) -> str:
    g = (g or "binary").lower().strip()
    if g in ("ternary", "discrete", "multi", "multiclass"):
        return "discrete"
    if g in ("continuous", "regression", "cont"):
        return "continuous"
    if g in ("binary", "bin"):
        return "binary"
    raise ValueError(f"Unknown granularity '{g}'. Use binary | discrete/ternary | continuous.")


def normalize_task(task: str | None) -> str:
    if task is None:
        return "all"
    t = TASK_MODE.get(str(task).lower().strip())
    if t is None:
        raise ValueError(f"Unknown task '{task}'. Use passive | active | pooled | all.")
    return t


def channel_setup(cfg: dict):
    drop = cfg.get("drop_suspect_channels", False)
    if drop:
        return CHANNELS_6, PAIRS_6, f"{len(CHANNELS_6)}"
    return CHANNELS_8, PAIRS_8, "8"


def expand_jobs(cfg: dict) -> list[tuple[str, list[str]]]:
    """Return [(job_name, [condition codes]), ...].

    Priority:
      1. ``condition_groups`` — explicit multi-condition jobs
         e.g. ``[{name: robot_passive, conditions: [RW]}, ...]``
      2. explicit ``conditions`` list (one job per condition)
      3. ``task`` + ``domains``  (passive / active / pooled)
    """
    groups = cfg.get("condition_groups")
    if groups:
        out = []
        for g in groups:
            if isinstance(g, str):
                out.append((g, [g]))
                continue
            name = g.get("name") or "+".join(g["conditions"])
            conds = list(g["conditions"])
            out.append((str(name), conds))
        return out

    if cfg.get("conditions"):
        return [(c, [c]) for c in cfg["conditions"]]

    task = normalize_task(cfg.get("task", "all"))
    domains = [d.upper() for d in cfg.get("domains", DOMAINS)]
    for d in domains:
        if d not in DOMAINS:
            raise ValueError(f"Unknown domain '{d}'. Use one of {DOMAINS}.")

    if task == "all":
        return [(c, [c]) for c in CONDS if c[0] in domains]

    if task == "pooled":
        jobs = []
        for d in domains:
            conds = [f"{d}W", f"{d}P"]
            jobs.append((f"{d}_pooled", conds))
        return jobs

    suffix = task  # W or P
    return [(f"{d}{suffix}", [f"{d}{suffix}"]) for d in domains]


def win_kwargs(cfg: dict) -> dict:
    g = normalize_granularity(cfg.get("granularity", "binary"))
    return dict(
        granularity=g,
        window_s=cfg.get("window_s", 8.0),
        step_s=cfg.get("step_s", 1.0),
        rate=cfg.get("rate_hz", 5.2),
        ambig_lo=cfg.get("ambig_lo", 0.25),
        ambig_hi=cfg.get("ambig_hi", 0.75),
        min_majority=cfg.get("min_majority", 0.5),
        min_windows=cfg.get("min_windows", 40),
        min_per_class=cfg.get("min_per_class", 15),
        min_std=cfg.get("min_std", 1e-6),
        temporal_shift=float(cfg.get("temporal_shift", 0.0)),
    )


def cv_kwargs(cfg: dict) -> dict:
    return dict(
        n_folds=cfg.get("n_folds", 8),
        window_s=cfg.get("window_s", 8.0),
        embargo_s=cfg.get("embargo_s", 4.0),
        granularity=normalize_granularity(cfg.get("granularity", "binary")),
    )

"""
Run offline ML evaluation tests (decoding / controls / power).

Uses the same models + purged-CV splits as prior OfflineNeuroloop tests
(shrinkage LDA / Ridge; contiguous blocks + embargo), with OfflineNeuroloop
data layout:

  DATA_PATH/fNIRS/FilteredData/{pid}_processed_{COND}.csv
  DATA_PATH/fNIRS/LabeledData/{pid}_{COND}_LabeledData.csv

Examples:
  python run_test.py --list
  python run_test.py configs/eval/within_subject_rw.yaml
  python run_test.py configs/eval/lopo_focus4.yaml configs/eval/focus_transfer.yaml
  python run_test.py --all

RL / integration sweeps live in run_test_rl.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval import get_test, list_tests  # noqa: E402

DEFAULT_DATA_PATH = Path(
    os.environ.get(
        "DATA_PATH",
        "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData",
    )
)
REPO_FILTERED = REPO_ROOT / "data" / "fNIRS" / "FilteredData"

DEFAULT_TESTS = [
    REPO_ROOT / "configs/eval/within_subject.yaml",
    REPO_ROOT / "configs/eval/within_subject_6ch.yaml",
    REPO_ROOT / "configs/eval/lopo.yaml",
    REPO_ROOT / "configs/eval/lopo_focus4.yaml",
    REPO_ROOT / "configs/eval/focus_transfer.yaml",
    REPO_ROOT / "configs/eval/positive_control.yaml",
]


def default_paths() -> dict:
    """Prefer repo ``data/fNIRS/FilteredData`` when present; labels from ParticipantData."""
    processed = REPO_FILTERED if REPO_FILTERED.is_dir() and any(REPO_FILTERED.glob("*.csv")) else (
        DEFAULT_DATA_PATH / "fNIRS" / "FilteredData"
    )
    return {
        "processed": str(processed),
        "labeled": str(DEFAULT_DATA_PATH / "fNIRS" / "LabeledData"),
        "results": str(REPO_ROOT / "results" / "eval"),
    }


def _resolve_path(p: str | Path) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def load_cfg(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "test" not in cfg:
        raise ValueError(f"{path}: config must be a mapping with a 'test' field")
    paths = dict(default_paths())
    paths.update(cfg.get("paths") or {})
    for key in ("processed", "labeled", "results"):
        if key in paths and paths[key]:
            paths[key] = _resolve_path(paths[key])
    cfg["paths"] = paths
    cfg["_config_path"] = str(path)
    return cfg


def out_path(cfg: dict, config_path: Path) -> Path:
    results = Path(cfg["paths"].get("results", REPO_ROOT / "results" / "eval"))
    results.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [cfg["test"], config_path.stem]
    g = cfg.get("granularity")
    if g:
        parts.insert(1, str(g))
    if cfg.get("task") and not cfg.get("conditions"):
        parts.insert(1, str(cfg["task"]))
    if cfg.get("drop_suspect_channels"):
        parts.append("intensity")
    name = "_".join(parts) + f"_{stamp}.csv"
    return results / name


def run_one(config_path: Path) -> Path | None:
    cfg = load_cfg(config_path)
    name = cfg["test"]
    print("\n" + "=" * 72)
    print(f"TEST: {name}")
    print(f"CONFIG: {config_path}")
    print(f"granularity: {cfg.get('granularity', 'binary')}  task: {cfg.get('task', '-')}  "
          f"conditions: {cfg.get('conditions', cfg.get('domains', '-'))}")
    print("=" * 72)

    fn = get_test(name)
    df = fn(cfg)
    if df is None or len(df) == 0:
        print("(no rows returned)")
        return None

    path = out_path(cfg, config_path)
    df.to_csv(path, index=False)
    print(f"\nsaved → {path}")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Run OfflineNeuroloop ML evaluation tests")
    p.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="YAML config(s) under configs/eval/",
    )
    p.add_argument("--list", action="store_true", help="list registered tests")
    p.add_argument(
        "--all",
        action="store_true",
        help="run the default eval suite (excludes power — run that config explicitly)",
    )
    args = p.parse_args(argv)

    if args.list:
        print("Registered tests:")
        for t in list_tests():
            print(f"  - {t}")
        print("\nExample configs:")
        for c in sorted((REPO_ROOT / "configs" / "eval").glob("*.yaml")):
            print(f"  - {c.relative_to(REPO_ROOT)}")
        return 0

    if args.all:
        configs = DEFAULT_TESTS
    elif args.configs:
        configs = [c if c.is_absolute() else REPO_ROOT / c for c in args.configs]
    else:
        p.print_help()
        print("\nTip: python run_test.py --list")
        print("     python run_test.py configs/eval/within_subject_rw.yaml")
        return 1

    saved = []
    for c in configs:
        if not c.exists():
            print(f"missing config: {c}", file=sys.stderr)
            return 1
        out = run_one(c)
        if out is not None:
            saved.append(out)

    print("\n" + "=" * 72)
    print(f"done: {len(saved)} result file(s)")
    for s in saved:
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

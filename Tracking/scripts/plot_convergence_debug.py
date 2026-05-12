from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._convergence_config import load_convergence_config  # noqa: E402


def _read_csv(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except Exception:
                    pass
            rows.append(parsed)
    return rows


def _result_env_step(run_dir: Path) -> float | None:
    result_path = run_dir / "exp" / "result.pkl"
    if not result_path.exists():
        return None
    try:
        with result_path.open("rb") as f:
            result = pickle.load(f)
        if isinstance(result, dict) and result.get("env_step") is not None:
            return float(result["env_step"])
    except Exception:
        return None
    return None


def add_true_env_step(rows: List[Dict[str, float]], run_dir: Path) -> List[Dict[str, float]]:
    """Add eval_event_index/true_env_step robustly when DI logs duplicate evaluator rows.

    DI-engine may write one row per evaluator episode at the same logged train_step.
    For paper plots, each distinct train_step is treated as one eval event and is
    linearly mapped to the final env_step stored in exp/result.pkl.
    """
    if not rows:
        return rows
    final_env_step = _result_env_step(run_dir)
    if final_env_step is None or final_env_step <= 0:
        for idx, row in enumerate(rows):
            row.setdefault("eval_event_index", float(idx))
            row.setdefault("true_env_step", row.get("train_step", float(idx)))
        return rows

    event_keys: List[float] = []
    key_to_index: Dict[float, int] = {}
    for idx, row in enumerate(rows):
        key = row.get("train_step", float(idx))
        if key not in key_to_index:
            key_to_index[key] = len(event_keys)
            event_keys.append(key)
    denom = max(1, len(event_keys))
    for idx, row in enumerate(rows):
        event_idx = key_to_index.get(row.get("train_step", float(idx)), idx)
        row["eval_event_index"] = float(event_idx + 1)
        row["true_env_step"] = float(final_env_step) * float(event_idx + 1) / float(denom)
    return rows


def write_grouped_true_env_step_csv(rows: List[Dict[str, float]], out: Path) -> None:
    if not rows:
        return
    grouped: Dict[int, Dict[str, List[float]]] = {}
    for row in rows:
        event = int(row.get("eval_event_index", 0))
        bucket = grouped.setdefault(event, {})
        for key, value in row.items():
            bucket.setdefault(key, []).append(float(value))
    out_rows: List[Dict[str, float]] = []
    for event in sorted(grouped):
        out_row: Dict[str, float] = {"eval_event_index": float(event)}
        for key, values in grouped[event].items():
            vals = [float(v) for v in values]
            out_row[key] = float(sum(vals) / max(1, len(vals)))
        out_rows.append(out_row)
    keys = sorted({key for row in out_rows for key in row})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(out_rows)


def _plot_metric(rows: List[Dict[str, float]], x_key: str, y_key: str, out: Path, title: str) -> None:
    if not rows or y_key not in rows[0]:
        return
    x = [row.get(x_key, idx) for idx, row in enumerate(rows)]
    y = [row.get(y_key, 0.0) for row in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, linewidth=1.8)
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/auv6dof_convergence_debug.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    load_convergence_config(args.config)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    learning = _read_csv(run_dir / "learning_curve.csv")
    eval_rows = add_true_env_step(_read_csv(run_dir / "eval_detail.csv"), run_dir)
    eval_curve = _read_csv(run_dir / "eval_curve.csv")
    write_grouped_true_env_step_csv(eval_rows, out_dir / "eval_detail_grouped_true_env_step.csv")
    _plot_metric(learning, "episode", "reward", out_dir / "episode_return.png", "Episode return")
    _plot_metric(eval_curve, "episode", "reward", out_dir / "eval_return.png", "Eval return")
    for key in (
        "mean_tracking_error",
        "mean_target_lost",
        "mean_tracking_reward",
        "mean_observation_reward",
        "mean_control_cost",
        "mean_action_delta_norm",
    ):
        _plot_metric(eval_rows, "true_env_step", key, out_dir / f"{key}.png", key)
    print({"output_dir": out_dir.as_posix()})


if __name__ == "__main__":
    main()

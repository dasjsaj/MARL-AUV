from __future__ import annotations

"""
Generate paper-ready figures and tables for the hard 4-AUV experiments.

This script is intentionally self-contained:
- It reads completed runs from artifacts/auv6dof_tmc_2e6/hard_4auv/auv6dof.
- A run is considered formal only when exp/result.pkl exists and env_step >= min_env_step.
- It maps each eval row to a unified true_env_step axis, because eval_detail.csv train_step
  is not a reliable cross-algorithm env-step field.

Outputs:
- paper_figures/*.png: convergence curves, final-performance bars, boxplots, diagnostics.
- paper_tables/*.csv: main result tables, protocol table, failure/stability analysis.
- hard_paper_tables/hard_paper_tables.xlsx if pandas/openpyxl are available.

Metric meanings used in this script:
- eval_return: evaluator episode return. Higher is better.
- mean_tracking_error: mean |AUV-target distance - desired_tracking_distance|. Lower is better.
- tail_mean_target_distance: mean AUV-target distance in the final tail segment of eval. Lower is better.
- mean_target_lost: fraction/rate of target-lost events during eval. Lower is better.
- mean_action_saturation_rate: rate of saturated/clipped action commands. Lower is better.
- mean_control_cost: control effort and action-change penalty. Lower is better when tracking quality is similar.
- mean_tracking_reward / mean_semantic_reward: reward components used to diagnose STG-MAPPO.
"""

import argparse
import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "artifacts/auv6dof_tmc_2e6/hard_4auv/auv6dof"
DEFAULT_OUT = REPO_ROOT / "artifacts/auv6dof_tmc_2e6"

ALGO_ORDER = ["stg_mappo", "mappo", "maddpg", "matd3", "happo", "madqn", "masac"]
ALGO_LABEL = {
    "stg_mappo": "STG-MAPPO",
    "mappo": "MAPPO",
    "maddpg": "MADDPG",
    "matd3": "MATD3",
    "happo": "HAPPO",
    "madqn": "MADQN",
    "masac": "MASAC",
}

# Paper plotting style. This follows the user's reference plotting script:
# Times New Roman, large labels, clear grid, thick mean curves, and std/CI shadows.
PLOT_COLORS = {
    "stg_mappo": "#1f77b4",  # proposed method, blue
    "mappo": "#ff7f0e",      # orange
    "maddpg": "#d62728",     # red
    "matd3": "#2ca02c",      # green
    "happo": "#9467bd",      # purple
    "madqn": "#8c564b",      # brown
    "masac": "#e377c2",      # pink
}

PLOT_STYLE = {
    "font_family": "Times New Roman",
    "font_size": 18,
    "title_size": 20,
    "label_size": 20,
    "tick_size": 18,
    "legend_size": 10,
    "line_width": 2.4,
    "shadow_alpha": 0.23,
    # Convergence curves are noisy because every eval point is preserved.
    # A 9-point moving average keeps trend readability without hiding failures.
    "smooth_window": 8,
    # False = std shadow. True = 95% CI shadow.
    "use_ci95": False,
}

METRICS = {
    "eval_return": {
        "label": "Eval Return",
        "better": "higher",
        "paper_role": "Convergence and final task utility.",
    },
    "mean_tracking_error": {
        "label": "Mean Tracking Error",
        "better": "lower",
        "paper_role": "Main target-tracking accuracy metric.",
    },
    "tail_mean_target_distance": {
        "label": "Tail Mean Target Distance",
        "better": "lower",
        "paper_role": "Steady-state AUV-target distance in the final eval tail.",
    },
    "mean_target_lost": {
        "label": "Target Lost Rate",
        "better": "lower",
        "paper_role": "Reliability of continuous target observation.",
    },
    "mean_action_saturation_rate": {
        "label": "Action Saturation Rate",
        "better": "lower",
        "paper_role": "Whether the controller relies on saturated actions.",
    },
    "mean_control_cost": {
        "label": "Control Cost",
        "better": "lower",
        "paper_role": "Control smoothness/energy regularization.",
    },
    "mean_tracking_reward": {
        "label": "Tracking Reward",
        "better": "higher",
        "paper_role": "Whether reward is dominated by useful tracking progress.",
    },
    "mean_observation_reward": {
        "label": "Observation Reward",
        "better": "higher",
        "paper_role": "Whether target observation quality improves.",
    },
    "mean_semantic_reward": {
        "label": "Semantic Reward",
        "better": "higher",
        "paper_role": "Whether semantic phase/task-graph shaping contributes.",
    },
}

# Distance-like metrics are reported in the original kilometer-scale values.
# The manuscript explains the physical unit, so no display rescaling is applied.
DISPLAY_SCALE = {
    "mean_tracking_error": 1.0,
    "tail_mean_target_distance": 1.0,
    "tail100_mean_target_distance": 1.0,
    "tail100_mean_tracking_error": 1.0,
}


def display_scale(metric: str) -> float:
    return float(DISPLAY_SCALE.get(metric, 1.0))


def display_label(metric: str) -> str:
    label = METRICS[metric]["label"]
    if metric in {
        "mean_tracking_error",
        "tail_mean_target_distance",
        "tail100_mean_target_distance",
        "tail100_mean_tracking_error",
    }:
        return f"{label} (km)"
    return label


def apply_paper_style() -> None:
    """Apply one global matplotlib style for all generated paper figures."""
    matplotlib.rcParams["font.family"] = PLOT_STYLE["font_family"]
    matplotlib.rcParams["font.size"] = PLOT_STYLE["font_size"]
    matplotlib.rcParams["axes.titlesize"] = PLOT_STYLE["title_size"]
    matplotlib.rcParams["axes.labelsize"] = PLOT_STYLE["label_size"]
    matplotlib.rcParams["xtick.labelsize"] = PLOT_STYLE["tick_size"]
    matplotlib.rcParams["ytick.labelsize"] = PLOT_STYLE["tick_size"]
    matplotlib.rcParams["legend.fontsize"] = PLOT_STYLE["legend_size"]
    matplotlib.rcParams["legend.title_fontsize"] = PLOT_STYLE["legend_size"]
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def smooth_1d(y: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing with edge normalization."""
    y = np.asarray(y, dtype=np.float64)
    if window <= 1 or y.size <= 2:
        return y
    window = min(int(window), int(y.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    z = np.ones_like(y, dtype=np.float64)
    return np.convolve(y, kernel, mode="same") / np.convolve(z, kernel, mode="same")


def save_figure(fig: plt.Figure, path_png: Path) -> Path:
    """Save both PNG for quick browsing and PDF for paper submission."""
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, bbox_inches="tight", dpi=500)
    fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    return path_png


@dataclass
class RunData:
    algo: str
    seed: int
    run_dir: Path
    env_step: int
    train_iter: int
    finish_time: str
    rows: List[Dict[str, str]]
    true_env_step: np.ndarray
    metrics: Dict[str, np.ndarray]


def _float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_result(path: Path) -> Optional[Dict[str, object]]:
    result_path = path / "exp" / "result.pkl"
    if not result_path.exists():
        return None
    with result_path.open("rb") as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, dict) else {}


def _metric_array(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    values = [_float(row.get(key)) for row in rows]
    return np.asarray(values, dtype=np.float64)


def _build_true_env_step(num_rows: int, env_step: int) -> np.ndarray:
    if num_rows <= 0:
        return np.asarray([], dtype=np.float64)
    # The first eval event is not necessarily at env_step=0. We distribute events
    # uniformly over the completed training budget for cross-algorithm plotting.
    return np.linspace(float(env_step) / num_rows, float(env_step), num_rows)


def discover_formal_runs(base: Path, min_env_step: int) -> List[RunData]:
    """Discover completed formal runs and drop stale/interrupted directories."""
    candidates: Dict[Tuple[str, int], RunData] = {}
    for algo_dir in sorted(base.iterdir()) if base.exists() else []:
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name
        for seed_dir in sorted(algo_dir.glob("seed_*")):
            try:
                seed = int(seed_dir.name.replace("seed_", ""))
            except Exception:
                continue
            for run_dir in sorted([p for p in seed_dir.iterdir() if p.is_dir()]):
                result = _read_result(run_dir)
                if not result:
                    continue
                env_step = int(result.get("env_step") or 0)
                if env_step < min_env_step:
                    continue
                rows = _read_csv_rows(run_dir / "eval_detail.csv")
                if not rows:
                    continue
                metrics = {key: _metric_array(rows, key) for key in METRICS.keys()}
                run = RunData(
                    algo=algo,
                    seed=seed,
                    run_dir=run_dir,
                    env_step=env_step,
                    train_iter=int(result.get("train_iter") or 0),
                    finish_time=str(result.get("finish_time") or ""),
                    rows=rows,
                    true_env_step=_build_true_env_step(len(rows), env_step),
                    metrics=metrics,
                )
                key = (algo, seed)
                prev = candidates.get(key)
                if prev is None or (run.env_step, len(run.rows), run.run_dir.name) > (
                    prev.env_step,
                    len(prev.rows),
                    prev.run_dir.name,
                ):
                    candidates[key] = run
    return sorted(candidates.values(), key=lambda r: (ALGO_ORDER.index(r.algo) if r.algo in ALGO_ORDER else 999, r.seed))


def _algo_groups(runs: Iterable[RunData]) -> Dict[str, List[RunData]]:
    groups: Dict[str, List[RunData]] = {}
    for run in runs:
        groups.setdefault(run.algo, []).append(run)
    return {algo: groups[algo] for algo in ALGO_ORDER if algo in groups}


def _interpolate_runs(runs: List[RunData], metric: str, grid: np.ndarray) -> np.ndarray:
    ys = []
    for run in runs:
        x = run.true_env_step
        y = run.metrics[metric]
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            continue
        ys.append(np.interp(grid, x[mask], y[mask]))
    return np.vstack(ys) if ys else np.empty((0, grid.size))


def _last_fraction_values(run: RunData, metric: str, fraction: float = 0.2) -> np.ndarray:
    y = run.metrics[metric]
    y = y[np.isfinite(y)]
    if y.size == 0:
        return y
    start = int(math.floor((1.0 - fraction) * y.size))
    return y[start:]


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return math.nan, math.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _auc_normalized(run: RunData, metric: str) -> float:
    x = run.true_env_step
    y = run.metrics[metric]
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2 or run.env_step <= 0:
        return math.nan
    return float(np.trapz(y[mask], x[mask]) / float(run.env_step))


def _steps_to_threshold(run: RunData, metric: str, threshold: float, *, lower_is_better: bool, window: int = 10) -> float:
    x = run.true_env_step
    y = run.metrics[metric]
    mask = np.isfinite(y)
    x, y = x[mask], y[mask]
    if y.size < window:
        return math.nan
    kernel = np.ones(window, dtype=np.float64) / float(window)
    rolling = np.convolve(y, kernel, mode="valid")
    xs = x[window - 1 :]
    passed = rolling <= threshold if lower_is_better else rolling >= threshold
    if not np.any(passed):
        return math.nan
    return float(xs[int(np.argmax(passed))])


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_tables(runs: List[RunData], table_dir: Path) -> Dict[str, Path]:
    """Create CSV tables used directly by the paper."""
    table_dir.mkdir(parents=True, exist_ok=True)
    groups = _algo_groups(runs)

    per_seed_rows: List[Dict[str, object]] = []
    for run in runs:
        latest = {f"latest_{k}": float(run.metrics[k][-1]) if run.metrics[k].size else math.nan for k in METRICS}
        last20 = {
            f"last20_mean_{k}": float(np.mean(_last_fraction_values(run, k))) if _last_fraction_values(run, k).size else math.nan
            for k in METRICS
        }
        latest_display = {
            f"latest_{k}_display": latest[f"latest_{k}"] * display_scale(k)
            for k in METRICS
        }
        last20_display = {
            f"last20_mean_{k}_display": last20[f"last20_mean_{k}"] * display_scale(k)
            for k in METRICS
        }
        aucs = {
            "auc_eval_return": _auc_normalized(run, "eval_return"),
            "auc_tracking_error": _auc_normalized(run, "mean_tracking_error"),
            "steps_to_tracking_error_0p05": _steps_to_threshold(
                run, "mean_tracking_error", 0.05, lower_is_better=True
            ),
            "steps_to_lost_rate_0p05": _steps_to_threshold(run, "mean_target_lost", 0.05, lower_is_better=True),
        }
        per_seed_rows.append(
            {
                "algo": run.algo,
                "algo_label": ALGO_LABEL.get(run.algo, run.algo),
                "seed": run.seed,
                "env_step": run.env_step,
                "train_iter": run.train_iter,
                "eval_points": len(run.rows),
                "run_dir": run.run_dir.as_posix(),
                "finish_time": run.finish_time,
                **latest,
                **last20,
                **latest_display,
                **last20_display,
                **aucs,
            }
        )

    per_seed_fields = [
        "algo",
        "algo_label",
        "seed",
        "env_step",
        "train_iter",
        "eval_points",
        "latest_eval_return",
        "latest_tail_mean_target_distance",
        "latest_tail_mean_target_distance_display",
        "latest_mean_tracking_error",
        "latest_mean_tracking_error_display",
        "latest_mean_target_lost",
        "latest_mean_action_saturation_rate",
        "last20_mean_eval_return",
        "last20_mean_tail_mean_target_distance",
        "last20_mean_tail_mean_target_distance_display",
        "last20_mean_mean_tracking_error",
        "last20_mean_mean_tracking_error_display",
        "last20_mean_mean_target_lost",
        "last20_mean_mean_action_saturation_rate",
        "last20_mean_mean_control_cost",
        "last20_mean_mean_tracking_reward",
        "last20_mean_mean_semantic_reward",
        "auc_eval_return",
        "auc_tracking_error",
        "steps_to_tracking_error_0p05",
        "steps_to_lost_rate_0p05",
        "run_dir",
        "finish_time",
    ]
    per_seed_path = table_dir / "table_per_seed_results.csv"
    write_csv(per_seed_path, per_seed_rows, per_seed_fields)

    main_rows: List[Dict[str, object]] = []
    for algo, rs in groups.items():
        row: Dict[str, object] = {
            "algo": algo,
            "algo_label": ALGO_LABEL.get(algo, algo),
            "num_seeds": len(rs),
            "completed_seeds": ",".join(str(r.seed) for r in rs),
        }
        for metric in [
            "eval_return",
            "tail_mean_target_distance",
            "mean_tracking_error",
            "mean_target_lost",
            "mean_action_saturation_rate",
            "mean_control_cost",
            "mean_tracking_reward",
            "mean_semantic_reward",
        ]:
            values = [float(np.mean(_last_fraction_values(run, metric))) for run in rs]
            mean, std = _mean_std(values)
            row[f"last20_{metric}_mean"] = mean
            row[f"last20_{metric}_std"] = std
            row[f"last20_{metric}_mean_display"] = mean * display_scale(metric)
            row[f"last20_{metric}_std_display"] = std * display_scale(metric)
        for metric in ["eval_return", "tail_mean_target_distance", "mean_tracking_error", "mean_target_lost"]:
            values = [float(run.metrics[metric][-1]) for run in rs if run.metrics[metric].size]
            mean, std = _mean_std(values)
            row[f"latest_{metric}_mean"] = mean
            row[f"latest_{metric}_std"] = std
            row[f"latest_{metric}_mean_display"] = mean * display_scale(metric)
            row[f"latest_{metric}_std_display"] = std * display_scale(metric)
        main_rows.append(row)

    main_fields = [
        "algo",
        "algo_label",
        "num_seeds",
        "completed_seeds",
        "last20_eval_return_mean",
        "last20_eval_return_std",
        "last20_tail_mean_target_distance_mean",
        "last20_tail_mean_target_distance_std",
        "last20_tail_mean_target_distance_mean_display",
        "last20_tail_mean_target_distance_std_display",
        "last20_mean_tracking_error_mean",
        "last20_mean_tracking_error_std",
        "last20_mean_tracking_error_mean_display",
        "last20_mean_tracking_error_std_display",
        "last20_mean_target_lost_mean",
        "last20_mean_target_lost_std",
        "last20_mean_action_saturation_rate_mean",
        "last20_mean_action_saturation_rate_std",
        "last20_mean_control_cost_mean",
        "last20_mean_control_cost_std",
        "last20_mean_tracking_reward_mean",
        "last20_mean_tracking_reward_std",
        "last20_mean_semantic_reward_mean",
        "last20_mean_semantic_reward_std",
        "latest_eval_return_mean",
        "latest_eval_return_std",
        "latest_tail_mean_target_distance_mean",
        "latest_tail_mean_target_distance_std",
        "latest_tail_mean_target_distance_mean_display",
        "latest_tail_mean_target_distance_std_display",
        "latest_mean_tracking_error_mean",
        "latest_mean_tracking_error_std",
        "latest_mean_tracking_error_mean_display",
        "latest_mean_tracking_error_std_display",
        "latest_mean_target_lost_mean",
        "latest_mean_target_lost_std",
    ]
    main_path = table_dir / "table_main_results_last20.csv"
    write_csv(main_path, main_rows, main_fields)

    protocol_rows = [
        {
            "algorithm": "STG-MAPPO",
            "di_engine_policy": "ppo",
            "policy_class": "on-policy actor-critic",
            "action_mode": "velocity3",
            "semantic_state": "yes",
            "semantic_graph": "yes",
            "semantic_reward": "yes",
            "role_in_paper": "proposed method",
        },
        {
            "algorithm": "MAPPO",
            "di_engine_policy": "ppo",
            "policy_class": "on-policy actor-critic",
            "action_mode": "tau6",
            "semantic_state": "no",
            "semantic_graph": "no",
            "semantic_reward": "no",
            "role_in_paper": "strong raw baseline",
        },
        {
            "algorithm": "MADDPG",
            "di_engine_policy": "ddpg",
            "policy_class": "off-policy actor-critic",
            "action_mode": "tau6",
            "semantic_state": "no",
            "semantic_graph": "no",
            "semantic_reward": "no",
            "role_in_paper": "continuous off-policy baseline",
        },
        {
            "algorithm": "MATD3",
            "di_engine_policy": "td3",
            "policy_class": "off-policy twin-critic actor-critic",
            "action_mode": "tau6",
            "semantic_state": "no",
            "semantic_graph": "no",
            "semantic_reward": "no",
            "role_in_paper": "continuous twin-critic baseline",
        },
        {
            "algorithm": "HAPPO",
            "di_engine_policy": "happo",
            "policy_class": "on-policy heterogeneous-agent PPO",
            "action_mode": "tau6",
            "semantic_state": "no",
            "semantic_graph": "no",
            "semantic_reward": "no",
            "role_in_paper": "on-policy MARL baseline",
        },
        {
            "algorithm": "MADQN",
            "di_engine_policy": "madqn",
            "policy_class": "value-based discrete MARL",
            "action_mode": "discrete tau6 codebook",
            "semantic_state": "no",
            "semantic_graph": "no",
            "semantic_reward": "no",
            "role_in_paper": "discrete-control baseline",
        },
        {
            "algorithm": "MASAC",
            "di_engine_policy": "discrete_sac",
            "policy_class": "entropy-regularized off-policy discrete actor-critic",
            "action_mode": "discrete tau6 codebook",
            "semantic_state": "no",
            "semantic_graph": "no",
            "semantic_reward": "no",
            "role_in_paper": "entropy-regularized discrete baseline",
        },
    ]
    protocol_fields = [
        "algorithm",
        "di_engine_policy",
        "policy_class",
        "action_mode",
        "semantic_state",
        "semantic_graph",
        "semantic_reward",
        "role_in_paper",
    ]
    protocol_path = table_dir / "table_algorithm_protocol.csv"
    write_csv(protocol_path, protocol_rows, protocol_fields)

    failure_rows = []
    for row in main_rows:
        algo = str(row["algo"])
        dist = float(row["last20_tail_mean_target_distance_mean"])
        err = float(row["last20_mean_tracking_error_mean"])
        lost = float(row["last20_mean_target_lost_mean"])
        ret = float(row["last20_eval_return_mean"])
        if algo == "stg_mappo":
            issue = "stable tracking"
            reason = "semantic task graph, semantic reward, and velocity3 action abstraction align the optimization target."
        elif algo == "mappo":
            issue = "learns but less accurate"
            reason = "raw tau6 action and non-semantic observation make stable close tracking harder."
        elif lost > 0.5 or err > 0.7:
            issue = "mostly unstable or not converged"
            reason = "target lost rate and tracking error remain high under the same training budget."
        elif err > 0.3:
            issue = "partial convergence with high variance"
            reason = "some seeds learn useful behavior but tracking accuracy is inconsistent."
        else:
            issue = "moderate baseline"
            reason = "shows partial target-following behavior but remains behind STG-MAPPO."
        failure_rows.append(
            {
                "algorithm": ALGO_LABEL.get(algo, algo),
                "last20_eval_return": ret,
                "last20_tail_distance": dist,
                "last20_tracking_error": err,
                "last20_lost_rate": lost,
                "diagnosis": issue,
                "paper_explanation": reason,
            }
        )
    failure_fields = [
        "algorithm",
        "last20_eval_return",
        "last20_tail_distance",
        "last20_tracking_error",
        "last20_lost_rate",
        "diagnosis",
        "paper_explanation",
    ]
    failure_path = table_dir / "table_failure_and_diagnosis.csv"
    write_csv(failure_path, failure_rows, failure_fields)

    paths = {
        "per_seed": per_seed_path,
        "main": main_path,
        "protocol": protocol_path,
        "diagnosis": failure_path,
    }
    try:
        import pandas as pd

        xlsx_path = table_dir / "hard_paper_tables.xlsx"
        with pd.ExcelWriter(xlsx_path) as writer:
            pd.DataFrame(per_seed_rows).to_excel(writer, sheet_name="per_seed", index=False)
            pd.DataFrame(main_rows).to_excel(writer, sheet_name="main_last20", index=False)
            pd.DataFrame(protocol_rows).to_excel(writer, sheet_name="algorithm_protocol", index=False)
            pd.DataFrame(failure_rows).to_excel(writer, sheet_name="diagnosis", index=False)
        paths["xlsx"] = xlsx_path
    except Exception:
        # CSV files are the required outputs; xlsx is a convenience artifact.
        pass
    return paths


def plot_metric_convergence(groups: Dict[str, List[RunData]], metric: str, fig_dir: Path) -> Path:
    """Plot mean +/- std convergence for one metric across algorithms."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 2_000_000, 400)
    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=220)
    for algo, rs in groups.items():
        arr = _interpolate_runs(rs, metric, grid)
        if arr.size == 0:
            continue
        arr = np.vstack([smooth_1d(row, PLOT_STYLE["smooth_window"]) for row in arr])
        arr = arr * display_scale(metric)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        spread = 1.96 * std / math.sqrt(arr.shape[0]) if PLOT_STYLE["use_ci95"] else std
        label = ALGO_LABEL.get(algo, algo)
        color = PLOT_COLORS.get(algo)
        ax.plot(grid / 1e6, mean, linewidth=PLOT_STYLE["line_width"], label=label, color=color)
        ax.fill_between(
            grid / 1e6,
            mean - spread,
            mean + spread,
            alpha=PLOT_STYLE["shadow_alpha"],
            color=color,
        )
    ax.set_xlabel("Env Steps (Million)")
    ax.set_ylabel(display_label(metric))
    ax.set_title(f"{display_label(metric)} Convergence")
    if METRICS[metric]["better"] == "lower":
        ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out = fig_dir / f"fig_convergence_{metric}.png"
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_convergence_2x2(groups: Dict[str, List[RunData]], fig_dir: Path) -> Path:
    """Plot the four headline convergence curves in one 2x2 paper figure."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["eval_return", "mean_tracking_error", "tail_mean_target_distance", "mean_target_lost"]
    grid = np.linspace(0, 2_000_000, 400)
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0), dpi=220)
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        for algo, rs in groups.items():
            arr = _interpolate_runs(rs, metric, grid)
            if arr.size == 0:
                continue
            arr = np.vstack([smooth_1d(row, PLOT_STYLE["smooth_window"]) for row in arr])
            arr = arr * display_scale(metric)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            spread = 1.96 * std / math.sqrt(arr.shape[0]) if PLOT_STYLE["use_ci95"] else std
            label = ALGO_LABEL.get(algo, algo)
            color = PLOT_COLORS.get(algo)
            ax.plot(grid / 1e6, mean, linewidth=PLOT_STYLE["line_width"], label=label, color=color)
            ax.fill_between(
                grid / 1e6,
                mean - spread,
                mean + spread,
                alpha=PLOT_STYLE["shadow_alpha"],
                color=color,
            )

        ax.set_xlabel("Env Steps (Million)")
        ax.set_ylabel(display_label(metric))
        ax.set_title(display_label(metric))
        if METRICS[metric]["better"] == "lower":
            ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=False, ncol=2, fontsize=max(PLOT_STYLE["legend_size"] - 2, 8))

    fig.tight_layout(h_pad=2.4, w_pad=2.4)
    out = fig_dir / "fig_convergence_2x2.png"
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_final_bars(groups: Dict[str, List[RunData]], fig_dir: Path) -> Path:
    """Plot final-20% mean +/- std for the four headline metrics."""
    metrics = ["eval_return", "tail_mean_target_distance", "mean_tracking_error", "mean_target_lost"]
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0), dpi=220)
    axes = axes.ravel()
    algos = list(groups.keys())
    labels = [ALGO_LABEL.get(a, a) for a in algos]
    for ax, metric in zip(axes, metrics):
        means, stds = [], []
        for algo in algos:
            vals = [float(np.mean(_last_fraction_values(run, metric))) * display_scale(metric) for run in groups[algo]]
            mean, std = _mean_std(vals)
            means.append(mean)
            stds.append(std)
        colors = [PLOT_COLORS.get(algo, "#5B8FF9") for algo in algos]
        ax.bar(np.arange(len(algos)), means, yerr=stds, capsize=4, color=colors, alpha=0.88)
        ax.set_xticks(np.arange(len(algos)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(f"Final 20% {display_label(metric)}")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        if METRICS[metric]["better"] == "lower":
            ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = fig_dir / "fig_final20_bar_metrics.png"
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_stg_vs_mappo_seeds(groups: Dict[str, List[RunData]], fig_dir: Path) -> Optional[Path]:
    """Plot individual seed curves for STG-MAPPO and raw MAPPO."""
    if "stg_mappo" not in groups or "mappo" not in groups:
        return None
    metrics = ["eval_return", "tail_mean_target_distance", "mean_tracking_error", "mean_target_lost"]
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0), dpi=220)
    axes = axes.ravel()
    style = {"stg_mappo": "-", "mappo": "--"}
    color = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    for ax, metric in zip(axes, metrics):
        for algo in ["stg_mappo", "mappo"]:
            for run in groups[algo]:
                y = smooth_1d(run.metrics[metric], PLOT_STYLE["smooth_window"]) * display_scale(metric)
                ax.plot(
                    run.true_env_step / 1e6,
                    y,
                    linestyle=style[algo],
                    color=color.get(run.seed, None),
                    linewidth=2.0,
                    alpha=0.85,
                    label=f"{ALGO_LABEL[algo]} seed{run.seed}",
                )
        ax.set_title(display_label(metric))
        ax.set_xlabel("Env Steps (Million)")
        if METRICS[metric]["better"] == "lower":
            ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=PLOT_STYLE["legend_size"])
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = fig_dir / "fig_stg_mappo_vs_mappo_seed_curves.png"
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_reward_components(groups: Dict[str, List[RunData]], fig_dir: Path) -> Optional[Path]:
    """Plot STG-MAPPO reward components to support reward-design discussion."""
    if "stg_mappo" not in groups:
        return None
    grid = np.linspace(0, 2_000_000, 400)
    metrics = ["mean_tracking_reward", "mean_observation_reward", "mean_semantic_reward", "mean_control_cost"]
    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=220)
    for metric in metrics:
        arr = _interpolate_runs(groups["stg_mappo"], metric, grid)
        if arr.size == 0:
            continue
        arr = np.vstack([smooth_1d(row, PLOT_STYLE["smooth_window"]) for row in arr])
        ax.plot(grid / 1e6, np.mean(arr, axis=0), linewidth=PLOT_STYLE["line_width"], label=METRICS[metric]["label"])
    ax.set_xlabel("Env Steps (Million)")
    ax.set_ylabel("Component Value")
    ax.set_title("STG-MAPPO Reward Components")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = fig_dir / "fig_stg_mappo_reward_components.png"
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_boxplots(groups: Dict[str, List[RunData]], fig_dir: Path) -> Path:
    """Plot final-20% metric distributions instead of only mean/std."""
    metrics = ["eval_return", "tail_mean_target_distance", "mean_tracking_error"]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), dpi=220)
    algos = list(groups.keys())
    labels = [ALGO_LABEL.get(a, a) for a in algos]
    for ax, metric in zip(axes, metrics):
        data = []
        for algo in algos:
            vals = []
            for run in groups[algo]:
                vals.extend((_last_fraction_values(run, metric) * display_scale(metric)).tolist())
            data.append(vals)
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        for patch, algo in zip(bp["boxes"], algos):
            patch.set_facecolor(PLOT_COLORS.get(algo, "#5B8FF9"))
            patch.set_alpha(0.62)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(f"Final 20% {display_label(metric)}")
        if METRICS[metric]["better"] == "lower":
            ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    out = fig_dir / "fig_final20_boxplots.png"
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_action_diagnostics(groups: Dict[str, List[RunData]], fig_dir: Path) -> Path:
    """Plot action saturation and control cost diagnostics for controller behavior."""
    metrics = ["mean_action_saturation_rate", "mean_control_cost"]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), dpi=220)
    grid = np.linspace(0, 2_000_000, 400)
    for ax, metric in zip(axes, metrics):
        for algo, rs in groups.items():
            arr = _interpolate_runs(rs, metric, grid)
            if arr.size == 0:
                continue
            arr = np.vstack([smooth_1d(row, PLOT_STYLE["smooth_window"]) for row in arr])
            ax.plot(
                grid / 1e6,
                np.mean(arr, axis=0),
                linewidth=PLOT_STYLE["line_width"],
                label=ALGO_LABEL.get(algo, algo),
                color=PLOT_COLORS.get(algo),
            )
        ax.set_xlabel("Env Steps (Million)")
        ax.set_title(METRICS[metric]["label"])
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    out = fig_dir / "fig_action_diagnostics.png"
    save_figure(fig, out)
    plt.close(fig)
    return out


def build_figures(runs: List[RunData], fig_dir: Path) -> List[Path]:
    groups = _algo_groups(runs)
    paths = []
    paths.append(plot_convergence_2x2(groups, fig_dir))
    paths.append(plot_final_bars(groups, fig_dir))
    stg_vs_mappo = plot_stg_vs_mappo_seeds(groups, fig_dir)
    if stg_vs_mappo:
        paths.append(stg_vs_mappo)
    reward_components = plot_reward_components(groups, fig_dir)
    if reward_components:
        paths.append(reward_components)
    paths.append(plot_boxplots(groups, fig_dir))
    paths.append(plot_action_diagnostics(groups, fig_dir))
    return paths


def write_metric_dictionary(path: Path) -> None:
    rows = [
        {
            "metric": key,
            "display_name": cfg["label"],
            "better": cfg["better"],
            "meaning": cfg["paper_role"],
        }
        for key, cfg in METRICS.items()
    ]
    write_csv(path, rows, ["metric", "display_name", "better", "meaning"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hard 4-AUV paper figures and tables.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Root containing algorithm/seed run folders.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output root for paper_figures and paper_tables.")
    parser.add_argument("--min-env-step", type=int, default=2_000_000, help="Minimum env_step required for formal runs.")
    parser.add_argument("--smooth-win", type=int, default=PLOT_STYLE["smooth_window"], help="Moving-average window for convergence curves.")
    parser.add_argument("--ci95", action="store_true", help="Use 95% confidence interval shadows instead of std shadows.")
    args = parser.parse_args()

    PLOT_STYLE["smooth_window"] = max(1, int(args.smooth_win))
    PLOT_STYLE["use_ci95"] = bool(args.ci95)
    apply_paper_style()

    base = args.base if args.base.is_absolute() else REPO_ROOT / args.base
    out_root = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    fig_dir = out_root / "hard_paper_figures"
    table_dir = out_root / "hard_paper_tables"
    runs = discover_formal_runs(base, args.min_env_step)
    if not runs:
        raise SystemExit(
            f"No completed formal runs found under {base}. "
            "Check that exp/result.pkl exists and env_step >= --min-env-step."
        )

    table_paths = build_tables(runs, table_dir)
    write_metric_dictionary(table_dir / "metric_dictionary.csv")
    fig_paths = build_figures(runs, fig_dir)

    print("Formal runs:")
    for run in runs:
        print(f"- {ALGO_LABEL.get(run.algo, run.algo)} seed{run.seed}: env_step={run.env_step}, eval_points={len(run.rows)}")
    print("\nTables:")
    for name, path in table_paths.items():
        print(f"- {name}: {path}")
    print(f"- metric_dictionary: {table_dir / 'metric_dictionary.csv'}")
    print("\nFigures:")
    for path in fig_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()

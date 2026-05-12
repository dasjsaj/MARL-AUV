from __future__ import annotations

"""
Generate paper-ready ablation figures and tables for the medium 4-AUV setting.

The ablation protocol compares:
- MAPPO-raw-tau6: original raw observation/state and tau6 action baseline.
- MAPPO-velocity3-nonsemantic: only the low-dimensional velocity3 action abstraction.
- MAPPO-semantic-state-only: semantic/diagnostic observation features without semantic reward.
- STG-MAPPO-full: semantic state, semantic graph/diagnostics, semantic reward, and velocity3.

Outputs:
- ablation_paper_figures/*.png and *.pdf
- ablation_paper_tables/*.csv
- ablation_paper_tables/ablation_paper_tables.xlsx when pandas/openpyxl are available

Distances are plotted in km without multiplying by 100.
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
DEFAULT_TMC_ROOT = REPO_ROOT / "artifacts/auv6dof_tmc_2e6"
DEFAULT_MAIN_MEDIUM = DEFAULT_TMC_ROOT / "medium_4auv/auv6dof"
DEFAULT_ABLATION = DEFAULT_TMC_ROOT / "ablation_medium_4auv/auv6dof"
DEFAULT_OUT = DEFAULT_TMC_ROOT / "ablation_medium_4auv"


VARIANT_ORDER = [
    "mappo_raw_tau6",
    "mappo_velocity3_nonsemantic",
    "mappo_semantic_state_only",
    "stg_mappo_full",
]

VARIANT_LABEL = {
    "mappo_raw_tau6": "MAPPO-raw-tau6",
    "mappo_velocity3_nonsemantic": "MAPPO-velocity3",
    "mappo_semantic_state_only": "MAPPO-semantic-state",
    "stg_mappo_full": "STG-MAPPO-full",
}

PLOT_COLORS = {
    "mappo_raw_tau6": "#7f7f7f",
    "mappo_velocity3_nonsemantic": "#ff7f0e",
    "mappo_semantic_state_only": "#2ca02c",
    "stg_mappo_full": "#1f77b4",
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
    "smooth_window": 8,
    "use_ci95": False,
}

METRICS = {
    "eval_return": {
        "label": "Eval Return",
        "better": "higher",
        "paper_role": "Overall evaluation return and convergence quality.",
    },
    "mean_tracking_error": {
        "label": "Mean Tracking Error",
        "better": "lower",
        "paper_role": "Mean |AUV-target distance - desired tracking distance|.",
    },
    "tail_mean_target_distance": {
        "label": "Tail Mean Target Distance",
        "better": "lower",
        "paper_role": "Steady-state AUV-target distance in the final eval tail.",
    },
    "mean_target_lost": {
        "label": "Target Lost Rate",
        "better": "lower",
        "paper_role": "Target loss frequency during evaluation.",
    },
    "mean_action_saturation_rate": {
        "label": "Action Saturation Rate",
        "better": "lower",
        "paper_role": "Fraction of actions close to clipping/saturation.",
    },
    "mean_control_cost": {
        "label": "Control Cost",
        "better": "lower",
        "paper_role": "Action magnitude/change penalty for control smoothness.",
    },
    "mean_tracking_reward": {
        "label": "Tracking Reward",
        "better": "higher",
        "paper_role": "Reward component associated with useful tracking behavior.",
    },
    "mean_semantic_reward": {
        "label": "Semantic Reward",
        "better": "higher",
        "paper_role": "Semantic phase/task-graph shaping reward.",
    },
}

CONVERGENCE_METRICS = [
    "eval_return",
    "mean_tracking_error",
    "tail_mean_target_distance",
    "mean_target_lost",
]

FINAL_BAR_METRICS = [
    "eval_return",
    "tail_mean_target_distance",
    "mean_target_lost",
    "mean_action_saturation_rate",
]


def apply_paper_style() -> None:
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


def display_label(metric: str) -> str:
    label = METRICS.get(metric, {}).get("label", metric)
    if metric in {"mean_tracking_error", "tail_mean_target_distance"}:
        return f"{label} (km)"
    return label


def smooth_1d(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if window <= 1 or y.size <= 2:
        return y
    window = min(int(window), int(y.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    z = np.ones_like(y, dtype=np.float64)
    return np.convolve(y, kernel, mode="same") / np.convolve(z, kernel, mode="same")


def save_figure(fig: plt.Figure, path_png: Path) -> Path:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, bbox_inches="tight", dpi=500)
    fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    return path_png


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


def _read_result(path: Path) -> Tuple[int, int, str]:
    if not path.exists():
        return 0, 0, ""
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        env_step = int(obj.get("env_step", 0) or 0)
        train_iter = int(obj.get("train_iter", obj.get("learner_step", 0)) or 0)
        finish_time = str(obj.get("finish_time", ""))
    else:
        env_step = int(getattr(obj, "env_step", 0) or 0)
        train_iter = int(getattr(obj, "train_iter", 0) or 0)
        finish_time = str(getattr(obj, "finish_time", ""))
    return env_step, train_iter, finish_time


def _find_eval_detail(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "eval_detail.csv"
    if direct.exists():
        return direct
    candidates = sorted(run_dir.glob("*eval_detail*.csv"))
    if candidates:
        return candidates[0]
    return None


@dataclass
class RunData:
    variant: str
    label: str
    seed: int
    run_dir: Path
    env_step: int
    train_iter: int
    finish_time: str
    rows: List[Dict[str, str]]
    true_env_step: np.ndarray
    metrics: Dict[str, np.ndarray]


def parse_seed(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.split("_", 1)[1])
            except Exception:
                return None
    return None


def load_run(variant: str, run_dir: Path, min_env_step: int) -> Optional[RunData]:
    seed = parse_seed(run_dir)
    if seed is None:
        return None
    result_path = run_dir / "exp/result.pkl"
    env_step, train_iter, finish_time = _read_result(result_path)
    if env_step < min_env_step:
        return None
    eval_path = _find_eval_detail(run_dir)
    if eval_path is None:
        return None
    rows = _read_csv_rows(eval_path)
    if not rows:
        return None

    # Map each eval event to the completed true env-step budget. DI-engine's train_step
    # field is not comparable across on-policy/off-policy algorithms.
    n = len(rows)
    true_env_step = np.linspace(float(env_step) / float(n), float(env_step), n)
    metrics: Dict[str, np.ndarray] = {}
    for metric in METRICS:
        metrics[metric] = np.asarray([_float(r.get(metric)) for r in rows], dtype=np.float64)

    return RunData(
        variant=variant,
        label=VARIANT_LABEL[variant],
        seed=seed,
        run_dir=run_dir,
        env_step=env_step,
        train_iter=train_iter,
        finish_time=finish_time,
        rows=rows,
        true_env_step=true_env_step,
        metrics=metrics,
    )


def discover_completed_runs(
    base: Path,
    algo: str,
    run_name_contains: str,
    variant: str,
    min_env_step: int,
) -> List[RunData]:
    runs_by_seed: Dict[int, RunData] = {}
    algo_root = base / algo
    for result_path in sorted(algo_root.glob("seed_*/**/exp/result.pkl")):
        run_dir = result_path.parents[1]
        if run_name_contains and run_name_contains not in run_dir.name:
            continue
        run = load_run(variant, run_dir, min_env_step=min_env_step)
        if run is None:
            continue
        old = runs_by_seed.get(run.seed)
        if old is None or (run.env_step, run.run_dir.stat().st_mtime) > (
            old.env_step,
            old.run_dir.stat().st_mtime,
        ):
            runs_by_seed[run.seed] = run
    return [runs_by_seed[s] for s in sorted(runs_by_seed)]


def collect_all_runs(main_medium: Path, ablation: Path, min_env_step: int) -> Dict[str, List[RunData]]:
    out: Dict[str, List[RunData]] = {
        "mappo_raw_tau6": discover_completed_runs(
            main_medium, "mappo", "tmc_medium_2e6", "mappo_raw_tau6", min_env_step
        ),
        "mappo_velocity3_nonsemantic": discover_completed_runs(
            ablation,
            "mappo",
            "tmc_ablation_mappo_velocity3_nonsemantic",
            "mappo_velocity3_nonsemantic",
            min_env_step,
        ),
        "mappo_semantic_state_only": discover_completed_runs(
            ablation,
            "mappo",
            "tmc_ablation_mappo_semantic_state_only",
            "mappo_semantic_state_only",
            min_env_step,
        ),
        "stg_mappo_full": discover_completed_runs(
            ablation, "stg_mappo", "tmc_ablation_stg_mappo_full", "stg_mappo_full", min_env_step
        ),
    }

    # If the dedicated STG ablation run has not been launched, use the completed
    # medium main STG-MAPPO runs as the full method under the same scenario.
    if not out["stg_mappo_full"]:
        out["stg_mappo_full"] = discover_completed_runs(
            main_medium, "stg_mappo", "tmc_medium_2e6", "stg_mappo_full", min_env_step
        )
    return out


def interpolate_run_metric(run: RunData, metric: str, grid: np.ndarray) -> np.ndarray:
    y = run.metrics.get(metric)
    if y is None or y.size == 0:
        return np.full_like(grid, math.nan, dtype=np.float64)
    finite = np.isfinite(y)
    if finite.sum() == 0:
        return np.full_like(grid, math.nan, dtype=np.float64)
    x = run.true_env_step[finite]
    yy = y[finite]
    if x.size == 1:
        return np.full_like(grid, yy[0], dtype=np.float64)
    return np.interp(grid, x, yy, left=yy[0], right=yy[-1])


def aggregate_on_grid(
    runs_by_variant: Dict[str, List[RunData]],
    metric: str,
    grid: np.ndarray,
    smooth_window: int,
    use_ci95: bool,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for variant in VARIANT_ORDER:
        runs = runs_by_variant.get(variant, [])
        if not runs:
            continue
        curves = np.vstack([interpolate_run_metric(r, metric, grid) for r in runs])
        curves = np.vstack([smooth_1d(c, smooth_window) for c in curves])
        mean = np.nanmean(curves, axis=0)
        std = np.nanstd(curves, axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)
        spread = 1.96 * std / math.sqrt(curves.shape[0]) if use_ci95 else std
        out[variant] = (mean, spread)
    return out


def last_fraction_mean(run: RunData, metric: str, fraction: float) -> float:
    y = run.metrics.get(metric)
    if y is None or y.size == 0:
        return math.nan
    n = max(1, int(math.ceil(y.size * fraction)))
    return float(np.nanmean(y[-n:]))


def build_tables(runs_by_variant: Dict[str, List[RunData]], out_dir: Path, last_fraction: float) -> None:
    table_dir = out_dir / "ablation_paper_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for variant in VARIANT_ORDER:
        runs = runs_by_variant.get(variant, [])
        for run in runs:
            row: Dict[str, object] = {
                "variant": variant,
                "label": run.label,
                "seed": run.seed,
                "env_step": run.env_step,
                "num_eval_points": len(run.rows),
                "run_dir": str(run.run_dir),
            }
            for metric in METRICS:
                latest = float(run.metrics[metric][-1]) if run.metrics[metric].size else math.nan
                row[f"latest_{metric}"] = latest
                row[f"last20_{metric}"] = last_fraction_mean(run, metric, last_fraction)
            per_seed_rows.append(row)

        summary: Dict[str, object] = {
            "variant": variant,
            "label": VARIANT_LABEL[variant],
            "num_seeds": len(runs),
            "seeds": ",".join(str(r.seed) for r in runs),
        }
        for metric in FINAL_BAR_METRICS + ["mean_tracking_reward", "mean_semantic_reward", "mean_control_cost"]:
            vals = np.asarray([last_fraction_mean(r, metric, last_fraction) for r in runs], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            summary[f"last20_{metric}_mean"] = float(np.mean(vals)) if vals.size else math.nan
            summary[f"last20_{metric}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        summary_rows.append(summary)

    def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        keys = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(table_dir / "table_ablation_per_seed.csv", per_seed_rows)
    write_csv(table_dir / "table_ablation_last20_summary.csv", summary_rows)

    try:
        import pandas as pd

        with pd.ExcelWriter(table_dir / "ablation_paper_tables.xlsx") as writer:
            pd.DataFrame(per_seed_rows).to_excel(writer, sheet_name="per_seed", index=False)
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="last20_summary", index=False)
    except Exception as exc:
        print(f"[Warn] Excel export skipped: {exc}")


def plot_convergence_2x2(
    runs_by_variant: Dict[str, List[RunData]],
    out_dir: Path,
    max_env_step: int,
    eval_points: int,
    smooth_window: int,
    use_ci95: bool,
) -> None:
    fig_dir = out_dir / "ablation_paper_figures"
    grid = np.linspace(0.0, float(max_env_step), eval_points)
    x = grid / 1_000_000.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, metric in zip(axes.ravel(), CONVERGENCE_METRICS):
        curves = aggregate_on_grid(runs_by_variant, metric, grid, smooth_window, use_ci95)
        for variant in VARIANT_ORDER:
            if variant not in curves:
                continue
            mean, spread = curves[variant]
            color = PLOT_COLORS[variant]
            ax.plot(
                x,
                mean,
                label=VARIANT_LABEL[variant],
                color=color,
                linewidth=PLOT_STYLE["line_width"],
            )
            ax.fill_between(
                x,
                mean - spread,
                mean + spread,
                color=color,
                alpha=PLOT_STYLE["shadow_alpha"],
                linewidth=0.0,
            )
        ax.set_xlabel("Env Steps (Million)")
        ax.set_ylabel(display_label(metric))
        ax.grid(True, linestyle="--", alpha=0.55)
        ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_ablation_convergence_2x2.png")
    plt.close(fig)


def plot_final20_bars(
    runs_by_variant: Dict[str, List[RunData]],
    out_dir: Path,
    last_fraction: float,
) -> None:
    fig_dir = out_dir / "ablation_paper_figures"
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    variants = [v for v in VARIANT_ORDER if runs_by_variant.get(v)]
    labels = [VARIANT_LABEL[v] for v in variants]
    x = np.arange(len(variants))

    for ax, metric in zip(axes.ravel(), FINAL_BAR_METRICS):
        means: List[float] = []
        stds: List[float] = []
        colors: List[str] = []
        for variant in variants:
            vals = np.asarray(
                [last_fraction_mean(r, metric, last_fraction) for r in runs_by_variant[variant]],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if vals.size else math.nan)
            stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            colors.append(PLOT_COLORS[variant])
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.86, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(display_label(metric))
        ax.grid(True, axis="y", linestyle="--", alpha=0.55)
    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_ablation_final20_bar_metrics.png")
    plt.close(fig)


def write_run_manifest(runs_by_variant: Dict[str, List[RunData]], out_dir: Path, min_env_step: int) -> None:
    table_dir = out_dir / "ablation_paper_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "ablation_run_manifest.csv"
    rows: List[Dict[str, object]] = []
    for variant in VARIANT_ORDER:
        runs = runs_by_variant.get(variant, [])
        for run in runs:
            rows.append(
                {
                    "variant": variant,
                    "label": run.label,
                    "seed": run.seed,
                    "env_step": run.env_step,
                    "num_eval_points": len(run.rows),
                    "complete_by_min_env_step": run.env_step >= min_env_step,
                    "run_dir": str(run.run_dir),
                }
            )
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def print_status(runs_by_variant: Dict[str, List[RunData]], min_env_step: int) -> None:
    print("\n[Ablation runs]")
    for variant in VARIANT_ORDER:
        runs = runs_by_variant.get(variant, [])
        if not runs:
            print(f"- {VARIANT_LABEL[variant]}: no completed runs found")
            continue
        seeds = ", ".join(f"{r.seed}(eval={len(r.rows)}, step={r.env_step})" for r in runs)
        print(f"- {VARIANT_LABEL[variant]}: {seeds}")
        if len(runs) < 3:
            print(f"  [Warn] fewer than 3 complete seeds at min_env_step={min_env_step}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-medium", type=Path, default=DEFAULT_MAIN_MEDIUM)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-env-step", type=int, default=2_000_000)
    parser.add_argument("--max-env-step", type=int, default=2_000_000)
    parser.add_argument("--eval-points", type=int, default=400)
    parser.add_argument("--smooth-win", type=int, default=PLOT_STYLE["smooth_window"])
    parser.add_argument("--ci95", action="store_true")
    parser.add_argument("--last-fraction", type=float, default=0.20)
    args = parser.parse_args()

    apply_paper_style()
    runs_by_variant = collect_all_runs(
        main_medium=args.main_medium,
        ablation=args.ablation,
        min_env_step=args.min_env_step,
    )
    print_status(runs_by_variant, min_env_step=args.min_env_step)

    args.out.mkdir(parents=True, exist_ok=True)
    write_run_manifest(runs_by_variant, args.out, min_env_step=args.min_env_step)
    build_tables(runs_by_variant, args.out, last_fraction=args.last_fraction)
    plot_convergence_2x2(
        runs_by_variant,
        args.out,
        max_env_step=args.max_env_step,
        eval_points=args.eval_points,
        smooth_window=args.smooth_win,
        use_ci95=args.ci95 or bool(PLOT_STYLE["use_ci95"]),
    )
    plot_final20_bars(runs_by_variant, args.out, last_fraction=args.last_fraction)

    print("\n[Done]")
    print(f"Figures: {args.out / 'ablation_paper_figures'}")
    print(f"Tables : {args.out / 'ablation_paper_tables'}")


if __name__ == "__main__":
    main()

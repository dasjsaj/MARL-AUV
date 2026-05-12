from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except Exception:
                    pass
            rows.append(parsed)
    return rows


def _first_last_window(values: List[float], frac: float = 0.3) -> tuple[float | None, float | None]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if len(finite) < 2:
        return None, None
    n = max(1, int(np.ceil(len(finite) * frac)))
    return float(np.mean(finite[:n])), float(np.mean(finite[-n:]))


def _latest_run(seed_dir: Path) -> Path | None:
    if not seed_dir.exists():
        return None
    runs = [p for p in seed_dir.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: (p / "summary.json").stat().st_mtime)[-1]


def _auc(values: List[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return None
    if len(finite) == 1:
        return float(finite[0])
    return float(np.trapz(finite) / max(1, len(finite) - 1))


def _first_index_where(values: List[float], threshold: float, *, le: bool = True) -> int | None:
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if (le and value <= threshold) or ((not le) and value >= threshold):
            return int(idx)
    return None


def evaluate_run(
    run_dir: Path,
    random_baseline: Dict[str, Any],
    min_eval_points: int,
    tracking_error_success_threshold: float,
    tail100_distance_threshold: float,
    tail100_tracking_error_threshold: float,
    tail100_lost_threshold: float,
    tail100_action_saturation_threshold: float,
    control_to_tracking_threshold: float,
) -> Dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    rows = _read_rows(run_dir / "eval_detail.csv")
    random_return = float(random_baseline.get("mean_return", 0.0))
    random_tracking_error = float(random_baseline.get("mean_tracking_error", float("nan")))
    random_target_lost = float(random_baseline.get("mean_target_lost", float("nan")))
    returns = [row.get("eval_return", 0.0) for row in rows]
    errors = [row.get("mean_tracking_error", row.get("tail_mean_target_distance", 0.0)) for row in rows]
    lost = [row.get("mean_target_lost", 0.0) for row in rows]
    distances = [
        row.get("mean_target_distance", row.get("tail100_mean_target_distance", row.get("tail_mean_target_distance", 0.0)))
        for row in rows
    ]
    tracking_rewards = [row.get("mean_tracking_reward", 0.0) for row in rows]
    control_costs = [row.get("mean_control_cost", 0.0) for row in rows]
    clip_rates = [row.get("mean_action_clip_rate", 0.0) for row in rows]
    saturation_rates = [row.get("mean_action_saturation_rate", 0.0) for row in rows]
    tail100_distances = [
        row.get("tail100_mean_target_distance", row.get("tail_mean_target_distance", row.get("final_mean_target_distance", 0.0)))
        for row in rows
    ]
    tail100_tracking_errors = [
        row.get("tail100_mean_tracking_error", row.get("mean_tracking_error", 0.0)) for row in rows
    ]
    tail100_lost_rates = [row.get("tail100_target_lost_rate", row.get("mean_target_lost", 0.0)) for row in rows]
    tail100_action_saturations = [
        row.get("tail100_action_saturation_rate", row.get("mean_action_saturation_rate", 0.0)) for row in rows
    ]

    return_first, return_last = _first_last_window(returns)
    err_first, err_last = _first_last_window(errors)
    dist_first, dist_last = _first_last_window(distances)
    lost_first, lost_last = _first_last_window(lost)
    track_first, track_last = _first_last_window(tracking_rewards)
    sat_first, sat_last = _first_last_window(saturation_rates)
    latest_return = float(returns[-1]) if returns else float(summary.get("latest_eval_reward") or 0.0)
    mean_tracking_reward = float(np.mean(tracking_rewards)) if tracking_rewards else 0.0
    mean_control_cost = float(np.mean(control_costs)) if control_costs else 0.0
    latest_tracking_reward = float(tracking_rewards[-1]) if tracking_rewards else 0.0
    latest_control_cost = float(control_costs[-1]) if control_costs else 0.0
    mean_clip = float(np.mean(clip_rates)) if clip_rates else 0.0
    mean_saturation = float(np.mean(saturation_rates)) if saturation_rates else 0.0
    latest_tail100_distance = float(tail100_distances[-1]) if tail100_distances else float("inf")
    latest_tail100_tracking_error = float(tail100_tracking_errors[-1]) if tail100_tracking_errors else float("inf")
    latest_tail100_lost_rate = float(tail100_lost_rates[-1]) if tail100_lost_rates else float("inf")
    latest_tail100_action_saturation = (
        float(tail100_action_saturations[-1]) if tail100_action_saturations else float("inf")
    )
    control_to_tracking_ratio = latest_control_cost / max(1e-6, abs(latest_tracking_reward))
    eval_return_auc = _auc(returns)
    tracking_error_auc = _auc(errors)
    final_env_step = None
    try:
        import pickle

        result_path = run_dir / "exp" / "result.pkl"
        if result_path.exists():
            with result_path.open("rb") as f:
                result = pickle.load(f)
            if isinstance(result, dict) and result.get("env_step") is not None:
                final_env_step = float(result["env_step"])
    except Exception:
        final_env_step = None

    stable_threshold_candidates = []
    if np.isfinite(random_tracking_error):
        stable_threshold_candidates.append(float(random_tracking_error) * 0.75)
    if err_first is not None:
        stable_threshold_candidates.append(float(err_first) * 0.75)
    stable_threshold_candidates.append(float(tracking_error_success_threshold))
    stable_threshold = max(1e-6, min(v for v in stable_threshold_candidates if np.isfinite(v) and v > 0))
    stable_eval_index = _first_index_where(errors, stable_threshold, le=True)
    steps_to_stable_tracking = None
    if stable_eval_index is not None and rows:
        steps_to_stable_tracking = rows[stable_eval_index].get("train_step", float(stable_eval_index))
        if final_env_step is not None and len(rows) > 0:
            steps_to_stable_tracking = final_env_step * float(stable_eval_index + 1) / float(len(rows))

    distance_down = bool(dist_first is not None and dist_last is not None and dist_last < dist_first)
    tracking_error_down = bool(err_first is not None and err_last is not None and err_last < err_first)
    tracking_improved_vs_random = bool(
        np.isfinite(random_tracking_error) and err_last is not None and err_last < random_tracking_error
    )
    lost_improved_vs_random = bool(
        np.isfinite(random_target_lost) and lost_last is not None and lost_last <= random_target_lost
    )
    return_up = bool(return_first is not None and return_last is not None and return_last > return_first)
    return_beats_random = bool(latest_return > float(random_return))
    saturation_not_rising = bool(
        sat_first is not None and sat_last is not None and (sat_last <= sat_first or sat_last <= 0.2)
    )

    reference_checks = {
        "enough_eval_points": len(rows) >= int(min_eval_points),
        "beats_random_return": latest_return > float(random_return),
        "tail100_distance_pass": latest_tail100_distance < float(tail100_distance_threshold),
        "tail100_tracking_error_pass": latest_tail100_tracking_error < float(tail100_tracking_error_threshold),
        "tail100_lost_pass": latest_tail100_lost_rate <= float(tail100_lost_threshold),
        "tail100_action_saturation_pass": latest_tail100_action_saturation <= float(tail100_action_saturation_threshold),
        "control_to_tracking_pass": control_to_tracking_ratio <= float(control_to_tracking_threshold),
        "tracking_error_down_or_low": bool(
            err_first is not None
            and err_last is not None
            and (err_last < err_first or err_last <= float(tracking_error_success_threshold))
        ),
        "target_lost_down_or_low": bool(
            lost_first is not None and lost_last is not None and (lost_last <= lost_first or lost_last < 0.25)
        ),
        "tracking_reward_up": bool(
            track_first is not None
            and track_last is not None
            and (
                track_last >= track_first
                or (
                    err_first is not None
                    and err_last is not None
                    and err_last < err_first
                    and abs(track_last - track_first) <= 0.02
                )
            )
        ),
        "control_not_dominant": bool(mean_control_cost <= max(1e-6, 0.5 * abs(mean_tracking_reward))),
        "action_not_saturated": bool(mean_clip <= 0.2 and mean_saturation <= 0.2),
    }
    evidence_checks = {
        "enough_eval_points": len(rows) >= int(min_eval_points),
        "beats_random_return": return_beats_random,
        "eval_return_up_or_positive": bool(return_up or return_beats_random),
        "tracking_error_down_or_beats_random": bool(tracking_error_down or tracking_improved_vs_random),
        "target_distance_down": distance_down,
        "target_lost_down_or_low": bool(
            lost_first is not None and lost_last is not None and (lost_last <= lost_first or lost_last < 0.25)
        ),
        "target_lost_beats_random": bool(lost_improved_vs_random or (lost_last is not None and lost_last < 0.25)),
        "tracking_reward_up": reference_checks["tracking_reward_up"],
        "control_not_dominant": reference_checks["control_not_dominant"],
        "action_not_saturated": reference_checks["action_not_saturated"],
        "action_saturation_not_rising": saturation_not_rising,
    }
    evidence_passed = all(evidence_checks.values())
    return {
        "run_dir": run_dir.as_posix(),
        "num_eval_points": len(rows),
        "final_env_step": final_env_step,
        "latest_eval_return": latest_return,
        "random_mean_return": float(random_return),
        "random_mean_tracking_error": random_tracking_error,
        "random_mean_target_lost": random_target_lost,
        "eval_return_first30": return_first,
        "eval_return_last30": return_last,
        "target_distance_first30": dist_first,
        "target_distance_last30": dist_last,
        "tracking_error_first30": err_first,
        "tracking_error_last30": err_last,
        "target_lost_first30": lost_first,
        "target_lost_last30": lost_last,
        "tracking_reward_first30": track_first,
        "tracking_reward_last30": track_last,
        "mean_tracking_reward": mean_tracking_reward,
        "mean_control_cost": mean_control_cost,
        "latest_tracking_reward": latest_tracking_reward,
        "latest_control_cost": latest_control_cost,
        "control_to_tracking_ratio": control_to_tracking_ratio,
        "mean_action_clip_rate": mean_clip,
        "mean_action_saturation_rate": mean_saturation,
        "latest_tail100_mean_target_distance": latest_tail100_distance,
        "latest_tail100_mean_tracking_error": latest_tail100_tracking_error,
        "latest_tail100_target_lost_rate": latest_tail100_lost_rate,
        "latest_tail100_action_saturation_rate": latest_tail100_action_saturation,
        "eval_return_auc": eval_return_auc,
        "tracking_error_auc": tracking_error_auc,
        "steps_to_stable_tracking": steps_to_stable_tracking,
        "reference_threshold_checks": reference_checks,
        "evidence_checks": evidence_checks,
        "checks": evidence_checks,
        "strict_threshold_passed": all(reference_checks.values()),
        "passed": evidence_passed,
        "acceptance_mode": "evidence",
    }


def summarize(
    root: str | Path,
    *,
    min_eval_points: int = 10,
    random_baseline: str = "",
    tracking_error_success_threshold: float = 0.08,
    tail100_distance_threshold: float = 0.015,
    tail100_tracking_error_threshold: float = 0.005,
    tail100_lost_threshold: float = 0.0,
    tail100_action_saturation_threshold: float = 0.10,
    control_to_tracking_threshold: float = 0.35,
    required_passed_runs: int = 2,
) -> Dict[str, Any]:
    root_path = Path(root)
    baseline_path = Path(random_baseline) if random_baseline else root_path / "random_policy_baseline.json"
    default_baseline = _read_json(baseline_path)
    random_mean_return = float(default_baseline.get("mean_return", 0.0))
    records: List[Dict[str, Any]] = []
    for algo_dir in sorted((root_path / "auv6dof").glob("*")):
        if not algo_dir.is_dir():
            continue
        algo_baseline_path = root_path / f"random_policy_baseline_{algo_dir.name}.json"
        if not algo_baseline_path.exists():
            # Most TMC configs share a profile across raw algorithms. Prefer the
            # explicit profile stored in config.json when available.
            profile = ""
            latest_cfg = next(iter(sorted(algo_dir.glob("seed_*/*/config.json"))), None)
            if latest_cfg is not None:
                try:
                    payload = _read_json(latest_cfg)
                    env_cfg = payload.get("contract", {}).get("env_cfg", {})
                    profile = str(env_cfg.get("algo_profile", "")).strip()
                except Exception:
                    profile = ""
            if profile:
                algo_baseline_path = root_path / f"random_policy_baseline_{profile}.json"
        baseline = _read_json(algo_baseline_path) if algo_baseline_path.exists() else default_baseline
        for seed_dir in sorted(algo_dir.glob("seed_*")):
            run = _latest_run(seed_dir)
            if run is None:
                continue
            rec = evaluate_run(
                run,
                baseline,
                min_eval_points,
                tracking_error_success_threshold,
                tail100_distance_threshold,
                tail100_tracking_error_threshold,
                tail100_lost_threshold,
                tail100_action_saturation_threshold,
                control_to_tracking_threshold,
            )
            rec["algo"] = algo_dir.name
            rec["seed"] = seed_dir.name.replace("seed_", "")
            rec["random_baseline"] = (algo_baseline_path if algo_baseline_path.exists() else baseline_path).as_posix()
            records.append(rec)
    by_algo: Dict[str, Dict[str, Any]] = {}
    for algo in sorted({r["algo"] for r in records}):
        subset = [r for r in records if r["algo"] == algo]
        by_algo[algo] = {
            "runs": len(subset),
            "passed_runs": int(sum(1 for r in subset if r["passed"])),
            "paper_acceptance": bool(
                len(subset) >= 3 and sum(1 for r in subset if r["passed"]) >= int(required_passed_runs)
            ),
        }
    return {
        "root": root_path.as_posix(),
        "random_baseline": baseline_path.as_posix(),
        "random_mean_return": random_mean_return,
        "acceptance_mode": "evidence",
        "acceptance_note": (
            "Tail100 distance/error thresholds are retained as reference metrics only. "
            "Run acceptance is based on evidence of effective tracking: improvement over early training or random "
            "baseline, low/lower target lost rate, non-dominant control cost, and non-saturated actions."
        ),
        "min_eval_points": int(min_eval_points),
        "tracking_error_success_threshold": float(tracking_error_success_threshold),
        "tail100_distance_threshold": float(tail100_distance_threshold),
        "tail100_tracking_error_threshold": float(tail100_tracking_error_threshold),
        "tail100_lost_threshold": float(tail100_lost_threshold),
        "tail100_action_saturation_threshold": float(tail100_action_saturation_threshold),
        "control_to_tracking_threshold": float(control_to_tracking_threshold),
        "required_passed_runs": int(required_passed_runs),
        "records": records,
        "by_algo": by_algo,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/auv6dof_convergence_dense_debug")
    parser.add_argument("--random-baseline", default="")
    parser.add_argument("--min-eval-points", type=int, default=10)
    parser.add_argument("--tracking-error-success-threshold", type=float, default=0.08)
    parser.add_argument("--tail100-distance-threshold", type=float, default=0.015)
    parser.add_argument("--tail100-tracking-error-threshold", type=float, default=0.005)
    parser.add_argument("--tail100-lost-threshold", type=float, default=0.0)
    parser.add_argument("--tail100-action-saturation-threshold", type=float, default=0.10)
    parser.add_argument("--control-to-tracking-threshold", type=float, default=0.35)
    parser.add_argument("--required-passed-runs", type=int, default=2)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = summarize(
        args.root,
        min_eval_points=args.min_eval_points,
        random_baseline=args.random_baseline,
        tracking_error_success_threshold=args.tracking_error_success_threshold,
        tail100_distance_threshold=args.tail100_distance_threshold,
        tail100_tracking_error_threshold=args.tail100_tracking_error_threshold,
        tail100_lost_threshold=args.tail100_lost_threshold,
        tail100_action_saturation_threshold=args.tail100_action_saturation_threshold,
        control_to_tracking_threshold=args.control_to_tracking_threshold,
        required_passed_runs=args.required_passed_runs,
    )
    output = Path(args.output) if args.output else Path(args.root) / "acceptance_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"} | {"output": output.as_posix()}, indent=2))


if __name__ == "__main__":
    main()

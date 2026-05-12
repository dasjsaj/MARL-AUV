from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_orchestrator import run_experiment  # noqa: E402
from scripts._convergence_config import (  # noqa: E402
    algo_env_overrides,
    algo_profile_name,
    env_cfg_from_config,
    load_convergence_config,
)
from scripts.evaluate_random_policy import run_episode as run_random_episode  # noqa: E402
from scripts.evaluate_trained_policy import evaluate_trained_policy  # noqa: E402
from scripts.plot_convergence_debug import (  # noqa: E402
    _plot_metric,
    _read_csv,
    add_true_env_step,
    write_grouped_true_env_step_csv,
)
from scripts.summarize_convergence_acceptance import summarize  # noqa: E402


ALGO_DISPLAY = {
    "stg_mappo": "STG-MAPPO",
    "maddpg": "MADDPG",
    "matd3": "MATD3",
    "mappo": "MAPPO",
    "happo": "HAPPO",
    "atoc": "ATOC",
    "madqn": "MADQN",
    "masac": "MASAC",
}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_csv_list(raw: str, default: Iterable[str]) -> List[str]:
    if not raw.strip():
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str, default: Iterable[int]) -> List[int]:
    if not raw.strip():
        return list(default)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _random_baseline(
    cfg: Dict[str, Any],
    episodes: int,
    seed_base: int,
    *,
    algo: str | None = None,
    output_name: str = "random_policy_baseline",
) -> Dict[str, Any]:
    env_cfg = env_cfg_from_config(cfg, algo=algo)
    rows = [run_random_episode(env_cfg, seed_base + i) for i in range(int(episodes))]
    summary = {
        "episodes": int(episodes),
        "algo": str(algo or ""),
        "algo_profile": algo_profile_name(cfg, algo),
        "semantic_enabled": bool(
            env_cfg.get("obs", {}).get("include_semantic_features", False)
            or env_cfg.get("obs", {}).get("include_semantic_graph_features", False)
        ),
        "action_mode": str(env_cfg.get("action_control_mode", "tau6")),
        "mean_return": float(np.mean([row["episode_return"] for row in rows])),
        "std_return": float(np.std([row["episode_return"] for row in rows])),
        "mean_tracking_error": float(np.mean([row.get("tracking_error", 0.0) for row in rows])),
        "mean_target_lost": float(np.mean([row.get("target_lost", 0.0) for row in rows])),
        "rows": rows,
    }
    output = Path(cfg.get("output_root", "artifacts")) / f"{output_name}.json"
    _write_json(output, summary)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {k: v for k, v in summary.items() if k != "rows"} | {"output": output.as_posix()}


def _random_baselines_for_algos(
    cfg: Dict[str, Any], algos: List[str], episodes: int, seed_base: int
) -> Dict[str, Dict[str, Any]]:
    """Create one random baseline per distinct env/action/semantic profile."""
    outputs: Dict[str, Dict[str, Any]] = {}
    seen_profiles: Dict[str, str] = {}
    for idx, algo in enumerate(algos):
        profile = algo_profile_name(cfg, algo)
        if profile in seen_profiles:
            outputs[algo] = outputs[seen_profiles[profile]]
            continue
        name = f"random_policy_baseline_{profile}"
        outputs[algo] = _random_baseline(
            cfg,
            episodes,
            seed_base + 100 * idx,
            algo=algo,
            output_name=name,
        )
        seen_profiles[profile] = algo

    # Backward-compatible default baseline for summarizers that accept one file.
    if algos:
        first = outputs[algos[0]]
        default_path = Path(cfg.get("output_root", "artifacts")) / "random_policy_baseline.json"
        source = Path(first["output"])
        if source.exists():
            default_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return outputs


def _plot_run(run_dir: Path) -> Dict[str, str]:
    out_dir = run_dir / "plots"
    learning = _read_csv(run_dir / "learning_curve.csv")
    eval_rows = add_true_env_step(_read_csv(run_dir / "eval_detail.csv"), run_dir)
    eval_curve = _read_csv(run_dir / "eval_curve.csv")
    write_grouped_true_env_step_csv(eval_rows, out_dir / "eval_detail_grouped_true_env_step.csv")
    outputs: Dict[str, str] = {}
    specs = [
        (learning, "episode", "reward", "episode_return.png", "Episode return"),
        (eval_curve, "episode", "reward", "eval_return.png", "Eval return"),
    ]
    for key in (
        "mean_tracking_error",
        "tail100_mean_target_distance",
        "tail100_mean_tracking_error",
        "tail100_target_lost_rate",
        "mean_target_lost",
        "mean_tracking_reward",
        "mean_observation_reward",
        "mean_control_cost",
        "mean_action_delta_norm",
    ):
        specs.append((eval_rows, "train_step", key, f"{key}.png", key))
    for rows, x_key, y_key, filename, title in specs:
        if rows is eval_rows:
            x_key = "true_env_step"
        out = out_dir / filename
        _plot_metric(rows, x_key, y_key, out, title)
        if out.exists():
            outputs[y_key] = out.as_posix()
    return outputs


def _run_one(
    cfg: Dict[str, Any],
    algo: str,
    seed: int,
    max_env_step: int,
    run_tag_suffix: str,
) -> Dict[str, Any]:
    policy_overrides: Dict[str, Any] = dict(cfg.get("policy_overrides", {}).get(algo, {}))
    env_overrides = algo_env_overrides(cfg, algo)
    env_overrides["algo_profile"] = algo_profile_name(cfg, algo)
    env_overrides["semantic_enabled"] = bool(
        env_overrides.get("obs", {}).get("include_semantic_features", False)
        or env_overrides.get("obs", {}).get("include_semantic_graph_features", False)
    )
    env_overrides["action_mode_for_report"] = str(env_overrides.get("action_control_mode", "tau6"))
    run_dir = run_experiment(
        env_name=str(cfg.get("env", "auv6dof")),
        algo_name=algo,
        seed=int(seed),
        max_env_step=int(max_env_step),
        n_agent=int(cfg.get("n_agent", 4)),
        episode_length=int(cfg.get("episode_length", 200)),
        collector_env_num=int(cfg.get("collector_env_num", 1)),
        evaluator_env_num=int(cfg.get("evaluator_env_num", 1)),
        eval_interval_steps=int(cfg.get("eval_interval_steps", 400)),
        eval_horizon_steps=int(cfg.get("eval_horizon_steps", cfg.get("episode_length", 200))),
        codebook_size=int(cfg.get("codebook_size", 125)),
        discrete_level=int(cfg.get("discrete_level", 3)),
        output_root=str(cfg.get("output_root", "artifacts/auv6dof_convergence_dense_debug")),
        run_tag=f"{cfg.get('run_tag', 'dense_debug')}_{run_tag_suffix}",
        env_overrides=env_overrides,
        policy_overrides=policy_overrides,
        print_config=False,
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    plots = _plot_run(run_dir)
    try:
        trained_eval = evaluate_trained_policy(
            run_dir,
            config="",
            episodes=int(cfg.get("trained_eval_episodes", 5)),
            seed_base=3000,
            output=(run_dir / "trained_policy_eval.json").as_posix(),
        )
    except Exception as exc:
        trained_eval = {"skipped": True, "reason": str(exc)}
        _write_json(run_dir / "trained_policy_eval.json", trained_eval)
    return {
        "algo": algo,
        "algo_display": ALGO_DISPLAY.get(algo, algo.upper()),
        "seed": int(seed),
        "algo_profile": algo_profile_name(cfg, algo),
        "semantic_enabled": bool(env_overrides.get("semantic_enabled", False)),
        "action_mode": str(env_overrides.get("action_control_mode", "tau6")),
        "run_dir": run_dir.as_posix(),
        "summary": summary,
        "plots": plots,
        "trained_eval": trained_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/auv6dof_convergence_dense_debug.json")
    parser.add_argument("--stage", choices=["random", "seed0", "multiseed", "formal", "paper", "all"], default="seed0")
    parser.add_argument("--algos", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-env-step", type=int, default=0)
    parser.add_argument("--random-episodes", type=int, default=10)
    parser.add_argument("--min-eval-points", type=int, default=0)
    parser.add_argument("--trained-eval-episodes", type=int, default=0)
    args = parser.parse_args()

    cfg = load_convergence_config(args.config)
    algos = _parse_csv_list(args.algos, cfg.get("algos", ["maddpg", "mappo"]))
    default_seeds = [0] if args.stage == "seed0" else cfg.get("seeds", [0, 1, 2])
    seeds = _parse_int_list(args.seeds, default_seeds)
    max_env_step = int(args.max_env_step or cfg.get("max_env_step", 50000))
    if args.trained_eval_episodes > 0:
        cfg["trained_eval_episodes"] = int(args.trained_eval_episodes)

    manifest: Dict[str, Any] = {
        "config": str(args.config),
        "stage": args.stage,
        "algos": algos,
        "seeds": seeds,
        "max_env_step": max_env_step,
        "runs": [],
    }

    if args.stage in {"random", "all", "seed0", "multiseed", "formal", "paper"}:
        manifest["random_baselines"] = _random_baselines_for_algos(
            cfg, algos, args.random_episodes, seed_base=1000
        )
        if algos:
            manifest["random_baseline"] = manifest["random_baselines"].get(algos[0], {})

    if args.stage != "random":
        for algo in algos:
            for seed in seeds:
                manifest["runs"].append(_run_one(cfg, algo, int(seed), max_env_step, args.stage))

    root = Path(cfg.get("output_root", "artifacts/auv6dof_convergence_dense_debug"))
    acceptance_cfg = dict(cfg.get("acceptance", {}))
    min_eval_points = int(args.min_eval_points or acceptance_cfg.get("min_eval_points", 10))
    manifest["acceptance"] = summarize(
        root,
        min_eval_points=min_eval_points,
        tracking_error_success_threshold=float(acceptance_cfg.get("tracking_error_success_threshold", 0.08)),
        tail100_distance_threshold=float(acceptance_cfg.get("tail100_distance_threshold", 0.015)),
        tail100_tracking_error_threshold=float(acceptance_cfg.get("tail100_tracking_error_threshold", 0.005)),
        tail100_lost_threshold=float(acceptance_cfg.get("tail100_lost_threshold", 0.0)),
        tail100_action_saturation_threshold=float(acceptance_cfg.get("tail100_action_saturation_threshold", 0.10)),
        control_to_tracking_threshold=float(acceptance_cfg.get("control_to_tracking_threshold", 0.35)),
        required_passed_runs=int(acceptance_cfg.get("required_passed_runs", 2)),
    )
    output = root / f"longtest_manifest_{args.stage}.json"
    _write_json(output, manifest)
    print(json.dumps({"output": output.as_posix(), "acceptance": manifest["acceptance"].get("by_algo", {})}, indent=2))


if __name__ == "__main__":
    main()

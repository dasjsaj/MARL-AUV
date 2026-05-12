from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_paper_3e6_excel import collect_rows, export_workbook  # noqa: E402
from scripts.run_convergence_longtest import (  # noqa: E402
    _parse_csv_list,
    _parse_int_list,
    _random_baselines_for_algos,
    _run_one,
    _write_json,
)
from scripts._convergence_config import load_convergence_config  # noqa: E402
from scripts.summarize_convergence_acceptance import summarize  # noqa: E402


DEFAULT_MEDIUM = "configs/auv6dof_tmc_medium_2e6.json"
DEFAULT_HARD = "configs/auv6dof_tmc_hard_2e6.json"
DEFAULT_SCALE = "configs/auv6dof_tmc_scale_medium_2e6.json"
DEFAULT_ABLATION = "configs/auv6dof_tmc_ablation_2e6.json"


def _manifest_root() -> Path:
    return Path("artifacts/auv6dof_tmc_2e6/logs")


def _run_cfg(
    cfg: Dict[str, Any],
    *,
    algos: Iterable[str],
    seeds: Iterable[int],
    max_env_step: int,
    random_episodes: int,
    stage: str,
) -> Dict[str, Any]:
    algos = [str(a) for a in algos]
    seeds = [int(s) for s in seeds]
    manifest: Dict[str, Any] = {
        "stage": stage,
        "scenario": cfg.get("scenario", ""),
        "output_root": cfg.get("output_root", ""),
        "max_env_step": int(max_env_step),
        "algos": algos,
        "seeds": seeds,
        "runs": [],
    }
    manifest["random_baselines"] = _random_baselines_for_algos(cfg, algos, int(random_episodes), seed_base=1000)
    for algo in algos:
        for seed in seeds:
            manifest["runs"].append(_run_one(cfg, algo, seed, int(max_env_step), stage))

    acceptance_cfg = dict(cfg.get("acceptance", {}))
    manifest["acceptance"] = summarize(
        cfg.get("output_root", "artifacts/auv6dof_tmc_2e6"),
        min_eval_points=int(acceptance_cfg.get("min_eval_points", 300)),
        tracking_error_success_threshold=float(acceptance_cfg.get("tracking_error_success_threshold", 0.08)),
        tail100_distance_threshold=float(acceptance_cfg.get("tail100_distance_threshold", 0.015)),
        tail100_tracking_error_threshold=float(acceptance_cfg.get("tail100_tracking_error_threshold", 0.005)),
        tail100_lost_threshold=float(acceptance_cfg.get("tail100_lost_threshold", 0.0)),
        tail100_action_saturation_threshold=float(acceptance_cfg.get("tail100_action_saturation_threshold", 0.10)),
        control_to_tracking_threshold=float(acceptance_cfg.get("control_to_tracking_threshold", 0.35)),
        required_passed_runs=int(acceptance_cfg.get("required_passed_runs", 2)),
    )
    out = _manifest_root() / f"manifest_{stage}.json"
    _write_json(out, manifest)
    return manifest | {"manifest": out.as_posix()}


def _export_excel() -> Path:
    roots = {
        "medium_4auv": Path("artifacts/auv6dof_tmc_2e6/medium_4auv"),
        "hard_4auv": Path("artifacts/auv6dof_tmc_2e6/hard_4auv"),
        "scale_medium": Path("artifacts/auv6dof_tmc_2e6/scale_medium"),
        "ablation_medium_4auv": Path("artifacts/auv6dof_tmc_2e6/ablation_medium_4auv"),
    }
    rows = collect_rows(roots)
    out = Path("artifacts/auv6dof_tmc_2e6/logs/tmc_2e6_live_summary.xlsx")
    export_workbook(rows, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[
            "stage1-stg-medium",
            "main-medium",
            "main-hard",
            "main-all",
            "scale",
            "ablation",
            "all",
            "export",
        ],
        default="stage1-stg-medium",
    )
    parser.add_argument("--medium-config", default=DEFAULT_MEDIUM)
    parser.add_argument("--hard-config", default=DEFAULT_HARD)
    parser.add_argument("--scale-config", default=DEFAULT_SCALE)
    parser.add_argument("--ablation-config", default=DEFAULT_ABLATION)
    parser.add_argument("--algos", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-env-step", type=int, default=0)
    parser.add_argument("--random-episodes", type=int, default=10)
    parser.add_argument("--scale-n-agents", default="")
    args = parser.parse_args()

    manifests: List[Dict[str, Any]] = []
    if args.phase in {"stage1-stg-medium", "main-medium", "main-all", "all"}:
        cfg = load_convergence_config(args.medium_config)
        default_algos = ["stg_mappo"] if args.phase == "stage1-stg-medium" else list(cfg.get("algos", []))
        algos = _parse_csv_list(args.algos, default_algos)
        seeds = _parse_int_list(args.seeds, cfg.get("seeds", [0, 1, 2]))
        max_env_step = int(args.max_env_step or cfg.get("max_env_step", 2000000))
        manifests.append(
            _run_cfg(
                cfg,
                algos=algos,
                seeds=seeds,
                max_env_step=max_env_step,
                random_episodes=args.random_episodes,
                stage=args.phase,
            )
        )

    if args.phase in {"main-hard", "main-all", "all"}:
        cfg = load_convergence_config(args.hard_config)
        algos = _parse_csv_list(args.algos, list(cfg.get("algos", [])))
        seeds = _parse_int_list(args.seeds, cfg.get("seeds", [0, 1, 2]))
        max_env_step = int(args.max_env_step or cfg.get("max_env_step", 2000000))
        manifests.append(
            _run_cfg(
                cfg,
                algos=algos,
                seeds=seeds,
                max_env_step=max_env_step,
                random_episodes=args.random_episodes,
                stage="main-hard",
            )
        )

    if args.phase in {"scale", "all"}:
        base = load_convergence_config(args.scale_config)
        algos = _parse_csv_list(args.algos, list(base.get("algos", [])))
        seeds = _parse_int_list(args.seeds, base.get("seeds", [0, 1, 2]))
        n_values = _parse_int_list(args.scale_n_agents, base.get("scale_n_agents", [2, 4, 6, 8]))
        max_env_step = int(args.max_env_step or base.get("max_env_step", 2000000))
        for n_agent in n_values:
            cfg = copy.deepcopy(base)
            cfg["n_agent"] = int(n_agent)
            cfg["output_root"] = f"artifacts/auv6dof_tmc_2e6/scale_medium/n_agent_{n_agent}"
            cfg["run_tag"] = f"tmc_scale_medium_{n_agent}auv_2e6"
            manifests.append(
                _run_cfg(
                    cfg,
                    algos=algos,
                    seeds=seeds,
                    max_env_step=max_env_step,
                    random_episodes=args.random_episodes,
                    stage=f"scale-{n_agent}auv",
                )
            )

    if args.phase in {"ablation", "all"}:
        base = load_convergence_config(args.ablation_config)
        seeds = _parse_int_list(args.seeds, base.get("seeds", [0, 1, 2]))
        max_env_step = int(args.max_env_step or base.get("max_env_step", 2000000))
        profiles = dict(base.get("ablation_profiles", {}))
        selected = _parse_csv_list(args.algos, list(profiles.keys()))
        for name in selected:
            spec = profiles[name]
            algo = str(spec.get("algo", "mappo"))
            cfg = copy.deepcopy(base)
            cfg["algos"] = [algo]
            cfg["algo_env_overrides"] = {algo: dict(spec.get("env_overrides", {}))}
            cfg["run_tag"] = f"tmc_ablation_{name}_2e6"
            manifests.append(
                _run_cfg(
                    cfg,
                    algos=[algo],
                    seeds=seeds,
                    max_env_step=max_env_step,
                    random_episodes=args.random_episodes,
                    stage=f"ablation-{name}",
                )
            )

    excel = _export_excel()
    output = _manifest_root() / f"suite_{args.phase}_summary.json"
    _write_json(output, {"phase": args.phase, "manifests": manifests, "excel": excel.as_posix()})
    print(json.dumps({"output": output.as_posix(), "excel": excel.as_posix(), "manifests": len(manifests)}, indent=2))


if __name__ == "__main__":
    main()

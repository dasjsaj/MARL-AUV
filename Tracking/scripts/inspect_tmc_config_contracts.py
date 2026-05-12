from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_orchestrator import build_run_contract  # noqa: E402
from scripts._convergence_config import algo_env_overrides, algo_profile_name, load_convergence_config  # noqa: E402


def _parse_csv(raw: str, default: List[str]) -> List[str]:
    if not raw.strip():
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def inspect_contract(cfg: Dict[str, Any], algo: str, n_agent: int | None = None) -> Dict[str, Any]:
    env_overrides = algo_env_overrides(cfg, algo)
    contract = build_run_contract(
        env_name=str(cfg.get("env", "auv6dof")),
        algo_name=algo,
        seed=0,
        max_env_step=int(cfg.get("max_env_step", 2000000)),
        n_agent=int(n_agent or cfg.get("n_agent", 4)),
        episode_length=int(cfg.get("episode_length", 500)),
        collector_env_num=int(cfg.get("collector_env_num", 4)),
        evaluator_env_num=int(cfg.get("evaluator_env_num", 2)),
        eval_interval_steps=int(cfg.get("eval_interval_steps", 5000)),
        eval_horizon_steps=int(cfg.get("eval_horizon_steps", 500)),
        codebook_size=int(cfg.get("codebook_size", 125)),
        discrete_level=int(cfg.get("discrete_level", 5)),
        env_overrides=env_overrides,
    )
    env_cfg = contract["env_cfg"]
    obs_cfg = env_cfg.get("obs", {})
    reward_cfg = env_cfg.get("reward", {})
    shape_cfg = contract["shape_cfg"]
    return {
        "algo": algo,
        "n_agent": int(shape_cfg["n_agent"]),
        "algo_profile": algo_profile_name(cfg, algo),
        "semantic_enabled": bool(
            obs_cfg.get("include_semantic_features", False)
            or obs_cfg.get("include_semantic_graph_features", False)
            or reward_cfg.get("version", "") == "semantic_tracking_band"
        ),
        "action_control_mode": env_cfg.get("action_control_mode"),
        "agent_obs_dim": int(shape_cfg["agent_obs_dim"]),
        "global_obs_dim": int(shape_cfg["global_obs_dim"]),
        "continuous_action_dim": int(shape_cfg["action_dim_continuous"]),
        "discrete_action_dim": int(shape_cfg["action_dim_discrete"]),
        "reward_version": reward_cfg.get("version"),
        "w_semantic_reward": reward_cfg.get("w_semantic_reward"),
        "include_tracking_diagnostics": bool(obs_cfg.get("include_tracking_diagnostics", False)),
        "include_semantic_features": bool(obs_cfg.get("include_semantic_features", False)),
        "include_semantic_graph_features": bool(obs_cfg.get("include_semantic_graph_features", False)),
        "max_env_step": int(contract["train_cfg"]["max_env_step"]),
        "eval_interval_steps": int(contract["eval_cfg"]["eval_freq"]),
        "eval_horizon_steps": int(contract["eval_cfg"]["eval_horizon_steps"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/auv6dof_tmc_medium_2e6.json")
    parser.add_argument("--algos", default="")
    parser.add_argument("--n-agent", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    cfg = load_convergence_config(args.config)
    algos = _parse_csv(args.algos, list(cfg.get("algos", [])))
    rows = [inspect_contract(cfg, algo, args.n_agent or None) for algo in algos]
    payload = {"config": args.config, "rows": rows}
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

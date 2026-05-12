from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_orchestrator import run_experiment  # noqa: E402
from scripts._convergence_config import load_convergence_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/auv6dof_convergence_debug.json")
    parser.add_argument("--algo", choices=["maddpg", "mappo"], default="maddpg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-env-step", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_convergence_config(args.config)
    policy_overrides: Dict[str, Any] = dict(cfg.get("policy_overrides", {}).get(args.algo, {}))
    result = run_experiment(
        env_name=str(cfg.get("env", "auv6dof")),
        algo_name=args.algo,
        seed=args.seed,
        max_env_step=int(args.max_env_step or cfg.get("max_env_step", 2000)),
        n_agent=int(cfg.get("n_agent", 4)),
        episode_length=int(cfg.get("episode_length", 200)),
        collector_env_num=int(cfg.get("collector_env_num", 1)),
        evaluator_env_num=int(cfg.get("evaluator_env_num", 1)),
        eval_interval_steps=int(cfg.get("eval_interval_steps", 1000)),
        eval_horizon_steps=int(cfg.get("eval_horizon_steps", cfg.get("episode_length", 200))),
        output_root=str(cfg.get("output_root", "artifacts/auv6dof_convergence_debug")),
        run_tag=f"{cfg.get('run_tag', 'debug')}_smoke_{args.algo}",
        env_overrides=dict(cfg.get("env_overrides", {})),
        policy_overrides=policy_overrides,
        print_config=False,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

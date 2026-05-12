from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auv6dof.gym_env import AUV6DOFGymEnv  # noqa: E402
from scripts._convergence_config import env_cfg_from_config, load_convergence_config  # noqa: E402


def run_episode(env_cfg: Dict[str, Any], seed: int) -> Dict[str, float]:
    env = AUV6DOFGymEnv(env_cfg)
    _, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    steps = 0
    sums: Dict[str, float] = {}
    target_distance_tail: List[float] = []
    tracking_error_tail: List[float] = []
    target_lost_tail: List[float] = []
    action_norm_tail: List[float] = []
    while not done:
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += float(np.sum(reward))
        for key in ("tracking_error", "target_distance", "target_lost", "tracking_reward", "control_cost", "action_norm"):
            sums[key] = sums.get(key, 0.0) + float(info.get(key, 0.0))
        target_distance_tail.append(float(info.get("target_distance", 0.0)))
        tracking_error_tail.append(float(info.get("tracking_error", 0.0)))
        target_lost_tail.append(float(info.get("target_lost", 0.0)))
        action_norm_tail.append(float(info.get("action_norm", 0.0)))
        steps += 1
        done = bool(terminated or truncated)
    env.close()
    out = {"episode_return": total, "steps": float(steps)}
    for key, value in sums.items():
        out[key] = value / max(1, steps)
    tail_n = 100
    out["tail100_mean_target_distance"] = float(np.mean(target_distance_tail[-tail_n:])) if target_distance_tail else 0.0
    out["tail100_mean_tracking_error"] = float(np.mean(tracking_error_tail[-tail_n:])) if tracking_error_tail else 0.0
    out["tail100_target_lost_rate"] = float(np.mean(target_lost_tail[-tail_n:])) if target_lost_tail else 0.0
    out["tail100_action_norm"] = float(np.mean(action_norm_tail[-tail_n:])) if action_norm_tail else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/auv6dof_convergence_debug.json")
    parser.add_argument("--algo", default="")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_convergence_config(args.config)
    env_cfg = env_cfg_from_config(cfg, algo=args.algo or None)
    rows: List[Dict[str, float]] = [run_episode(env_cfg, args.seed_base + i) for i in range(args.episodes)]
    summary = {
        "episodes": args.episodes,
        "algo": args.algo,
        "mean_return": float(np.mean([row["episode_return"] for row in rows])),
        "std_return": float(np.std([row["episode_return"] for row in rows])),
        "mean_tracking_error": float(np.mean([row.get("tracking_error", 0.0) for row in rows])),
        "mean_target_distance": float(np.mean([row.get("target_distance", 0.0) for row in rows])),
        "mean_tail100_target_distance": float(np.mean([row.get("tail100_mean_target_distance", 0.0) for row in rows])),
        "mean_target_lost": float(np.mean([row.get("target_lost", 0.0) for row in rows])),
        "rows": rows,
    }
    output = Path(args.output or Path(cfg.get("output_root", "artifacts")) / "random_policy_baseline.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": output.as_posix(), "csv": csv_path.as_posix(), **{k: v for k, v in summary.items() if k != "rows"}}, indent=2))


if __name__ == "__main__":
    main()

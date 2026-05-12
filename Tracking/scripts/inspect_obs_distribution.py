from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auv6dof.gym_env import AUV6DOFGymEnv  # noqa: E402
from scripts._convergence_config import env_cfg_from_config, load_convergence_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/auv6dof_convergence_debug.json")
    parser.add_argument("--algo", default="")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    cfg = load_convergence_config(args.config)
    env = AUV6DOFGymEnv(env_cfg_from_config(cfg, algo=args.algo or None))
    obs, _ = env.reset(seed=0)
    samples = [obs["agent_state"]]
    for step in range(args.steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        samples.append(obs["agent_state"])
        if terminated or truncated:
            obs, _ = env.reset(seed=step + 1)
    env.close()
    arr = np.concatenate(samples, axis=0)
    summary = {
        "shape": list(arr.shape),
        "global_min": float(np.min(arr)),
        "global_max": float(np.max(arr)),
        "abs_gt_1_rate": float(np.mean(np.abs(arr) > 1.0)),
        "abs_gt_2_rate": float(np.mean(np.abs(arr) > 2.0)),
        "dim_mean": np.mean(arr, axis=0).round(6).tolist(),
        "dim_std": np.std(arr, axis=0).round(6).tolist(),
        "dim_min": np.min(arr, axis=0).round(6).tolist(),
        "dim_max": np.max(arr, axis=0).round(6).tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

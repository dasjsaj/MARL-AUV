from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

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
    env.reset(seed=0)
    values: Dict[str, list[float]] = {}
    for _ in range(args.steps):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        terms = info.get("reward_terms_mean", {})
        for key, value in terms.items():
            values.setdefault(key, []).append(float(value))
        if terminated or truncated:
            env.reset()
    env.close()
    summary: Dict[str, Dict[str, float]] = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "finite": float(np.isfinite(arr).all()),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

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
    args = parser.parse_args()
    cfg = load_convergence_config(args.config)
    env_cfg = env_cfg_from_config(cfg, algo=args.algo or None)
    env_cfg["episode_length"] = 1
    env = AUV6DOFGymEnv(env_cfg)
    results: Dict[str, Any] = {}
    for name, axis, value in [("zero", -1, 0.0), ("pos_x", 0, 1.0), ("neg_x", 0, -1.0), ("pos_z", 2, 1.0)]:
        env.reset(seed=42)
        before = np.stack([agent.state.p_pos.copy() for agent in env.world.agents], axis=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        if axis >= 0:
            action[:, axis] = value
        _, reward, _, _, info = env.step(action)
        after = np.stack([agent.state.p_pos.copy() for agent in env.world.agents], axis=0)
        results[name] = {
            "mean_delta": np.mean(after - before, axis=0).tolist(),
            "reward_sum": float(np.sum(reward)),
            "action_norm": float(info.get("action_norm", 0.0)),
        }
    env.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

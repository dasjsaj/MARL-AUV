from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auv6dof.gym_env import AUV6DOFGymEnv  # noqa: E402
from scripts._convergence_config import env_cfg_from_config, load_convergence_config  # noqa: E402


class SimpleMADDPGActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.out(x))


class SimpleMAPPOActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.enc = nn.Linear(obs_dim, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.enc(obs))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.mu(x))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_ckpt(run_dir: Path, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(explicit)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        best = _read_json(summary_path).get("best_ckpt")
        if isinstance(best, str) and best.strip() and Path(best).exists():
            return Path(best)
    candidates: List[Path] = []
    for pattern in ("ckpt_best*.pth.tar", "ckpt_best*.pth", "*.pth.tar", "*.pth", "*.pt"):
        candidates.extend((run_dir / "exp" / "ckpt").glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint under {run_dir / 'exp' / 'ckpt'}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _load_actor(run_dir: Path, ckpt: Path) -> tuple[nn.Module, str, int, int]:
    cfg = _read_json(run_dir / "config.json")
    contract = cfg.get("contract", {})
    algo = str(contract.get("algo_cfg", {}).get("algo_name", "")).lower()
    shape = contract.get("shape_cfg", {})
    obs_dim = int(shape.get("agent_obs_dim", 68))
    action_dim = int(shape.get("action_dim_continuous", 3))
    data = torch.load(ckpt.as_posix(), map_location="cpu")
    state = data.get("model", data) if isinstance(data, dict) else data
    if algo == "maddpg":
        actor = SimpleMADDPGActor(obs_dim, action_dim)
        actor.load_state_dict(
            {
                "fc1.weight": state["actor.0.weight"],
                "fc1.bias": state["actor.0.bias"],
                "fc2.weight": state["actor.2.main.0.weight"],
                "fc2.bias": state["actor.2.main.0.bias"],
                "out.weight": state["actor.2.last.weight"],
                "out.bias": state["actor.2.last.bias"],
            }
        )
    elif algo in {"mappo", "stg_mappo"}:
        actor = SimpleMAPPOActor(obs_dim, action_dim)
        actor.load_state_dict(
            {
                "enc.weight": state["actor_encoder.0.weight"],
                "enc.bias": state["actor_encoder.0.bias"],
                "fc1.weight": state["actor_head.main.0.weight"],
                "fc1.bias": state["actor_head.main.0.bias"],
                "fc2.weight": state["actor_head.main.2.weight"],
                "fc2.bias": state["actor_head.main.2.bias"],
                "mu.weight": state["actor_head.mu.weight"],
                "mu.bias": state["actor_head.mu.bias"],
            }
        )
    else:
        raise ValueError(f"Unsupported algo for direct evaluation: {algo}")
    actor.eval()
    return actor, algo, obs_dim, action_dim


def _env_cfg_from_run(run_dir: Path, config_path: str) -> Dict[str, Any]:
    if config_path:
        return env_cfg_from_config(load_convergence_config(config_path))
    cfg = _read_json(run_dir / "config.json")
    env_cfg = dict(cfg.get("contract", {}).get("env_cfg", {}))
    env_cfg.pop("artifact_dir", None)
    env_cfg["print_episode_reward"] = False
    return env_cfg


def _rollout(actor: nn.Module, env_cfg: Dict[str, Any], seed: int) -> Dict[str, float]:
    env = AUV6DOFGymEnv(env_cfg)
    obs, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    steps = 0
    sums: Dict[str, float] = {}
    target_distance_tail: List[float] = []
    tracking_error_tail: List[float] = []
    target_lost_tail: List[float] = []
    action_norm_tail: List[float] = []
    while not done:
        obs_tensor = torch.as_tensor(np.asarray(obs["agent_state"], dtype=np.float32))
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy().astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(np.sum(reward))
        for key in (
            "tracking_error",
            "target_distance",
            "target_lost",
            "tracking_reward",
            "observation_reward",
            "control_cost",
            "action_norm",
            "action_delta_norm",
        ):
            sums[key] = sums.get(key, 0.0) + float(info.get(key, 0.0))
        target_distance_tail.append(float(info.get("target_distance", 0.0)))
        tracking_error_tail.append(float(info.get("tracking_error", 0.0)))
        target_lost_tail.append(float(info.get("target_lost", 0.0)))
        action_norm_tail.append(float(info.get("action_norm", 0.0)))
        steps += 1
        done = bool(terminated or truncated)
    env.close()
    row = {"episode_return": total, "steps": float(steps)}
    for key, value in sums.items():
        row[key] = value / max(1, steps)
    tail_n = 100
    row["tail100_mean_target_distance"] = float(np.mean(target_distance_tail[-tail_n:])) if target_distance_tail else 0.0
    row["tail100_mean_tracking_error"] = float(np.mean(tracking_error_tail[-tail_n:])) if tracking_error_tail else 0.0
    row["tail100_target_lost_rate"] = float(np.mean(target_lost_tail[-tail_n:])) if target_lost_tail else 0.0
    row["tail100_action_norm"] = float(np.mean(action_norm_tail[-tail_n:])) if action_norm_tail else 0.0
    return row


def evaluate_trained_policy(
    run_dir: str | Path,
    *,
    episodes: int = 10,
    seed_base: int = 2000,
    config: str = "",
    ckpt: str = "",
    output: str = "",
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    ckpt_path = _discover_ckpt(run_path, ckpt)
    actor, algo, _, _ = _load_actor(run_path, ckpt_path)
    env_cfg = _env_cfg_from_run(run_path, config)
    rows = [_rollout(actor, env_cfg, int(seed_base + i)) for i in range(int(episodes))]
    summary = {
        "algo": algo,
        "episodes": int(episodes),
        "checkpoint": ckpt_path.as_posix(),
        "mean_return": float(np.mean([row["episode_return"] for row in rows])),
        "std_return": float(np.std([row["episode_return"] for row in rows])),
        "mean_tracking_error": float(np.mean([row.get("tracking_error", 0.0) for row in rows])),
        "mean_target_distance": float(np.mean([row.get("target_distance", 0.0) for row in rows])),
        "mean_tail100_target_distance": float(np.mean([row.get("tail100_mean_target_distance", 0.0) for row in rows])),
        "mean_tail100_tracking_error": float(np.mean([row.get("tail100_mean_tracking_error", 0.0) for row in rows])),
        "mean_target_lost": float(np.mean([row.get("target_lost", 0.0) for row in rows])),
        "mean_tracking_reward": float(np.mean([row.get("tracking_reward", 0.0) for row in rows])),
        "mean_control_cost": float(np.mean([row.get("control_cost", 0.0) for row in rows])),
        "mean_action_norm": float(np.mean([row.get("action_norm", 0.0) for row in rows])),
        "rows": rows,
    }
    output_path = Path(output) if output else run_path / "trained_policy_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with output_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {k: v for k, v in summary.items() if k != "rows"} | {"output": output_path.as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_trained_policy(
                args.run_dir,
                episodes=args.episodes,
                seed_base=args.seed_base,
                config=args.config,
                ckpt=args.ckpt,
                output=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

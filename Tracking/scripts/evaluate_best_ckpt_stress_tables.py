from __future__ import annotations

"""
Evaluate best checkpoints under stress-test conditions and export paper tables.

The script does not train. It loads each run's ckpt_best checkpoint, rolls out
the policy in AUV6DOF Gym directly, and writes medium/hard CSV/XLSX tables.
"""

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
from itertools import product

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auv6dof.gym_env import AUV6DOFGymEnv  # noqa: E402
from scripts._convergence_config import deep_update  # noqa: E402


ALGO_ORDER = ["stg_mappo", "mappo", "maddpg", "matd3", "happo", "madqn", "masac"]
ALGO_LABEL = {
    "stg_mappo": "STG-MAPPO",
    "mappo": "MAPPO",
    "maddpg": "MADDPG",
    "matd3": "MATD3",
    "happo": "HAPPO",
    "madqn": "MADQN",
    "masac": "MASAC",
}


def _build_discrete_action_codebook(
    action_dim: int,
    *,
    discrete_level: int = 3,
    codebook_size: int = 125,
    action_scale: float = 1.0,
) -> Tuple[np.ndarray, int]:
    levels = np.array([-1.0, 0.0, 1.0], dtype=np.float32) if int(discrete_level) == 3 else np.linspace(
        -1.0, 1.0, num=max(2, int(discrete_level)), dtype=np.float32
    )
    full = np.asarray(list(product(levels.tolist(), repeat=max(1, int(action_dim)))), dtype=np.float32)
    full_size = int(full.shape[0])
    target = max(1, min(int(codebook_size), full_size))
    if target >= full_size:
        selected = full
    else:
        idx = np.round(np.linspace(0, full_size - 1, num=target)).astype(np.int64)
        idx = np.unique(idx)
        if idx.size < target:
            existing = set(idx.tolist())
            fill = [i for i in range(full_size) if i not in existing][: target - idx.size]
            idx = np.concatenate([idx, np.asarray(fill, dtype=np.int64)])
            idx.sort()
        selected = full[idx]
    zero = np.zeros((1, max(1, int(action_dim))), dtype=np.float32)
    if not np.any(np.all(np.isclose(selected, zero), axis=1)):
        selected = selected.copy()
        selected[0] = zero[0]
    return (selected * float(action_scale)).astype(np.float32), full_size


@dataclass
class RunSpec:
    scenario: str
    algo: str
    seed: int
    run_dir: Path
    ckpt: Path
    env_step: int
    env_cfg: Dict[str, Any]
    discrete_action: bool
    codebook_size: int
    discrete_level: int
    action_scale: float


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_get(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _discover_ckpt(run_dir: Path) -> Path:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        best = _read_json(summary_path).get("best_ckpt")
        if isinstance(best, str) and best.strip() and Path(best).exists():
            return Path(best)
    ckpt_dir = run_dir / "exp" / "ckpt"
    candidates: List[Path] = []
    for pattern in ("ckpt_best*.pth.tar", "ckpt_best*.pth", "*.pth.tar", "*.pth", "*.pt"):
        candidates.extend(ckpt_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint under {ckpt_dir}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _env_step(run_dir: Path) -> int:
    p = run_dir / "exp" / "result.pkl"
    if not p.exists():
        return 0
    with p.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return int(obj.get("env_step") or obj.get("envstep") or 0)
    return 0


def _env_cfg_from_run(run_dir: Path) -> Tuple[Dict[str, Any], bool, int, int, float, str]:
    cfg = _read_json(run_dir / "config.json")
    contract = cfg.get("contract", {})
    env_cfg = dict(contract.get("env_cfg", {}))
    env_cfg.pop("artifact_dir", None)
    env_cfg["print_episode_reward"] = False
    algo = str(_nested_get(contract, ["algo_cfg", "algo_name"], "")).lower()
    discrete_action = bool(env_cfg.pop("discrete_action", False))
    codebook_size = int(env_cfg.pop("codebook_size", 125))
    discrete_level = int(env_cfg.pop("discrete_level", 5))
    action_scale = float(env_cfg.get("action_scale", 1.0))
    return env_cfg, discrete_action, codebook_size, discrete_level, action_scale, algo


def discover_runs(root: Path, scenario: str, min_env_step: int) -> List[RunSpec]:
    base = root / f"{scenario}_4auv" / "auv6dof"
    specs: List[RunSpec] = []
    for algo in ALGO_ORDER:
        for seed_dir in sorted((base / algo).glob("seed_*")):
            try:
                seed = int(seed_dir.name.split("_")[-1])
            except Exception:
                continue
            candidates = []
            for result in seed_dir.glob("*/exp/result.pkl"):
                run_dir = result.parent.parent
                step = _env_step(run_dir)
                if step >= int(min_env_step):
                    candidates.append((step, run_dir))
            if not candidates:
                continue
            step, run_dir = sorted(candidates, key=lambda x: (x[0], x[1].stat().st_mtime))[-1]
            env_cfg, discrete, codebook, level, scale, config_algo = _env_cfg_from_run(run_dir)
            specs.append(
                RunSpec(
                    scenario=scenario,
                    algo=config_algo or algo,
                    seed=seed,
                    run_dir=run_dir,
                    ckpt=_discover_ckpt(run_dir),
                    env_step=int(step),
                    env_cfg=env_cfg,
                    discrete_action=discrete,
                    codebook_size=codebook,
                    discrete_level=level,
                    action_scale=scale,
                )
            )
    return specs


class ContinuousActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.out(x))


class MAPPOActor(nn.Module):
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


class HAPPOAgentActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.enc0 = nn.Linear(obs_dim, 128)
        self.enc1 = nn.Linear(128, 128)
        self.enc2 = nn.Linear(128, 64)
        self.head0 = nn.Linear(64, 64)
        self.head1 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.enc0(obs))
        x = torch.relu(self.enc1(x))
        x = torch.relu(self.enc2(x))
        x = torch.relu(self.head0(x))
        x = torch.relu(self.head1(x))
        return torch.tanh(self.mu(x))


class HAPPOActor(nn.Module):
    def __init__(self, actors: List[HAPPOAgentActor]) -> None:
        super().__init__()
        self.actors = nn.ModuleList(actors)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        outs = [actor(obs[i : i + 1]) for i, actor in enumerate(self.actors)]
        return torch.cat(outs, dim=0)


class MASACDiscreteActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(obs_dim, 64)
        self.head0 = nn.Linear(64, 64)
        self.head1 = nn.Linear(64, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc(obs))
        x = torch.relu(self.head0(x))
        return self.head1(x)


class MADQNDiscreteActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, n_agent: int) -> None:
        super().__init__()
        self.n_agent = int(n_agent)
        self.fc0 = nn.Linear(obs_dim, 256)
        self.fc1 = nn.Linear(256, 256)
        self.gru = nn.GRUCell(256, 256)
        self.head0 = nn.Linear(256, 256)
        self.head1 = nn.Linear(256, action_dim)
        self.register_buffer("hidden", torch.zeros(self.n_agent, 256))

    def reset_hidden(self) -> None:
        self.hidden.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc0(obs))
        x = torch.relu(self.fc1(x))
        self.hidden = self.gru(x, self.hidden)
        q = torch.relu(self.head0(self.hidden))
        return self.head1(q)


def _state_dict(ckpt: Path) -> Dict[str, torch.Tensor]:
    data = torch.load(ckpt.as_posix(), map_location="cpu")
    state = data.get("model", data) if isinstance(data, dict) else data
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format: {ckpt}")
    return state


def load_actor(spec: RunSpec) -> Tuple[nn.Module, str]:
    cfg = _read_json(spec.run_dir / "config.json")
    shape = cfg.get("contract", {}).get("shape_cfg", {})
    obs_dim = int(shape.get("agent_obs_dim", 51))
    action_dim_cont = int(shape.get("action_dim_continuous", 6))
    action_dim_dis = int(shape.get("action_dim_discrete", 125))
    n_agent = int(shape.get("n_agent", 4))
    state = _state_dict(spec.ckpt)
    algo = spec.algo.lower()

    if algo in {"mappo", "stg_mappo"}:
        actor = MAPPOActor(obs_dim, action_dim_cont)
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
        actor.eval()
        return actor, "continuous"

    if algo in {"maddpg", "matd3"}:
        actor = ContinuousActor(obs_dim, 512, action_dim_cont)
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
        actor.eval()
        return actor, "continuous"

    if algo == "happo":
        actors: List[HAPPOAgentActor] = []
        for i in range(n_agent):
            actor_i = HAPPOAgentActor(obs_dim, action_dim_cont)
            prefix = f"agent_models.{i}."
            actor_i.load_state_dict(
                {
                    "enc0.weight": state[prefix + "actor_encoder.init.weight"],
                    "enc0.bias": state[prefix + "actor_encoder.init.bias"],
                    "enc1.weight": state[prefix + "actor_encoder.main.0.weight"],
                    "enc1.bias": state[prefix + "actor_encoder.main.0.bias"],
                    "enc2.weight": state[prefix + "actor_encoder.main.2.weight"],
                    "enc2.bias": state[prefix + "actor_encoder.main.2.bias"],
                    "head0.weight": state[prefix + "actor_head.main.0.weight"],
                    "head0.bias": state[prefix + "actor_head.main.0.bias"],
                    "head1.weight": state[prefix + "actor_head.main.2.weight"],
                    "head1.bias": state[prefix + "actor_head.main.2.bias"],
                    "mu.weight": state[prefix + "actor_head.mu.weight"],
                    "mu.bias": state[prefix + "actor_head.mu.bias"],
                }
            )
            actors.append(actor_i)
        actor = HAPPOActor(actors)
        actor.eval()
        return actor, "continuous"

    if algo == "masac":
        actor = MASACDiscreteActor(obs_dim, action_dim_dis)
        actor.load_state_dict(
            {
                "fc.weight": state["actor.0.weight"],
                "fc.bias": state["actor.0.bias"],
                "head0.weight": state["actor.2.Q.0.0.weight"],
                "head0.bias": state["actor.2.Q.0.0.bias"],
                "head1.weight": state["actor.2.Q.1.0.weight"],
                "head1.bias": state["actor.2.Q.1.0.bias"],
            }
        )
        actor.eval()
        return actor, "discrete"

    if algo == "madqn":
        actor = MADQNDiscreteActor(obs_dim, action_dim_dis, n_agent=n_agent)
        actor.load_state_dict(
            {
                "fc0.weight": state["current._q_network.encoder.init.weight"],
                "fc0.bias": state["current._q_network.encoder.init.bias"],
                "fc1.weight": state["current._q_network.encoder.main.0.weight"],
                "fc1.bias": state["current._q_network.encoder.main.0.bias"],
                "gru.weight_ih": state["current._q_network.rnn.weight_ih"],
                "gru.weight_hh": state["current._q_network.rnn.weight_hh"],
                "gru.bias_ih": state["current._q_network.rnn.bias_ih"],
                "gru.bias_hh": state["current._q_network.rnn.bias_hh"],
                "head0.weight": state["current._q_network.head.Q.0.0.weight"],
                "head0.bias": state["current._q_network.head.Q.0.0.bias"],
                "head1.weight": state["current._q_network.head.Q.1.0.weight"],
                "head1.bias": state["current._q_network.head.Q.1.0.bias"],
            },
            strict=False,
        )
        actor.eval()
        return actor, "discrete_recurrent"

    raise ValueError(f"Unsupported algo: {algo}")


def stress_conditions(scenario: str) -> Dict[str, Dict[str, Any]]:
    if scenario == "medium":
        return {
            "Nominal-medium": {"reset": {"curriculum_stage": "medium"}},
            "Fast target": {
                "reset": {"curriculum_stage": "medium", "medium_target_speed_range": [0.008, 0.014]}
            },
            "Far initialization": {"reset": {"curriculum_stage": "hard"}},
            "Limited sensing": {"reset": {"curriculum_stage": "medium"}, "reward": {"sensor_range": 0.30}},
            "Communication degraded": {
                "reset": {"curriculum_stage": "medium"},
                "reward": {"sensor_range": 0.30, "w_communication_reward": 0.0},
            },
            "Combined stress": {
                "reset": {"curriculum_stage": "hard", "hard_target_speed_range": [0.008, 0.014]},
                "reward": {"sensor_range": 0.30},
            },
        }
    return {
        "Nominal-hard": {"reset": {"curriculum_stage": "hard"}},
        "Faster target": {"reset": {"curriculum_stage": "hard", "hard_target_speed_range": [0.014, 0.020]}},
        "Farther initialization": {"reset": {"curriculum_stage": "hard", "min_init_separation": 0.14}},
        "Limited sensing": {"reset": {"curriculum_stage": "hard"}, "reward": {"sensor_range": 0.30}},
        "Communication degraded": {
            "reset": {"curriculum_stage": "hard"},
            "reward": {"sensor_range": 0.30, "w_communication_reward": 0.0},
        },
        "Combined stress": {
            "reset": {"curriculum_stage": "hard", "hard_target_speed_range": [0.014, 0.020], "min_init_separation": 0.14},
            "reward": {"sensor_range": 0.30},
        },
    }


def _metric_row(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if arr.size > 1 else 0.0)


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "load_failed"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_env_cfg(base: Dict[str, Any], patch: Dict[str, Any], horizon: int) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    cfg["episode_length"] = int(horizon)
    cfg["print_episode_reward"] = False
    deep_update(cfg, json.loads(json.dumps(patch)))
    reset = dict(cfg.get("reset", {}))
    reset.pop("train_curriculum_stage", None)
    reset.pop("eval_curriculum_stage", None)
    cfg["reset"] = reset
    return cfg


def _action_from_actor(
    actor: nn.Module,
    actor_kind: str,
    obs_agent_state: np.ndarray,
    env: AUV6DOFGymEnv,
    spec: RunSpec,
    action_map: Optional[np.ndarray],
) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs_agent_state, dtype=torch.float32)
    with torch.no_grad():
        out = actor(obs_tensor)
    if actor_kind == "continuous":
        action = out.cpu().numpy().astype(np.float32)
        return np.clip(action * float(spec.action_scale), -1.0, 1.0)
    q = out.cpu().numpy()
    idx = np.argmax(q, axis=1).astype(np.int64)
    if action_map is None:
        raise RuntimeError("Missing discrete action map")
    action_cont = action_map[idx]
    # Match DI wrapper behavior: decoded codebook action is scaled again before
    # entering the Gym env.
    return np.clip(action_cont * float(spec.action_scale), -1.0, 1.0).astype(np.float32)


def rollout(
    spec: RunSpec,
    actor: nn.Module,
    actor_kind: str,
    condition: str,
    patch: Dict[str, Any],
    episodes: int,
    seed_base: int,
    horizon: int,
) -> List[Dict[str, float]]:
    cfg = build_env_cfg(spec.env_cfg, patch, horizon)
    env = AUV6DOFGymEnv(cfg)
    action_map: Optional[np.ndarray] = None
    if actor_kind.startswith("discrete"):
        action_map, _ = _build_discrete_action_codebook(
            env.action_dim,
            discrete_level=spec.discrete_level,
            codebook_size=spec.codebook_size,
            action_scale=spec.action_scale,
        )

    rows: List[Dict[str, float]] = []
    for ep in range(int(episodes)):
        if hasattr(actor, "reset_hidden"):
            actor.reset_hidden()  # type: ignore[attr-defined]
        obs, _ = env.reset(seed=int(seed_base + ep))
        done = False
        total = 0.0
        steps = 0
        sums: Dict[str, float] = {}
        tail: Dict[str, List[float]] = {"target_distance": [], "tracking_error": [], "target_lost": [], "action_norm": []}
        while not done:
            action = _action_from_actor(actor, actor_kind, np.asarray(obs["agent_state"], dtype=np.float32), env, spec, action_map)
            action_saturation = float(np.mean(np.abs(action) >= 0.95))
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(np.sum(np.asarray(reward, dtype=np.float32)))
            for key in (
                "tracking_error",
                "target_distance",
                "target_lost",
                "tracking_reward",
                "observation_reward",
                "communication_quality",
                "control_cost",
                "action_norm",
                "action_delta_norm",
            ):
                sums[key] = sums.get(key, 0.0) + float(info.get(key, 0.0))
            sums["action_saturation_rate"] = sums.get("action_saturation_rate", 0.0) + action_saturation
            for key in tail:
                tail[key].append(float(info.get(key, 0.0)))
            steps += 1
            done = bool(terminated or truncated)
        row: Dict[str, float] = {"episode_return": total, "steps": float(steps)}
        for key, value in sums.items():
            row[key] = value / max(1, steps)
        tail_n = min(100, len(tail["target_distance"]))
        row["tail100_mean_target_distance"] = float(np.mean(tail["target_distance"][-tail_n:]))
        row["tail100_mean_tracking_error"] = float(np.mean(tail["tracking_error"][-tail_n:]))
        row["tail100_target_lost_rate"] = float(np.mean(tail["target_lost"][-tail_n:]))
        row["tail100_action_norm"] = float(np.mean(tail["action_norm"][-tail_n:]))
        rows.append(row)
    env.close()
    return rows


def aggregate_rows(rows: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    keys = [
        "episode_return",
        "tail100_mean_target_distance",
        "tail100_mean_tracking_error",
        "tail100_target_lost_rate",
        "action_norm",
        "tail100_action_norm",
        "action_saturation_rate",
        "control_cost",
        "communication_quality",
    ]
    return {key: _metric_row([float(r.get(key, 0.0)) for r in rows]) for key in keys}


def evaluate_scenario(
    root: Path,
    scenario: str,
    *,
    episodes: int,
    seed_base: int,
    horizon: int,
    min_env_step: int,
    algos: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    conditions: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    specs = discover_runs(root, scenario, min_env_step=min_env_step)
    if algos:
        algos_set = {a.lower() for a in algos}
        specs = [s for s in specs if s.algo.lower() in algos_set]
    if seeds:
        seeds_set = {int(s) for s in seeds}
        specs = [s for s in specs if s.seed in seeds_set]
    cond_map = stress_conditions(scenario)
    if conditions:
        cond_map = {k: v for k, v in cond_map.items() if k in set(conditions)}

    detail_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for spec in specs:
        try:
            actor, actor_kind = load_actor(spec)
        except Exception as exc:
            for cond in cond_map:
                failures.append(
                    {
                        "scenario": scenario,
                        "algo": spec.algo,
                        "seed": spec.seed,
                        "condition": cond,
                        "stage": "load_actor",
                        "error": repr(exc),
                    }
                )
            continue
        for cond_name, patch in cond_map.items():
            try:
                ep_rows = rollout(
                    spec,
                    actor,
                    actor_kind,
                    cond_name,
                    patch,
                    episodes=episodes,
                    seed_base=seed_base + spec.seed * 10000,
                    horizon=horizon,
                )
                for ep, row in enumerate(ep_rows):
                    detail_rows.append(
                        {
                            "scenario": scenario,
                            "algo": spec.algo,
                            "algo_label": ALGO_LABEL.get(spec.algo, spec.algo),
                            "seed": spec.seed,
                            "condition": cond_name,
                            "episode": ep,
                            "run_dir": spec.run_dir.as_posix(),
                            "checkpoint": spec.ckpt.as_posix(),
                            **row,
                        }
                    )
                agg = aggregate_rows(ep_rows)
                table_rows.append(
                    {
                        "scenario": scenario,
                        "algorithm": ALGO_LABEL.get(spec.algo, spec.algo),
                        "algo": spec.algo,
                        "seed": spec.seed,
                        "condition": cond_name,
                        "eval_return": _fmt(*agg["episode_return"], digits=2),
                        "tail100_distance_km": _fmt(*agg["tail100_mean_target_distance"], digits=3),
                        "tail100_tracking_error_km": _fmt(*agg["tail100_mean_tracking_error"], digits=3),
                        "target_lost_rate": _fmt(*agg["tail100_target_lost_rate"], digits=3),
                        "action_norm": _fmt(*agg["action_norm"], digits=3),
                        "action_saturation_rate": _fmt(*agg["action_saturation_rate"], digits=3),
                        "control_cost": _fmt(*agg["control_cost"], digits=3),
                        "communication_quality": _fmt(*agg["communication_quality"], digits=3),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "scenario": scenario,
                        "algo": spec.algo,
                        "seed": spec.seed,
                        "condition": cond_name,
                        "stage": "rollout",
                        "error": repr(exc),
                        "run_dir": spec.run_dir.as_posix(),
                    }
                )

    return detail_rows, table_rows, failures


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _parse_csv(text: str) -> Optional[List[str]]:
    if not text.strip():
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_int_csv(text: str) -> Optional[List[int]]:
    vals = _parse_csv(text)
    if vals is None:
        return None
    return [int(x) for x in vals]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best checkpoints under medium/hard stress tests.")
    parser.add_argument("--root", type=Path, default=Path("artifacts/auv6dof_tmc_2e6"))
    parser.add_argument("--scenarios", default="medium,hard")
    parser.add_argument("--algos", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed-base", type=int, default=3000)
    parser.add_argument("--min-env-step", type=int, default=2_000_000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/auv6dof_tmc_2e6/best_ckpt_stress_tables"))
    args = parser.parse_args()

    scenarios = _parse_csv(args.scenarios) or ["medium", "hard"]
    algos = _parse_csv(args.algos)
    seeds = _parse_int_csv(args.seeds)
    conditions = _parse_csv(args.conditions)

    args.out.mkdir(parents=True, exist_ok=True)
    all_detail: List[Dict[str, Any]] = []
    all_tables: Dict[str, List[Dict[str, Any]]] = {}
    all_failures: List[Dict[str, Any]] = []

    for scenario in scenarios:
        detail, table, failures = evaluate_scenario(
            args.root,
            scenario,
            episodes=args.episodes,
            seed_base=args.seed_base,
            horizon=args.horizon,
            min_env_step=args.min_env_step,
            algos=algos,
            seeds=seeds,
            conditions=conditions,
        )
        all_detail.extend(detail)
        all_tables[scenario] = table
        all_failures.extend(failures)
        _write_csv(args.out / f"{scenario}_stress_table.csv", table)

    _write_csv(args.out / "stress_eval_detail.csv", all_detail)
    _write_csv(args.out / "stress_eval_failures.csv", all_failures)

    try:
        import pandas as pd

        xlsx = args.out / "best_ckpt_stress_tables.xlsx"
        with pd.ExcelWriter(xlsx) as writer:
            for scenario, rows in all_tables.items():
                pd.DataFrame(rows).to_excel(writer, sheet_name=f"{scenario}_table", index=False)
            pd.DataFrame(all_detail).to_excel(writer, sheet_name="detail", index=False)
            pd.DataFrame(all_failures).to_excel(writer, sheet_name="failures", index=False)
    except Exception:
        pass

    summary = {
        "out": args.out.as_posix(),
        "scenarios": scenarios,
        "episodes": int(args.episodes),
        "detail_rows": len(all_detail),
        "table_rows": sum(len(v) for v in all_tables.values()),
        "failures": len(all_failures),
    }
    (args.out / "stress_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

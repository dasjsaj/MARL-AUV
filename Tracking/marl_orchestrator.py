from __future__ import annotations

"""
OpenMARL 统一编排层（AUV6DOF + Tracking）。

这是本项目的核心调度模块，负责把环境、算法、配置和训练入口统一起来。

模块职责：
1) 环境注册与维度推断
   - 统一管理 `tracking` 与 `auv6dof` 环境。
   - 自动推断 `n_agent/obs_dim/action_dim`，减少手工维度错误。
2) 算法注册
   - 集中维护 13 个 MARL 算法的策略类型、训练流水线、动作模式和默认超参。
3) 配置契约输出
   - 统一生成 `env_cfg/algo_cfg/train_cfg/eval_cfg/seed_cfg`。
   - 旧脚本不再复制配置，统一由本模块构建。
4) 训练执行与结果落盘
   - 调用 DI-engine `serial_pipeline/serial_pipeline_onpolicy`。
   - 写出 `config.json/seed.json/learning_curve.csv/eval_curve.csv/summary.json`。

兼容性说明：
- 旧入口脚本（`Tracking_DDPG.py` 等）通过 `run_legacy_entry` 转发到本模块。
- 在受限 Windows 环境中，支持 no-op tensorboard writer 以规避权限问题。
"""

import argparse
import csv
import json
import os
import sys
import time
import types
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    gym = None

try:
    from easydict import EasyDict
except Exception:  # pragma: no cover
    class EasyDict(dict):
        """Fallback EasyDict when dependency is unavailable."""

        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, item):
            del self[item]


try:
    from ditk import logging as ditk_logging
except Exception:  # pragma: no cover
    import logging as ditk_logging


ditk_logging.getLogger().setLevel(getattr(ditk_logging, "WARNING", 30))
if gym is not None:
    sys.modules.setdefault("gym", gym)

warnings.filterwarnings(
    "ignore",
    message="`torch\\.utils\\._pytree\\._register_pytree_node` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings("ignore", message="Gym has been unmaintained since 2022.*")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

DEFAULT_CODEBOOK_SIZE = 125
DEFAULT_DISCRETE_LEVEL = 3
DEFAULT_OUTPUT_ROOT = "artifacts/runs"


def _to_plain(obj: Any) -> Any:
    """Convert nested objects into plain JSON-serializable values."""
    if isinstance(obj, EasyDict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.generic):
        return obj.item()
    if callable(obj):
        return getattr(obj, "__name__", str(obj))
    return obj


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_plain(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_tensorboard_writer() -> None:
    """Inject a no-op tensorboard writer for restricted Windows environments."""
    if os.environ.get("OPENMARL_USE_NOOP_TB", "1") == "0":
        return

    class _NoopSummaryWriter:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def close(self):
            return None

        def flush(self):
            return None

        def __getattr__(self, _name):
            def _noop(*args, **kwargs):
                del args, kwargs
                return None

            return _noop

    stub = types.ModuleType("tensorboardX")
    stub.SummaryWriter = _NoopSummaryWriter
    sys.modules["tensorboardX"] = stub


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``patch`` into ``base``."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _inject_hparam_overrides(policy_cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply dotted-path hyperparameter overrides to policy config."""
    if not overrides:
        return policy_cfg

    for key, value in overrides.items():
        if "." in key:
            cursor = policy_cfg
            parts = key.split(".")
            for part in parts[:-1]:
                if part not in cursor or not isinstance(cursor[part], dict):
                    cursor[part] = {}
                cursor = cursor[part]
            cursor[parts[-1]] = value
            continue

        if key in policy_cfg:
            policy_cfg[key] = value
            continue
        policy_cfg.setdefault("learn", {})[key] = value
    return policy_cfg


def _infer_tracking_shapes(n_agent_fallback: int = 8) -> Dict[str, int]:
    """Infer dimensions from legacy Tracking env; fallback to deterministic defaults."""
    try:
        from make_world import MakeWorld
        from multienvironment import Multiagent

        world_res = MakeWorld()
        world = world_res.make_world()
        env = Multiagent(world, world_res, MakeWorld.rest_world, MakeWorld.reward, MakeWorld.observation)
        n_agent = len(env.agents)
        obs_dim = int(env.observation_space[0].shape[0])
        act_dim = int(env.action_space[0].shape[0])
        return {
            "n_agent": n_agent,
            "agent_obs_dim": obs_dim,
            "global_obs_dim": n_agent * obs_dim,
            "action_dim_continuous": act_dim,
            "action_dim_discrete": int(3**act_dim),
        }
    except Exception:
        try:
            from Tracking.make_world import MakeWorld
            from Tracking.multienvironment import Multiagent

            world_res = MakeWorld()
            world = world_res.make_world()
            env = Multiagent(world, world_res, MakeWorld.rest_world, MakeWorld.reward, MakeWorld.observation)
            n_agent = len(env.agents)
            obs_dim = int(env.observation_space[0].shape[0])
            act_dim = int(env.action_space[0].shape[0])
            return {
                "n_agent": n_agent,
                "agent_obs_dim": obs_dim,
                "global_obs_dim": n_agent * obs_dim,
                "action_dim_continuous": act_dim,
                "action_dim_discrete": int(3**act_dim),
            }
        except Exception:
            n_agent = int(n_agent_fallback)
            obs_dim = 33
            act_dim = 3
            return {
                "n_agent": n_agent,
                "agent_obs_dim": obs_dim,
                "global_obs_dim": n_agent * obs_dim,
                "action_dim_continuous": act_dim,
                "action_dim_discrete": int(3**act_dim),
            }


def _infer_auv6dof_shapes(n_agent_fallback: int = 4) -> Dict[str, int]:
    """Infer dimensions from AUV6DOF Gym env; fallback to deterministic defaults."""
    try:
        try:
            from auv6dof.gym_env import AUV6DOFGymEnv
        except Exception:
            from Tracking.auv6dof.gym_env import AUV6DOFGymEnv

        env = AUV6DOFGymEnv({"n_agent": n_agent_fallback})
        obs_dim = int(env.observation_space["agent_state"].shape[-1])
        n_agent = int(env.observation_space["agent_state"].shape[0])
        act_dim = int(env.action_space.shape[-1])
        env.close()
        return {
            "n_agent": n_agent,
            "agent_obs_dim": obs_dim,
            "global_obs_dim": n_agent * obs_dim,
            "action_dim_continuous": act_dim,
            "action_dim_discrete": int(3**act_dim),
        }
    except Exception:
        n_agent = int(n_agent_fallback)
        # default features:
        # self_pos(3)+att_sin_cos(6)+self_vel(3)+self_omega(3)+target_rel(3)+other_rel(3*(n-1))
        # + boundary_margin(3) + target_velocity(3) + relative_velocity(3)
        # + target_rel_body(3) + relative_velocity_body(3) + los_unit_body(3) + prev_action(6)
        obs_dim = 15 + 3 + 3 * (n_agent - 1) + 3 + 3 + 3 + 3 + 3 + 6 + 3
        act_dim = 6
        return {
            "n_agent": n_agent,
            "agent_obs_dim": obs_dim,
            "global_obs_dim": n_agent * obs_dim,
            "action_dim_continuous": act_dim,
            "action_dim_discrete": int(3**act_dim),
        }


ENV_REGISTRY: Dict[str, Dict[str, Any]] = {
    "tracking": {
        "env_type": "tracking_di",
        "import_names": ["di_envs.tracking_di_env"],
        "shape_infer_fn": _infer_tracking_shapes,
        "default_n_agent": 8,
    },
    "auv6dof": {
        "env_type": "auv6dof_di",
        "import_names": ["di_envs.auv6dof_di_env"],
        "shape_infer_fn": _infer_auv6dof_shapes,
        "default_n_agent": 4,
    },
}


def _algo_entry(
    policy_type: str,
    pipeline_type: str,
    collector_type: str,
    action_mode: str,
    exp_name: str,
    max_env_step: int,
    *,
    requires_action_mask: bool = False,
    collector_env_num: int = 1,
    evaluator_env_num: int = 1,
    env_manager_type: str = "subprocess",
    policy_template: Optional[Dict[str, Any]] = None,
    tuning_space: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "policy_type": policy_type,
        "pipeline_type": pipeline_type,
        "collector_type": collector_type,
        "action_mode": action_mode,
        "requires_action_mask": requires_action_mask,
        "exp_name": exp_name,
        "env_manager_type": env_manager_type,
        "max_env_step": max_env_step,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "policy_template": policy_template or {},
        "tuning_space": tuning_space or {},
    }


_DISCRETE_Q_POLICY = {
    "cuda": False,
    "learn": {"update_per_collect": 100, "batch_size": 64, "learning_rate": 3e-4, "target_update_theta": 0.001, "discount_factor": 0.95},
    "collect": {"n_sample": 600, "unroll_len": 16},
    "eval": {"evaluator": {"eval_freq": 500}},
    "other": {"eps": {"type": "exp", "start": 1.0, "end": 0.05, "decay": 200000}},
}

ALGO_REGISTRY: Dict[str, Dict[str, Any]] = {
    "maddpg": _algo_entry("ddpg", "offpolicy", "sample", "continuous", "MADDPG-Data", int(1.5e6), collector_env_num=4, evaluator_env_num=2, policy_template={"cuda": False, "on_policy": False, "multi_agent": True, "reward_batch_norm": True, "random_collect_size": 10000, "learn": {"update_per_collect": 50, "batch_size": 512, "learning_rate_actor": 3e-4, "learning_rate_critic": 3e-4, "target_theta": 0.005, "discount_factor": 0.95, "ignore_done": False}, "collect": {"n_sample": 3200, "unroll_len": 1, "noise_sigma": 0.15}, "eval": {"evaluator": {"eval_freq": 500}}, "other": {"replay_buffer": {"replay_buffer_size": int(1e6)}}}, tuning_space={"learning_rate_actor": [1e-4, 3e-4, 5e-4], "learning_rate_critic": [1e-4, 3e-4, 5e-4]}),
    "matd3": _algo_entry("td3", "offpolicy", "sample", "continuous", "MATD3-Data", int(1.5e6), collector_env_num=4, evaluator_env_num=2, policy_template={"cuda": False, "on_policy": False, "multi_agent": True, "reward_batch_norm": True, "random_collect_size": 10000, "learn": {"update_per_collect": 50, "batch_size": 512, "learning_rate_actor": 3e-4, "learning_rate_critic": 3e-4, "target_theta": 0.005, "discount_factor": 0.95, "actor_update_freq": 2, "noise": True, "noise_sigma": 0.2, "noise_range": {"min": -0.5, "max": 0.5}}, "collect": {"n_sample": 3200, "unroll_len": 1, "noise_sigma": 0.15}, "eval": {"evaluator": {"eval_freq": 500}}, "other": {"replay_buffer": {"replay_buffer_size": int(1e6)}}}),
    "masac": _algo_entry("discrete_sac", "offpolicy", "sample", "discrete", "MASAC-Data", int(1.5e6), collector_env_num=4, evaluator_env_num=2, requires_action_mask=True, policy_template={"cuda": False, "on_policy": False, "multi_agent": True, "random_collect_size": 10000, "learn": {"update_per_collect": 50, "batch_size": 512, "learning_rate_q": 3e-4, "learning_rate_policy": 3e-4, "learning_rate_alpha": 3e-5, "target_theta": 0.005, "discount_factor": 0.95, "alpha": 0.2, "auto_alpha": True}, "collect": {"n_sample": 3200, "unroll_len": 1}, "eval": {"evaluator": {"eval_freq": 500}}, "other": {"eps": {"type": "linear", "start": 1.0, "end": 0.05, "decay": 200000}, "replay_buffer": {"replay_buffer_size": int(1e6)}}}),
    "mappo": _algo_entry("ppo", "onpolicy", "sample", "continuous", "MAPPO-Data", int(3e6), collector_env_num=8, evaluator_env_num=2, requires_action_mask=False, policy_template={"cuda": False, "multi_agent": True, "action_space": "continuous", "learn": {"epoch_per_collect": 10, "batch_size": 800, "learning_rate": 5e-4, "value_weight": 0.5, "entropy_weight": 0.001, "clip_ratio": 0.2, "adv_norm": True, "value_norm": True, "ppo_param_init": True, "grad_clip_type": "clip_norm", "grad_clip_value": 5, "ignore_done": False, "tanh_squash_action": True}, "collect": {"n_sample": 6400, "unroll_len": 1, "discount_factor": 0.95, "gae_lambda": 0.95, "tanh_squash_action": True, "action_range": {"min": -1.0, "max": 1.0}}, "eval": {"evaluator": {"eval_freq": 1000}}, "other": {}}),
    "stg_mappo": _algo_entry("ppo", "onpolicy", "sample", "continuous", "STG-MAPPO-Data", int(3e6), collector_env_num=8, evaluator_env_num=2, requires_action_mask=False, policy_template={"cuda": False, "multi_agent": True, "action_space": "continuous", "learn": {"epoch_per_collect": 10, "batch_size": 800, "learning_rate": 3e-4, "value_weight": 0.5, "entropy_weight": 0.001, "clip_ratio": 0.2, "adv_norm": True, "value_norm": True, "ppo_param_init": True, "grad_clip_type": "clip_norm", "grad_clip_value": 5, "ignore_done": False, "tanh_squash_action": True}, "collect": {"n_sample": 6400, "unroll_len": 1, "discount_factor": 0.95, "gae_lambda": 0.95, "tanh_squash_action": True, "action_range": {"min": -1.0, "max": 1.0}}, "eval": {"evaluator": {"eval_freq": 1000}}, "other": {}}),
    "happo": _algo_entry("happo", "onpolicy", "sample", "continuous", "HAPPO-Data", int(3e6), collector_env_num=8, evaluator_env_num=2, requires_action_mask=False, env_manager_type="base", policy_template={"cuda": False, "multi_agent": True, "agent_num": 4, "action_space": "continuous", "learn": {"epoch_per_collect": 10, "batch_size": 800, "learning_rate": 5e-4, "critic_learning_rate": 5e-4, "value_weight": 0.5, "entropy_weight": 0.001, "clip_ratio": 0.2, "adv_norm": True, "value_norm": True, "ppo_param_init": True, "grad_clip_type": "clip_norm", "grad_clip_value": 3, "ignore_done": False, "tanh_squash_action": True}, "collect": {"n_sample": 6400, "unroll_len": 1, "discount_factor": 0.95, "gae_lambda": 0.95, "tanh_squash_action": True, "action_range": {"min": -1.0, "max": 1.0}}, "eval": {"evaluator": {"eval_freq": 1000}}, "other": {}}),
    "madqn": _algo_entry("madqn", "offpolicy", "episode", "discrete", "MADQN-Data", int(1.5e6), requires_action_mask=True, collector_env_num=4, evaluator_env_num=2, policy_template={"cuda": False, "nstep": 3, "learn": {"update_per_collect": 20, "batch_size": 64, "learning_rate": 3e-4, "clip_value": 5, "target_update_theta": 0.008, "discount_factor": 0.95}, "collect": {"collector": {"get_train_sample": True}, "n_episode": 16, "unroll_len": 10}, "eval": {"evaluator": {"eval_freq": 500}}, "other": {"eps": {"type": "linear", "start": 1.0, "end": 0.05, "decay": 50000}, "replay_buffer": {"replay_buffer_size": 30000, "max_reuse": 1e9, "max_staleness": 1e9}}}),
    "qmix": _algo_entry("qmix", "offpolicy", "episode", "discrete", "QMIX-Data", int(1.5e6), collector_env_num=4, evaluator_env_num=2, requires_action_mask=True, policy_template={**deepcopy(_DISCRETE_Q_POLICY), "collect": {"n_episode": 16, "unroll_len": 10}, "other": {"eps": {"type": "linear", "start": 1.0, "end": 0.05, "decay": 50000}, "replay_buffer": {"replay_buffer_size": 30000, "max_reuse": 1e9, "max_staleness": 1e9}}}),
    "vdn": _algo_entry("qmix", "offpolicy", "episode", "discrete", "VDN-Data", int(1.5e6), collector_env_num=4, evaluator_env_num=2, requires_action_mask=True, policy_template={**deepcopy(_DISCRETE_Q_POLICY), "collect": {"n_episode": 16, "unroll_len": 10}, "other": {"eps": {"type": "linear", "start": 1.0, "end": 0.05, "decay": 50000}, "replay_buffer": {"replay_buffer_size": 30000, "max_reuse": 1e9, "max_staleness": 1e9}}}),
    "wqmix": _algo_entry("wqmix", "offpolicy", "episode", "discrete", "WQMIX-Data", int(1.5e6), collector_env_num=4, evaluator_env_num=2, requires_action_mask=True, policy_template={**deepcopy(_DISCRETE_Q_POLICY), "learn": {**deepcopy(_DISCRETE_Q_POLICY["learn"]), "wqmix_ow": True, "alpha": 0.5}, "collect": {"n_episode": 16, "unroll_len": 10}, "other": {"eps": {"type": "linear", "start": 1.0, "end": 0.05, "decay": 200000}, "replay_buffer": {"replay_buffer_size": 30000, "max_reuse": 1e9, "max_staleness": 1e9}}}),
    "qtran": _algo_entry("qtran", "offpolicy", "episode", "discrete", "QTRAN-Data", int(1.5e6), collector_env_num=4, evaluator_env_num=2, requires_action_mask=True, policy_template={**deepcopy(_DISCRETE_Q_POLICY), "learn": {**deepcopy(_DISCRETE_Q_POLICY["learn"]), "td_weight": 1, "opt_weight": 0.1, "nopt_min_weight": 1e-4}, "collect": {"n_episode": 16, "unroll_len": 10}, "other": {"eps": {"type": "linear", "start": 1.0, "end": 0.05, "decay": 50000}, "replay_buffer": {"replay_buffer_size": 30000, "max_reuse": 1e9, "max_staleness": 1e9}}}),
    "coma": _algo_entry("qmix", "offpolicy", "episode", "discrete", "COMA-Compat-Data", int(1.5e6), collector_env_num=4, evaluator_env_num=2, requires_action_mask=True, policy_template={**deepcopy(_DISCRETE_Q_POLICY), "collect": {"n_episode": 16, "unroll_len": 10}, "other": {"eps": {"type": "linear", "start": 1.0, "end": 0.05, "decay": 50000}, "replay_buffer": {"replay_buffer_size": 30000, "max_reuse": 1e9, "max_staleness": 1e9}}}),
    "collaq": _algo_entry("collaq", "offpolicy", "episode", "discrete", "COLLAQ-Data", int(1.5e6), collector_env_num=4, evaluator_env_num=2, requires_action_mask=True, policy_template={"cuda": False, "learn": {"update_per_collect": 20, "batch_size": 64, "learning_rate": 3e-4, "target_update_theta": 0.008, "discount_factor": 0.95, "clip_value": 5, "double_q": False, "collaq_loss_weight": 1.0}, "collect": {"n_episode": 16, "unroll_len": 10}, "eval": {"evaluator": {"eval_freq": 500}}, "other": {"eps": {"type": "linear", "start": 1.0, "end": 0.05, "decay": 50000}, "replay_buffer": {"replay_buffer_size": 30000, "max_reuse": 1e9, "max_staleness": 1e9}}}),
    "atoc": _algo_entry("atoc", "offpolicy", "sample", "continuous", "ATOC-Data", int(1.5e6), collector_env_num=4, evaluator_env_num=2, requires_action_mask=False, policy_template={"cuda": False, "priority": False, "learn": {"update_per_collect": 10, "batch_size": 128, "learning_rate_actor": 3e-4, "learning_rate_critic": 3e-4, "ignore_done": False, "target_theta": 0.005, "discount_factor": 0.95, "communication": True, "actor_update_freq": 1, "noise": True, "noise_sigma": 0.15, "noise_range": {"min": -0.5, "max": 0.5}}, "collect": {"n_sample": 1600, "noise_sigma": 0.2, "unroll_len": 1}, "eval": {"evaluator": {"eval_freq": 500}}, "other": {"replay_buffer": {"replay_buffer_size": 200000}}}),
}

# 统一公开算法展示顺序，供脚本、README 和看板复用。
ALGO_ORDER = [
    "atoc",
    "collaq",
    "coma",
    "happo",
    "maddpg",
    "madqn",
    "mappo",
    "stg_mappo",
    "masac",
    "qmix",
    "qtran",
    "vdn",
    "wqmix",
    "matd3",
]

TRAJECTORY_EVAL_ALGOS = {"maddpg", "matd3"}


def _parse_csv_values(raw: str) -> List[str]:
    """Parse comma-separated CLI values into a clean list."""
    if not raw:
        return []
    return [v.strip() for v in raw.split(",") if v.strip()]


def _discover_latest_run_dir(
    output_root: str,
    env_name: str,
    algo_name: str,
    *,
    seed: Optional[int] = None,
    require_checkpoint: bool = False,
) -> Path:
    """Find the latest run directory for the given env/algo(/seed) tuple."""
    base_dir = Path(output_root) / env_name / algo_name
    if seed is not None:
        base_dir = base_dir / f"seed_{seed}"
    if not base_dir.exists():
        raise FileNotFoundError(f"Run root not found: {base_dir.as_posix()}")

    candidates = {cfg.parent.resolve() for cfg in base_dir.rglob("config.json") if cfg.is_file()}
    if require_checkpoint:
        candidates = {
            run_dir
            for run_dir in candidates
            if (run_dir / "exp" / "ckpt").exists()
            and any(p.is_file() for p in (run_dir / "exp" / "ckpt").rglob("*"))
        }
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    if not candidates:
        if require_checkpoint:
            raise FileNotFoundError(
                f"No completed run with checkpoint found under {base_dir.as_posix()}"
            )
        raise FileNotFoundError(f"No run directory with config.json found under {base_dir.as_posix()}")
    return candidates[-1]


def _resolve_run_dir(
    explicit_run_dir: str,
    *,
    output_root: str,
    env_name: str,
    algo_name: str,
    seed: Optional[int],
    require_checkpoint: bool = False,
) -> Path:
    """Resolve an explicit run dir or auto-discover the latest matching run."""
    if explicit_run_dir.strip():
        run_dir = Path(explicit_run_dir).expanduser().resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run dir not found: {run_dir.as_posix()}")
        return run_dir
    return _discover_latest_run_dir(
        output_root,
        env_name,
        algo_name,
        seed=seed,
        require_checkpoint=require_checkpoint,
    )


def _build_model_config(algo_name: str, shapes: Dict[str, int]) -> Dict[str, Any]:
    """Build per-algorithm model shape config from inferred environment dimensions."""
    n_agent = int(shapes["n_agent"])
    agent_obs_dim = int(shapes["agent_obs_dim"])
    global_obs_dim = int(shapes["global_obs_dim"])
    action_dim_cont = int(shapes["action_dim_continuous"])
    action_dim_dis = int(shapes["action_dim_discrete"])

    if algo_name == "maddpg":
        return {
            "agent_obs_shape": agent_obs_dim,
            "global_obs_shape": global_obs_dim,
            "action_shape": action_dim_cont,
            "agent_num": n_agent,
            "action_space": "regression",
            "actor_head_hidden_size": 512,
            "critic_head_hidden_size": 512,
            "twin_critic": False,
            "critic_use_joint_action": True,
        }
    if algo_name == "matd3":
        return {
            "agent_obs_shape": agent_obs_dim,
            "global_obs_shape": global_obs_dim,
            "action_shape": action_dim_cont,
            "action_space": "regression",
            "actor_head_hidden_size": 512,
            "critic_head_hidden_size": 512,
            "twin_critic": True,
        }
    if algo_name == "masac":
        return {
            "agent_obs_shape": agent_obs_dim,
            "global_obs_shape": global_obs_dim,
            "action_shape": action_dim_dis,
            "twin_critic": True,
        }
    if algo_name == "madqn":
        return {"obs_shape": agent_obs_dim, "global_obs_shape": global_obs_dim, "agent_num": n_agent, "action_shape": action_dim_dis, "global_cooperation": True, "hidden_size_list": [256, 256]}
    if algo_name in {"mappo", "stg_mappo", "happo"}:
        return {
            "action_space": "continuous",
            "bound_type": "tanh",
            "sigma_type": "independent",
            "agent_num": n_agent,
            "agent_obs_shape": agent_obs_dim,
            "global_obs_shape": global_obs_dim + agent_obs_dim,
            "action_shape": action_dim_cont,
        }
    if algo_name == "coma":
        return {
            "agent_num": n_agent,
            "obs_shape": agent_obs_dim,
            "global_obs_shape": global_obs_dim,
            "action_shape": action_dim_dis,
            "hidden_size_list": [128, 128, 64],
            "mixer": True,
            "lstm_type": "gru",
            "dueling": False,
        }
    if algo_name == "atoc":
        return {
            "obs_shape": agent_obs_dim,
            "action_shape": action_dim_cont,
            "n_agent": n_agent,
            "communication": True,
            "thought_size": 16,
            "agent_per_group": max(1, min(n_agent // 2, 5)),
        }
    if algo_name == "collaq":
        # Split observation into self/ally slices with robust clipping.
        self_high = min(12, agent_obs_dim)
        ally_low = self_high
        ally_high = min(agent_obs_dim, ally_low + 3 * max(0, n_agent - 1))
        if ally_high <= ally_low:
            ally_low = max(0, agent_obs_dim // 2)
            ally_high = agent_obs_dim
        return {
            "agent_num": n_agent,
            "obs_shape": agent_obs_dim,
            "alone_obs_shape": agent_obs_dim,
            "global_obs_shape": global_obs_dim,
            "action_shape": action_dim_dis,
            "hidden_size_list": [128, 128, 64],
            "attention": False,
            "self_feature_range": [0, self_high],
            "ally_feature_range": [ally_low, ally_high],
            "attention_size": 32,
            "mixer": True,
            "lstm_type": "gru",
            "dueling": False,
        }
    if algo_name == "qtran":
        return {"agent_num": n_agent, "obs_shape": agent_obs_dim, "global_obs_shape": global_obs_dim, "action_shape": action_dim_dis, "hidden_size_list": [128], "embedding_size": 64, "lstm_type": "gru", "dueling": False}
    if algo_name == "vdn":
        return {"agent_num": n_agent, "obs_shape": agent_obs_dim, "global_obs_shape": global_obs_dim, "action_shape": action_dim_dis, "hidden_size_list": [128, 128, 64], "mixer": False}
    if algo_name == "qmix":
        return {"agent_num": n_agent, "obs_shape": agent_obs_dim, "global_obs_shape": global_obs_dim, "action_shape": action_dim_dis, "hidden_size_list": [128, 128, 64], "mixer": True}
    if algo_name == "wqmix":
        return {"agent_num": n_agent, "obs_shape": agent_obs_dim, "global_obs_shape": global_obs_dim, "action_shape": action_dim_dis, "hidden_size_list": [128, 128, 64]}
    raise KeyError(f"Unsupported algo model config: {algo_name}")


def _prepare_run_dir(output_root: str, env_name: str, algo_name: str, seed: int, run_tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = run_tag.strip() if run_tag else timestamp
    run_dir = Path(output_root) / env_name / algo_name / f"seed_{seed}" / tag
    suffix = 1
    while run_dir.exists():
        run_dir = Path(output_root) / env_name / algo_name / f"seed_{seed}" / f"{tag}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_run_contract(
    env_name: str,
    algo_name: str,
    *,
    seed: int = 0,
    max_env_step: Optional[int] = None,
    n_agent: Optional[int] = None,
    episode_length: int = 400,
    shared_reward: bool = False,
    print_episode_reward: bool = True,
    collector_env_num: Optional[int] = None,
    evaluator_env_num: Optional[int] = None,
    eval_interval_steps: Optional[int] = None,
    eval_horizon_steps: int = 0,
    codebook_size: int = DEFAULT_CODEBOOK_SIZE,
    discrete_level: int = DEFAULT_DISCRETE_LEVEL,
    env_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build normalized run contract used by all scripts/runners."""
    env_spec = ENV_REGISTRY[env_name]
    algo_spec = ALGO_REGISTRY[algo_name]
    inferred = env_spec["shape_infer_fn"](n_agent or env_spec["default_n_agent"])
    if n_agent is not None:
        inferred["n_agent"] = int(n_agent)
        inferred["global_obs_dim"] = int(n_agent) * int(inferred["agent_obs_dim"])

    discrete_action = algo_spec["action_mode"] == "discrete"
    if discrete_action:
        inferred["action_dim_discrete"] = int(codebook_size)

    collector_num = int(collector_env_num or algo_spec["collector_env_num"])
    evaluator_num = int(evaluator_env_num or algo_spec["evaluator_env_num"])
    eval_freq_default = int(algo_spec["policy_template"].get("eval", {}).get("evaluator", {}).get("eval_freq", 100))
    eval_freq = int(eval_interval_steps or eval_freq_default)
    env_manager_type = str(algo_spec["env_manager_type"])
    if os.environ.get("OPENMARL_FORCE_BASE_MANAGER", "1") == "1":
        env_manager_type = "base"

    # Cooperative tracking defaults to shared reward, but callers can override it.
    shared_reward_enabled = True if env_name == "auv6dof" else bool(shared_reward)
    env_cfg: Dict[str, Any] = {
        "collector_env_num": collector_num,
        "evaluator_env_num": evaluator_num,
        "n_evaluator_episode": evaluator_num,
        "stop_value": 1e10,
        "n_agent": int(inferred["n_agent"]),
        "episode_length": int(episode_length),
        "shared_reward": shared_reward_enabled,
        "print_episode_reward": bool(print_episode_reward),
        "discrete_action": discrete_action,
        "agent_obs_only": algo_name in {"atoc"},
        "agent_specific_global_state": algo_name in {"mappo", "stg_mappo", "happo", "masac"},
        "global_state_per_agent": algo_name in {"maddpg", "matd3", "madqn", "masac"},
        "wrap_obs_key": algo_name in {"masac"},
        "augment_coma_obs": False,
        "codebook_size": int(codebook_size),
        "discrete_level": int(discrete_level),
        "eval_horizon_steps": int(eval_horizon_steps),
        "tail_eval_steps": 5,
        "eval_interval_steps": int(eval_freq),
        "verbose_eval": True,
        "eval_log_format": "pretty",
        "action_scale": 0.18 if env_name == "auv6dof" else 1.0,
        "action_control_mode": "tau6",
        "control_mode": "direct_tau",
        "residual_baseline_scale": 0.25,
        "residual_los_speed": 0.25,
        "residual_vel_gain": 0.35,
        "residual_att_gain": 0.20,
        "residual_rate_gain": 0.12,
        "boundary_guard_enabled": True,
        "boundary_guard_ratio": 0.70,
        "boundary_guard_gain": 0.55,
        "boundary_outward_damping": 0.15,
        "separation_guard_enabled": True,
        "separation_guard_distance": 0.12,
        "separation_guard_gain": 0.45,
    }
    if env_name == "auv6dof":
        env_cfg.update(
            {
                "boundary_limit": 1.0,
                "dt": 0.1,
                "auv_model": {
                    "profile": "remus100_mss",
                    "overrides": {},
                },
                "normalize_obs": True,
                "normalize_obs_in_eval": False,
                "normalize_obs_eval_mode": "frozen",
                "reward": {
                    "version": "v2_fast_converge",
                    "d_target_min": 0.015,
                    "d_auv_min": 0.015,
                    "near_target_scale": 0.2,
                    "collision_scale": 5.0,
                    "pos_weight": 0.95,
                    "col_weight": 0.05,
                    "distance_clip": 1.2,
                    "near_distance": 0.10,
                    "safe_distance": 0.08,
                    "success_distance": 0.025,
                    "boundary_soft_ratio": 0.75,
                    "progress_clip": 0.04,
                    "closing_speed_clip": 0.08,
                    "attitude_angle_clip": 0.75,
                    "angular_rate_clip": 1.2,
                    "w_centroid_distance": 0.0,
                    "w_centroid_progress": 0.0,
                    "w_centroid_near": 0.0,
                    "w_centroid_success": 0.0,
                    "w_distance": 0.25,
                    "w_progress": 0.40,
                    "w_closing_speed": 0.20,
                    "w_near": 0.10,
                    "w_success": 0.10,
                    "w_separation": 0.03,
                    "w_boundary": 0.03,
                    "w_collision": 0.03,
                    "w_oob": 0.02,
                    "w_unstable": 0.01,
                    "w_attitude_stability": 0.01,
                    "w_angular_rate": 0.01,
                    "w_action_energy": 0.015,
                    "w_action_smooth": 0.015,
                    "w_tracking_group": 0.85,
                    "w_safety_group": 0.12,
                    "w_action_group": 0.03,
                },
                "obs": {
                    "include_target_velocity": True,
                    "include_relative_velocity": True,
                    "include_target_rel_body": True,
                    "include_relative_velocity_body": True,
                    "include_los_unit_body": True,
                    "include_prev_action": True,
                    "use_attitude_sin_cos": True,
                    "include_boundary_margin": True,
                    "normalize_physical": True,
                },
                "reset": {
                    "curriculum_stage": "auto",
                    "train_curriculum_stage": "auto",
                    "eval_curriculum_stage": "medium",
                    "min_init_separation": 0.10,
                    "auto_easy_episodes": 800,
                    "auto_medium_episodes": 2000,
                    "easy_target_speed_range": [0.001, 0.004],
                    "medium_target_speed_range": [0.003, 0.008],
                    "hard_target_speed_range": [0.006, 0.014],
                },
            }
        )
    if env_overrides:
        _deep_update(env_cfg, deepcopy(env_overrides))
    if env_name == "auv6dof":
        # Keep model shape and environment observation contract in sync when obs flags are overridden.
        obs_cfg = env_cfg.get("obs", {})
        if str(env_cfg.get("action_control_mode", "tau6")).strip().lower() == "velocity3":
            inferred["action_dim_continuous"] = 3
        include_target_velocity = bool(obs_cfg.get("include_target_velocity", True))
        include_relative_velocity = bool(obs_cfg.get("include_relative_velocity", True))
        include_target_rel_body = bool(obs_cfg.get("include_target_rel_body", True))
        include_relative_velocity_body = bool(obs_cfg.get("include_relative_velocity_body", True))
        include_los_unit_body = bool(obs_cfg.get("include_los_unit_body", True))
        include_prev_action = bool(obs_cfg.get("include_prev_action", True))
        include_tracking_diagnostics = bool(obs_cfg.get("include_tracking_diagnostics", False))
        include_semantic_features = bool(obs_cfg.get("include_semantic_features", False))
        include_semantic_graph_features = bool(obs_cfg.get("include_semantic_graph_features", False))
        use_attitude_sin_cos = bool(obs_cfg.get("use_attitude_sin_cos", True))
        include_boundary_margin = bool(obs_cfg.get("include_boundary_margin", True))
        att_dim = 6 if use_attitude_sin_cos else 3
        obs_dim = 3 + att_dim + 3 + 3 + 3 + 3 * max(0, int(inferred["n_agent"]) - 1)
        if include_boundary_margin:
            obs_dim += 3
        if include_target_velocity:
            obs_dim += 3
        if include_relative_velocity:
            obs_dim += 3
        if include_target_rel_body:
            obs_dim += 3
        if include_relative_velocity_body:
            obs_dim += 3
        if include_los_unit_body:
            obs_dim += 3
        if include_prev_action:
            obs_dim += 6
        if include_tracking_diagnostics:
            obs_dim += 8
        if include_semantic_features:
            obs_dim += 9
        if include_semantic_graph_features:
            obs_dim += 6
        inferred["agent_obs_dim"] = int(obs_dim)
        inferred["global_obs_dim"] = int(inferred["n_agent"]) * int(obs_dim)

    algo_cfg: Dict[str, Any] = {
        "algo_name": algo_name,
        "policy_type": algo_spec["policy_type"],
        "pipeline_type": algo_spec["pipeline_type"],
        "collector_type": algo_spec["collector_type"],
        "action_mode": algo_spec["action_mode"],
        "requires_action_mask": bool(algo_spec["requires_action_mask"]),
        "exp_name": algo_spec["exp_name"],
        "default_hparams": deepcopy(algo_spec["policy_template"]),
        "default_train_budget": {"max_env_step": int(algo_spec["max_env_step"]), "collector_env_num": int(algo_spec["collector_env_num"]), "evaluator_env_num": int(algo_spec["evaluator_env_num"])},
        "tuning_space": deepcopy(algo_spec["tuning_space"]),
    }
    train_cfg: Dict[str, Any] = {
        "max_env_step": int(max_env_step or algo_spec["max_env_step"]),
        "collector_env_num": collector_num,
        "evaluator_env_num": evaluator_num,
        "env_manager_type": env_manager_type,
    }
    eval_cfg: Dict[str, Any] = {
        "n_evaluator_episode": evaluator_num,
        "eval_freq": eval_freq,
        "eval_horizon_steps": int(eval_horizon_steps),
    }
    seed_cfg: Dict[str, Any] = {"seed": int(seed)}

    return {"env_cfg": env_cfg, "algo_cfg": algo_cfg, "train_cfg": train_cfg, "eval_cfg": eval_cfg, "seed_cfg": seed_cfg, "shape_cfg": inferred, "env_spec": env_spec}


def build_experiment_config(
    env_name: str,
    algo_name: str,
    *,
    seed: int = 0,
    max_env_step: Optional[int] = None,
    n_agent: Optional[int] = None,
    episode_length: int = 400,
    shared_reward: bool = False,
    print_episode_reward: bool = True,
    collector_env_num: Optional[int] = None,
    evaluator_env_num: Optional[int] = None,
    eval_interval_steps: Optional[int] = None,
    eval_horizon_steps: int = 0,
    codebook_size: int = DEFAULT_CODEBOOK_SIZE,
    discrete_level: int = DEFAULT_DISCRETE_LEVEL,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    run_tag: str = "",
    env_overrides: Optional[Dict[str, Any]] = None,
    policy_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], EasyDict, EasyDict, int, Path]:
    """Build runtime config triplet and create artifact directory."""
    contract = build_run_contract(
        env_name=env_name,
        algo_name=algo_name,
        seed=seed,
        max_env_step=max_env_step,
        n_agent=n_agent,
        episode_length=episode_length,
        shared_reward=shared_reward,
        print_episode_reward=print_episode_reward,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        eval_interval_steps=eval_interval_steps,
        eval_horizon_steps=eval_horizon_steps,
        codebook_size=codebook_size,
        discrete_level=discrete_level,
        env_overrides=env_overrides,
    )
    run_dir = _prepare_run_dir(output_root, env_name, algo_name, seed, run_tag)

    env_cfg = deepcopy(contract["env_cfg"])
    algo_cfg = contract["algo_cfg"]
    train_cfg = contract["train_cfg"]
    shape_cfg = contract["shape_cfg"]
    env_spec = contract["env_spec"]
    env_cfg["artifact_dir"] = run_dir.as_posix()

    policy_cfg = deepcopy(algo_cfg["default_hparams"])
    policy_cfg["model"] = {**policy_cfg.get("model", {}), **_build_model_config(algo_name, shape_cfg)}
    if algo_name == "masac":
        policy_cfg.setdefault("learn", {})["target_entropy"] = -float(shape_cfg["action_dim_discrete"])
    if algo_name in {"madqn", "coma", "collaq", "qmix", "vdn", "wqmix", "qtran", "stg_mappo", "happo"}:
        policy_cfg["agent_num"] = int(shape_cfg["n_agent"])
    if "collect" in policy_cfg:
        policy_cfg["collect"]["env_num"] = int(train_cfg["collector_env_num"])
    policy_cfg.setdefault("eval", {})
    policy_cfg["eval"]["env_num"] = int(train_cfg["evaluator_env_num"])
    policy_cfg["eval"].setdefault("evaluator", {})
    policy_cfg["eval"]["evaluator"]["eval_freq"] = int(contract["eval_cfg"]["eval_freq"])
    policy_cfg = _inject_hparam_overrides(policy_cfg, policy_overrides)

    main_config = EasyDict({"exp_name": (run_dir / "exp").as_posix(), "env": env_cfg, "policy": policy_cfg})
    create_config = EasyDict({"env": {"import_names": env_spec["import_names"], "type": env_spec["env_type"]}, "env_manager": {"type": train_cfg["env_manager_type"]}, "policy": {"type": algo_cfg["policy_type"]}, "collector": {"type": algo_cfg["collector_type"]}})
    if algo_cfg["collector_type"] == "episode":
        create_config["collector"]["get_train_sample"] = True
    max_steps = int(train_cfg["max_env_step"])

    _write_json(run_dir / "config.json", {"contract": contract, "main_config": main_config, "create_config": create_config, "max_env_step": max_steps})
    _write_json(run_dir / "seed.json", {"seed": int(seed)})
    return contract, main_config, create_config, max_steps, run_dir


def _discover_latest_checkpoint(exp_dir: Path) -> Optional[Path]:
    if not exp_dir.exists():
        return None
    best_candidates: List[Path] = []
    for pattern in ("ckpt_best*.pth", "ckpt_best*.pth.tar", "ckpt_best*.pt"):
        best_candidates.extend(exp_dir.rglob(pattern))
    if best_candidates:
        best_candidates = sorted(best_candidates, key=lambda p: p.stat().st_mtime)
        return best_candidates[-1]

    ckpts: List[Path] = []
    for pattern in ("*.pth", "*.pth.tar", "*.pt"):
        ckpts.extend(exp_dir.rglob(pattern))
    ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def _write_curve(path: Path, rewards: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards.tolist()):
            writer.writerow([i, r])


def _read_reward_curve_csv(path: Path) -> np.ndarray:
    """Parse monitor curve csv with schema ``episode,reward``."""
    rows: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                rows.append(float(row[1]))
            except Exception:
                continue
    return np.asarray(rows, dtype=np.float32)


def _load_reward_trace(run_dir: Path) -> np.ndarray:
    """Load reward trajectory from monitor artifacts with retries."""
    for _ in range(40):
        npy_files = sorted(run_dir.glob("*reward_data.npy"), key=lambda p: p.stat().st_mtime)
        for npy_file in reversed(npy_files):
            try:
                rewards = np.asarray(np.load(npy_file), dtype=np.float32).reshape(-1)
                if rewards.size > 0:
                    return rewards
            except Exception:
                continue

        curve_files = sorted(run_dir.glob("*reward_curve.csv"), key=lambda p: p.stat().st_mtime)
        for curve_file in reversed(curve_files):
            try:
                rewards = _read_reward_curve_csv(curve_file)
                if rewards.size > 0:
                    return rewards
            except Exception:
                continue
        time.sleep(0.2)
    return np.asarray([], dtype=np.float32)


def _load_eval_detail_raw(run_dir: Path) -> List[Dict[str, Any]]:
    """Load evaluator raw episode records written by evaluator env instance(s)."""
    records: List[Dict[str, Any]] = []
    raw_files = sorted(run_dir.glob("auv6dof_eval_detail_raw*.csv"), key=lambda p: p.stat().st_mtime)
    for raw in raw_files:
        try:
            with raw.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(
                        {
                            "eval_index": int(row.get("eval_index", 0)),
                            "train_step": int(row.get("train_step", 0)),
                            "eval_return": float(row.get("eval_return", row.get("episode_return", 0.0))),
                            "episode_return": float(row.get("episode_return", 0.0)),
                            "eval_steps": int(row.get("eval_steps", row.get("episode_steps", 0))),
                            "episode_steps": int(row.get("episode_steps", 0)),
                            "horizon_steps": int(row.get("horizon_steps", 0)),
                            "collision": int(row.get("collision", 0)),
                            "out_of_bounds": int(row.get("out_of_bounds", 0)),
                            "unstable": int(row.get("unstable", 0)),
                            "final_centroid_target_distance": float(
                                row.get("final_centroid_target_distance", row.get("final_mean_target_distance", 0.0))
                            ),
                            "mean_centroid_target_distance_over_episode": float(
                                row.get(
                                    "mean_centroid_target_distance_over_episode",
                                    row.get("mean_target_distance_over_episode", 0.0),
                                )
                            ),
                            "tail_mean_centroid_target_distance": float(
                                row.get(
                                    "tail_mean_centroid_target_distance",
                                    row.get("final_centroid_target_distance", 0.0),
                                )
                            ),
                            "final_mean_target_distance": float(
                                row.get("final_mean_target_distance", 0.0)
                            ),
                            "mean_target_distance": float(
                                row.get("mean_target_distance", row.get("mean_target_distance_over_episode", 0.0))
                            ),
                            "mean_target_distance_over_episode": float(
                                row.get("mean_target_distance_over_episode", 0.0)
                            ),
                            "tail_mean_target_distance": float(
                                row.get("tail_mean_target_distance", row.get("final_mean_target_distance", 0.0))
                            ),
                            "tail100_mean_target_distance": float(
                                row.get(
                                    "tail100_mean_target_distance",
                                    row.get("tail_mean_target_distance", row.get("final_mean_target_distance", 0.0)),
                                )
                            ),
                            "tail100_std_target_distance": float(row.get("tail100_std_target_distance", 0.0)),
                            "tail100_mean_tracking_error": float(
                                row.get("tail100_mean_tracking_error", row.get("mean_tracking_error", 0.0))
                            ),
                            "tail100_target_lost_rate": float(
                                row.get("tail100_target_lost_rate", row.get("mean_target_lost", 0.0))
                            ),
                            "tail100_action_norm": float(row.get("tail100_action_norm", 0.0)),
                            "tail100_action_saturation_rate": float(
                                row.get(
                                    "tail100_action_saturation_rate",
                                    row.get("mean_action_saturation_rate", 0.0),
                                )
                            ),
                            "mean_centroid_distance_term": float(row.get("mean_centroid_distance_term", 0.0)),
                            "mean_centroid_progress_term": float(row.get("mean_centroid_progress_term", 0.0)),
                            "mean_centroid_near_term": float(row.get("mean_centroid_near_term", 0.0)),
                            "mean_distance_term": float(row.get("mean_distance_term", 0.0)),
                            "mean_progress_term": float(row.get("mean_progress_term", 0.0)),
                            "mean_closing_speed_term": float(row.get("mean_closing_speed_term", 0.0)),
                            "mean_near_term": float(row.get("mean_near_term", 0.0)),
                            "mean_success_term": float(row.get("mean_success_term", 0.0)),
                            "mean_attitude_stability_term": float(row.get("mean_attitude_stability_term", 0.0)),
                            "mean_angular_rate_term": float(row.get("mean_angular_rate_term", 0.0)),
                            "mean_safety_term": float(row.get("mean_safety_term", 0.0)),
                            "mean_action_reg_term": float(row.get("mean_action_reg_term", 0.0)),
                            "mean_tracking_group_term": float(row.get("mean_tracking_group_term", 0.0)),
                            "mean_safety_group_term": float(row.get("mean_safety_group_term", 0.0)),
                            "mean_action_group_term": float(row.get("mean_action_group_term", 0.0)),
                            "mean_tracking_contrib": float(row.get("mean_tracking_contrib", 0.0)),
                            "mean_safety_contrib": float(row.get("mean_safety_contrib", 0.0)),
                            "mean_action_contrib": float(row.get("mean_action_contrib", 0.0)),
                            "mean_tracking_reward": float(row.get("mean_tracking_reward", 0.0)),
                            "mean_observation_reward": float(row.get("mean_observation_reward", 0.0)),
                            "mean_coordination_reward": float(row.get("mean_coordination_reward", 0.0)),
                            "mean_communication_reward": float(row.get("mean_communication_reward", 0.0)),
                            "mean_semantic_reward": float(row.get("mean_semantic_reward", 0.0)),
                            "mean_control_cost": float(row.get("mean_control_cost", 0.0)),
                            "mean_tracking_error": float(row.get("mean_tracking_error", 0.0)),
                            "mean_tracking_error_delta": float(row.get("mean_tracking_error_delta", 0.0)),
                            "mean_observation_confidence": float(row.get("mean_observation_confidence", 0.0)),
                            "mean_target_lost": float(row.get("mean_target_lost", 0.0)),
                            "mean_communication_quality": float(row.get("mean_communication_quality", 0.0)),
                            "mean_action_clip_rate": float(row.get("mean_action_clip_rate", 0.0)),
                            "mean_action_saturation_rate": float(row.get("mean_action_saturation_rate", 0.0)),
                            "mean_action_delta_norm": float(row.get("mean_action_delta_norm", 0.0)),
                        }
                    )
        except Exception:
            continue
    records = sorted(records, key=lambda item: item["eval_index"])
    return records


def _write_eval_detail(
    path: Path, eval_records: List[Dict[str, Any]], eval_interval_steps: int, eval_horizon_steps: int
) -> None:
    keys = [
        "eval_index",
        "train_step",
        "eval_return",
        "eval_steps",
        "horizon_steps",
        "final_centroid_target_distance",
        "mean_centroid_target_distance_over_episode",
        "tail_mean_centroid_target_distance",
        "collision",
        "out_of_bounds",
        "unstable",
        "final_mean_target_distance",
        "mean_target_distance",
        "mean_target_distance_over_episode",
        "tail_mean_target_distance",
        "tail100_mean_target_distance",
        "tail100_std_target_distance",
        "tail100_mean_tracking_error",
        "tail100_target_lost_rate",
        "tail100_action_norm",
        "tail100_action_saturation_rate",
        "mean_centroid_distance_term",
        "mean_centroid_progress_term",
        "mean_centroid_near_term",
        "mean_distance_term",
        "mean_progress_term",
        "mean_closing_speed_term",
        "mean_near_term",
        "mean_success_term",
        "mean_attitude_stability_term",
        "mean_angular_rate_term",
        "mean_safety_term",
        "mean_action_reg_term",
        "mean_tracking_group_term",
        "mean_safety_group_term",
        "mean_action_group_term",
        "mean_tracking_contrib",
        "mean_safety_contrib",
        "mean_action_contrib",
        "mean_tracking_reward",
        "mean_observation_reward",
        "mean_coordination_reward",
        "mean_communication_reward",
        "mean_semantic_reward",
        "mean_control_cost",
        "mean_tracking_error",
        "mean_tracking_error_delta",
        "mean_observation_confidence",
        "mean_target_lost",
        "mean_communication_quality",
        "mean_action_clip_rate",
        "mean_action_saturation_rate",
        "mean_action_delta_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for idx, rec in enumerate(eval_records):
            train_step = int(rec.get("train_step", 0))
            if train_step <= 0:
                train_step = (idx + 1) * int(eval_interval_steps) if eval_interval_steps > 0 else idx
            row = dict(rec)
            row["eval_index"] = idx
            row["train_step"] = train_step
            row["eval_return"] = float(rec.get("eval_return", rec.get("episode_return", 0.0)))
            row["eval_steps"] = int(rec.get("eval_steps", rec.get("episode_steps", 0)))
            row["horizon_steps"] = int(rec.get("horizon_steps", eval_horizon_steps))
            row["final_centroid_target_distance"] = float(
                rec.get("final_centroid_target_distance", rec.get("final_mean_target_distance", 0.0))
            )
            row["mean_centroid_target_distance_over_episode"] = float(
                rec.get("mean_centroid_target_distance_over_episode", rec.get("mean_target_distance_over_episode", 0.0))
            )
            row["tail_mean_centroid_target_distance"] = float(
                rec.get(
                    "tail_mean_centroid_target_distance",
                    rec.get("final_centroid_target_distance", rec.get("final_mean_target_distance", 0.0)),
                )
            )
            row["tail_mean_target_distance"] = float(
                rec.get("tail_mean_target_distance", rec.get("final_mean_target_distance", 0.0))
            )
            row["mean_target_distance"] = float(
                rec.get("mean_target_distance", rec.get("mean_target_distance_over_episode", 0.0))
            )
            row["tail100_mean_target_distance"] = float(
                rec.get(
                    "tail100_mean_target_distance",
                    rec.get("tail_mean_target_distance", rec.get("final_mean_target_distance", 0.0)),
                )
            )
            row["tail100_std_target_distance"] = float(rec.get("tail100_std_target_distance", 0.0))
            row["tail100_mean_tracking_error"] = float(
                rec.get("tail100_mean_tracking_error", rec.get("mean_tracking_error", 0.0))
            )
            row["tail100_target_lost_rate"] = float(
                rec.get("tail100_target_lost_rate", rec.get("mean_target_lost", 0.0))
            )
            row["tail100_action_norm"] = float(rec.get("tail100_action_norm", 0.0))
            row["tail100_action_saturation_rate"] = float(
                rec.get("tail100_action_saturation_rate", rec.get("mean_action_saturation_rate", 0.0))
            )
            writer.writerow([row.get(key, 0.0) for key in keys])


def _finalize_run(run_dir: Path, exp_name: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    rewards = _load_reward_trace(run_dir)
    learning_curve = run_dir / "learning_curve.csv"
    eval_curve = run_dir / "eval_curve.csv"
    eval_detail = run_dir / "eval_detail.csv"
    eval_records = _load_eval_detail_raw(run_dir)
    eval_interval_steps = int(contract.get("eval_cfg", {}).get("eval_freq", 0))
    eval_horizon_steps = int(contract.get("eval_cfg", {}).get("eval_horizon_steps", 0))

    eval_returns = np.asarray(
        [float(rec.get("eval_return", rec.get("episode_return", 0.0))) for rec in eval_records],
        dtype=np.float32,
    )
    eval_agent_mean_precisions = np.asarray(
        [float(rec.get("final_mean_target_distance", 0.0)) for rec in eval_records], dtype=np.float32
    )
    eval_tail_mean_precisions = np.asarray(
        [
            float(rec.get("tail_mean_target_distance", rec.get("final_mean_target_distance", 0.0)))
            for rec in eval_records
        ],
        dtype=np.float32,
    )
    eval_tail100_target_distances = np.asarray(
        [
            float(
                rec.get(
                    "tail100_mean_target_distance",
                    rec.get("tail_mean_target_distance", rec.get("final_mean_target_distance", 0.0)),
                )
            )
            for rec in eval_records
        ],
        dtype=np.float32,
    )
    eval_tail100_tracking_errors = np.asarray(
        [float(rec.get("tail100_mean_tracking_error", rec.get("mean_tracking_error", 0.0))) for rec in eval_records],
        dtype=np.float32,
    )
    eval_tail100_target_losts = np.asarray(
        [float(rec.get("tail100_target_lost_rate", rec.get("mean_target_lost", 0.0))) for rec in eval_records],
        dtype=np.float32,
    )
    eval_tail100_action_saturations = np.asarray(
        [
            float(rec.get("tail100_action_saturation_rate", rec.get("mean_action_saturation_rate", 0.0)))
            for rec in eval_records
        ],
        dtype=np.float32,
    )
    eval_centroid_precisions = np.asarray(
        [
            float(rec.get("final_centroid_target_distance", rec.get("final_mean_target_distance", 0.0)))
            for rec in eval_records
        ],
        dtype=np.float32,
    )
    eval_tail_centroid_precisions = np.asarray(
        [
            float(
                rec.get(
                    "tail_mean_centroid_target_distance",
                    rec.get("final_centroid_target_distance", rec.get("final_mean_target_distance", 0.0)),
                )
            )
            for rec in eval_records
        ],
        dtype=np.float32,
    )
    # Precision primary metric is agent mean target distance.
    eval_precisions = eval_tail_mean_precisions if eval_tail_mean_precisions.size else eval_agent_mean_precisions
    eval_collisions = np.asarray([float(rec.get("collision", 0.0)) for rec in eval_records], dtype=np.float32)
    eval_out_bounds = np.asarray([float(rec.get("out_of_bounds", 0.0)) for rec in eval_records], dtype=np.float32)
    eval_unstable = np.asarray([float(rec.get("unstable", 0.0)) for rec in eval_records], dtype=np.float32)
    mean_distance_terms = np.asarray(
        [float(rec.get("mean_distance_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_progress_terms = np.asarray(
        [float(rec.get("mean_progress_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_closing_speed_terms = np.asarray(
        [float(rec.get("mean_closing_speed_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_near_terms = np.asarray(
        [float(rec.get("mean_near_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_success_terms = np.asarray(
        [float(rec.get("mean_success_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_attitude_stability_terms = np.asarray(
        [float(rec.get("mean_attitude_stability_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_angular_rate_terms = np.asarray(
        [float(rec.get("mean_angular_rate_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_safety_terms = np.asarray(
        [float(rec.get("mean_safety_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_action_reg_terms = np.asarray(
        [float(rec.get("mean_action_reg_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_tracking_group_terms = np.asarray(
        [float(rec.get("mean_tracking_group_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_safety_group_terms = np.asarray(
        [float(rec.get("mean_safety_group_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_action_group_terms = np.asarray(
        [float(rec.get("mean_action_group_term", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_tracking_contribs = np.asarray(
        [float(rec.get("mean_tracking_contrib", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_safety_contribs = np.asarray(
        [float(rec.get("mean_safety_contrib", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_action_contribs = np.asarray(
        [float(rec.get("mean_action_contrib", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_tracking_rewards = np.asarray(
        [float(rec.get("mean_tracking_reward", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_observation_rewards = np.asarray(
        [float(rec.get("mean_observation_reward", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_coordination_rewards = np.asarray(
        [float(rec.get("mean_coordination_reward", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_communication_rewards = np.asarray(
        [float(rec.get("mean_communication_reward", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_semantic_rewards = np.asarray(
        [float(rec.get("mean_semantic_reward", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_control_costs = np.asarray(
        [float(rec.get("mean_control_cost", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_tracking_errors = np.asarray(
        [float(rec.get("mean_tracking_error", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_target_losts = np.asarray(
        [float(rec.get("mean_target_lost", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_observation_confidences = np.asarray(
        [float(rec.get("mean_observation_confidence", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_communication_qualities = np.asarray(
        [float(rec.get("mean_communication_quality", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_action_clip_rates = np.asarray(
        [float(rec.get("mean_action_clip_rate", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_action_saturation_rates = np.asarray(
        [float(rec.get("mean_action_saturation_rate", 0.0)) for rec in eval_records], dtype=np.float32
    )
    mean_action_delta_norms = np.asarray(
        [float(rec.get("mean_action_delta_norm", 0.0)) for rec in eval_records], dtype=np.float32
    )

    def _slope_last_n(values: np.ndarray, n: int = 50) -> float | None:
        if values.size < 2:
            return None
        tail = values[-min(n, values.size):].astype(np.float64)
        if tail.size < 2:
            return None
        x = np.arange(tail.size, dtype=np.float64)
        return float(np.polyfit(x, tail, 1)[0])

    violation_flags = (eval_collisions > 0) | (eval_out_bounds > 0) | (eval_unstable > 0)
    _write_curve(learning_curve, rewards)
    _write_curve(eval_curve, eval_returns if eval_returns.size > 0 else rewards)
    if eval_records:
        _write_eval_detail(eval_detail, eval_records, eval_interval_steps, eval_horizon_steps)
    ckpt = _discover_latest_checkpoint(Path(exp_name))
    last200_window = min(200, eval_precisions.size) if eval_precisions.size else 0
    precision_mean_last200 = (
        float(np.mean(eval_precisions[-last200_window:])) if last200_window > 0 else None
    )
    summary = {
        "num_episodes": int(rewards.shape[0]),
        "best_reward": float(np.max(rewards)) if rewards.size else None,
        "mean_reward": float(np.mean(rewards)) if rewards.size else None,
        "std_reward": float(np.std(rewards)) if rewards.size else None,
        "latest_reward": float(rewards[-1]) if rewards.size else None,
        "num_eval_points": int(eval_returns.shape[0]),
        "best_eval_reward": float(np.max(eval_returns)) if eval_returns.size else None,
        "latest_eval_reward": float(eval_returns[-1]) if eval_returns.size else None,
        "best_precision": float(np.min(eval_precisions)) if eval_precisions.size else None,
        "latest_precision": float(eval_precisions[-1]) if eval_precisions.size else None,
        "mean_precision": float(np.mean(eval_precisions)) if eval_precisions.size else None,
        "std_precision": float(np.std(eval_precisions)) if eval_precisions.size else None,
        "precision_slope_last50": _slope_last_n(eval_precisions, n=50),
        "precision_slope_last200": _slope_last_n(eval_precisions, n=200),
        "precision_mean_last200": precision_mean_last200,
        "centroid_slope_last200": _slope_last_n(eval_centroid_precisions, n=200),
        "centroid_mean_last200": float(np.mean(eval_centroid_precisions[-last200_window:]))
        if last200_window > 0
        else None,
        "best_centroid_precision": float(np.min(eval_centroid_precisions)) if eval_centroid_precisions.size else None,
        "latest_centroid_precision": float(eval_centroid_precisions[-1]) if eval_centroid_precisions.size else None,
        "mean_centroid_precision": float(np.mean(eval_centroid_precisions)) if eval_centroid_precisions.size else None,
        "std_centroid_precision": float(np.std(eval_centroid_precisions)) if eval_centroid_precisions.size else None,
        "best_tail_precision": float(np.min(eval_tail_mean_precisions)) if eval_tail_mean_precisions.size else None,
        "latest_tail_precision": float(eval_tail_mean_precisions[-1]) if eval_tail_mean_precisions.size else None,
        "mean_tail_precision": float(np.mean(eval_tail_mean_precisions)) if eval_tail_mean_precisions.size else None,
        "tail_precision_slope_last50": _slope_last_n(eval_tail_mean_precisions, n=50),
        "best_tail_centroid_precision": float(np.min(eval_tail_centroid_precisions))
        if eval_tail_centroid_precisions.size
        else None,
        "latest_tail_centroid_precision": float(eval_tail_centroid_precisions[-1])
        if eval_tail_centroid_precisions.size
        else None,
        "best_agent_mean_precision": float(np.min(eval_agent_mean_precisions))
        if eval_agent_mean_precisions.size
        else None,
        "latest_agent_mean_precision": float(eval_agent_mean_precisions[-1])
        if eval_agent_mean_precisions.size
        else None,
        "eval_reward_slope_last50": _slope_last_n(eval_returns, n=50),
        "safety_violation_rate": float(np.mean(violation_flags.astype(np.float32))) if violation_flags.size else None,
        "mean_collision": float(np.mean(eval_collisions)) if eval_collisions.size else None,
        "mean_out_of_bounds": float(np.mean(eval_out_bounds)) if eval_out_bounds.size else None,
        "mean_unstable": float(np.mean(eval_unstable)) if eval_unstable.size else None,
        "mean_distance_term": float(np.mean(mean_distance_terms)) if mean_distance_terms.size else None,
        "mean_progress_term": float(np.mean(mean_progress_terms)) if mean_progress_terms.size else None,
        "mean_closing_speed_term": float(np.mean(mean_closing_speed_terms))
        if mean_closing_speed_terms.size
        else None,
        "mean_near_term": float(np.mean(mean_near_terms)) if mean_near_terms.size else None,
        "mean_success_term": float(np.mean(mean_success_terms)) if mean_success_terms.size else None,
        "mean_attitude_stability_term": float(np.mean(mean_attitude_stability_terms))
        if mean_attitude_stability_terms.size
        else None,
        "mean_angular_rate_term": float(np.mean(mean_angular_rate_terms)) if mean_angular_rate_terms.size else None,
        "mean_safety_term": float(np.mean(mean_safety_terms)) if mean_safety_terms.size else None,
        "mean_action_reg_term": float(np.mean(mean_action_reg_terms)) if mean_action_reg_terms.size else None,
        "mean_tracking_group_term": float(np.mean(mean_tracking_group_terms))
        if mean_tracking_group_terms.size
        else None,
        "mean_safety_group_term": float(np.mean(mean_safety_group_terms)) if mean_safety_group_terms.size else None,
        "mean_action_group_term": float(np.mean(mean_action_group_terms)) if mean_action_group_terms.size else None,
        "mean_tracking_contrib": float(np.mean(mean_tracking_contribs)) if mean_tracking_contribs.size else None,
        "mean_safety_contrib": float(np.mean(mean_safety_contribs)) if mean_safety_contribs.size else None,
        "mean_action_contrib": float(np.mean(mean_action_contribs)) if mean_action_contribs.size else None,
        "mean_tracking_reward": float(np.mean(mean_tracking_rewards)) if mean_tracking_rewards.size else None,
        "mean_observation_reward": float(np.mean(mean_observation_rewards)) if mean_observation_rewards.size else None,
        "mean_coordination_reward": float(np.mean(mean_coordination_rewards)) if mean_coordination_rewards.size else None,
        "mean_communication_reward": float(np.mean(mean_communication_rewards))
        if mean_communication_rewards.size
        else None,
        "mean_semantic_reward": float(np.mean(mean_semantic_rewards)) if mean_semantic_rewards.size else None,
        "mean_control_cost": float(np.mean(mean_control_costs)) if mean_control_costs.size else None,
        "mean_tracking_error": float(np.mean(mean_tracking_errors)) if mean_tracking_errors.size else None,
        "latest_tail100_mean_target_distance": float(eval_tail100_target_distances[-1])
        if eval_tail100_target_distances.size
        else None,
        "best_tail100_mean_target_distance": float(np.min(eval_tail100_target_distances))
        if eval_tail100_target_distances.size
        else None,
        "mean_tail100_mean_target_distance": float(np.mean(eval_tail100_target_distances))
        if eval_tail100_target_distances.size
        else None,
        "latest_tail100_mean_tracking_error": float(eval_tail100_tracking_errors[-1])
        if eval_tail100_tracking_errors.size
        else None,
        "latest_tail100_target_lost_rate": float(eval_tail100_target_losts[-1])
        if eval_tail100_target_losts.size
        else None,
        "latest_tail100_action_saturation_rate": float(eval_tail100_action_saturations[-1])
        if eval_tail100_action_saturations.size
        else None,
        "mean_target_lost": float(np.mean(mean_target_losts)) if mean_target_losts.size else None,
        "mean_observation_confidence": float(np.mean(mean_observation_confidences))
        if mean_observation_confidences.size
        else None,
        "mean_communication_quality": float(np.mean(mean_communication_qualities))
        if mean_communication_qualities.size
        else None,
        "tracking_error_slope_last50": _slope_last_n(mean_tracking_errors, n=50),
        "target_lost_slope_last50": _slope_last_n(mean_target_losts, n=50),
        "mean_action_clip_rate": float(np.mean(mean_action_clip_rates)) if mean_action_clip_rates.size else None,
        "mean_action_saturation_rate": float(np.mean(mean_action_saturation_rates))
        if mean_action_saturation_rates.size
        else None,
        "mean_action_delta_norm": float(np.mean(mean_action_delta_norms)) if mean_action_delta_norms.size else None,
        "best_ckpt": ckpt.as_posix() if ckpt else None,
        "learning_curve": learning_curve.as_posix(),
        "eval_curve": eval_curve.as_posix(),
        "eval_detail": eval_detail.as_posix() if eval_records else None,
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def _wait_for_reward_artifact(run_dir: Path, timeout_s: float = 30.0) -> None:
    """Wait briefly for delayed reward artifacts flushed during env teardown."""
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        npy_ready = any(p.stat().st_size > 0 for p in run_dir.glob("*reward_data.npy"))
        csv_ready = any(p.stat().st_size > 0 for p in run_dir.glob("*reward_curve.csv"))
        if npy_ready or csv_ready:
            return
        time.sleep(0.2)


def run_experiment(
    env_name: str,
    algo_name: str,
    *,
    seed: int = 0,
    max_env_step: Optional[int] = None,
    n_agent: Optional[int] = None,
    episode_length: int = 400,
    shared_reward: bool = False,
    print_episode_reward: bool = True,
    collector_env_num: Optional[int] = None,
    evaluator_env_num: Optional[int] = None,
    eval_interval_steps: Optional[int] = None,
    eval_horizon_steps: int = 0,
    codebook_size: int = DEFAULT_CODEBOOK_SIZE,
    discrete_level: int = DEFAULT_DISCRETE_LEVEL,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    run_tag: str = "",
    env_overrides: Optional[Dict[str, Any]] = None,
    policy_overrides: Optional[Dict[str, Any]] = None,
    print_config: bool = False,
) -> Path:
    """Run one training experiment and persist standardized artifacts."""
    contract, main_config, create_config, max_steps, run_dir = build_experiment_config(
        env_name=env_name,
        algo_name=algo_name,
        seed=seed,
        max_env_step=max_env_step,
        n_agent=n_agent,
        episode_length=episode_length,
        shared_reward=shared_reward,
        print_episode_reward=print_episode_reward,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        eval_interval_steps=eval_interval_steps,
        eval_horizon_steps=eval_horizon_steps,
        codebook_size=codebook_size,
        discrete_level=discrete_level,
        output_root=output_root,
        run_tag=run_tag,
        env_overrides=env_overrides,
        policy_overrides=policy_overrides,
    )

    if print_config:
        print(json.dumps(_to_plain({"run_dir": run_dir, "contract": contract, "main_config": main_config, "create_config": create_config, "max_env_step": max_steps}), indent=2, ensure_ascii=False))
        return run_dir

    _ensure_tensorboard_writer()
    if ALGO_REGISTRY[algo_name]["pipeline_type"] == "onpolicy":
        from ding.entry import serial_pipeline_onpolicy

        serial_pipeline_onpolicy((main_config, create_config), seed=seed, max_env_step=max_steps)
    else:
        from ding.entry import serial_pipeline

        serial_pipeline((main_config, create_config), seed=seed, max_env_step=max_steps)

    _wait_for_reward_artifact(run_dir)
    summary = _finalize_run(run_dir, main_config.exp_name, contract)
    if int(summary.get("num_episodes") or 0) <= 0:
        for _ in range(60):
            rewards = _load_reward_trace(run_dir)
            if rewards.size > 0:
                _finalize_run(run_dir, main_config.exp_name, contract)
                break
            time.sleep(0.2)
    return run_dir


def run_legacy_entry(algo_name: str) -> None:
    """Compatibility shim used by legacy `Tracking_*.py` entry scripts."""
    run_experiment(env_name="tracking", algo_name=algo_name, seed=0, run_tag="legacy")


def run_matrix_experiments(
    env_name: str,
    algos: List[str],
    seeds: List[int],
    *,
    max_env_step: Optional[int] = None,
    n_agent: Optional[int] = None,
    episode_length: int = 400,
    shared_reward: bool = False,
    print_episode_reward: bool = True,
    collector_env_num: Optional[int] = None,
    evaluator_env_num: Optional[int] = None,
    eval_interval_steps: Optional[int] = None,
    eval_horizon_steps: int = 0,
    codebook_size: int = DEFAULT_CODEBOOK_SIZE,
    discrete_level: int = DEFAULT_DISCRETE_LEVEL,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    run_tag: str = "matrix",
    env_overrides: Optional[Dict[str, Any]] = None,
    print_config: bool = False,
) -> List[Dict[str, Any]]:
    """Run a batch of env/algo/seed experiments and return manifest records."""
    unknown_algos = [algo for algo in algos if algo not in ALGO_REGISTRY]
    if unknown_algos:
        raise KeyError(f"Unknown algorithms: {unknown_algos}")

    records: List[Dict[str, Any]] = []
    for algo_name in algos:
        for seed in seeds:
            run_dir = run_experiment(
                env_name=env_name,
                algo_name=algo_name,
                seed=seed,
                max_env_step=max_env_step,
                n_agent=n_agent,
                episode_length=episode_length,
                shared_reward=shared_reward,
                print_episode_reward=print_episode_reward,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                eval_interval_steps=eval_interval_steps,
                eval_horizon_steps=eval_horizon_steps,
                codebook_size=codebook_size,
                discrete_level=discrete_level,
                output_root=output_root,
                run_tag=run_tag,
                env_overrides=env_overrides,
                print_config=print_config,
            )
            records.append(
                {
                    "env": env_name,
                    "algo": algo_name,
                    "seed": int(seed),
                    "run_dir": Path(run_dir).as_posix(),
                }
            )
    return records


def run_checkpoint_evaluation(
    run_dir: str | Path,
    *,
    algo_name: Optional[str] = None,
    episodes: int = 20,
    eval_horizon_steps: int = 100,
    seed_base: int = 1000,
    ckpt: str = "",
) -> Dict[str, Any]:
    """Run offline checkpoint evaluation and export AUV trajectory artifacts."""
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    cfg_path = resolved_run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json under {resolved_run_dir.as_posix()}")

    config_payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    contract = config_payload.get("contract", {})
    contract_algo = contract.get("algo_cfg", {}).get("algo_name")
    resolved_algo = (contract_algo or algo_name or "").strip().lower()
    if not resolved_algo:
        raise ValueError(f"Unable to infer algorithm name for {resolved_run_dir.as_posix()}")
    if resolved_algo not in TRAJECTORY_EVAL_ALGOS:
        supported = ", ".join(sorted(TRAJECTORY_EVAL_ALGOS))
        raise ValueError(
            f"Checkpoint trajectory evaluation currently supports {supported}, got {resolved_algo}"
        )

    try:
        from scripts.eval_ckpt_3d_auv import evaluate_checkpoint_3d
    except Exception:  # pragma: no cover
        from Tracking.scripts.eval_ckpt_3d_auv import evaluate_checkpoint_3d

    result = evaluate_checkpoint_3d(
        run_dir=resolved_run_dir,
        episodes=episodes,
        eval_horizon_steps=eval_horizon_steps,
        seed_base=seed_base,
        ckpt=ckpt,
    )
    result["algo"] = resolved_algo
    result["run_dir"] = resolved_run_dir.as_posix()
    return result


def run_cli(argv: Optional[list[str]] = None) -> None:
    """CLI entry for unified MARL launching in Tracking/OpenMARL."""
    parser = argparse.ArgumentParser(description="Unified MARL launcher for Tracking and AUV6DOF environments.")
    parser.add_argument("--mode", default="train", choices=["train", "test", "train-test", "matrix"])
    parser.add_argument("--env", default="auv6dof", choices=sorted(ENV_REGISTRY.keys()))
    parser.add_argument("--algo", default="maddpg", choices=sorted(ALGO_REGISTRY.keys()))
    parser.add_argument("--algos", type=str, default=",".join(ALGO_ORDER))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-env-step", type=int, default=0)
    parser.add_argument("--total-train-steps", type=int, default=0)
    parser.add_argument("--n-agent", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=400)
    parser.add_argument("--eval-interval-steps", type=int, default=0)
    parser.add_argument("--eval-horizon-steps", type=int, default=0)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--test-seed-base", type=int, default=1000)
    parser.add_argument("--shared-reward", action="store_true")
    parser.add_argument("--no-print-episode-reward", action="store_true")
    parser.add_argument("--collector-env-num", type=int, default=0)
    parser.add_argument("--evaluator-env-num", type=int, default=0)
    parser.add_argument("--codebook-size", type=int, default=DEFAULT_CODEBOOK_SIZE)
    parser.add_argument("--discrete-level", type=int, default=DEFAULT_DISCRETE_LEVEL)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--env-overrides-json", type=str, default="")
    parser.add_argument("--action-scale", type=float, default=None)
    parser.add_argument("--eval-log-format", choices=["csv", "jsonl", "pretty"], default="")
    parser.add_argument("--normalize-obs-eval-mode", choices=["running", "frozen", "disabled"], default="")
    parser.add_argument("--verbose-eval", action="store_true", dest="verbose_eval")
    parser.add_argument("--no-verbose-eval", action="store_false", dest="verbose_eval")
    parser.set_defaults(verbose_eval=None)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(argv)
    env_overrides: Optional[Dict[str, Any]] = None
    if args.env_overrides_json.strip():
        env_overrides = json.loads(args.env_overrides_json)
    if env_overrides is None:
        env_overrides = {}
    if args.action_scale is not None:
        env_overrides["action_scale"] = float(args.action_scale)
    if args.eval_log_format.strip():
        env_overrides["eval_log_format"] = str(args.eval_log_format)
    if args.normalize_obs_eval_mode.strip():
        env_overrides["normalize_obs_eval_mode"] = str(args.normalize_obs_eval_mode)
    if args.verbose_eval is not None:
        env_overrides["verbose_eval"] = bool(args.verbose_eval)
    if args.eval_interval_steps and int(args.eval_interval_steps) > 0:
        env_overrides["eval_interval_steps"] = int(args.eval_interval_steps)
    if len(env_overrides) == 0:
        env_overrides = None

    train_steps = args.total_train_steps or args.max_env_step
    run_common = {
        "env_name": args.env,
        "algo_name": args.algo,
        "seed": args.seed,
        "max_env_step": train_steps or None,
        "n_agent": args.n_agent or None,
        "episode_length": args.episode_length,
        "eval_interval_steps": args.eval_interval_steps or None,
        "eval_horizon_steps": args.eval_horizon_steps,
        "shared_reward": args.shared_reward,
        "print_episode_reward": not args.no_print_episode_reward,
        "collector_env_num": args.collector_env_num or None,
        "evaluator_env_num": args.evaluator_env_num or None,
        "codebook_size": args.codebook_size,
        "discrete_level": args.discrete_level,
        "output_root": args.output_root,
        "run_tag": args.run_tag,
        "env_overrides": env_overrides,
        "print_config": args.print_config,
    }

    if args.mode == "train":
        run_dir = run_experiment(**run_common)
        print(json.dumps({"mode": args.mode, "run_dir": Path(run_dir).as_posix()}, indent=2, ensure_ascii=False))
        return

    if args.mode == "test":
        target_run_dir = _resolve_run_dir(
            args.run_dir,
            output_root=args.output_root,
            env_name=args.env,
            algo_name=args.algo,
            seed=args.seed,
            require_checkpoint=True,
        )
        result = run_checkpoint_evaluation(
            run_dir=target_run_dir,
            algo_name=args.algo,
            episodes=args.test_episodes,
            eval_horizon_steps=args.eval_horizon_steps or 100,
            seed_base=args.test_seed_base,
            ckpt=args.ckpt,
        )
        print(json.dumps({"mode": args.mode, **_to_plain(result)}, indent=2, ensure_ascii=False))
        return

    if args.mode == "train-test":
        run_dir = run_experiment(**run_common)
        if args.print_config:
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "run_dir": Path(run_dir).as_posix(),
                        "note": "print-config skips checkpoint evaluation because no training is executed",
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
        result = run_checkpoint_evaluation(
            run_dir=run_dir,
            algo_name=args.algo,
            episodes=args.test_episodes,
            eval_horizon_steps=args.eval_horizon_steps or 100,
            seed_base=args.test_seed_base,
            ckpt=args.ckpt,
        )
        print(
            json.dumps(
                {"mode": args.mode, "run_dir": Path(run_dir).as_posix(), "evaluation": _to_plain(result)},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    algos = _parse_csv_values(args.algos) or list(ALGO_ORDER)
    seeds = [int(v) for v in _parse_csv_values(args.seeds)] or [int(args.seed)]
    matrix_records = run_matrix_experiments(
        env_name=args.env,
        algos=algos,
        seeds=seeds,
        max_env_step=train_steps or None,
        n_agent=args.n_agent or None,
        episode_length=args.episode_length,
        shared_reward=args.shared_reward,
        print_episode_reward=not args.no_print_episode_reward,
        collector_env_num=args.collector_env_num or None,
        evaluator_env_num=args.evaluator_env_num or None,
        eval_interval_steps=args.eval_interval_steps or None,
        eval_horizon_steps=args.eval_horizon_steps,
        codebook_size=args.codebook_size,
        discrete_level=args.discrete_level,
        output_root=args.output_root,
        run_tag=args.run_tag or "matrix",
        env_overrides=env_overrides,
        print_config=args.print_config,
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "env": args.env,
                "num_runs": len(matrix_records),
                "runs": matrix_records,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run_cli()

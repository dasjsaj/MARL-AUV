from __future__ import annotations

"""
DI-engine BaseEnv adapter for the AUV6DOF Gym environment.

This wrapper keeps the lower Gym environment clean and focuses on DI-engine
contract responsibilities:
- reset/step API conversion to BaseEnvTimestep
- optional continuous/discrete action branch switch
- unified observation keys: agent_state/global_state/(optional action_mask)
- reward scalarization and episode metric accumulation
- standardized artifact output under cfg.artifact_dir
"""

import csv
import json
import os
from pathlib import Path
import time
from collections import deque
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY

from .action_codebook import build_discrete_action_codebook

try:
    from Tracking.auv6dof.gym_env import AUV6DOFGymEnv
    from Tracking.auv6dof.dynamics import rotation_zyx
except Exception:  # pragma: no cover
    from auv6dof.gym_env import AUV6DOFGymEnv
    from auv6dof.dynamics import rotation_zyx


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance of observations."""

    def __init__(self, shape: tuple, clip: float = 10.0) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = float(clip)

    def update(self, x: np.ndarray) -> None:
        """Update with a batch of shape (batch, *shape) or (*shape,)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[np.newaxis]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x.astype(np.float64) - self.mean) / np.sqrt(self.var + 1e-8),
            -self.clip,
            self.clip,
        ).astype(np.float32)


@ENV_REGISTRY.register("auv6dof_di")
class AUV6DOFDIEnv(BaseEnv):
    """
    DI-engine adapter for AUV 6DOF multi-agent target tracking tasks.
    """

    config = dict(
        n_agent=4,
        episode_length=400,
        eval_horizon_steps=0,
        tail_eval_steps=5,
        shared_reward=True,
        print_episode_reward=True,
        discrete_action=False,
        agent_obs_only=False,
        agent_specific_global_state=False,
        global_state_per_agent=False,
        wrap_obs_key=False,
        augment_coma_obs=False,
        normalize_obs=True,
        normalize_obs_in_eval=True,
        normalize_obs_eval_mode="frozen",
        action_scale=1.0,
        action_control_mode="tau6",
        control_mode="direct_tau",
        residual_baseline_scale=0.25,
        residual_los_speed=0.25,
        residual_vel_gain=0.35,
        residual_att_gain=0.20,
        residual_rate_gain=0.12,
        boundary_guard_enabled=True,
        boundary_guard_ratio=0.70,
        boundary_guard_gain=0.55,
        boundary_outward_damping=0.15,
        separation_guard_enabled=True,
        separation_guard_distance=0.12,
        separation_guard_gain=0.45,
        verbose_eval=True,
        eval_log_format="pretty",
        eval_interval_steps=0,
        codebook_size=125,
        discrete_level=3,
        artifact_dir="",
        boundary_limit=1.0,
        dt=0.1,
        reward=dict(
            version="v2_fast_converge",
            d_target_min=0.015,
            d_auv_min=0.015,
            near_target_scale=0.2,
            collision_scale=5.0,
            pos_weight=0.95,
            col_weight=0.05,
            distance_clip=1.2,
            near_distance=0.10,
            safe_distance=0.08,
            success_distance=0.025,
            boundary_soft_ratio=0.75,
            progress_clip=0.04,
            closing_speed_clip=0.08,
            attitude_angle_clip=0.75,
            angular_rate_clip=1.2,
            w_centroid_distance=0.0,
            w_centroid_progress=0.0,
            w_centroid_near=0.0,
            w_centroid_success=0.0,
            w_distance=0.25,
            w_progress=0.40,
            w_closing_speed=0.20,
            w_near=0.10,
            w_success=0.10,
            w_separation=0.03,
            w_boundary=0.03,
            w_collision=0.03,
            w_oob=0.02,
            w_unstable=0.01,
            w_attitude_stability=0.01,
            w_angular_rate=0.01,
            w_action_energy=0.015,
            w_action_smooth=0.015,
            w_tracking_group=0.85,
            w_safety_group=0.12,
            w_action_group=0.03,
        ),
        obs=dict(
            include_target_velocity=True,
            include_relative_velocity=True,
            include_target_rel_body=True,
            include_relative_velocity_body=True,
            include_los_unit_body=True,
            include_prev_action=True,
            use_attitude_sin_cos=True,
            include_boundary_margin=True,
            normalize_physical=True,
        ),
        reset=dict(
            curriculum_stage="auto",
            train_curriculum_stage="auto",
            eval_curriculum_stage="medium",
            min_init_separation=0.10,
            auto_easy_episodes=800,
            auto_medium_episodes=2000,
            easy_target_speed_range=(0.001, 0.004),
            medium_target_speed_range=(0.003, 0.008),
            hard_target_speed_range=(0.006, 0.014),
        ),
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        return EasyDict(cls.config)

    _EVAL_DETAIL_KEYS = [
        "eval_index",
        "train_step",
        "eval_return",
        "final_centroid_target_distance",
        "mean_centroid_target_distance_over_episode",
        "tail_mean_centroid_target_distance",
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
        "collision",
        "out_of_bounds",
        "unstable",
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

    def __init__(self, cfg: dict) -> None:
        self._cfg = EasyDict(cfg)
        self._n_agent = int(self._cfg.get("n_agent", 4))
        self._episode_length = int(self._cfg.get("episode_length", 400))
        self._eval_horizon_steps = int(self._cfg.get("eval_horizon_steps", 0))
        self._tail_eval_steps = max(1, int(self._cfg.get("tail_eval_steps", 5)))
        self._shared_reward = bool(self._cfg.get("shared_reward", True))
        self._print_episode_reward = bool(self._cfg.get("print_episode_reward", True))
        self._discrete_action = bool(self._cfg.get("discrete_action", False))
        self._agent_obs_only = bool(self._cfg.get("agent_obs_only", False))
        self._agent_specific_global_state = bool(self._cfg.get("agent_specific_global_state", False))
        self._global_state_per_agent = bool(self._cfg.get("global_state_per_agent", False))
        self._wrap_obs_key = bool(self._cfg.get("wrap_obs_key", False))
        self._augment_coma_obs = bool(self._cfg.get("augment_coma_obs", False))
        self._action_scale = float(self._cfg.get("action_scale", 1.0))
        self._control_mode = str(self._cfg.get("control_mode", "direct_tau")).strip().lower()
        if self._control_mode not in {"direct_tau", "residual_tau"}:
            self._control_mode = "direct_tau"
        self._residual_baseline_scale = float(self._cfg.get("residual_baseline_scale", 0.25))
        self._residual_los_speed = float(self._cfg.get("residual_los_speed", 0.25))
        self._residual_vel_gain = float(self._cfg.get("residual_vel_gain", 0.35))
        self._residual_att_gain = float(self._cfg.get("residual_att_gain", 0.20))
        self._residual_rate_gain = float(self._cfg.get("residual_rate_gain", 0.12))
        self._boundary_guard_enabled = bool(self._cfg.get("boundary_guard_enabled", True))
        self._boundary_guard_ratio = float(self._cfg.get("boundary_guard_ratio", 0.70))
        self._boundary_guard_gain = float(self._cfg.get("boundary_guard_gain", 0.55))
        self._boundary_outward_damping = float(self._cfg.get("boundary_outward_damping", 0.15))
        self._separation_guard_enabled = bool(self._cfg.get("separation_guard_enabled", True))
        self._separation_guard_distance = float(self._cfg.get("separation_guard_distance", 0.12))
        self._separation_guard_gain = float(self._cfg.get("separation_guard_gain", 0.45))
        self._codebook_size = int(self._cfg.get("codebook_size", 125))
        self._discrete_level = int(self._cfg.get("discrete_level", 3))
        self._is_evaluator = bool(self._cfg.get("is_evaluator", False))
        self._evaluator_id = int(self._cfg.get("evaluator_id", 0))
        self._verbose_eval = bool(self._cfg.get("verbose_eval", True))
        self._eval_log_format = str(self._cfg.get("eval_log_format", "pretty")).strip().lower()
        if self._eval_log_format not in {"csv", "jsonl", "pretty"}:
            self._eval_log_format = "pretty"
        self._eval_interval_steps = int(self._cfg.get("eval_interval_steps", 0))
        if self._is_evaluator and self._eval_horizon_steps > 0:
            self._episode_length = int(self._eval_horizon_steps)

        self._artifact_dir = str(self._cfg.get("artifact_dir", "")).strip()
        self._save_dir = Path(self._artifact_dir) if self._artifact_dir else Path.cwd()
        self._eval_jsonl_path = self._save_dir / f"auv6dof_eval_detail_raw_{self._evaluator_id}.jsonl"

        self._env = None
        self._action_map = None
        self._full_discrete_size = 0
        self._init_flag = False
        self._step_count = 0
        self._eval_episode_return = 0.0
        self._episode_count = 0
        self._episode_rewards: List[float] = []
        self._eval_episode_records: List[Dict[str, Any]] = []
        self._episode_collision = 0
        self._episode_out_of_bounds = 0
        self._episode_unstable = 0
        self._current_final_target_distance = 0.0
        self._current_final_centroid_distance = 0.0
        self._target_distance_sum = 0.0
        self._target_distance_count = 0
        self._centroid_distance_sum = 0.0
        self._centroid_distance_count = 0
        self._target_distance_tail = deque(maxlen=self._tail_eval_steps)
        self._centroid_distance_tail = deque(maxlen=self._tail_eval_steps)
        self._tracking_error_tail = deque(maxlen=self._tail_eval_steps)
        self._target_lost_tail = deque(maxlen=self._tail_eval_steps)
        self._action_norm_tail = deque(maxlen=self._tail_eval_steps)
        self._action_saturation_tail = deque(maxlen=self._tail_eval_steps)
        self._centroid_distance_term_sum = 0.0
        self._centroid_progress_term_sum = 0.0
        self._centroid_near_term_sum = 0.0
        self._distance_term_sum = 0.0
        self._progress_term_sum = 0.0
        self._closing_speed_term_sum = 0.0
        self._near_term_sum = 0.0
        self._success_term_sum = 0.0
        self._attitude_stability_term_sum = 0.0
        self._angular_rate_term_sum = 0.0
        self._safety_term_sum = 0.0
        self._action_reg_term_sum = 0.0
        self._tracking_group_term_sum = 0.0
        self._safety_group_term_sum = 0.0
        self._action_group_term_sum = 0.0
        self._tracking_contrib_sum = 0.0
        self._safety_contrib_sum = 0.0
        self._action_contrib_sum = 0.0
        self._tracking_reward_sum = 0.0
        self._observation_reward_sum = 0.0
        self._coordination_reward_sum = 0.0
        self._communication_reward_sum = 0.0
        self._semantic_reward_sum = 0.0
        self._control_cost_sum = 0.0
        self._tracking_error_sum = 0.0
        self._tracking_error_delta_sum = 0.0
        self._observation_confidence_sum = 0.0
        self._target_lost_sum = 0.0
        self._communication_quality_sum = 0.0
        self._action_clip_rate_sum = 0.0
        self._action_saturation_rate_sum = 0.0
        self._action_delta_norm_sum = 0.0
        self._action_stat_count = 0
        self._reward_term_count = 0
        self._seed = 0
        self._prev_action_onehot: np.ndarray | None = None
        self._prev_action_cont: np.ndarray | None = None
        self._obs_rms: RunningMeanStd | None = None
        self._normalize_obs = bool(self._cfg.get("normalize_obs", True))
        self._normalize_obs_in_eval = bool(self._cfg.get("normalize_obs_in_eval", True))
        raw_eval_mode = str(self._cfg.get("normalize_obs_eval_mode", "")).strip().lower()
        if raw_eval_mode in {"running", "frozen", "disabled"}:
            self._normalize_obs_eval_mode = raw_eval_mode
        else:
            self._normalize_obs_eval_mode = "running" if self._normalize_obs_in_eval else "disabled"
        self._eval_obs_stats_frozen = False

    def _build_env(self) -> None:
        """Instantiate underlying Gym env and build DI-compatible spaces."""
        reset_cfg = dict(self._cfg.get("reset", {}))
        train_stage = str(reset_cfg.pop("train_curriculum_stage", "")).strip().lower()
        eval_stage = str(reset_cfg.pop("eval_curriculum_stage", "")).strip().lower()
        if self._is_evaluator and eval_stage:
            reset_cfg["curriculum_stage"] = eval_stage
        elif (not self._is_evaluator) and train_stage:
            reset_cfg["curriculum_stage"] = train_stage

        env_cfg = {
            "n_agent": self._n_agent,
            "episode_length": self._episode_length,
            "boundary_limit": float(self._cfg.get("boundary_limit", 1.0)),
            "dt": float(self._cfg.get("dt", 0.1)),
            "action_control_mode": str(self._cfg.get("action_control_mode", "tau6")),
            "action_smoothing": float(self._cfg.get("action_smoothing", 0.0)),
            "max_action_delta": float(self._cfg.get("max_action_delta", 0.0)),
            "velocity_command_gain": float(self._cfg.get("velocity_command_gain", 1.0)),
            "attitude_damping_gain": float(self._cfg.get("attitude_damping_gain", 0.12)),
            "rate_damping_gain": float(self._cfg.get("rate_damping_gain", 0.10)),
            "auv_model": dict(self._cfg.get("auv_model", {})),
            "reward": dict(self._cfg.get("reward", {})),
            "obs": dict(self._cfg.get("obs", {})),
            "reset": reset_cfg,
        }
        self._env = AUV6DOFGymEnv(env_cfg)
        self._env.seed(self._seed)

        obs_space = self._env.observation_space
        act_dim = int(self._env.action_space.shape[-1])
        obs_dim = int(obs_space["agent_state"].shape[-1])
        self._prev_action_cont = np.zeros((self._n_agent, act_dim), dtype=np.float32)

        # Initialize running obs normalizer with per-feature statistics.
        if self._normalize_obs and self._obs_rms is None:
            self._obs_rms = RunningMeanStd(shape=(obs_dim,), clip=5.0)

        if self._discrete_action:
            self._action_map, self._full_discrete_size = build_discrete_action_codebook(
                act_dim,
                discrete_level=self._discrete_level,
                codebook_size=self._codebook_size,
                action_scale=self._action_scale,
            )
            discrete_dim = int(self._action_map.shape[0])
            self._action_space = gym.spaces.Discrete(discrete_dim)
        else:
            self._action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self._n_agent, act_dim), dtype=np.float32
            )

        effective_obs_dim = obs_dim
        if self._augment_coma_obs:
            effective_obs_dim = obs_dim + (discrete_dim if self._discrete_action else 0) + self._n_agent

        if self._agent_obs_only:
            self._observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._n_agent, effective_obs_dim), dtype=np.float32
            )
        else:
            global_state_space: gym.spaces.Space
            if self._global_state_per_agent:
                global_state_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._n_agent, self._n_agent * obs_dim),
                    dtype=np.float32,
                )
            elif self._agent_specific_global_state:
                global_state_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._n_agent, obs_dim + self._n_agent * obs_dim),
                    dtype=np.float32,
                )
            else:
                global_state_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._n_agent * obs_dim,),
                    dtype=np.float32,
                )
            obs_dict = {
                "agent_state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._n_agent, effective_obs_dim), dtype=np.float32
                ),
                "global_state": global_state_space,
                "agent_alone_state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._n_agent, effective_obs_dim), dtype=np.float32
                ),
                "agent_alone_padding_state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._n_agent, effective_obs_dim), dtype=np.float32
                ),
            }
            if self._discrete_action:
                obs_dict["action_mask"] = gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self._n_agent, discrete_dim), dtype=np.float32
                )
            base_space = gym.spaces.Dict(obs_dict)
            if self._wrap_obs_key:
                self._observation_space = gym.spaces.Dict({"obs": base_space})
            else:
                self._observation_space = base_space

        self._reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self._init_flag = True

    def _decode_discrete_action(self, action: np.ndarray) -> np.ndarray:
        """Map discrete action indices to continuous 6DOF control vectors."""
        idx = np.asarray(action).reshape(-1).astype(np.int64)
        if idx.shape[0] != self._n_agent:
            raise ValueError(f"discrete action first dim must be {self._n_agent}, got {idx.shape}")
        if np.any(idx < 0) or np.any(idx >= len(self._action_map)):
            raise ValueError(f"discrete action index out of range [0, {len(self._action_map)-1}]")
        return self._action_map[idx]

    def _process_obs(
        self, obs: Dict[str, np.ndarray], prev_action_onehot: np.ndarray | None = None
    ) -> Dict[str, np.ndarray] | np.ndarray:
        """Convert Gym observation dict into configured DI observation contract."""
        agent_state = np.asarray(obs["agent_state"], dtype=np.float32)
        # Running observation normalization — update stats and normalize.
        if self._normalize_obs and self._obs_rms is not None:
            should_normalize = True
            should_update = False
            if not self._is_evaluator:
                should_update = True
            else:
                if self._normalize_obs_eval_mode == "disabled":
                    should_normalize = False
                elif self._normalize_obs_eval_mode == "running":
                    should_update = True
                elif self._normalize_obs_eval_mode == "frozen":
                    should_update = not self._eval_obs_stats_frozen
            if should_normalize:
                if should_update:
                    self._obs_rms.update(agent_state)
                agent_state = self._obs_rms.normalize(agent_state)
        if self._augment_coma_obs:
            if prev_action_onehot is None:
                if self._prev_action_onehot is None:
                    action_dim = int(len(self._action_map)) if self._discrete_action else 0
                    self._prev_action_onehot = np.zeros((self._n_agent, action_dim), dtype=np.float32)
                prev_action_onehot = self._prev_action_onehot
            agent_id = np.eye(self._n_agent, dtype=np.float32)
            agent_state = np.concatenate([agent_state, prev_action_onehot.astype(np.float32), agent_id], axis=1)
        if self._agent_obs_only:
            return {"obs": agent_state} if self._wrap_obs_key else agent_state

        global_state = agent_state.reshape(-1).astype(np.float32)
        if self._global_state_per_agent:
            global_state = np.repeat(global_state[None, :], self._n_agent, axis=0).astype(np.float32)
        elif self._agent_specific_global_state:
            global_state = np.concatenate(
                [agent_state, np.repeat(global_state[None, :], self._n_agent, axis=0)], axis=1
            ).astype(np.float32)

        out = {
            "agent_state": agent_state,
            "global_state": global_state,
            "agent_alone_state": agent_state.copy(),
            "agent_alone_padding_state": agent_state.copy(),
        }
        if self._discrete_action:
            out["action_mask"] = np.ones((self._n_agent, len(self._action_map)), dtype=np.float32)
        if self._wrap_obs_key:
            return {"obs": out}
        return out

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset one episode and return DI-formatted initial observation."""
        if not self._init_flag:
            self._build_env()
        self._step_count = 0
        self._eval_episode_return = 0.0
        self._episode_collision = 0
        self._episode_out_of_bounds = 0
        self._episode_unstable = 0
        self._current_final_target_distance = 0.0
        self._current_final_centroid_distance = 0.0
        self._target_distance_sum = 0.0
        self._target_distance_count = 0
        self._centroid_distance_sum = 0.0
        self._centroid_distance_count = 0
        self._target_distance_tail.clear()
        self._centroid_distance_tail.clear()
        self._tracking_error_tail.clear()
        self._target_lost_tail.clear()
        self._action_norm_tail.clear()
        self._action_saturation_tail.clear()
        self._centroid_distance_term_sum = 0.0
        self._centroid_progress_term_sum = 0.0
        self._centroid_near_term_sum = 0.0
        self._distance_term_sum = 0.0
        self._progress_term_sum = 0.0
        self._closing_speed_term_sum = 0.0
        self._near_term_sum = 0.0
        self._success_term_sum = 0.0
        self._attitude_stability_term_sum = 0.0
        self._angular_rate_term_sum = 0.0
        self._safety_term_sum = 0.0
        self._action_reg_term_sum = 0.0
        self._tracking_group_term_sum = 0.0
        self._safety_group_term_sum = 0.0
        self._action_group_term_sum = 0.0
        self._tracking_contrib_sum = 0.0
        self._safety_contrib_sum = 0.0
        self._action_contrib_sum = 0.0
        self._tracking_reward_sum = 0.0
        self._observation_reward_sum = 0.0
        self._coordination_reward_sum = 0.0
        self._communication_reward_sum = 0.0
        self._semantic_reward_sum = 0.0
        self._control_cost_sum = 0.0
        self._tracking_error_sum = 0.0
        self._tracking_error_delta_sum = 0.0
        self._observation_confidence_sum = 0.0
        self._target_lost_sum = 0.0
        self._communication_quality_sum = 0.0
        self._action_clip_rate_sum = 0.0
        self._action_saturation_rate_sum = 0.0
        self._action_delta_norm_sum = 0.0
        self._action_stat_count = 0
        self._reward_term_count = 0
        if self._env is not None:
            self._prev_action_cont = np.zeros((self._n_agent, self._env.action_dim), dtype=np.float32)
        obs, _ = self._env.reset(seed=self._seed)
        if self._augment_coma_obs and self._discrete_action:
            self._prev_action_onehot = np.zeros((self._n_agent, len(self._action_map)), dtype=np.float32)
        return self._process_obs(obs, self._prev_action_onehot)

    def _current_target_distance(self) -> float:
        """Mean Euclidean distance from all AUV agents to the current target."""
        if self._env is None or not getattr(self._env.world, "targets", None):
            return 0.0
        target_pos = np.asarray(self._env.world.targets[0].state.p_pos, dtype=np.float64)
        agent_pos = np.stack(
            [np.asarray(agent.state.p_pos, dtype=np.float64) for agent in self._env.world.agents],
            axis=0,
        )
        return float(np.mean(np.linalg.norm(agent_pos - target_pos[None, :], axis=1)))

    def _current_centroid_target_distance(self) -> float:
        """Euclidean distance between AUV centroid and target."""
        if self._env is None or not getattr(self._env.world, "targets", None):
            return 0.0
        target_pos = np.asarray(self._env.world.targets[0].state.p_pos, dtype=np.float64)
        agent_pos = np.stack(
            [np.asarray(agent.state.p_pos, dtype=np.float64) for agent in self._env.world.agents],
            axis=0,
        )
        centroid = np.mean(agent_pos, axis=0)
        return float(np.linalg.norm(centroid - target_pos))

    def _compute_eval_train_step(self, eval_index: int) -> int:
        if self._eval_interval_steps > 0:
            return int((eval_index + 1) * self._eval_interval_steps)
        horizon = self._eval_horizon_steps if self._eval_horizon_steps > 0 else self._step_count
        return int((eval_index + 1) * max(1, int(horizon)))

    def _build_eval_record(self) -> Dict[str, Any]:
        eval_index = int(self._episode_count - 1)
        mean_target_distance = (
            float(self._target_distance_sum / self._target_distance_count)
            if self._target_distance_count > 0
            else float(self._current_final_target_distance)
        )
        mean_centroid_distance = (
            float(self._centroid_distance_sum / self._centroid_distance_count)
            if self._centroid_distance_count > 0
            else float(self._current_final_centroid_distance)
        )
        tail_mean_target_distance = (
            float(np.mean(np.asarray(self._target_distance_tail, dtype=np.float64)))
            if len(self._target_distance_tail) > 0
            else float(self._current_final_target_distance)
        )
        tail_mean_centroid_distance = (
            float(np.mean(np.asarray(self._centroid_distance_tail, dtype=np.float64)))
            if len(self._centroid_distance_tail) > 0
            else float(self._current_final_centroid_distance)
        )
        tail_target = np.asarray(self._target_distance_tail, dtype=np.float64)
        tail_tracking_error = np.asarray(self._tracking_error_tail, dtype=np.float64)
        tail_target_lost = np.asarray(self._target_lost_tail, dtype=np.float64)
        tail_action_norm = np.asarray(self._action_norm_tail, dtype=np.float64)
        tail_action_saturation = np.asarray(self._action_saturation_tail, dtype=np.float64)
        tail100_mean_target_distance = (
            float(np.mean(tail_target)) if tail_target.size > 0 else float(self._current_final_target_distance)
        )
        tail100_std_target_distance = float(np.std(tail_target)) if tail_target.size > 0 else 0.0
        tail100_mean_tracking_error = (
            float(np.mean(tail_tracking_error)) if tail_tracking_error.size > 0 else 0.0
        )
        tail100_target_lost_rate = float(np.mean(tail_target_lost)) if tail_target_lost.size > 0 else 0.0
        tail100_action_norm = float(np.mean(tail_action_norm)) if tail_action_norm.size > 0 else 0.0
        tail100_action_saturation_rate = (
            float(np.mean(tail_action_saturation)) if tail_action_saturation.size > 0 else 0.0
        )
        horizon_steps = int(self._eval_horizon_steps if self._eval_horizon_steps > 0 else self._step_count)
        return {
            "eval_index": eval_index,
            "train_step": self._compute_eval_train_step(eval_index),
            "eval_return": float(self._eval_episode_return),
            "episode_return": float(self._eval_episode_return),
            "eval_steps": int(self._step_count),
            "episode_steps": int(self._step_count),
            "horizon_steps": horizon_steps,
            "final_centroid_target_distance": float(self._current_final_centroid_distance),
            "mean_centroid_target_distance_over_episode": mean_centroid_distance,
            "tail_mean_centroid_target_distance": tail_mean_centroid_distance,
            "collision": int(self._episode_collision),
            "out_of_bounds": int(self._episode_out_of_bounds),
            "unstable": int(self._episode_unstable),
            "final_mean_target_distance": float(self._current_final_target_distance),
            "mean_target_distance": mean_target_distance,
            "mean_target_distance_over_episode": mean_target_distance,
            "tail_mean_target_distance": tail_mean_target_distance,
            "tail100_mean_target_distance": tail100_mean_target_distance,
            "tail100_std_target_distance": tail100_std_target_distance,
            "tail100_mean_tracking_error": tail100_mean_tracking_error,
            "tail100_target_lost_rate": tail100_target_lost_rate,
            "tail100_action_norm": tail100_action_norm,
            "tail100_action_saturation_rate": tail100_action_saturation_rate,
            "mean_centroid_distance_term": float(self._centroid_distance_term_sum / max(1, self._reward_term_count)),
            "mean_centroid_progress_term": float(self._centroid_progress_term_sum / max(1, self._reward_term_count)),
            "mean_centroid_near_term": float(self._centroid_near_term_sum / max(1, self._reward_term_count)),
            "mean_distance_term": float(self._distance_term_sum / max(1, self._reward_term_count)),
            "mean_progress_term": float(self._progress_term_sum / max(1, self._reward_term_count)),
            "mean_closing_speed_term": float(self._closing_speed_term_sum / max(1, self._reward_term_count)),
            "mean_near_term": float(self._near_term_sum / max(1, self._reward_term_count)),
            "mean_success_term": float(self._success_term_sum / max(1, self._reward_term_count)),
            "mean_attitude_stability_term": float(
                self._attitude_stability_term_sum / max(1, self._reward_term_count)
            ),
            "mean_angular_rate_term": float(self._angular_rate_term_sum / max(1, self._reward_term_count)),
            "mean_safety_term": float(self._safety_term_sum / max(1, self._reward_term_count)),
            "mean_action_reg_term": float(self._action_reg_term_sum / max(1, self._reward_term_count)),
            "mean_tracking_group_term": float(self._tracking_group_term_sum / max(1, self._reward_term_count)),
            "mean_safety_group_term": float(self._safety_group_term_sum / max(1, self._reward_term_count)),
            "mean_action_group_term": float(self._action_group_term_sum / max(1, self._reward_term_count)),
            "mean_tracking_contrib": float(self._tracking_contrib_sum / max(1, self._reward_term_count)),
            "mean_safety_contrib": float(self._safety_contrib_sum / max(1, self._reward_term_count)),
            "mean_action_contrib": float(self._action_contrib_sum / max(1, self._reward_term_count)),
            "mean_tracking_reward": float(self._tracking_reward_sum / max(1, self._reward_term_count)),
            "mean_observation_reward": float(self._observation_reward_sum / max(1, self._reward_term_count)),
            "mean_coordination_reward": float(self._coordination_reward_sum / max(1, self._reward_term_count)),
            "mean_communication_reward": float(self._communication_reward_sum / max(1, self._reward_term_count)),
            "mean_semantic_reward": float(self._semantic_reward_sum / max(1, self._reward_term_count)),
            "mean_control_cost": float(self._control_cost_sum / max(1, self._reward_term_count)),
            "mean_tracking_error": float(self._tracking_error_sum / max(1, self._reward_term_count)),
            "mean_tracking_error_delta": float(self._tracking_error_delta_sum / max(1, self._reward_term_count)),
            "mean_observation_confidence": float(self._observation_confidence_sum / max(1, self._reward_term_count)),
            "mean_target_lost": float(self._target_lost_sum / max(1, self._reward_term_count)),
            "mean_communication_quality": float(self._communication_quality_sum / max(1, self._reward_term_count)),
            "mean_action_clip_rate": float(self._action_clip_rate_sum / max(1, self._action_stat_count)),
            "mean_action_saturation_rate": float(self._action_saturation_rate_sum / max(1, self._action_stat_count)),
            "mean_action_delta_norm": float(self._action_delta_norm_sum / max(1, self._action_stat_count)),
        }

    @staticmethod
    def _fmt_metric(value: Any) -> str:
        if isinstance(value, (np.floating, float)):
            return f"{float(value):.6f}"
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        return str(value)

    def _emit_eval_detail(self, rec: Dict[str, Any]) -> None:
        if not (self._is_evaluator and self._verbose_eval):
            return
        if self._eval_log_format == "jsonl":
            payload = {key: rec.get(key, 0.0) for key in self._EVAL_DETAIL_KEYS}
            print(f"evaluate_detail_json {json.dumps(payload, ensure_ascii=False)}", flush=True)
            return
        if self._eval_log_format == "pretty":
            print(
                f"[EVAL] idx={self._fmt_metric(rec.get('eval_index', 0))} "
                f"step={self._fmt_metric(rec.get('train_step', 0))}",
                flush=True,
            )
            print(
                "  reward: "
                f"eval_return={self._fmt_metric(rec.get('eval_return', 0.0))} "
                f"final_centroid_dist={self._fmt_metric(rec.get('final_centroid_target_distance', 0.0))} "
                f"mean_centroid_dist={self._fmt_metric(rec.get('mean_centroid_target_distance_over_episode', 0.0))} "
                f"tail_mean_dist={self._fmt_metric(rec.get('tail_mean_target_distance', 0.0))}",
                flush=True,
            )
            print(
                "  safety: "
                f"collision={self._fmt_metric(rec.get('collision', 0))} "
                f"out_of_bounds={self._fmt_metric(rec.get('out_of_bounds', 0))} "
                f"unstable={self._fmt_metric(rec.get('unstable', 0))}",
                flush=True,
            )
            print(
                "  centroid_terms: "
                f"dist={self._fmt_metric(rec.get('mean_centroid_distance_term', 0.0))} "
                f"progress={self._fmt_metric(rec.get('mean_centroid_progress_term', 0.0))} "
                f"near={self._fmt_metric(rec.get('mean_centroid_near_term', 0.0))}",
                flush=True,
            )
            print(
                "  tracking_terms: "
                f"distance={self._fmt_metric(rec.get('mean_distance_term', 0.0))} "
                f"progress={self._fmt_metric(rec.get('mean_progress_term', 0.0))} "
                f"closing={self._fmt_metric(rec.get('mean_closing_speed_term', 0.0))} "
                f"near={self._fmt_metric(rec.get('mean_near_term', 0.0))} "
                f"success={self._fmt_metric(rec.get('mean_success_term', 0.0))}",
                flush=True,
            )
            print(
                "  action_terms: "
                f"reg={self._fmt_metric(rec.get('mean_action_reg_term', 0.0))} "
                f"clip_rate={self._fmt_metric(rec.get('mean_action_clip_rate', 0.0))} "
                f"sat_rate={self._fmt_metric(rec.get('mean_action_saturation_rate', 0.0))} "
                f"delta_norm={self._fmt_metric(rec.get('mean_action_delta_norm', 0.0))}",
                flush=True,
            )
            print(
                "  reward_groups: "
                f"tracking={self._fmt_metric(rec.get('mean_tracking_contrib', 0.0))} "
                f"safety={self._fmt_metric(rec.get('mean_safety_contrib', 0.0))} "
                f"action={self._fmt_metric(rec.get('mean_action_contrib', 0.0))}",
                flush=True,
            )
            print("", flush=True)
            return
        parts = [f"{key}={self._fmt_metric(rec.get(key, 0.0))}" for key in self._EVAL_DETAIL_KEYS]
        print("evaluate_detail," + ",".join(parts), flush=True)

    def _append_eval_jsonl(self, rec: Dict[str, Any]) -> None:
        if self._eval_log_format != "jsonl":
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        payload = {key: rec.get(key, 0.0) for key in self._EVAL_DETAIL_KEYS}
        for retry in range(6):
            try:
                with self._eval_jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                return
            except PermissionError:
                if retry >= 5:
                    raise
                time.sleep(0.2 * (retry + 1))

    def _save_eval_detail(self) -> None:
        if len(self._eval_episode_records) == 0:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        path = self._save_dir / "eval_detail.csv"
        rows: List[List[float | int | str]] = [
            [
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
        ]
        for rec in self._eval_episode_records:
            rows.append(
                [
                    int(rec.get("eval_index", 0)),
                    int(rec.get("train_step", 0)),
                    float(rec.get("eval_return", rec.get("episode_return", 0.0))),
                    int(rec.get("eval_steps", rec.get("episode_steps", 0))),
                    int(rec.get("horizon_steps", self._eval_horizon_steps or rec.get("episode_steps", 0))),
                    float(rec.get("final_centroid_target_distance", 0.0)),
                    float(rec.get("mean_centroid_target_distance_over_episode", 0.0)),
                    float(rec.get("tail_mean_centroid_target_distance", 0.0)),
                    int(rec.get("collision", 0)),
                    int(rec.get("out_of_bounds", 0)),
                    int(rec.get("unstable", 0)),
                    float(rec.get("final_mean_target_distance", 0.0)),
                    float(rec.get("mean_target_distance", rec.get("mean_target_distance_over_episode", 0.0))),
                    float(rec.get("mean_target_distance_over_episode", 0.0)),
                    float(rec.get("tail_mean_target_distance", 0.0)),
                    float(rec.get("mean_centroid_distance_term", 0.0)),
                    float(rec.get("mean_centroid_progress_term", 0.0)),
                    float(rec.get("mean_centroid_near_term", 0.0)),
                    float(rec.get("mean_distance_term", 0.0)),
                    float(rec.get("mean_progress_term", 0.0)),
                    float(rec.get("mean_closing_speed_term", 0.0)),
                    float(rec.get("mean_near_term", 0.0)),
                    float(rec.get("mean_success_term", 0.0)),
                    float(rec.get("mean_attitude_stability_term", 0.0)),
                    float(rec.get("mean_angular_rate_term", 0.0)),
                    float(rec.get("mean_safety_term", 0.0)),
                    float(rec.get("mean_action_reg_term", 0.0)),
                    float(rec.get("mean_tracking_group_term", 0.0)),
                    float(rec.get("mean_safety_group_term", 0.0)),
                    float(rec.get("mean_action_group_term", 0.0)),
                    float(rec.get("mean_tracking_contrib", 0.0)),
                    float(rec.get("mean_safety_contrib", 0.0)),
                    float(rec.get("mean_action_contrib", 0.0)),
                    float(rec.get("mean_tracking_reward", 0.0)),
                    float(rec.get("mean_observation_reward", 0.0)),
                    float(rec.get("mean_coordination_reward", 0.0)),
                    float(rec.get("mean_communication_reward", 0.0)),
                    float(rec.get("mean_semantic_reward", 0.0)),
                    float(rec.get("mean_control_cost", 0.0)),
                    float(rec.get("mean_tracking_error", 0.0)),
                    float(rec.get("mean_tracking_error_delta", 0.0)),
                    float(rec.get("mean_observation_confidence", 0.0)),
                    float(rec.get("mean_target_lost", 0.0)),
                    float(rec.get("mean_communication_quality", 0.0)),
                    float(rec.get("mean_action_clip_rate", 0.0)),
                    float(rec.get("mean_action_saturation_rate", 0.0)),
                    float(rec.get("mean_action_delta_norm", 0.0)),
                ]
            )
        self._write_csv_atomic(path, rows)

    def _residual_baseline_action(self) -> np.ndarray:
        """Weak 6DOF tau baseline used only in residual_tau mode."""
        if self._env is None or not getattr(self._env.world, "targets", None):
            return np.zeros((self._n_agent, 6), dtype=np.float32)
        target = self._env.world.targets[0]
        target_pos = np.asarray(target.state.p_pos, dtype=np.float64)
        target_vel = np.asarray(target.state.p_vel, dtype=np.float64)
        baseline = np.zeros((self._n_agent, 6), dtype=np.float64)
        for idx, agent in enumerate(self._env.world.agents):
            params = agent.auv6_params
            pos = np.asarray(agent.state.p_pos, dtype=np.float64)
            rel_world = target_pos - pos
            dist = float(np.linalg.norm(rel_world))
            if dist > 1e-9:
                los_world = rel_world / dist
            else:
                los_world = np.zeros(3, dtype=np.float64)
            rot_t = rotation_zyx(agent.state.phi, agent.state.theta, agent.state.psi).T
            los_body = rot_t @ los_world
            target_vel_body = rot_t @ target_vel
            body_vel = np.array([agent.state.u, agent.state.v, agent.state.w], dtype=np.float64)
            desired_body_vel = target_vel_body + float(self._residual_los_speed) * los_body
            vel_scale = np.maximum(np.array([params.umax, params.vmax, params.wmax], dtype=np.float64), 1e-6)
            force_norm = float(self._residual_vel_gain) * (desired_body_vel - body_vel) / vel_scale

            att = np.array([agent.state.phi, agent.state.theta, 0.0], dtype=np.float64)
            omega = np.array([agent.state.p, agent.state.q, agent.state.p_w], dtype=np.float64)
            omega_scale = np.maximum(np.array([params.pmax, params.qmax, params.rmax], dtype=np.float64), 1e-6)
            moment_norm = -float(self._residual_att_gain) * att - float(self._residual_rate_gain) * omega / omega_scale
            baseline[idx] = np.concatenate([force_norm, moment_norm], axis=0)
        return np.clip(baseline * float(self._residual_baseline_scale), -0.5, 0.5).astype(np.float32)

    def _apply_boundary_guard(self, action_norm: np.ndarray) -> np.ndarray:
        """Dampen outward force near walls while preserving the 6D action contract."""
        if (not self._boundary_guard_enabled) or self._env is None:
            return action_norm
        guarded = np.asarray(action_norm, dtype=np.float32).copy()
        limit = max(1e-6, float(getattr(self._env.world, "boundary_limit", 1.0)))
        guard_ratio = float(np.clip(self._boundary_guard_ratio, 0.0, 0.99))
        damping = float(np.clip(self._boundary_outward_damping, 0.0, 1.0))
        for idx, agent in enumerate(self._env.world.agents):
            pos = np.asarray(agent.state.p_pos, dtype=np.float64)
            if pos.shape[0] < 3:
                continue
            ratio = np.abs(pos[:3]) / limit
            if float(np.max(ratio)) <= guard_ratio:
                continue

            rot = rotation_zyx(agent.state.phi, agent.state.theta, agent.state.psi)
            force_world = rot @ guarded[idx, :3].astype(np.float64)
            correction_world = np.zeros(3, dtype=np.float64)
            for axis in range(3):
                if ratio[axis] <= guard_ratio:
                    continue
                sign = float(np.sign(pos[axis]) or 1.0)
                over = float(np.clip((ratio[axis] - guard_ratio) / max(1e-6, 1.0 - guard_ratio), 0.0, 1.0))
                if force_world[axis] * sign > 0.0:
                    force_world[axis] *= damping
                correction_world[axis] -= sign * float(self._boundary_guard_gain) * over
            guarded[idx, :3] = (rot.T @ (force_world + correction_world)).astype(np.float32)
        return np.clip(guarded, -1.0, 1.0)

    def _apply_separation_guard(self, action_norm: np.ndarray) -> np.ndarray:
        """Add weak pairwise repulsion to prevent multi-AUV collapse near target."""
        if (not self._separation_guard_enabled) or self._env is None:
            return action_norm
        guarded = np.asarray(action_norm, dtype=np.float32).copy()
        agents = list(self._env.world.agents)
        sep_dist = max(1e-6, float(self._separation_guard_distance))
        gain = float(self._separation_guard_gain)
        for idx, agent in enumerate(agents):
            pos = np.asarray(agent.state.p_pos, dtype=np.float64)
            repulse_world = np.zeros(3, dtype=np.float64)
            for jdx, other in enumerate(agents):
                if idx == jdx:
                    continue
                delta = pos - np.asarray(other.state.p_pos, dtype=np.float64)
                dist = float(np.linalg.norm(delta))
                if dist <= 1e-9 or dist >= sep_dist:
                    continue
                repulse_world += (delta / dist) * gain * ((sep_dist - dist) / sep_dist)
            if np.linalg.norm(repulse_world) <= 1e-12:
                continue
            rot = rotation_zyx(agent.state.phi, agent.state.theta, agent.state.psi)
            guarded[idx, :3] = guarded[idx, :3] + (rot.T @ repulse_world).astype(np.float32)
        return np.clip(guarded, -1.0, 1.0)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        """Run one step and package outputs into BaseEnvTimestep."""
        self._step_count += 1
        step_action_clip_rate = 0.0
        step_action_saturation_rate = 0.0
        step_action_delta_norm = 0.0
        if self._discrete_action:
            action_cont = self._decode_discrete_action(action)
            step_action_saturation_rate = float(np.mean(np.abs(action_cont) >= 0.999))
            if self._prev_action_cont is not None:
                step_action_delta_norm = float(
                    np.mean(np.linalg.norm(action_cont - self._prev_action_cont, axis=1))
                )
            self._prev_action_cont = action_cont.astype(np.float32)
            action_idx = np.asarray(action).reshape(-1).astype(np.int64)
            action_onehot = np.eye(len(self._action_map), dtype=np.float32)[action_idx]
        else:
            action_cont = np.asarray(action, dtype=np.float32)
            # Handle various shapes from different DI-engine policies:
            #   (n_agent, act_dim), (n_agent*act_dim,), or nested structures.
            if action_cont.ndim == 1:
                if action_cont.shape[0] == self._n_agent * self._env.action_dim:
                    action_cont = action_cont.reshape(self._n_agent, self._env.action_dim)
                elif action_cont.shape[0] == self._env.action_dim:
                    action_cont = np.repeat(action_cont[None, :], self._n_agent, axis=0)
            if action_cont.shape != (self._n_agent, self._env.action_dim):
                raise ValueError(
                    f"action shape must be ({self._n_agent}, {self._env.action_dim}), got {action_cont.shape}"
                )
            # Scale and clip continuous actions to valid range. In residual_tau,
            # the policy learns a residual around a weak 6DOF LOS stabilizer.
            action_scaled = action_cont * float(self._action_scale)
            if self._control_mode == "residual_tau":
                action_scaled = action_scaled + self._residual_baseline_action()
            action_scaled = self._apply_boundary_guard(action_scaled)
            action_scaled = self._apply_separation_guard(action_scaled)
            action_cont = np.clip(action_scaled, -1.0, 1.0)
            step_action_clip_rate = float(np.mean(np.abs(action_scaled) > 1.0))
            step_action_saturation_rate = float(np.mean(np.abs(action_cont) >= 0.999))
            if self._prev_action_cont is not None:
                step_action_delta_norm = float(
                    np.mean(np.linalg.norm(action_cont - self._prev_action_cont, axis=1))
                )
            self._prev_action_cont = action_cont.astype(np.float32)
            action_onehot = None

        obs, reward_n, terminated, truncated, info = self._env.step(action_cont)
        if self._augment_coma_obs and self._discrete_action:
            self._prev_action_onehot = action_onehot
        obs_out = self._process_obs(obs, self._prev_action_onehot)
        reward_n = np.asarray(reward_n, dtype=np.float32)

        reward_scalar = float(np.mean(reward_n) if self._shared_reward else np.sum(reward_n))

        done = bool(terminated or truncated)
        self._eval_episode_return += reward_scalar

        info = dict(info)
        step_collision = int(info.get("collision_events", 0))
        step_out = int(info.get("out_of_bounds_events", 0))
        step_unstable = int(info.get("instability_events", 0))
        current_target_distance = self._current_target_distance()
        current_centroid_distance = self._current_centroid_target_distance()
        reward_terms = info.get("reward_terms_mean", {})
        current_tracking_error = current_target_distance
        current_target_lost = 0.0
        if isinstance(reward_terms, dict):
            current_tracking_error = float(reward_terms.get("tracking_error", current_target_distance))
            current_target_lost = float(reward_terms.get("target_lost", 0.0))
        self._current_final_target_distance = current_target_distance
        self._current_final_centroid_distance = current_centroid_distance
        self._target_distance_sum += current_target_distance
        self._target_distance_count += 1
        self._centroid_distance_sum += current_centroid_distance
        self._centroid_distance_count += 1
        self._target_distance_tail.append(current_target_distance)
        self._centroid_distance_tail.append(current_centroid_distance)
        self._tracking_error_tail.append(current_tracking_error)
        self._target_lost_tail.append(current_target_lost)
        self._action_norm_tail.append(float(info.get("action_norm", 0.0)))
        self._action_saturation_tail.append(float(step_action_saturation_rate))
        self._action_clip_rate_sum += float(step_action_clip_rate)
        self._action_saturation_rate_sum += float(step_action_saturation_rate)
        self._action_delta_norm_sum += float(step_action_delta_norm)
        self._action_stat_count += 1
        if isinstance(reward_terms, dict):
            self._centroid_distance_term_sum += float(reward_terms.get("centroid_distance_term", 0.0))
            self._centroid_progress_term_sum += float(reward_terms.get("centroid_progress_term", 0.0))
            self._centroid_near_term_sum += float(reward_terms.get("centroid_near_term", 0.0))
            self._distance_term_sum += float(reward_terms.get("distance_term", 0.0))
            self._progress_term_sum += float(reward_terms.get("progress_term", 0.0))
            self._closing_speed_term_sum += float(reward_terms.get("closing_speed_term", 0.0))
            self._near_term_sum += float(reward_terms.get("near_term", 0.0))
            self._success_term_sum += float(reward_terms.get("success_term", 0.0))
            self._attitude_stability_term_sum += float(reward_terms.get("attitude_stability_term", 0.0))
            self._angular_rate_term_sum += float(reward_terms.get("angular_rate_term", 0.0))
            self._safety_term_sum += float(reward_terms.get("safety_term", 0.0))
            self._action_reg_term_sum += float(reward_terms.get("action_reg_term", 0.0))
            self._tracking_group_term_sum += float(reward_terms.get("tracking_group_term", 0.0))
            self._safety_group_term_sum += float(reward_terms.get("safety_group_term", 0.0))
            self._action_group_term_sum += float(reward_terms.get("action_group_term", 0.0))
            self._tracking_contrib_sum += float(reward_terms.get("tracking_contrib", 0.0))
            self._safety_contrib_sum += float(reward_terms.get("safety_contrib", 0.0))
            self._action_contrib_sum += float(reward_terms.get("action_contrib", 0.0))
            self._tracking_reward_sum += float(reward_terms.get("tracking_reward", reward_terms.get("tracking_group_term", 0.0)))
            self._observation_reward_sum += float(reward_terms.get("observation_reward", 0.0))
            self._coordination_reward_sum += float(reward_terms.get("coordination_reward", 0.0))
            self._communication_reward_sum += float(reward_terms.get("communication_reward", 0.0))
            self._semantic_reward_sum += float(reward_terms.get("semantic_reward", 0.0))
            self._control_cost_sum += float(reward_terms.get("control_cost", -reward_terms.get("action_group_term", 0.0)))
            self._tracking_error_sum += float(reward_terms.get("tracking_error", current_target_distance))
            self._tracking_error_delta_sum += float(reward_terms.get("tracking_error_delta", 0.0))
            self._observation_confidence_sum += float(reward_terms.get("observation_confidence", 0.0))
            self._target_lost_sum += float(reward_terms.get("target_lost", 0.0))
            self._communication_quality_sum += float(reward_terms.get("communication_quality", 0.0))
            self._reward_term_count += 1
        self._episode_collision += step_collision
        self._episode_out_of_bounds += step_out
        self._episode_unstable += step_unstable
        info["per_agent_reward"] = reward_n.tolist()
        info["constraint_violations"] = {
            "collision": step_collision,
            "out_of_bounds": step_out,
            "unstable": step_unstable,
        }
        info["tracking_metrics"] = {
            "current_mean_target_distance": current_target_distance,
            "current_centroid_target_distance": current_centroid_distance,
            "action_clip_rate": float(step_action_clip_rate),
            "action_saturation_rate": float(step_action_saturation_rate),
            "action_delta_norm": float(step_action_delta_norm),
        }
        if self._discrete_action:
            info["discrete_codebook_size"] = int(len(self._action_map))
            info["discrete_full_action_size"] = int(self._full_discrete_size)

        if done:
            self._episode_count += 1
            self._episode_rewards.append(self._eval_episode_return)
            info["eval_episode_return"] = self._eval_episode_return
            if self._is_evaluator:
                eval_record = self._build_eval_record()
                self._eval_episode_records.append(eval_record)
                self._save_eval_detail_raw()
                self._save_eval_detail()
                self._append_eval_jsonl(eval_record)
                self._emit_eval_detail(eval_record)
                if self._normalize_obs_eval_mode == "frozen":
                    self._eval_obs_stats_frozen = True
            if self._print_episode_reward:
                prefix = "evaluate: " if self._is_evaluator else ""
                print(f"{prefix}Episode {self._episode_count}, Total Reward: {self._eval_episode_return:.2f}")

        reward = np.asarray([reward_scalar], dtype=np.float32)
        return BaseEnvTimestep(obs_out, reward, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        del dynamic_seed
        self._seed = int(seed)
        np.random.seed(self._seed)
        if self._env is not None:
            self._env.seed(self._seed)

    def _save_rewards(self) -> None:
        self._save_dir.mkdir(parents=True, exist_ok=True)
        rewards_np = np.asarray(self._episode_rewards, dtype=np.float32)
        npy_path = self._save_dir / "auv6dof_reward_data.npy"
        csv_path = self._save_dir / "auv6dof_reward_curve.csv"

        for retry in range(6):
            try:
                with npy_path.open("wb") as f:
                    np.save(f, rewards_np)
                break
            except PermissionError:
                if retry >= 5:
                    raise
                time.sleep(0.2 * (retry + 1))

        rows = [["episode", "reward"]]
        rows.extend([[idx, float(value)] for idx, value in enumerate(rewards_np.tolist())])
        self._write_csv_atomic(csv_path, rows)

    def _write_csv_atomic(self, path: Path, rows: List[List[float | int | str]]) -> None:
        """Write a CSV file with retry for transient Windows file locks."""
        path.parent.mkdir(parents=True, exist_ok=True)
        for retry in range(6):
            try:
                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                return
            except PermissionError:
                if retry >= 5:
                    raise
                time.sleep(0.2 * (retry + 1))

    def _save_eval_detail_raw(self) -> None:
        """Persist evaluator episode records for later conversion to eval_detail.csv."""
        if len(self._eval_episode_records) == 0:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        path = self._save_dir / f"auv6dof_eval_detail_raw_{self._evaluator_id}.csv"
        rows: List[List[float | int | str]] = [
            [
                "eval_index",
                "train_step",
                "eval_return",
                "episode_return",
                "eval_steps",
                "episode_steps",
                "horizon_steps",
                "final_centroid_target_distance",
                "mean_centroid_target_distance_over_episode",
                "tail_mean_centroid_target_distance",
                "collision",
                "out_of_bounds",
                "unstable",
                "final_mean_target_distance",
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
        ]
        for rec in self._eval_episode_records:
            rows.append(
                [
                    int(rec.get("eval_index", 0)),
                    int(rec.get("train_step", 0)),
                    float(rec.get("eval_return", rec.get("episode_return", 0.0))),
                    float(rec.get("episode_return", rec.get("eval_return", 0.0))),
                    int(rec.get("eval_steps", rec.get("episode_steps", 0))),
                    int(rec.get("episode_steps", rec.get("eval_steps", 0))),
                    int(rec.get("horizon_steps", self._eval_horizon_steps or rec.get("episode_steps", 0))),
                    float(rec.get("final_centroid_target_distance", 0.0)),
                    float(rec.get("mean_centroid_target_distance_over_episode", 0.0)),
                    float(rec.get("tail_mean_centroid_target_distance", 0.0)),
                    int(rec.get("collision", 0)),
                    int(rec.get("out_of_bounds", 0)),
                    int(rec.get("unstable", 0)),
                    float(rec.get("final_mean_target_distance", 0.0)),
                    float(rec.get("mean_target_distance_over_episode", 0.0)),
                    float(rec.get("tail_mean_target_distance", 0.0)),
                    float(rec.get("tail100_mean_target_distance", rec.get("tail_mean_target_distance", 0.0))),
                    float(rec.get("tail100_std_target_distance", 0.0)),
                    float(rec.get("tail100_mean_tracking_error", rec.get("mean_tracking_error", 0.0))),
                    float(rec.get("tail100_target_lost_rate", rec.get("mean_target_lost", 0.0))),
                    float(rec.get("tail100_action_norm", rec.get("action_norm", 0.0))),
                    float(rec.get("tail100_action_saturation_rate", rec.get("mean_action_saturation_rate", 0.0))),
                    float(rec.get("mean_centroid_distance_term", 0.0)),
                    float(rec.get("mean_centroid_progress_term", 0.0)),
                    float(rec.get("mean_centroid_near_term", 0.0)),
                    float(rec.get("mean_distance_term", 0.0)),
                    float(rec.get("mean_progress_term", 0.0)),
                    float(rec.get("mean_closing_speed_term", 0.0)),
                    float(rec.get("mean_near_term", 0.0)),
                    float(rec.get("mean_success_term", 0.0)),
                    float(rec.get("mean_attitude_stability_term", 0.0)),
                    float(rec.get("mean_angular_rate_term", 0.0)),
                    float(rec.get("mean_safety_term", 0.0)),
                    float(rec.get("mean_action_reg_term", 0.0)),
                    float(rec.get("mean_tracking_group_term", 0.0)),
                    float(rec.get("mean_safety_group_term", 0.0)),
                    float(rec.get("mean_action_group_term", 0.0)),
                    float(rec.get("mean_tracking_contrib", 0.0)),
                    float(rec.get("mean_safety_contrib", 0.0)),
                    float(rec.get("mean_action_contrib", 0.0)),
                    float(rec.get("mean_tracking_reward", 0.0)),
                    float(rec.get("mean_observation_reward", 0.0)),
                    float(rec.get("mean_coordination_reward", 0.0)),
                    float(rec.get("mean_communication_reward", 0.0)),
                    float(rec.get("mean_semantic_reward", 0.0)),
                    float(rec.get("mean_control_cost", 0.0)),
                    float(rec.get("mean_tracking_error", 0.0)),
                    float(rec.get("mean_tracking_error_delta", 0.0)),
                    float(rec.get("mean_observation_confidence", 0.0)),
                    float(rec.get("mean_target_lost", 0.0)),
                    float(rec.get("mean_communication_quality", 0.0)),
                    float(rec.get("mean_action_clip_rate", 0.0)),
                    float(rec.get("mean_action_saturation_rate", 0.0)),
                    float(rec.get("mean_action_delta_norm", 0.0)),
                ]
            )
        self._write_csv_atomic(path, rows)

    def close(self) -> None:
        if (not self._is_evaluator) and len(self._episode_rewards) > 0:
            self._save_rewards()
        if self._is_evaluator:
            self._save_eval_detail_raw()
            self._save_eval_detail()
        if self._env is not None:
            self._env.close()
        self._env = None
        self._init_flag = False

    def __repr__(self) -> str:
        return "DI-engine AUV6DOF Env"

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        cfg = EasyDict(cfg)
        collector_env_num = int(cfg.pop("collector_env_num"))
        cfg.is_evaluator = False
        return [EasyDict(cfg) for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        cfg = EasyDict(cfg)
        evaluator_env_num = int(cfg.pop("evaluator_env_num"))
        cfg.is_evaluator = True
        return [EasyDict({**cfg, "evaluator_id": idx}) for idx in range(evaluator_env_num)]

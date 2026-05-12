from __future__ import annotations

"""
Gymnasium 风格的 AUV6DOF 多智能体环境封装。

该层只做两件事：
- 提供标准 Gym API（reset/step/seed/render/close）；
- 把归一化动作映射到物理力/力矩，再调用底层 world.step()。
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .scenario_v2 import AUV6DOFScenario, ObservationConfig, ResetConfig, RewardConfig


class AUV6DOFGymEnv(gym.Env):
    """AUV6DOF 任务的 Gym 主环境。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    config = dict(
        n_agent=4,
        episode_length=400,
        boundary_limit=1.0,
        dt=0.1,
        action_control_mode="tau6",
        action_smoothing=0.0,
        max_action_delta=0.0,
        velocity_command_gain=1.0,
        attitude_damping_gain=0.12,
        rate_damping_gain=0.10,
        auv_model=dict(
            profile="remus100_mss",
            overrides={},
        ),
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
            desired_tracking_distance=0.08,
            sensor_range=0.45,
            lost_distance=0.65,
            tracking_error_clip=0.8,
            w_tracking_reward=0.55,
            w_observation_reward=0.20,
            w_coordination_reward=0.10,
            w_communication_reward=0.05,
            w_semantic_reward=0.10,
            w_control_cost=0.08,
            tracking_band_lower=0.010,
            tracking_band_upper=0.015,
            w_band_stability=0.35,
            w_reacquire=0.20,
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
            include_tracking_diagnostics=False,
            include_semantic_features=False,
            include_semantic_graph_features=False,
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

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg = {} if cfg is None else dict(cfg)
        merged_cfg = dict(self.config)
        merged_cfg.update(cfg)

        reward_cfg_dict = dict(self.config["reward"])
        reward_cfg_dict.update(dict(merged_cfg.get("reward", {})))
        reward_cfg = RewardConfig(**reward_cfg_dict)
        obs_cfg_dict = dict(self.config["obs"])
        obs_cfg_dict.update(dict(merged_cfg.get("obs", {})))
        obs_cfg = ObservationConfig(**obs_cfg_dict)
        reset_cfg_dict = dict(self.config["reset"])
        reset_cfg_dict.update(dict(merged_cfg.get("reset", {})))
        # Stage split is handled by DI env wrapper; scenario reset config
        # only accepts curriculum_stage.
        reset_cfg_dict.pop("train_curriculum_stage", None)
        reset_cfg_dict.pop("eval_curriculum_stage", None)
        reset_cfg = ResetConfig(**reset_cfg_dict)
        auv_model_cfg = dict(self.config.get("auv_model", {}))
        auv_model_cfg.update(dict(merged_cfg.get("auv_model", {})))

        self.n_agent = int(merged_cfg["n_agent"])
        self.episode_length = int(merged_cfg["episode_length"])
        self.boundary_limit = float(merged_cfg["boundary_limit"])
        self.dt = float(merged_cfg["dt"])
        self.action_control_mode = str(merged_cfg.get("action_control_mode", "tau6")).strip().lower()
        if self.action_control_mode not in {"tau6", "velocity3"}:
            self.action_control_mode = "tau6"
        self.action_smoothing = float(np.clip(float(merged_cfg.get("action_smoothing", 0.0)), 0.0, 1.0))
        self.max_action_delta = float(max(0.0, merged_cfg.get("max_action_delta", 0.0)))
        self.velocity_command_gain = float(max(0.0, merged_cfg.get("velocity_command_gain", 1.0)))
        self.attitude_damping_gain = float(max(0.0, merged_cfg.get("attitude_damping_gain", 0.12)))
        self.rate_damping_gain = float(max(0.0, merged_cfg.get("rate_damping_gain", 0.10)))
        self._seed = 0

        # 场景对象负责 world 构造、观测与奖励逻辑。
        self.scenario = AUV6DOFScenario(
            n_agent=self.n_agent,
            n_target=1,
            n_landmark=3,
            boundary_limit=self.boundary_limit,
            reward_config=reward_cfg,
            obs_config=obs_cfg,
            reset_config=reset_cfg,
            auv_model=auv_model_cfg,
        )
        self.world = self.scenario.make_world()
        self.world.dt = self.dt

        obs_dim = len(self.scenario.observation(self.world.agents[0], self.world))
        self.agent_obs_dim = int(obs_dim)
        self.action_dim = 3 if self.action_control_mode == "velocity3" else 6
        self._prev_command_action = np.zeros((self.n_agent, self.action_dim), dtype=np.float64)
        self._last_action_debug: Dict[str, Any] = {}

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_agent, self.action_dim), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "agent_state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.n_agent, self.agent_obs_dim), dtype=np.float32
                ),
                "global_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_agent, self.n_agent * self.agent_obs_dim),
                    dtype=np.float32,
                ),
            }
        )

        self._step_count = 0
        self._episode_return = 0.0

    def seed(self, seed: int) -> None:
        """设置环境随机种子。"""
        self._seed = int(seed)
        np.random.seed(self._seed)

    def _build_obs(self) -> Dict[str, np.ndarray]:
        """构造当前时刻的多智能体观测字典。"""
        agent_state = np.stack(
            [self.scenario.observation(agent, self.world) for agent in self.world.agents], axis=0
        ).astype(np.float32)
        global_state_flat = np.concatenate([obs for obs in agent_state], axis=0).astype(np.float32)
        global_state = np.repeat(global_state_flat[None, :], self.n_agent, axis=0)
        return {"agent_state": agent_state, "global_state": global_state}

    def _smooth_action(self, action_norm: np.ndarray) -> np.ndarray:
        raw = np.clip(np.asarray(action_norm, dtype=np.float64), -1.0, 1.0)
        if self.max_action_delta > 0.0:
            delta = np.clip(raw - self._prev_command_action, -self.max_action_delta, self.max_action_delta)
            clipped = self._prev_command_action + delta
        else:
            clipped = raw
        alpha = self.action_smoothing
        smoothed = (1.0 - alpha) * clipped + alpha * self._prev_command_action
        self._last_action_debug = {
            "raw_action": raw.astype(np.float32),
            "clipped_action": clipped.astype(np.float32),
            "action_norm": float(np.mean(np.linalg.norm(smoothed, axis=1))),
            "action_delta_norm": float(np.mean(np.linalg.norm(smoothed - self._prev_command_action, axis=1))),
        }
        self._prev_command_action = smoothed.copy()
        return np.clip(smoothed, -1.0, 1.0)

    def _scale_action_to_tau(self, action_norm: np.ndarray) -> np.ndarray:
        """将 [-1,1] 归一化动作缩放为每个 agent 的真实力/力矩上限。"""
        action_norm = np.clip(action_norm, -1.0, 1.0)
        tau = np.zeros_like(action_norm, dtype=np.float64)
        for i, agent in enumerate(self.world.agents):
            params = agent.auv6_params
            max_tau = np.array(
                [
                    params.taumax_x,
                    params.taumax_y,
                    params.taumax_z,
                    params.taumax_k,
                    params.taumax_m,
                    params.taumax_n,
                ],
                dtype=np.float64,
            )
            tau[i] = action_norm[i] * max_tau
        self._last_action_debug["mapped_action"] = tau.astype(np.float32)
        return tau

    def _velocity_action_to_tau(self, action_norm: np.ndarray) -> np.ndarray:
        from .dynamics import rotation_zyx

        action_norm = self._smooth_action(action_norm)
        tau = np.zeros((self.n_agent, 6), dtype=np.float64)
        for i, agent in enumerate(self.world.agents):
            params = agent.auv6_params
            max_world_vel = np.array([params.umax, params.vmax, params.wmax], dtype=np.float64)
            desired_world_vel = action_norm[i] * max_world_vel
            current_world_vel = (
                np.asarray(agent.state.p_vel, dtype=np.float64)
                if agent.state.p_vel is not None
                else np.zeros(3, dtype=np.float64)
            )
            vel_error_world = desired_world_vel - current_world_vel
            rot_t = rotation_zyx(agent.state.phi, agent.state.theta, agent.state.psi).T
            vel_error_body = rot_t @ vel_error_world
            vel_scale = np.maximum(np.array([params.umax, params.vmax, params.wmax], dtype=np.float64), 1e-6)
            force_norm = np.clip(self.velocity_command_gain * vel_error_body / vel_scale, -1.0, 1.0)
            tau[i, :3] = force_norm * np.array([params.taumax_x, params.taumax_y, params.taumax_z], dtype=np.float64)

            att = np.array([agent.state.phi, agent.state.theta, 0.0], dtype=np.float64)
            omega = np.array([agent.state.p, agent.state.q, agent.state.p_w], dtype=np.float64)
            omega_scale = np.maximum(np.array([params.pmax, params.qmax, params.rmax], dtype=np.float64), 1e-6)
            moment_norm = -self.attitude_damping_gain * att - self.rate_damping_gain * omega / omega_scale
            tau[i, 3:] = np.clip(moment_norm, -1.0, 1.0) * np.array(
                [params.taumax_k, params.taumax_m, params.taumax_n], dtype=np.float64
            )
        self._last_action_debug["mapped_action"] = tau.astype(np.float32)
        return tau

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境并返回初始观测。"""
        del options
        if seed is not None:
            self.seed(seed)
        else:
            np.random.seed(self._seed)
        self.scenario.reset_world(self.world)
        self._step_count = 0
        self._episode_return = 0.0
        self._prev_command_action = np.zeros((self.n_agent, self.action_dim), dtype=np.float64)
        self._last_action_debug = {}
        obs = self._build_obs()
        info = {"n_agent": self.n_agent}
        return obs, info

    def step(self, action: np.ndarray):
        """执行一步仿真并返回 Gym 五元组。"""
        action = np.asarray(action, dtype=np.float64)
        if action.shape == (self.action_dim,):
            action = np.repeat(action[None, :], self.n_agent, axis=0)
        if action.shape != (self.n_agent, self.action_dim):
            raise ValueError(
                f"action shape must be ({self.n_agent}, {self.action_dim}), got {action.shape}"
            )

        # Map normalized policy actions to executable 6DOF control.
        if self.action_control_mode == "velocity3":
            tau_actions = self._velocity_action_to_tau(action)
        else:
            action = self._smooth_action(action)
            tau_actions = self._scale_action_to_tau(action)
        for i, agent in enumerate(self.world.agents):
            agent.action.u = tau_actions[i].astype(np.float32)

        metrics = self.world.step()
        reward_n = np.asarray([self.scenario.reward(agent, self.world) for agent in self.world.agents], dtype=np.float32)

        self._step_count += 1
        self._episode_return += float(np.sum(reward_n))
        terminated = False
        truncated = self._step_count >= self.episode_length

        obs = self._build_obs()
        info: Dict[str, Any] = dict(metrics)
        info["per_agent_reward"] = reward_n.tolist()
        info["reward_terms_mean"] = self.scenario.get_last_reward_terms_mean()
        info["target_position"] = (
            self.world.targets[0].state.p_pos.tolist() if self.world.targets else [0.0, 0.0, 0.0]
        )
        info["target_velocity"] = (
            self.world.targets[0].state.p_vel.tolist() if self.world.targets else [0.0, 0.0, 0.0]
        )
        for key, value in self._last_action_debug.items():
            info[key] = value.tolist() if isinstance(value, np.ndarray) else value
        reward_terms = info.get("reward_terms_mean", {})
        if isinstance(reward_terms, dict):
            for key in (
                "tracking_reward",
                "observation_reward",
                "coordination_reward",
                "communication_reward",
                "semantic_reward",
                "control_cost",
                "total_reward",
                "tracking_error",
                "tracking_error_delta",
                "target_distance",
                "observation_confidence",
                "target_lost",
                "communication_quality",
                "action_norm",
                "action_delta_norm",
            ):
                info[key] = float(reward_terms.get(key, info.get(key, 0.0)))
        if truncated:
            info["episode_return"] = float(self._episode_return)
        return obs, reward_n, terminated, truncated, info

    def render(self):
        """返回可视化所需的关键位姿数据。"""
        return {
            "agent_positions": [agent.state.p_pos.copy() for agent in self.world.agents],
            "target_positions": [target.state.p_pos.copy() for target in self.world.targets],
        }

    def close(self) -> None:
        """释放环境资源（当前实现无额外句柄）。"""
        return None

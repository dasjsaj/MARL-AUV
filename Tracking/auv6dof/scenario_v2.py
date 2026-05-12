from __future__ import annotations

"""
AUV6DOF scenario with curriculum reset, richer observation features, and
reward shaping designed for faster and more stable MARL convergence.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dynamics import Agent, Landmark, Target, UniformCurrent3D, World, build_auv6_params, rotation_zyx


@dataclass
class RewardConfig:
    """Reward settings. Supports both legacy(v1) and shaped(v2) modes."""

    version: str = "v2_fast_converge"

    # Legacy v1 settings (kept for A/B baseline compatibility)
    d_target_min: float = 0.015
    d_auv_min: float = 0.015
    near_target_scale: float = 0.2
    collision_scale: float = 5.0
    pos_weight: float = 0.95
    col_weight: float = 0.05

    # v2 shaping settings — tuned for stable MARL convergence.
    # Key principle: keep total reward in [-1, +1] range per step to
    # avoid value function saturation and allow gradient signal to flow.
    distance_clip: float = 1.2
    near_distance: float = 0.10
    safe_distance: float = 0.08
    success_distance: float = 0.025
    boundary_soft_ratio: float = 0.75
    progress_clip: float = 0.04
    closing_speed_clip: float = 0.08
    attitude_angle_clip: float = 0.75
    angular_rate_clip: float = 1.2
    # Disable centroid objective in the main reward path.
    w_centroid_distance: float = 0.0
    w_centroid_progress: float = 0.0
    w_centroid_near: float = 0.0
    w_centroid_success: float = 0.0
    # Individual tracking group (sum=0.85).
    w_distance: float = 0.25
    w_progress: float = 0.40
    w_closing_speed: float = 0.20
    w_near: float = 0.10
    w_success: float = 0.10
    # Safety group (sum=0.12).
    w_separation: float = 0.03
    w_boundary: float = 0.03
    w_collision: float = 0.03
    w_oob: float = 0.02
    w_unstable: float = 0.01
    w_attitude_stability: float = 0.01
    w_angular_rate: float = 0.01
    # Action regularization group (sum=0.03).
    w_action_energy: float = 0.015
    w_action_smooth: float = 0.015
    # Group-level weights (tracking:safety:action = 0.85:0.12:0.03).
    w_tracking_group: float = 0.85
    w_safety_group: float = 0.12
    w_action_group: float = 0.03
    # v3 target-tracking reward used by convergence debug/formal configs.
    desired_tracking_distance: float = 0.08
    sensor_range: float = 0.45
    lost_distance: float = 0.65
    tracking_error_clip: float = 0.8
    w_tracking_reward: float = 0.55
    w_observation_reward: float = 0.20
    w_coordination_reward: float = 0.10
    w_communication_reward: float = 0.05
    w_semantic_reward: float = 0.10
    w_control_cost: float = 0.08
    tracking_band_lower: float = 0.010
    tracking_band_upper: float = 0.015
    w_band_stability: float = 0.35
    w_reacquire: float = 0.20


@dataclass
class ObservationConfig:
    """Observation feature switches."""

    include_target_velocity: bool = True
    include_relative_velocity: bool = True
    include_target_rel_body: bool = True
    include_relative_velocity_body: bool = True
    include_los_unit_body: bool = True
    include_prev_action: bool = True
    use_attitude_sin_cos: bool = True
    include_boundary_margin: bool = True
    normalize_physical: bool = True
    include_tracking_diagnostics: bool = False
    include_semantic_features: bool = False
    include_semantic_graph_features: bool = False


@dataclass
class ResetConfig:
    """Curriculum reset controls."""

    curriculum_stage: str = "auto"
    min_init_separation: float = 0.10
    auto_easy_episodes: int = 800
    auto_medium_episodes: int = 2000
    easy_target_speed_range: Tuple[float, float] = (0.001, 0.004)
    medium_target_speed_range: Tuple[float, float] = (0.003, 0.008)
    hard_target_speed_range: Tuple[float, float] = (0.006, 0.014)


class AUV6DOFScenario:
    def __init__(
        self,
        n_agent: int = 4,
        n_target: int = 1,
        n_landmark: int = 3,
        boundary_limit: float = 1.0,
        reward_config: Optional[RewardConfig] = None,
        obs_config: Optional[ObservationConfig] = None,
        reset_config: Optional[ResetConfig] = None,
        auv_model: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_agent = int(n_agent)
        self.n_target = int(n_target)
        self.n_landmark = int(n_landmark)
        self.boundary_limit = float(boundary_limit)
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.obs_config = obs_config if obs_config is not None else ObservationConfig()
        self.reset_config = reset_config if reset_config is not None else ResetConfig()
        auv_model = {} if auv_model is None else dict(auv_model)
        self.auv_model_profile = str(auv_model.get("profile", "remus100_mss")).strip() or "remus100_mss"
        overrides = auv_model.get("overrides", {})
        self.auv_model_overrides = dict(overrides) if isinstance(overrides, dict) else {}

        self._prev_target_distance: Dict[int, float] = {}
        self._prev_centroid_distance: float = 0.0
        self._prev_action_norm: Dict[int, np.ndarray] = {}
        self._last_reward_terms: Dict[int, Dict[str, float]] = {}
        self._tracking_error_history: Dict[int, List[float]] = {}
        self._lost_steps: Dict[int, int] = {}
        self._reset_count: int = 0
        # Cache centroid terms once per world step so all agents share the same
        # centroid progress signal. This avoids per-agent update order bias.
        self._centroid_step_time: Optional[float] = None
        self._centroid_step_terms: Dict[str, float] = {}
        self._centroid_step_agent_count: int = 0

    def _agent_index(self, agent: Agent) -> int:
        if agent.name.startswith("agent "):
            try:
                return int(agent.name.split(" ")[1])
            except Exception:
                pass
        return max(0, min(self.n_agent - 1, id(agent) % max(1, self.n_agent)))

    def _random_unit_vector(self) -> np.ndarray:
        vec = np.random.normal(0.0, 1.0, size=3).astype(np.float64)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return vec / norm

    def _effective_stage(self) -> str:
        stage = str(self.reset_config.curriculum_stage).strip().lower()
        if stage != "auto":
            return stage
        if self._reset_count < int(self.reset_config.auto_easy_episodes):
            return "easy"
        if self._reset_count < int(self.reset_config.auto_medium_episodes):
            return "medium"
        return "hard"

    def _stage_profile(self, stage: str) -> Dict[str, Tuple[float, float] | float]:
        if stage == "legacy":
            return {
                "target_pos_bound": (0.8660254, 0.8660254),
                "agent_radius": (0.0, 0.0),
                "target_speed": (0.0173205, 0.0173205),
                "min_init_separation": 0.0,
            }
        if stage == "hard":
            return {
                "target_pos_bound": (0.45, 0.75),
                "agent_radius": (0.4, 0.7),
                "target_speed": self.reset_config.hard_target_speed_range,
                "min_init_separation": 0.06,
            }
        if stage == "medium":
            return {
                "target_pos_bound": (0.25, 0.55),
                "agent_radius": (0.25, 0.45),
                "target_speed": self.reset_config.medium_target_speed_range,
                "min_init_separation": 0.09,
            }
        # easy by default — agents start very close to target so policy
        # can discover positive reward early in training.
        return {
            "target_pos_bound": (0.05, 0.20),
            "agent_radius": (0.08, 0.20),
            "target_speed": self.reset_config.easy_target_speed_range,
            "min_init_separation": 0.10,
        }

    def make_world(self) -> World:
        world = World()
        world.boundary_limit = self.boundary_limit
        world.agents = [Agent() for _ in range(self.n_agent)]
        world.targets = [Target() for _ in range(self.n_target)]
        world.landmarks = [Landmark() for _ in range(self.n_landmark)]

        for i, agent in enumerate(world.agents):
            agent.name = f"agent {i}"
            agent.collide = True
            agent.size = 0.011
            agent.color = np.array([0.35, 0.35, 0.85], dtype=np.float32)
            agent.movable = True
            agent.auv6_params = build_auv6_params(
                profile=self.auv_model_profile,
                overrides=self.auv_model_overrides,
            )
            agent.current3d = UniformCurrent3D(0.0, 0.0, 0.0)

        for i, target in enumerate(world.targets):
            target.name = f"target {i}"
            target.size = 0.011
            target.color = np.array([0.85, 0.35, 0.35], dtype=np.float32)
            target.movable = True

        self.reset_world(world)
        return world

    def _sample_agent_positions(
        self, target_pos: np.ndarray, radius_low: float, radius_high: float, min_sep: float
    ) -> List[np.ndarray]:
        positions: List[np.ndarray] = []
        for _idx in range(self.n_agent):
            accepted = False
            for _ in range(256):
                radius = float(np.random.uniform(radius_low, radius_high))
                candidate = target_pos + self._random_unit_vector() * radius
                candidate = np.clip(candidate, -0.9 * self.boundary_limit, 0.9 * self.boundary_limit)
                if all(float(np.linalg.norm(candidate - p)) >= min_sep for p in positions):
                    positions.append(candidate.astype(np.float64))
                    accepted = True
                    break
            if not accepted:
                fallback = np.random.uniform(
                    -0.9 * self.boundary_limit, 0.9 * self.boundary_limit, size=3
                ).astype(np.float64)
                positions.append(fallback)
        return positions

    def reset_world(self, world: World) -> None:
        stage = self._effective_stage()
        if stage == "legacy":
            for i, agent in enumerate(world.agents):
                agent.goal = world.targets[i % len(world.targets)] if world.targets else None
                agent.state.p_pos = np.random.uniform(-0.6, -0.5, 3).astype(np.float64)
                agent.state.p_vel = np.zeros(3, dtype=np.float64)
                agent.state.u = 0.0
                agent.state.v = 0.0
                agent.state.w = 0.0
                agent.state.p = 0.0
                agent.state.q = 0.0
                agent.state.p_w = 0.0
                agent.state.phi = 0.0
                agent.state.theta = 0.0
                agent.state.psi = 0.0
                agent.action.u = np.zeros(6, dtype=np.float32)

            if world.targets:
                world.targets[0].state.p_pos = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
                world.targets[0].state.p_vel = np.array([0.01, 0.01, 0.01], dtype=np.float64)
                world.targets[0].state.p_w = 0.0

            for landmark in world.landmarks:
                landmark.state.p_pos = np.random.uniform(-0.2, 0.2, 3).astype(np.float64)
                landmark.size = 0.03
                landmark.color = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            self._prev_target_distance.clear()
            self._prev_centroid_distance = 0.0
            self._prev_action_norm.clear()
            self._tracking_error_history.clear()
            self._lost_steps.clear()
            self._last_reward_terms.clear()
            self._centroid_step_time = None
            self._centroid_step_terms.clear()
            self._centroid_step_agent_count = 0
            world.time = 0.0
            world.last_metrics = {
                "collision_events": 0,
                "out_of_bounds_events": 0,
                "instability_events": 0,
            }
            return

        profile = self._stage_profile(stage)
        pos_low, pos_high = profile["target_pos_bound"]  # type: ignore[index]
        speed_low, speed_high = profile["target_speed"]  # type: ignore[index]
        radius_low, radius_high = profile["agent_radius"]  # type: ignore[index]
        stage_min_sep = float(profile.get("min_init_separation", 0.0))  # type: ignore[arg-type]
        min_sep = max(float(self.reset_config.min_init_separation), stage_min_sep)

        target_pos = self._random_unit_vector() * float(np.random.uniform(float(pos_low), float(pos_high)))
        target_speed = float(np.random.uniform(float(speed_low), float(speed_high)))
        target_vel = self._random_unit_vector() * target_speed

        if world.targets:
            world.targets[0].state.p_pos = target_pos.astype(np.float64)
            world.targets[0].state.p_vel = target_vel.astype(np.float64)
            world.targets[0].state.p_w = 0.0

        agent_positions = self._sample_agent_positions(
            target_pos, float(radius_low), float(radius_high), min_sep=min_sep
        )
        for i, agent in enumerate(world.agents):
            agent.goal = world.targets[i % len(world.targets)] if world.targets else None
            agent.state.p_pos = agent_positions[i]
            agent.state.p_vel = np.zeros(3, dtype=np.float64)
            agent.state.u = 0.0
            agent.state.v = 0.0
            agent.state.w = 0.0
            agent.state.p = 0.0
            agent.state.q = 0.0
            agent.state.p_w = 0.0
            agent.state.phi = 0.0
            agent.state.theta = 0.0
            agent.state.psi = 0.0
            agent.action.u = np.zeros(6, dtype=np.float32)

        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-0.2, 0.2, 3).astype(np.float64)
            landmark.size = 0.03
            landmark.color = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self._prev_target_distance.clear()
        self._prev_centroid_distance = 0.0
        self._prev_action_norm.clear()
        self._tracking_error_history.clear()
        self._lost_steps.clear()
        self._last_reward_terms.clear()
        self._centroid_step_time = None
        self._centroid_step_terms.clear()
        self._centroid_step_agent_count = 0

        world.time = 0.0
        world.last_metrics = {
            "collision_events": 0,
            "out_of_bounds_events": 0,
            "instability_events": 0,
        }
        self._reset_count += 1

    def _ensure_centroid_step_terms(self, world: World, cfg: RewardConfig) -> None:
        """
        Compute centroid-level terms once per world step and reuse for all agents.
        """
        step_time = float(world.time)
        if (
            self._centroid_step_time is not None
            and np.isclose(self._centroid_step_time, step_time)
            and self._centroid_step_terms
        ):
            return

        target = world.targets[0]
        all_agent_pos = np.stack([np.asarray(item.state.p_pos, dtype=np.float64) for item in world.agents], axis=0)
        centroid = np.mean(all_agent_pos, axis=0)
        centroid_rel = np.asarray(target.state.p_pos, dtype=np.float64) - centroid
        centroid_dist = float(np.linalg.norm(centroid_rel))
        centroid_clipped = min(centroid_dist, float(cfg.distance_clip))
        centroid_distance_term = -centroid_clipped / max(1e-6, float(cfg.distance_clip))

        prev_centroid_dist = float(self._prev_centroid_distance if self._prev_centroid_distance > 0 else centroid_dist)
        centroid_progress_raw = prev_centroid_dist - centroid_dist
        progress_clip = max(1e-6, float(cfg.progress_clip))
        centroid_progress_term = float(np.clip(centroid_progress_raw / progress_clip, -1.0, 1.0))
        centroid_near_term = float(
            max(0.0, (float(cfg.near_distance) - centroid_dist) / max(1e-6, float(cfg.near_distance)))
        )
        centroid_success_term = float(
            max(0.0, (float(cfg.success_distance) - centroid_dist) / max(1e-6, float(cfg.success_distance)))
        )
        self._centroid_step_terms = {
            "centroid_dist": float(centroid_dist),
            "centroid_distance_term": float(centroid_distance_term),
            "centroid_progress_term": float(centroid_progress_term),
            "centroid_near_term": float(centroid_near_term),
            "centroid_success_term": float(centroid_success_term),
        }
        self._centroid_step_time = step_time
        self._centroid_step_agent_count = 0

    def _normalized_action(self, agent: Agent) -> np.ndarray:
        tau = np.asarray(agent.action.u, dtype=np.float64).reshape(-1)
        params = agent.auv6_params
        max_tau = np.asarray(
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
        max_tau = np.maximum(max_tau, 1e-6)
        return np.clip(tau / max_tau, -1.0, 1.0)

    @staticmethod
    def _weighted_group_value(terms_and_weights: List[Tuple[float, float]]) -> float:
        weighted_sum = float(sum(weight * term for term, weight in terms_and_weights))
        weight_sum = float(sum(abs(weight) for _, weight in terms_and_weights))
        if weight_sum <= 1e-9:
            return 0.0
        return float(weighted_sum / weight_sum)

    def build_neighbor_features(self, agent: Agent, world: World) -> np.ndarray:
        """Return nearest-neighbor relative position used by the compact tracking obs."""
        rels: List[np.ndarray] = []
        for other in world.agents:
            if other is agent:
                continue
            rels.append(np.asarray(other.state.p_pos - agent.state.p_pos, dtype=np.float64))
        if not rels:
            return np.zeros(3, dtype=np.float64)
        dists = [float(np.linalg.norm(item)) for item in rels]
        return rels[int(np.argmin(dists))]

    def _tracking_measurements(self, agent: Agent, world: World) -> Dict[str, float]:
        cfg = self.reward_config
        agent_idx = self._agent_index(agent)
        if not world.targets:
            return {
                "distance": 0.0,
                "error": 0.0,
                "delta": 0.0,
                "distance_rate": 0.0,
                "error_ma": 0.0,
                "confidence": 0.0,
                "lost": 1.0,
                "lost_steps_norm": 1.0,
            }
        target = world.targets[0]
        rel = np.asarray(target.state.p_pos - agent.state.p_pos, dtype=np.float64)
        dist = float(np.linalg.norm(rel))
        prev_dist = float(self._prev_target_distance.get(agent_idx, dist))
        delta = prev_dist - dist
        los = rel / max(1e-9, dist)
        agent_vel = (
            np.asarray(agent.state.p_vel, dtype=np.float64)
            if agent.state.p_vel is not None
            else np.zeros(3, dtype=np.float64)
        )
        target_vel = np.asarray(target.state.p_vel, dtype=np.float64)
        distance_rate = float(np.dot(target_vel - agent_vel, los))
        sensor_range = max(1e-6, float(cfg.sensor_range))
        confidence = float(np.clip(1.0 - dist / sensor_range, 0.0, 1.0))
        lost = float(dist > float(cfg.lost_distance))
        prev_lost = int(self._lost_steps.get(agent_idx, 0))
        lost_steps = prev_lost + 1 if lost > 0.5 else 0
        error = abs(dist - float(cfg.desired_tracking_distance))
        history = self._tracking_error_history.get(agent_idx, [])
        error_ma = float(np.mean(np.asarray((history + [error])[-5:], dtype=np.float64)))
        return {
            "distance": dist,
            "error": error,
            "delta": delta,
            "distance_rate": distance_rate,
            "error_ma": error_ma,
            "confidence": confidence,
            "lost": lost,
            "lost_steps_norm": float(np.clip(lost_steps / 25.0, 0.0, 1.0)),
        }

    def build_tracking_features(self, agent: Agent, world: World) -> np.ndarray:
        cfg = self.reward_config
        m = self._tracking_measurements(agent, world)
        distance_clip = max(1e-6, float(cfg.distance_clip))
        error_clip = max(1e-6, float(cfg.tracking_error_clip))
        progress_clip = max(1e-6, float(cfg.progress_clip))
        return np.array(
            [
                np.clip(m["distance"] / distance_clip, 0.0, 2.0),
                np.clip(m["error"] / error_clip, 0.0, 2.0),
                np.clip(m["delta"] / progress_clip, -1.0, 1.0),
                np.clip(-m["distance_rate"] / max(1e-6, float(cfg.closing_speed_clip)), -1.0, 1.0),
                np.clip(m["error_ma"] / error_clip, 0.0, 2.0),
                m["confidence"],
                m["lost"],
                m["lost_steps_norm"],
            ],
            dtype=np.float64,
        )

    def build_semantic_features(self, agent: Agent, world: World) -> np.ndarray:
        m = self._tracking_measurements(agent, world)
        cfg = self.reward_config
        phase = np.zeros(4, dtype=np.float64)
        if m["lost"] > 0.5:
            phase[3] = 1.0
        elif m["distance"] <= float(cfg.near_distance):
            phase[2] = 1.0
        elif m["delta"] > 0.0:
            phase[1] = 1.0
        else:
            phase[0] = 1.0
        quality = np.zeros(4, dtype=np.float64)
        q_idx = 0 if m["confidence"] >= 0.75 else 1 if m["confidence"] >= 0.40 else 2 if m["confidence"] > 0.0 else 3
        quality[q_idx] = 1.0
        target_pos = np.asarray(world.targets[0].state.p_pos, dtype=np.float64) if world.targets else np.zeros(3)
        all_dist = [float(np.linalg.norm(target_pos - np.asarray(item.state.p_pos, dtype=np.float64))) for item in world.agents]
        own_rank_best = float(self._agent_index(agent) == int(np.argmin(all_dist))) if all_dist else 0.0
        return np.concatenate([phase, quality, np.array([own_rank_best], dtype=np.float64)], axis=0)

    def build_semantic_graph_features(self, agent: Agent, world: World) -> np.ndarray:
        """Rule-based local task-graph summary for semantic MARL variants.

        The features are deliberately compact and deterministic: they encode
        neighbor availability, link quality, neighbor tracking quality, and
        whether this AUV should behave like a primary tracking node. They are
        not encirclement features and do not use angular coverage objectives.
        """
        cfg = self.reward_config
        own = self._tracking_measurements(agent, world)
        target_pos = np.asarray(world.targets[0].state.p_pos, dtype=np.float64) if world.targets else np.zeros(3)
        link_qualities: List[float] = []
        neighbor_errors: List[float] = []
        for other in world.agents:
            if other is agent:
                continue
            rel_dist = float(np.linalg.norm(np.asarray(other.state.p_pos - agent.state.p_pos, dtype=np.float64)))
            link_qualities.append(float(np.clip(1.0 - rel_dist / max(1e-6, float(cfg.sensor_range)), 0.0, 1.0)))
            other_dist = float(np.linalg.norm(target_pos - np.asarray(other.state.p_pos, dtype=np.float64)))
            neighbor_errors.append(abs(other_dist - float(cfg.desired_tracking_distance)))

        if link_qualities:
            mean_link = float(np.mean(link_qualities))
            best_link = float(np.max(link_qualities))
            active_neighbors = float(np.mean([q > 0.1 for q in link_qualities]))
        else:
            mean_link = 0.0
            best_link = 0.0
            active_neighbors = 0.0
        neighbor_error_mean = float(np.mean(neighbor_errors)) if neighbor_errors else float(cfg.tracking_error_clip)
        neighbor_error_norm = float(np.clip(neighbor_error_mean / max(1e-6, float(cfg.tracking_error_clip)), 0.0, 1.0))
        own_error_norm = float(np.clip(float(own["error"]) / max(1e-6, float(cfg.tracking_error_clip)), 0.0, 1.0))
        local_advantage = float(np.clip(neighbor_error_norm - own_error_norm, -1.0, 1.0))
        primary_role = float(local_advantage > 0.0 and own["confidence"] >= 0.4)
        return np.array(
            [
                active_neighbors,
                mean_link,
                best_link,
                neighbor_error_norm,
                local_advantage,
                primary_role,
            ],
            dtype=np.float64,
        )

    def observation(self, agent: Agent, world: World) -> np.ndarray:
        self_pos = np.asarray(agent.state.p_pos, dtype=np.float64)
        self_pos_raw = self_pos.copy()
        params = agent.auv6_params
        rotation_t = rotation_zyx(agent.state.phi, agent.state.theta, agent.state.psi).T
        vel_scale = np.maximum(
            np.array([params.umax, params.vmax, params.wmax], dtype=np.float64),
            1e-6,
        )
        omega_scale = np.maximum(
            np.array([params.pmax, params.qmax, params.rmax], dtype=np.float64),
            1e-6,
        )
        pos_scale = max(1e-6, float(self.boundary_limit))
        if self.obs_config.use_attitude_sin_cos:
            self_att = np.array(
                [
                    np.sin(agent.state.phi),
                    np.cos(agent.state.phi),
                    np.sin(agent.state.theta),
                    np.cos(agent.state.theta),
                    np.sin(agent.state.psi),
                    np.cos(agent.state.psi),
                ],
                dtype=np.float64,
            )
        else:
            self_att = np.array([agent.state.phi, agent.state.theta, agent.state.psi], dtype=np.float64)
        self_vel_body = np.array([agent.state.u, agent.state.v, agent.state.w], dtype=np.float64)
        self_omega = np.array([agent.state.p, agent.state.q, agent.state.p_w], dtype=np.float64)

        target_rel = np.zeros(3, dtype=np.float64)
        target_rel_body = np.zeros(3, dtype=np.float64)
        los_unit_body = np.zeros(3, dtype=np.float64)
        target_vel = np.zeros(3, dtype=np.float64)
        relative_velocity_world = np.zeros(3, dtype=np.float64)
        relative_velocity_body = np.zeros(3, dtype=np.float64)
        self_vel_world = (
            np.asarray(agent.state.p_vel, dtype=np.float64)
            if agent.state.p_vel is not None
            else np.zeros(3, dtype=np.float64)
        )
        if world.targets:
            target_rel = np.asarray(world.targets[0].state.p_pos - agent.state.p_pos, dtype=np.float64)
            target_rel_body = rotation_t @ target_rel
            target_dist = float(np.linalg.norm(target_rel_body))
            if target_dist > 1e-9:
                los_unit_body = target_rel_body / target_dist
            target_vel = np.asarray(world.targets[0].state.p_vel, dtype=np.float64)
            relative_velocity_world = target_vel - self_vel_world
            relative_velocity_body = rotation_t @ relative_velocity_world

        other_rel: List[np.ndarray] = []
        for other in world.agents:
            if other is agent:
                continue
            other_rel.append(np.asarray(other.state.p_pos - agent.state.p_pos, dtype=np.float64))

        if self.obs_config.normalize_physical:
            self_pos = np.clip(self_pos / pos_scale, -2.0, 2.0)
            self_vel_body = np.clip(self_vel_body / vel_scale, -2.0, 2.0)
            self_omega = np.clip(self_omega / omega_scale, -2.0, 2.0)
            target_rel = np.clip(target_rel / pos_scale, -2.0, 2.0)
            target_rel_body = np.clip(target_rel_body / pos_scale, -2.0, 2.0)
            other_rel = [np.clip(item / pos_scale, -2.0, 2.0) for item in other_rel]
            target_vel = np.clip(target_vel / vel_scale, -2.0, 2.0)
            relative_velocity_world = np.clip(relative_velocity_world / vel_scale, -2.0, 2.0)
            relative_velocity_body = np.clip(relative_velocity_body / vel_scale, -2.0, 2.0)

        features: List[np.ndarray] = [self_pos, self_att, self_vel_body, self_omega, target_rel, *other_rel]
        if self.obs_config.include_boundary_margin:
            boundary_margin = (float(self.boundary_limit) - np.abs(self_pos_raw)) / max(1e-6, float(self.boundary_limit))
            features.append(np.clip(boundary_margin, -1.0, 1.0))
        if self.obs_config.include_target_velocity:
            features.append(target_vel)
        if self.obs_config.include_relative_velocity:
            features.append(relative_velocity_world)
        if self.obs_config.include_target_rel_body:
            features.append(target_rel_body)
        if self.obs_config.include_relative_velocity_body:
            features.append(relative_velocity_body)
        if self.obs_config.include_los_unit_body:
            features.append(los_unit_body)
        if self.obs_config.include_prev_action:
            agent_idx = self._agent_index(agent)
            features.append(self._prev_action_norm.get(agent_idx, np.zeros(6, dtype=np.float64)))
        if self.obs_config.include_tracking_diagnostics:
            features.append(self.build_tracking_features(agent, world))
        if self.obs_config.include_semantic_features:
            features.append(self.build_semantic_features(agent, world))
        if self.obs_config.include_semantic_graph_features:
            features.append(self.build_semantic_graph_features(agent, world))
        return np.concatenate(features, axis=0)

    def _legacy_reward(self, agent: Agent, world: World) -> float:
        cfg = self.reward_config
        rel_target = np.asarray(world.targets[0].state.p_pos - agent.state.p_pos, dtype=np.float64)
        dist = float(np.linalg.norm(rel_target))

        min_auv_dist = float("inf")
        for other in world.agents:
            if other is agent:
                continue
            d = float(np.linalg.norm(other.state.p_pos - agent.state.p_pos))
            min_auv_dist = min(min_auv_dist, d)

        if dist > cfg.d_target_min:
            pos_rew = -dist
        else:
            pos_rew = -cfg.near_target_scale * dist

        col_rew = 0.0
        if min_auv_dist < cfg.d_auv_min:
            col_rew = (min_auv_dist - cfg.d_auv_min) * cfg.collision_scale

        return float(cfg.pos_weight * pos_rew + cfg.col_weight * col_rew)

    def _tracking_v3_reward(self, agent: Agent, world: World) -> float:
        cfg = self.reward_config
        agent_idx = self._agent_index(agent)
        m = self._tracking_measurements(agent, world)
        dist = float(m["distance"])
        error = float(m["error"])
        progress = float(np.clip(m["delta"] / max(1e-6, float(cfg.progress_clip)), -1.0, 1.0))
        distance_score = 1.0 - np.clip(error / max(1e-6, float(cfg.tracking_error_clip)), 0.0, 1.0)
        closing_score = float(
            np.clip((-float(m["distance_rate"])) / max(1e-6, float(cfg.closing_speed_clip)), -1.0, 1.0)
        )
        in_band = float(error <= float(cfg.desired_tracking_distance))
        tracking_reward = float(np.clip(0.45 * distance_score + 0.40 * progress + 0.15 * closing_score, -1.0, 1.0))

        observation_confidence = float(m["confidence"])
        target_lost = float(m["lost"])
        observation_reward = float(np.clip(observation_confidence - target_lost, -1.0, 1.0))

        target_pos = np.asarray(world.targets[0].state.p_pos, dtype=np.float64)
        all_dists = np.asarray(
            [float(np.linalg.norm(target_pos - np.asarray(item.state.p_pos, dtype=np.float64))) for item in world.agents],
            dtype=np.float64,
        )
        neighbor_mean_error = float(np.mean(np.abs(all_dists - float(cfg.desired_tracking_distance))))
        coordination_reward = float(
            np.clip(1.0 - neighbor_mean_error / max(1e-6, float(cfg.tracking_error_clip)), 0.0, 1.0)
        )
        communication_quality = float(np.clip(1.0 - neighbor_mean_error / max(1e-6, float(cfg.sensor_range)), 0.0, 1.0))

        if target_lost > 0.5:
            semantic_reward = progress
        elif dist <= float(cfg.near_distance):
            semantic_reward = 0.5 * in_band + 0.5 * observation_confidence
        else:
            semantic_reward = 0.7 * progress + 0.3 * closing_score
        semantic_reward = float(np.clip(semantic_reward, -1.0, 1.0))

        action_norm = self._normalized_action(agent)
        prev_action_norm = self._prev_action_norm.get(agent_idx, np.zeros_like(action_norm))
        action_delta = action_norm - prev_action_norm
        control_cost = float(np.clip(0.65 * np.mean(np.square(action_norm)) + 0.35 * np.mean(np.square(action_delta)), 0.0, 1.0))

        total_reward = (
            float(cfg.w_tracking_reward) * tracking_reward
            + float(cfg.w_observation_reward) * observation_reward
            + float(cfg.w_coordination_reward) * coordination_reward
            + float(cfg.w_communication_reward) * communication_quality
            + float(cfg.w_semantic_reward) * semantic_reward
            - float(cfg.w_control_cost) * control_cost
        )
        total_reward = float(np.clip(total_reward, -2.0, 2.0))

        self._prev_target_distance[agent_idx] = dist
        self._prev_action_norm[agent_idx] = action_norm
        history = self._tracking_error_history.setdefault(agent_idx, [])
        history.append(error)
        if len(history) > 16:
            del history[:-16]
        self._lost_steps[agent_idx] = int(round(m["lost_steps_norm"] * 25.0)) if target_lost > 0.5 else 0

        self._last_reward_terms[agent_idx] = {
            "centroid_distance_term": 0.0,
            "centroid_progress_term": 0.0,
            "centroid_near_term": 0.0,
            "centroid_success_term": 0.0,
            "distance_term": float(distance_score),
            "progress_term": float(progress),
            "closing_speed_term": float(closing_score),
            "near_term": float(in_band),
            "success_term": float(in_band and observation_confidence > 0.5),
            "attitude_stability_term": 0.0,
            "angular_rate_term": 0.0,
            "safety_term": 0.0,
            "action_reg_term": -float(control_cost),
            "tracking_group_term": float(tracking_reward),
            "safety_group_term": 0.0,
            "action_group_term": -float(control_cost),
            "tracking_contrib": float(cfg.w_tracking_reward) * tracking_reward,
            "safety_contrib": 0.0,
            "action_contrib": -float(cfg.w_control_cost) * control_cost,
            "tracking_reward": float(tracking_reward),
            "observation_reward": float(observation_reward),
            "coordination_reward": float(coordination_reward),
            "communication_reward": float(communication_quality),
            "semantic_reward": float(semantic_reward),
            "control_cost": float(control_cost),
            "total_reward": float(total_reward),
            "tracking_error": float(error),
            "tracking_error_delta": float(m["delta"]),
            "target_distance": float(dist),
            "observation_confidence": float(observation_confidence),
            "target_lost": float(target_lost),
            "communication_quality": float(communication_quality),
            "action_norm": float(np.linalg.norm(action_norm)),
            "action_delta_norm": float(np.linalg.norm(action_delta)),
        }
        return total_reward

    def _tracking_band_semantic_reward(self, agent: Agent, world: World) -> float:
        cfg = self.reward_config
        agent_idx = self._agent_index(agent)
        m = self._tracking_measurements(agent, world)
        dist = float(m["distance"])
        desired = float(cfg.desired_tracking_distance)
        lower = min(float(cfg.tracking_band_lower), desired)
        upper = max(float(cfg.tracking_band_upper), desired)
        if upper <= lower:
            lower, upper = desired * 0.7, desired
        band_error = max(0.0, lower - dist, dist - upper)
        tracking_error = abs(dist - desired)
        progress = float(np.clip(m["delta"] / max(1e-6, float(cfg.progress_clip)), -1.0, 1.0))
        closing_score = float(
            np.clip((-float(m["distance_rate"])) / max(1e-6, float(cfg.closing_speed_clip)), -1.0, 1.0)
        )
        band_score = float(np.clip(1.0 - band_error / max(1e-6, float(cfg.tracking_error_clip)), 0.0, 1.0))
        in_band = float(band_error <= 1e-9)
        too_close_penalty = float(np.clip((lower - dist) / max(1e-6, lower), 0.0, 1.0))

        observation_confidence = float(m["confidence"])
        target_lost = float(m["lost"])
        observation_reward = float(np.clip(observation_confidence - target_lost, -1.0, 1.0))

        target_pos = np.asarray(world.targets[0].state.p_pos, dtype=np.float64)
        all_dists = np.asarray(
            [float(np.linalg.norm(target_pos - np.asarray(item.state.p_pos, dtype=np.float64))) for item in world.agents],
            dtype=np.float64,
        )
        all_band_errors = np.asarray([max(0.0, lower - d, d - upper) for d in all_dists], dtype=np.float64)
        coordination_reward = float(
            np.clip(1.0 - float(np.mean(all_band_errors)) / max(1e-6, float(cfg.tracking_error_clip)), 0.0, 1.0)
        )
        communication_quality = float(
            np.clip(1.0 - float(np.mean(all_band_errors)) / max(1e-6, float(cfg.sensor_range)), 0.0, 1.0)
        )

        if target_lost > 0.5:
            semantic_reward = progress
            reacquire_reward = progress
        elif in_band > 0.5:
            semantic_reward = 0.55 * observation_confidence + 0.45 * (1.0 - too_close_penalty)
            reacquire_reward = 0.0
        else:
            semantic_reward = 0.65 * progress + 0.35 * closing_score - 0.25 * too_close_penalty
            reacquire_reward = 0.5 * max(0.0, progress)
        semantic_reward = float(np.clip(semantic_reward, -1.0, 1.0))
        tracking_reward = float(
            np.clip(
                float(cfg.w_band_stability) * band_score
                + 0.30 * progress
                + 0.15 * closing_score
                + float(cfg.w_reacquire) * reacquire_reward
                - 0.20 * too_close_penalty,
                -1.0,
                1.0,
            )
        )

        action_norm = self._normalized_action(agent)
        prev_action_norm = self._prev_action_norm.get(agent_idx, np.zeros_like(action_norm))
        action_delta = action_norm - prev_action_norm
        control_cost = float(np.clip(0.60 * np.mean(np.square(action_norm)) + 0.40 * np.mean(np.square(action_delta)), 0.0, 1.0))

        total_reward = (
            float(cfg.w_tracking_reward) * tracking_reward
            + float(cfg.w_observation_reward) * observation_reward
            + float(cfg.w_coordination_reward) * coordination_reward
            + float(cfg.w_communication_reward) * communication_quality
            + float(cfg.w_semantic_reward) * semantic_reward
            - float(cfg.w_control_cost) * control_cost
        )
        total_reward = float(np.clip(total_reward, -2.0, 2.0))

        self._prev_target_distance[agent_idx] = dist
        self._prev_action_norm[agent_idx] = action_norm
        history = self._tracking_error_history.setdefault(agent_idx, [])
        history.append(tracking_error)
        if len(history) > 16:
            del history[:-16]
        self._lost_steps[agent_idx] = int(round(m["lost_steps_norm"] * 25.0)) if target_lost > 0.5 else 0

        self._last_reward_terms[agent_idx] = {
            "centroid_distance_term": 0.0,
            "centroid_progress_term": 0.0,
            "centroid_near_term": 0.0,
            "centroid_success_term": 0.0,
            "distance_term": float(band_score),
            "progress_term": float(progress),
            "closing_speed_term": float(closing_score),
            "near_term": float(in_band),
            "success_term": float(in_band and observation_confidence > 0.5),
            "attitude_stability_term": 0.0,
            "angular_rate_term": 0.0,
            "safety_term": 0.0,
            "action_reg_term": -float(control_cost),
            "tracking_group_term": float(tracking_reward),
            "safety_group_term": 0.0,
            "action_group_term": -float(control_cost),
            "tracking_contrib": float(cfg.w_tracking_reward) * tracking_reward,
            "safety_contrib": 0.0,
            "action_contrib": -float(cfg.w_control_cost) * control_cost,
            "tracking_reward": float(tracking_reward),
            "observation_reward": float(observation_reward),
            "coordination_reward": float(coordination_reward),
            "communication_reward": float(communication_quality),
            "semantic_reward": float(semantic_reward),
            "control_cost": float(control_cost),
            "total_reward": float(total_reward),
            "tracking_error": float(tracking_error),
            "tracking_error_delta": float(m["delta"]),
            "target_distance": float(dist),
            "observation_confidence": float(observation_confidence),
            "target_lost": float(target_lost),
            "communication_quality": float(communication_quality),
            "action_norm": float(np.linalg.norm(action_norm)),
            "action_delta_norm": float(np.linalg.norm(action_delta)),
            "band_error": float(band_error),
            "band_score": float(band_score),
            "too_close_penalty": float(too_close_penalty),
        }
        return total_reward

    def reward(self, agent: Agent, world: World) -> float:
        if not world.targets:
            return 0.0

        cfg = self.reward_config
        if str(cfg.version).strip().lower() in {"semantic_tracking_band", "tracking_band_semantic", "stg_tracking"}:
            return self._tracking_band_semantic_reward(agent, world)
        if str(cfg.version).strip().lower() in {"v3_tracking", "tracking_v3", "convergence_v3"}:
            return self._tracking_v3_reward(agent, world)
        if str(cfg.version).strip().lower() in {"v1", "v1_legacy", "legacy"}:
            value = self._legacy_reward(agent, world)
            agent_idx = self._agent_index(agent)
            self._last_reward_terms[agent_idx] = {
                "centroid_distance_term": 0.0,
                "centroid_progress_term": 0.0,
                "centroid_near_term": 0.0,
                "centroid_success_term": 0.0,
                "distance_term": float(value),
                "progress_term": 0.0,
                "closing_speed_term": 0.0,
                "near_term": 0.0,
                "success_term": 0.0,
                "separation_term": 0.0,
                "boundary_term": 0.0,
                "collision_term": 0.0,
                "oob_term": 0.0,
                "unstable_term": 0.0,
                "attitude_stability_term": 0.0,
                "angular_rate_term": 0.0,
                "action_energy_term": 0.0,
                "action_smooth_term": 0.0,
                "tracking_group_term": 0.0,
                "safety_group_term": 0.0,
                "action_group_term": 0.0,
                "tracking_contrib": 0.0,
                "safety_contrib": 0.0,
                "action_contrib": 0.0,
                "action_reg_term": 0.0,
                "safety_term": 0.0,
                "total_reward": float(value),
            }
            return float(value)

        agent_idx = self._agent_index(agent)
        target = world.targets[0]
        self._ensure_centroid_step_terms(world, cfg)
        centroid_dist = float(self._centroid_step_terms["centroid_dist"])
        centroid_distance_term = float(self._centroid_step_terms["centroid_distance_term"])
        centroid_progress_term = float(self._centroid_step_terms["centroid_progress_term"])
        centroid_near_term = float(self._centroid_step_terms["centroid_near_term"])
        centroid_success_term = float(self._centroid_step_terms["centroid_success_term"])
        progress_clip = max(1e-6, float(cfg.progress_clip))

        rel_target = np.asarray(target.state.p_pos - agent.state.p_pos, dtype=np.float64)
        dist = float(np.linalg.norm(rel_target))
        clipped_dist = min(dist, float(cfg.distance_clip))
        distance_term = -clipped_dist / max(1e-6, float(cfg.distance_clip))

        prev_dist = float(self._prev_target_distance.get(agent_idx, dist))
        progress_raw = prev_dist - dist
        # Symmetric progress signal: reward approaching, penalize retreating equally.
        progress_term = float(np.clip(progress_raw / progress_clip, -1.0, 1.0))
        self._prev_target_distance[agent_idx] = dist

        los_world = rel_target / max(1e-9, dist)
        agent_vel_world = (
            np.asarray(agent.state.p_vel, dtype=np.float64)
            if agent.state.p_vel is not None
            else np.zeros(3, dtype=np.float64)
        )
        target_vel_world = np.asarray(target.state.p_vel, dtype=np.float64)
        dist_rate = float(np.dot(target_vel_world - agent_vel_world, los_world))
        closing_speed_term = float(
            np.clip((-dist_rate) / max(1e-6, float(cfg.closing_speed_clip)), -1.0, 1.0)
        )

        near_term = float(max(0.0, (float(cfg.near_distance) - dist) / max(1e-6, float(cfg.near_distance))))
        success_term = float(
            max(0.0, (float(cfg.success_distance) - dist) / max(1e-6, float(cfg.success_distance)))
        )

        min_auv_dist = float("inf")
        for other in world.agents:
            if other is agent:
                continue
            min_auv_dist = min(min_auv_dist, float(np.linalg.norm(other.state.p_pos - agent.state.p_pos)))
        separation_term = -float(
            max(0.0, (float(cfg.safe_distance) - min_auv_dist) / max(1e-6, float(cfg.safe_distance)))
        )

        boundary_ratio = float(np.max(np.abs(np.asarray(agent.state.p_pos, dtype=np.float64)))) / max(
            1e-6, float(world.boundary_limit)
        )
        if boundary_ratio <= float(cfg.boundary_soft_ratio):
            boundary_term = 0.0
        else:
            boundary_over = np.clip(
                (boundary_ratio - float(cfg.boundary_soft_ratio))
                / max(1e-6, 1.0 - float(cfg.boundary_soft_ratio)),
                0.0,
                1.0,
            )
            boundary_term = -float(boundary_over * boundary_over)

        collision_term = -1.0 if int(world.last_metrics.get("collision_events", 0)) > 0 else 0.0
        oob_term = -1.0 if int(world.last_metrics.get("out_of_bounds_events", 0)) > 0 else 0.0

        unstable_term = -1.0 if int(world.last_metrics.get("instability_events", 0)) > 0 else 0.0
        roll_pitch_norm = np.linalg.norm([float(agent.state.phi), float(agent.state.theta)])
        attitude_stability_term = -float(
            np.clip(roll_pitch_norm / max(1e-6, float(cfg.attitude_angle_clip)), 0.0, 1.0)
        )
        angular_rate_norm = np.linalg.norm([float(agent.state.p), float(agent.state.q), float(agent.state.p_w)])
        angular_rate_term = -float(
            np.clip(angular_rate_norm / max(1e-6, float(cfg.angular_rate_clip)), 0.0, 1.0)
        )

        action_norm = self._normalized_action(agent)
        action_energy_term = -float(np.mean(np.square(action_norm)))
        prev_action_norm = self._prev_action_norm.get(agent_idx, np.zeros_like(action_norm))
        action_smooth_term = -float(np.mean(np.square(action_norm - prev_action_norm)))
        self._prev_action_norm[agent_idx] = action_norm

        safety_term = (
            separation_term
            + boundary_term
            + collision_term
            + oob_term
            + unstable_term
            + attitude_stability_term
            + angular_rate_term
        )
        action_reg_term = action_energy_term + action_smooth_term
        tracking_group_term = self._weighted_group_value(
            [
                (distance_term, float(cfg.w_distance)),
                (progress_term, float(cfg.w_progress)),
                (closing_speed_term, float(cfg.w_closing_speed)),
                (near_term, float(cfg.w_near)),
                (success_term, float(cfg.w_success)),
            ]
        )
        safety_group_term = self._weighted_group_value(
            [
                (separation_term, float(cfg.w_separation)),
                (boundary_term, float(cfg.w_boundary)),
                (collision_term, float(cfg.w_collision)),
                (oob_term, float(cfg.w_oob)),
                (unstable_term, float(cfg.w_unstable)),
                (attitude_stability_term, float(cfg.w_attitude_stability)),
                (angular_rate_term, float(cfg.w_angular_rate)),
            ]
        )
        action_group_term = self._weighted_group_value(
            [
                (action_energy_term, float(cfg.w_action_energy)),
                (action_smooth_term, float(cfg.w_action_smooth)),
            ]
        )
        tracking_contrib = float(cfg.w_tracking_group) * tracking_group_term
        safety_contrib = float(cfg.w_safety_group) * safety_group_term
        action_contrib = float(cfg.w_action_group) * action_group_term
        total_reward = tracking_contrib + safety_contrib + action_contrib
        # Clamp reward to avoid extreme outliers that destabilize value networks.
        total_reward = float(np.clip(total_reward, -5.0, 5.0))

        self._last_reward_terms[agent_idx] = {
            "centroid_distance_term": float(centroid_distance_term),
            "centroid_progress_term": float(centroid_progress_term),
            "centroid_near_term": float(centroid_near_term),
            "centroid_success_term": float(centroid_success_term),
            "distance_term": float(distance_term),
            "progress_term": float(progress_term),
            "closing_speed_term": float(closing_speed_term),
            "near_term": float(near_term),
            "success_term": float(success_term),
            "separation_term": float(separation_term),
            "boundary_term": float(boundary_term),
            "collision_term": float(collision_term),
            "oob_term": float(oob_term),
            "unstable_term": float(unstable_term),
            "attitude_stability_term": float(attitude_stability_term),
            "angular_rate_term": float(angular_rate_term),
            "action_energy_term": float(action_energy_term),
            "action_smooth_term": float(action_smooth_term),
            "tracking_group_term": float(tracking_group_term),
            "safety_group_term": float(safety_group_term),
            "action_group_term": float(action_group_term),
            "tracking_contrib": float(tracking_contrib),
            "safety_contrib": float(safety_contrib),
            "action_contrib": float(action_contrib),
            "action_reg_term": float(action_reg_term),
            "safety_term": float(safety_term),
            "total_reward": float(total_reward),
        }
        self._centroid_step_agent_count += 1
        if self._centroid_step_agent_count >= self.n_agent:
            self._prev_centroid_distance = centroid_dist
        return float(total_reward)

    def get_last_reward_terms_mean(self) -> Dict[str, float]:
        if not self._last_reward_terms:
            return {
                "distance_term": 0.0,
                "progress_term": 0.0,
                "closing_speed_term": 0.0,
                "near_term": 0.0,
                "success_term": 0.0,
                "attitude_stability_term": 0.0,
                "angular_rate_term": 0.0,
                "safety_term": 0.0,
                "action_reg_term": 0.0,
                "tracking_group_term": 0.0,
                "safety_group_term": 0.0,
                "action_group_term": 0.0,
                "tracking_contrib": 0.0,
                "safety_contrib": 0.0,
                "action_contrib": 0.0,
                "tracking_reward": 0.0,
                "observation_reward": 0.0,
                "coordination_reward": 0.0,
                "communication_reward": 0.0,
                "semantic_reward": 0.0,
                "control_cost": 0.0,
                "total_reward": 0.0,
                "tracking_error": 0.0,
                "tracking_error_delta": 0.0,
                "target_distance": 0.0,
                "observation_confidence": 0.0,
                "target_lost": 0.0,
                "communication_quality": 0.0,
                "action_norm": 0.0,
                "action_delta_norm": 0.0,
            }
        keys = [
            "centroid_distance_term",
            "centroid_progress_term",
            "centroid_near_term",
            "centroid_success_term",
            "distance_term",
            "progress_term",
            "closing_speed_term",
            "near_term",
            "success_term",
            "attitude_stability_term",
            "angular_rate_term",
            "safety_term",
            "action_reg_term",
            "tracking_group_term",
            "safety_group_term",
            "action_group_term",
            "tracking_contrib",
            "safety_contrib",
            "action_contrib",
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
        ]
        out: Dict[str, float] = {}
        for key in keys:
            vals = [float(item.get(key, 0.0)) for item in self._last_reward_terms.values()]
            out[key] = float(np.mean(np.asarray(vals, dtype=np.float64)))
        return out

from __future__ import annotations

"""
AUV 六自由度动力学核心模块。

职责：
- 定义 world/entity 数据结构；
- 计算 6DOF 刚体动力学更新；
- 统计碰撞、越界、失稳等约束事件。
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


def wrap_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi) 区间。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation_zyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """按 ZYX 欧拉角构造从体坐标到惯性坐标的旋转矩阵。"""
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    rz = np.array([[c_psi, -s_psi, 0.0], [s_psi, c_psi, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[c_theta, 0.0, s_theta], [0.0, 1.0, 0.0], [-s_theta, 0.0, c_theta]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c_phi, -s_phi], [0.0, s_phi, c_phi]])
    return rz @ ry @ rx


def euler_rate_map_zyx(phi: float, theta: float, eps: float = 1e-6) -> np.ndarray:
    """把角速度 [p,q,r] 映射成欧拉角速度 [phi_dot,theta_dot,psi_dot]。"""
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    if abs(c_theta) < eps:
        c_theta = float(np.copysign(eps, c_theta if c_theta != 0.0 else 1.0))
    t_theta = s_theta / c_theta
    return np.array(
        [
            [1.0, s_phi * t_theta, c_phi * t_theta],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta],
        ],
        dtype=np.float64,
    )


@dataclass
class AUV6Params:
    """单个 AUV 的惯性、阻尼、加速度限制与力矩上限参数。"""
    m: float = 10.0
    ix: float = 30.0
    iy: float = 35.0
    iz: float = 40.0

    xu_dot: float = -0.5
    yv_dot: float = -0.5
    zw_dot: float = -0.5
    kp_dot: float = -1.0
    mq_dot: float = -2.0
    nr_dot: float = -5.0

    xu: float = 2.0
    yv: float = 1.5
    zw: float = 2.0
    kp: float = -1.0
    mq: float = -1.5
    nr: float = -2.0

    xuu: float = 1.0
    yvv: float = 0.8
    zww: float = 1.0
    kpp: float = -0.5
    mqq: float = -0.5
    nrr: float = -2.0

    umax: float = 0.5
    vmax: float = 0.3
    wmax: float = 0.5
    pmax: float = 0.8
    qmax: float = 0.8
    rmax: float = 0.8

    taumax_x: float = 2.0
    taumax_y: float = 1.5
    taumax_z: float = 2.0
    taumax_k: float = 3.0
    taumax_m: float = 3.0
    taumax_n: float = 3.0


_LEGACY_FAST_DEFAULTS: Dict[str, float] = asdict(AUV6Params())

# REMUS 100 reference profile (Fossen/MSS-style practical defaults)
# mapped into this simplified 6DOF integrator contract.
_REMUS100_MSS_DEFAULTS: Dict[str, float] = {
    "m": 31.9,
    "ix": 0.16,
    "iy": 4.1,
    "iz": 4.1,
    "xu_dot": -2.0,
    "yv_dot": -35.0,
    "zw_dot": -35.0,
    "kp_dot": -0.12,
    "mq_dot": -4.5,
    "nr_dot": -4.5,
    "xu": 4.0,
    "yv": 8.0,
    "zw": 10.0,
    "kp": -0.8,
    "mq": -3.0,
    "nr": -2.5,
    "xuu": 18.0,
    "yvv": 30.0,
    "zww": 35.0,
    "kpp": -0.25,
    "mqq": -1.2,
    "nrr": -1.0,
    "umax": 1.5,
    "vmax": 0.5,
    "wmax": 0.5,
    "pmax": 1.2,
    "qmax": 1.2,
    "rmax": 1.2,
    "taumax_x": 35.0,
    "taumax_y": 20.0,
    "taumax_z": 20.0,
    "taumax_k": 8.0,
    "taumax_m": 10.0,
    "taumax_n": 10.0,
}


def build_auv6_params(profile: str = "remus100_mss", overrides: Optional[Mapping[str, Any]] = None) -> AUV6Params:
    """
    Build AUV6Params by profile name plus optional field overrides.
    Supported profiles:
      - remus100_mss (default)
      - legacy_fast
    """
    key = str(profile).strip().lower()
    if key in {"remus100_mss", "remus100", "mss_remus100"}:
        base = dict(_REMUS100_MSS_DEFAULTS)
    elif key in {"legacy_fast", "legacy"}:
        base = dict(_LEGACY_FAST_DEFAULTS)
    else:
        raise ValueError(f"Unknown auv profile: {profile}")

    if overrides:
        valid_keys = set(_LEGACY_FAST_DEFAULTS.keys())
        unknown = [k for k in overrides.keys() if k not in valid_keys]
        if unknown:
            raise ValueError(f"Unknown auv param override keys: {unknown}")
        for k, v in overrides.items():
            base[k] = float(v)
    return AUV6Params(**base)


class UniformCurrent3D:
    """简化的均匀流场模型。"""
    def __init__(self, uc: float = 0.0, vc: float = 0.0, wc: float = 0.0):
        self.uc = float(uc)
        self.vc = float(vc)
        self.wc = float(wc)

    def __call__(self, x: float, y: float, z: float, t: float) -> Tuple[float, float, float]:
        del x, y, z, t
        return self.uc, self.vc, self.wc


class EntityState:
    def __init__(self) -> None:
        self.p_pos: Optional[np.ndarray] = None
        self.p_vel: Optional[np.ndarray] = None
        self.p_w: Optional[float] = None


class Entity:
    def __init__(self) -> None:
        self.name = ""
        self.size = 0.001
        self.movable = False
        self.collide = True
        self.color = None
        self.max_speed = 0.05
        self.state = EntityState()


class AgentState(EntityState):
    def __init__(self) -> None:
        super().__init__()
        self.p_com = False
        self.u = 0.0
        self.v = 0.0
        self.w = 0.0
        self.p = 0.0
        self.q = 0.0
        self.phi = 0.0
        self.theta = 0.0
        self.psi = 0.0
        self.p_w = 0.0


class Action:
    def __init__(self) -> None:
        self.u = np.zeros(6, dtype=np.float32)


class Agent(Entity):
    def __init__(self) -> None:
        super().__init__()
        self.movable = True
        self.silent = False
        self.u_range = 1.0
        self.state = AgentState()
        self.action = Action()
        self.em = None
        self.u_noise = None
        self.initial_mass = 1.0
        self.control_mode6 = "tau6"
        self.current3d = UniformCurrent3D()
        self.auv6_params = AUV6Params()
        self.goal = None

    @property
    def mass(self) -> float:
        return self.initial_mass


class TargetState(EntityState):
    def __init__(self) -> None:
        super().__init__()
        self.p_vel = np.zeros(3, dtype=np.float64)


class Target(Entity):
    def __init__(self) -> None:
        super().__init__()
        self.movable = True
        self.state = TargetState()


class Landmark(Entity):
    pass


class World:
    """仿真世界对象，聚合实体并推进一步物理更新。"""

    def __init__(self) -> None:
        self.agents: List[Agent] = []
        self.targets: List[Target] = []
        self.landmarks: List[Landmark] = []
        self.dim_p = 3
        self.dt = 0.1
        self.contact_force = 1e2
        self.contact_margin = 1e-3
        self.boundary_limit = 1.0
        self.time = 0.0
        self.last_metrics: Dict[str, int] = {
            "collision_events": 0,
            "out_of_bounds_events": 0,
            "instability_events": 0,
        }

    @property
    def entities(self) -> List[Entity]:
        return self.agents + self.landmarks + self.targets

    def step(self) -> Dict[str, int]:
        """推进一个仿真步长，并返回约束事件统计。"""
        ext_forces = [np.zeros(3, dtype=np.float64) for _ in self.agents]
        collisions = 0
        out_of_bounds = 0
        instability_events = 0

        for i, entity_a in enumerate(self.agents):
            for j in range(i + 1, len(self.agents)):
                entity_b = self.agents[j]
                f_a, f_b, collided = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    ext_forces[i] += f_a
                if f_b is not None:
                    ext_forces[j] += f_b
                if collided:
                    collisions += 1

        for i, agent in enumerate(self.agents):
            if agent.u_noise:
                ext_forces[i] += np.random.randn(3) * float(agent.u_noise)
            is_stable = self._integrate_agent_6dof(agent, ext_force_world=ext_forces[i])
            if not is_stable:
                instability_events += 1
            if self._apply_agent_boundary_response(agent):
                out_of_bounds += 1

        for target in self.targets:
            if target.state.p_pos is None:
                continue
            target.state.p_pos = target.state.p_pos + target.state.p_vel * self.dt
            for axis in range(3):
                if abs(target.state.p_pos[axis]) > self.boundary_limit:
                    target.state.p_pos[axis] = np.clip(
                        target.state.p_pos[axis], -self.boundary_limit, self.boundary_limit
                    )
                    target.state.p_vel[axis] *= -1.0

        self.time += self.dt
        self.last_metrics = {
            "collision_events": collisions,
            "out_of_bounds_events": out_of_bounds,
            "instability_events": instability_events,
        }
        return self.last_metrics

    def _apply_agent_boundary_response(
        self,
        agent: Agent,
        restitution: float = 0.2,
        tangential_damping: float = 0.85,
        body_velocity_damping: float = 0.7,
        angular_damping: float = 0.8,
    ) -> bool:
        """Keep agent in bounds with clamp + reflection + damping."""
        if agent.state.p_pos is None:
            return False

        pos = np.asarray(agent.state.p_pos, dtype=np.float64).copy()
        vel = (
            np.asarray(agent.state.p_vel, dtype=np.float64).copy()
            if agent.state.p_vel is not None
            else np.zeros(3, dtype=np.float64)
        )
        hit = False

        for axis in range(3):
            axis_hit = False
            if pos[axis] > self.boundary_limit:
                pos[axis] = self.boundary_limit
                if vel[axis] > 0.0:
                    vel[axis] = -abs(vel[axis]) * float(restitution)
                axis_hit = True
            elif pos[axis] < -self.boundary_limit:
                pos[axis] = -self.boundary_limit
                if vel[axis] < 0.0:
                    vel[axis] = abs(vel[axis]) * float(restitution)
                axis_hit = True

            if axis_hit:
                hit = True
                for other_axis in range(3):
                    if other_axis != axis:
                        vel[other_axis] *= float(tangential_damping)

        if not hit:
            return False

        agent.state.p_pos = pos
        agent.state.p_vel = vel
        params = agent.auv6_params
        agent.state.u = float(np.clip(agent.state.u * body_velocity_damping, -params.umax, params.umax))
        agent.state.v = float(np.clip(agent.state.v * body_velocity_damping, -params.vmax, params.vmax))
        agent.state.w = float(np.clip(agent.state.w * body_velocity_damping, -params.wmax, params.wmax))
        agent.state.p = float(np.clip(agent.state.p * angular_damping, -params.pmax, params.pmax))
        agent.state.q = float(np.clip(agent.state.q * angular_damping, -params.qmax, params.qmax))
        agent.state.p_w = float(np.clip(agent.state.p_w * angular_damping, -params.rmax, params.rmax))
        return True

    def _integrate_agent_6dof(self, agent: Agent, ext_force_world: Optional[np.ndarray] = None) -> bool:
        """对单个 agent 执行 6DOF 动力学积分；若数值失稳则返回 False。"""
        params = agent.auv6_params
        state = agent.state
        dt = float(self.dt)
        eps = 1e-9

        if state.p_pos is None:
            state.p_pos = np.zeros(3, dtype=np.float64)
        if state.p_vel is None:
            state.p_vel = np.zeros(3, dtype=np.float64)

        x, y, z = state.p_pos.astype(np.float64)
        phi, theta, psi = float(state.phi), float(state.theta), float(state.psi)
        u, v, w = float(state.u), float(state.v), float(state.w)
        p_rate, q_rate, r_rate = float(state.p), float(state.q), float(state.p_w)

        rotation = rotation_zyx(phi, theta, psi)
        rotation_t = rotation.T

        uc_n, vc_n, wc_n = agent.current3d(x, y, z, self.time)
        uc_b, vc_b, wc_b = (rotation_t @ np.array([uc_n, vc_n, wc_n], dtype=np.float64)).tolist()
        ur, vr, wr = u - uc_b, v - vc_b, w - wc_b

        # 控制量采用体坐标系下的 6 维力/力矩输入 [Fx,Fy,Fz,K,M,N]。
        action_tau = np.asarray(agent.action.u, dtype=np.float64).reshape(-1)
        if action_tau.size < 6:
            action_tau = np.pad(action_tau, (0, 6 - action_tau.size), constant_values=0.0)
        x_tau, y_tau, z_tau, k_tau, m_tau, n_tau = action_tau[:6].tolist()

        if ext_force_world is not None:
            ext_force_world = np.asarray(ext_force_world, dtype=np.float64).reshape(-1)
            if ext_force_world.size >= 3:
                ext_force_body = rotation_t @ ext_force_world[:3]
                x_tau += ext_force_body[0]
                y_tau += ext_force_body[1]
                z_tau += ext_force_body[2]

        # 施加执行器饱和约束，避免动作越界导致积分爆炸。
        x_tau = float(np.clip(x_tau, -params.taumax_x, params.taumax_x))
        y_tau = float(np.clip(y_tau, -params.taumax_y, params.taumax_y))
        z_tau = float(np.clip(z_tau, -params.taumax_z, params.taumax_z))
        k_tau = float(np.clip(k_tau, -params.taumax_k, params.taumax_k))
        m_tau = float(np.clip(m_tau, -params.taumax_m, params.taumax_m))
        n_tau = float(np.clip(n_tau, -params.taumax_n, params.taumax_n))

        m_x = max(eps, float(params.m - params.xu_dot))
        m_y = max(eps, float(params.m - params.yv_dot))
        m_z = max(eps, float(params.m - params.zw_dot))
        i_x = max(eps, float(params.ix - params.kp_dot))
        i_y = max(eps, float(params.iy - params.mq_dot))
        i_z = max(eps, float(params.iz - params.nr_dot))

        c_x = params.m * (r_rate * v - q_rate * w)
        c_y = params.m * (p_rate * w - r_rate * u)
        c_z = params.m * (q_rate * u - p_rate * v)
        k_x = (params.iy - params.iz) * q_rate * r_rate
        k_y = (params.iz - params.ix) * p_rate * r_rate
        k_z = (params.ix - params.iy) * p_rate * q_rate

        d_x = abs(params.xu) * ur + abs(params.xuu) * abs(ur) * ur
        d_y = abs(params.yv) * vr + abs(params.yvv) * abs(vr) * vr
        d_z = abs(params.zw) * wr + abs(params.zww) * abs(wr) * wr
        d_p = abs(params.kp) * p_rate + abs(params.kpp) * abs(p_rate) * p_rate
        d_q = abs(params.mq) * q_rate + abs(params.mqq) * abs(q_rate) * q_rate
        d_r = abs(params.nr) * r_rate + abs(params.nrr) * abs(r_rate) * r_rate

        u_dot = (x_tau - d_x - c_x) / m_x
        v_dot = (y_tau - d_y - c_y) / m_y
        w_dot = (z_tau - d_z - c_z) / m_z
        p_dot = (k_tau - d_p - k_x) / i_x
        q_dot = (m_tau - d_q - k_y) / i_y
        r_dot = (n_tau - d_r - k_z) / i_z

        u = np.clip(u + u_dot * dt, -params.umax, params.umax)
        v = np.clip(v + v_dot * dt, -params.vmax, params.vmax)
        w = np.clip(w + w_dot * dt, -params.wmax, params.wmax)
        p_rate = np.clip(p_rate + p_dot * dt, -params.pmax, params.pmax)
        q_rate = np.clip(q_rate + q_dot * dt, -params.qmax, params.qmax)
        r_rate = np.clip(r_rate + r_dot * dt, -params.rmax, params.rmax)

        euler_rate_map = euler_rate_map_zyx(phi, theta)
        phi_dot, theta_dot, psi_dot = (euler_rate_map @ np.array([p_rate, q_rate, r_rate], dtype=np.float64)).tolist()
        linear_velocity_world = rotation @ np.array([u, v, w], dtype=np.float64)

        x += linear_velocity_world[0] * dt
        y += linear_velocity_world[1] * dt
        z += linear_velocity_world[2] * dt
        phi = wrap_angle(phi + phi_dot * dt)
        theta = wrap_angle(theta + theta_dot * dt)
        psi = wrap_angle(psi + psi_dot * dt)

        new_pos = np.array([x, y, z], dtype=np.float64)
        new_vel = linear_velocity_world.astype(np.float64)
        finite_ok = bool(
            np.all(np.isfinite(new_pos))
            and np.all(np.isfinite(new_vel))
            and np.isfinite([u, v, w, p_rate, q_rate, r_rate, phi, theta, psi]).all()
        )
        # 任何 NaN/Inf 都判定为失稳，交由上层记录 instability_events。
        if not finite_ok:
            return False

        state.p_pos = new_pos
        state.p_vel = new_vel
        state.u, state.v, state.w = float(u), float(v), float(w)
        state.p, state.q, state.p_w = float(p_rate), float(q_rate), float(r_rate)
        state.phi, state.theta, state.psi = float(phi), float(theta), float(psi)
        return True

    def get_collision_force(self, entity_a: Entity, entity_b: Entity) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        if (not entity_a.collide) or (not entity_b.collide):
            return None, None, False
        if entity_a is entity_b:
            return None, None, False

        delta_pos = np.asarray(entity_a.state.p_pos) - np.asarray(entity_b.state.p_pos)
        dist = float(np.linalg.norm(delta_pos))
        dist = max(dist, 1e-9)
        dist_min = float(entity_a.size + entity_b.size)
        k = self.contact_margin
        penetration = np.logaddexp(0.0, -(dist - dist_min) / k) * k
        force = self.contact_force * (delta_pos / dist) * penetration
        force_a = force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        collided = dist < dist_min
        return force_a, force_b, collided

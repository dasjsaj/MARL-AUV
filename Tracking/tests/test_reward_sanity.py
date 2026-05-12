from __future__ import annotations

import numpy as np

from Tracking.auv6dof.gym_env import AUV6DOFGymEnv


def _reward_for_first_agent(env: AUV6DOFGymEnv) -> float:
    return float(env.scenario.reward(env.world.agents[0], env.world))


def test_tracking_reward_prefers_reasonable_distance() -> None:
    env = AUV6DOFGymEnv(
        {
            "n_agent": 4,
            "episode_length": 5,
            "action_control_mode": "velocity3",
            "reward": {"version": "tracking_v3"},
            "reset": {"curriculum_stage": "easy"},
        }
    )
    env.reset(seed=2)
    target = env.world.targets[0]
    agent = env.world.agents[0]
    target.state.p_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    agent.state.p_pos = np.array([0.08, 0.0, 0.0], dtype=np.float64)
    near_reward = _reward_for_first_agent(env)
    env.scenario._last_reward_terms.clear()
    env.scenario._prev_target_distance.clear()
    agent.state.p_pos = np.array([0.75, 0.0, 0.0], dtype=np.float64)
    far_reward = _reward_for_first_agent(env)
    assert near_reward > far_reward
    env.close()

from __future__ import annotations

import numpy as np

from Tracking.auv6dof.gym_env import AUV6DOFGymEnv


def test_env_step_velocity3_action_effect_and_info() -> None:
    env = AUV6DOFGymEnv(
        {
            "n_agent": 4,
            "episode_length": 5,
            "action_control_mode": "velocity3",
            "reward": {"version": "tracking_v3"},
            "obs": {"include_tracking_diagnostics": True, "include_semantic_features": True},
            "reset": {"curriculum_stage": "easy"},
        }
    )
    env.reset(seed=1)
    before = np.stack([agent.state.p_pos.copy() for agent in env.world.agents], axis=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[:, 0] = 1.0
    obs, reward, terminated, truncated, info = env.step(action)
    after = np.stack([agent.state.p_pos.copy() for agent in env.world.agents], axis=0)
    assert obs["agent_state"].shape[0] == 4
    assert reward.shape == (4,)
    assert not terminated
    assert not truncated
    assert float(np.mean(after[:, 0] - before[:, 0])) > 0.0
    for key in ("raw_action", "mapped_action", "clipped_action", "tracking_reward", "control_cost", "tracking_error"):
        assert key in info
    env.close()

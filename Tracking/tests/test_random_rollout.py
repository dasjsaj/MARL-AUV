from __future__ import annotations

import numpy as np

from Tracking.auv6dof.gym_env import AUV6DOFGymEnv


def test_random_rollout_velocity3_no_nan() -> None:
    env = AUV6DOFGymEnv(
        {
            "n_agent": 4,
            "episode_length": 20,
            "action_control_mode": "velocity3",
            "reward": {"version": "tracking_v3"},
            "obs": {"include_tracking_diagnostics": True, "include_semantic_features": True},
            "reset": {"curriculum_stage": "easy"},
        }
    )
    obs, _ = env.reset(seed=3)
    total = 0.0
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += float(np.sum(reward))
        assert np.isfinite(obs["agent_state"]).all()
        assert np.isfinite(reward).all()
        assert "tracking_error" in info
        if terminated or truncated:
            break
    assert np.isfinite(total)
    env.close()

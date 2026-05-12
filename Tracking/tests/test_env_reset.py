from __future__ import annotations

import numpy as np

from Tracking.auv6dof.gym_env import AUV6DOFGymEnv


def test_env_reset_velocity3_obs_finite() -> None:
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
    obs, info = env.reset(seed=0)
    assert info["n_agent"] == 4
    assert obs["agent_state"].shape[0] == 4
    assert obs["global_state"].shape == (4, 4 * obs["agent_state"].shape[1])
    assert env.action_space.shape == (4, 3)
    assert np.isfinite(obs["agent_state"]).all()
    assert np.isfinite(obs["global_state"]).all()
    env.close()

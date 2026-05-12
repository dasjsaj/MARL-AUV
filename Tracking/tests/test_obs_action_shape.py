from __future__ import annotations

from Tracking.marl_orchestrator import build_run_contract


def test_maddpg_velocity3_contract_shape() -> None:
    contract = build_run_contract(
        env_name="auv6dof",
        algo_name="maddpg",
        n_agent=4,
        episode_length=200,
        env_overrides={
            "action_control_mode": "velocity3",
            "reward": {"version": "tracking_v3"},
            "obs": {"include_tracking_diagnostics": True, "include_semantic_features": True},
        },
    )
    assert contract["shape_cfg"]["action_dim_continuous"] == 3
    assert contract["shape_cfg"]["agent_obs_dim"] > 51
    assert contract["shape_cfg"]["global_obs_dim"] == 4 * contract["shape_cfg"]["agent_obs_dim"]


def test_mappo_velocity3_contract_shape() -> None:
    contract = build_run_contract(
        env_name="auv6dof",
        algo_name="mappo",
        n_agent=4,
        episode_length=200,
        env_overrides={
            "action_control_mode": "velocity3",
            "reward": {"version": "tracking_v3"},
            "obs": {"include_tracking_diagnostics": True, "include_semantic_features": True},
        },
    )
    assert contract["shape_cfg"]["action_dim_continuous"] == 3
    assert contract["env_cfg"]["agent_specific_global_state"] is True
    assert contract["shape_cfg"]["agent_obs_dim"] > 51

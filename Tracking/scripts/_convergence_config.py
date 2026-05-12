from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_convergence_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[1] / cfg_path
    if not cfg_path.exists():
        cfg_path = Path(path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    parent = cfg.pop("extends", "")
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = cfg_path.parent / parent_path
        base = load_convergence_config(parent_path)
        return deep_update(base, cfg)
    return cfg


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``patch`` into ``base`` and return ``base``."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def algo_env_overrides(cfg: Dict[str, Any], algo: str | None = None) -> Dict[str, Any]:
    """Return global env overrides merged with the requested algorithm profile."""
    merged: Dict[str, Any] = dict(cfg.get("env_overrides", {}))
    if algo:
        per_algo = cfg.get("algo_env_overrides", {})
        deep_update(merged, dict(per_algo.get(str(algo).lower(), {})))
    return merged


def algo_profile_name(cfg: Dict[str, Any], algo: str | None = None) -> str:
    overrides = algo_env_overrides(cfg, algo)
    explicit = str(overrides.get("algo_profile", "")).strip()
    if explicit:
        return explicit
    if str(algo or "").lower() == "stg_mappo":
        return "stg_semantic_velocity3"
    obs = overrides.get("obs", {}) if isinstance(overrides.get("obs"), dict) else {}
    semantic = bool(obs.get("include_semantic_features", False) or obs.get("include_semantic_graph_features", False))
    action_mode = str(overrides.get("action_control_mode", "tau6")).strip().lower()
    return ("semantic" if semantic else "raw") + f"_{action_mode}"


def env_cfg_from_config(cfg: Dict[str, Any], algo: str | None = None) -> Dict[str, Any]:
    env_cfg: Dict[str, Any] = {
        "n_agent": int(cfg.get("n_agent", 4)),
        "episode_length": int(cfg.get("episode_length", 200)),
    }
    deep_update(env_cfg, algo_env_overrides(cfg, algo))
    return env_cfg

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_orchestrator import (  # noqa: E402
    ALGO_REGISTRY,
    _ensure_tensorboard_writer,
    _finalize_run,
    _wait_for_reward_artifact,
)


def _to_easydict(value: Any) -> Any:
    if isinstance(value, dict):
        return EasyDict({k: _to_easydict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_easydict(v) for v in value]
    return value


def _latest_checkpoint(exp_dir: Path, ckpt_name: str) -> Path:
    ckpt_dir = exp_dir / "ckpt"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {ckpt_dir}")
    if ckpt_name != "latest":
        path = ckpt_dir / ckpt_name
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        return path.resolve()
    candidates = [p for p in ckpt_dir.glob("*.pth.tar") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume an interrupted TMC 2e6 DI-engine run in-place.")
    parser.add_argument("--run-dir", required=True, help="Existing run directory containing config.json and exp/ckpt.")
    parser.add_argument("--max-env-step", type=int, default=2_000_000)
    parser.add_argument("--ckpt", default="latest", help="Checkpoint filename under exp/ckpt, or latest by mtime.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "config.json"
    seed_path = run_dir / "seed.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    if not seed_path.exists():
        raise FileNotFoundError(f"Missing seed.json: {seed_path}")

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    contract = payload["contract"]
    main_config = _to_easydict(payload["main_config"])
    create_config = _to_easydict(payload["create_config"])
    seed = int(json.loads(seed_path.read_text(encoding="utf-8")).get("seed", 0))
    algo_name = str(contract.get("algo_cfg", {}).get("algo_name") or "").lower()
    if not algo_name:
        raise ValueError("Cannot infer algo_name from contract.algo_cfg.algo_name")

    ckpt = _latest_checkpoint(Path(main_config.exp_name), args.ckpt)
    main_config.policy.learn.resume_training = True
    if "hook" not in main_config.policy.learn:
        main_config.policy.learn.hook = EasyDict()
    main_config.policy.learn.hook.load_ckpt_before_run = ckpt.as_posix()
    # Save more often after resuming so another interruption does not lose a long segment.
    main_config.policy.learn.hook.save_ckpt_after_iter = min(
        int(main_config.policy.learn.hook.get("save_ckpt_after_iter", 10000) or 10000), 2000
    )

    print(
        json.dumps(
            {
                "run_dir": run_dir.as_posix(),
                "exp_name": str(main_config.exp_name),
                "algo": algo_name,
                "seed": seed,
                "max_env_step": int(args.max_env_step),
                "resume_checkpoint": ckpt.as_posix(),
                "resume_training": bool(main_config.policy.learn.resume_training),
                "save_ckpt_after_iter": int(main_config.policy.learn.hook.save_ckpt_after_iter),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return

    _ensure_tensorboard_writer()
    if ALGO_REGISTRY[algo_name]["pipeline_type"] == "onpolicy":
        from ding.entry import serial_pipeline_onpolicy

        serial_pipeline_onpolicy((main_config, create_config), seed=seed, max_env_step=int(args.max_env_step))
    else:
        from ding.entry import serial_pipeline

        serial_pipeline((main_config, create_config), seed=seed, max_env_step=int(args.max_env_step))

    _wait_for_reward_artifact(run_dir)
    summary = _finalize_run(run_dir, str(main_config.exp_name), contract)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

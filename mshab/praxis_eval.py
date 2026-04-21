"""Run MS-HAB subtask evaluation against a remote Praxis policy server."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import mshab.envs  # noqa: F401 - registers gym envs
from mshab.envs.make import EnvConfig, make_env
from praxis_client import PolicyClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="set_table")
    parser.add_argument(
        "--subtask",
        required=True,
        choices=("pick", "place", "open", "close"),
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--task-alias", required=True)
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--ms-asset-dir", default=None)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", default="depth")
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--stationary-base", action="store_true")
    parser.add_argument("--stationary-torso", action="store_true")
    parser.add_argument("--stationary-head", dest="stationary_head", action="store_true")
    parser.add_argument(
        "--no-stationary-head",
        dest="stationary_head",
        action="store_false",
    )
    parser.set_defaults(stationary_head=True)
    parser.add_argument("--praxis-host", required=True)
    parser.add_argument("--praxis-port", type=int, required=True)
    parser.add_argument("--praxis-policy-kwargs-json", default=None)
    parser.add_argument("--metrics-output-path", required=True)
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--no-save-video", action="store_true")
    return parser.parse_args()


def resolve_ms_asset_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get("MS_ASSET_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "data" / "maniskill_assets"


def rearrange_root(ms_asset_dir: Path) -> Path:
    return (
        ms_asset_dir
        / "data"
        / "scene_datasets"
        / "replica_cad_dataset"
        / "rearrange"
    )


def task_plan_path(*, root: Path, task: str, subtask: str, split: str, target: str) -> Path:
    return root / "task_plans" / task / subtask / split / f"{target}.json"


def spawn_data_path(*, root: Path, task: str, subtask: str, split: str) -> Path:
    return root / "spawn_data" / task / subtask / split / "spawn_data.pt"


def max_episode_steps_for_subtask(subtask: str) -> int:
    if subtask == "navigate":
        return 1000
    return 200


def approx_save_video_freq(num_episodes: int, max_videos: int) -> int:
    if max_videos <= 0:
        return 1
    return max(1, int(math.ceil(num_episodes / max_videos)))


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def flatten_depth_stack(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = to_numpy(value).astype(np.float32, copy=False)
    if array.ndim == 5 and array.shape[2] == 1:
        return array[:, :, 0, :, :]
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0, :, :]
    return array


def build_remote_observations(
    obs: dict[str, Any], *, task_description: str
) -> list[dict[str, Any]]:
    state_batch = to_numpy(obs["state"]).astype(np.float32, copy=False)
    pixels = obs.get("pixels", {})
    if not isinstance(pixels, dict):
        pixels = {}
    head_batch = flatten_depth_stack(
        pixels.get("fetch_head_depth", obs.get("fetch_head_depth"))
    )
    hand_batch = flatten_depth_stack(
        pixels.get("fetch_hand_depth", obs.get("fetch_hand_depth"))
    )

    batch_size = int(state_batch.shape[0])
    observations: list[dict[str, Any]] = []
    for index in range(batch_size):
        item: dict[str, Any] = {
            "observation.state": state_batch[index],
            "task": task_description,
        }
        if head_batch is not None:
            item["observation.images.fetch_head_depth"] = head_batch[index]
        if hand_batch is not None:
            item["observation.images.fetch_hand_depth"] = hand_batch[index]
        observations.append(item)
    return observations


def extend_done_values(dest: list[Any], value: Any, done_mask: np.ndarray) -> None:
    array = to_numpy(value)
    dest.extend(array[done_mask].tolist())


def main() -> None:
    args = parse_args()
    ms_asset_dir = resolve_ms_asset_dir(args.ms_asset_dir)
    rearrange_dir = rearrange_root(ms_asset_dir)
    plan_fp = task_plan_path(
        root=rearrange_dir,
        task=args.task,
        subtask=args.subtask,
        split=args.split,
        target=args.target,
    )
    spawn_fp = spawn_data_path(
        root=rearrange_dir,
        task=args.task,
        subtask=args.subtask,
        split=args.split,
    )
    if not plan_fp.exists():
        raise FileNotFoundError(f"Missing task plan file: {plan_fp}")
    if not spawn_fp.exists():
        raise FileNotFoundError(f"Missing spawn data file: {spawn_fp}")

    metrics_output_path = Path(args.metrics_output_path).expanduser().resolve()
    record_dir = Path(args.record_dir).expanduser().resolve()
    record_dir.mkdir(parents=True, exist_ok=True)
    policy_kwargs = (
        json.loads(args.praxis_policy_kwargs_json)
        if args.praxis_policy_kwargs_json
        else {}
    )
    save_video = bool(args.max_videos > 0 and not args.no_save_video)
    env_cfg = EnvConfig(
        env_id=f"{args.subtask.capitalize()}SubtaskTrain-v0",
        num_envs=int(args.num_envs),
        max_episode_steps=max_episode_steps_for_subtask(args.subtask),
        obs_mode=str(args.obs_mode),
        frame_stack=int(args.frame_stack),
        cat_state=True,
        cat_pixels=False,
        stationary_base=bool(args.stationary_base),
        stationary_torso=bool(args.stationary_torso),
        stationary_head=bool(args.stationary_head),
        task_plan_fp=str(plan_fp),
        spawn_data_fp=str(spawn_fp),
        record_video=save_video,
        save_video_freq=approx_save_video_freq(
            num_episodes=int(args.num_episodes),
            max_videos=int(args.max_videos),
        )
        if save_video
        else None,
        env_kwargs={
            "require_build_configs_repeated_equally_across_envs": False,
            "add_event_tracker_info": True,
        },
    )

    envs = None
    client = None
    try:
        start = time.time()
        envs = make_env(env_cfg, video_path=record_dir if save_video else None)
        device = envs.unwrapped.device
        obs, _info = envs.reset(seed=int(args.seed), options={"reconfigure": True})
        client = PolicyClient(host=args.praxis_host, port=int(args.praxis_port))
        ready, info = client.health_check()
        if not ready:
            raise RuntimeError(f"Praxis policy server is not ready: {info}")
        client.reset()

        sum_rewards: list[float] = []
        success_once: list[bool] = []
        success_at_end: list[bool] = []
        lengths: list[int] = []

        while len(lengths) < int(args.num_episodes):
            observations = build_remote_observations(
                obs, task_description=str(args.task_description)
            )
            action = np.asarray(
                client.predict_observations(
                    observations,
                    policy_kwargs=policy_kwargs,
                ),
                dtype=np.float32,
            )
            if action.ndim == 1:
                action = action[None, :]
            obs, _reward, _term, _trunc, infos = envs.step(
                torch.as_tensor(action, device=device)
            )
            done_mask = to_numpy(
                infos.get("_episode", np.zeros((args.num_envs,), dtype=bool))
            ).astype(bool)
            if np.any(done_mask):
                episode_info = infos["episode"]
                extend_done_values(sum_rewards, episode_info["r"], done_mask)
                extend_done_values(lengths, episode_info["l"], done_mask)
                extend_done_values(success_once, episode_info["s_o"], done_mask)
                extend_done_values(success_at_end, episode_info["s_e"], done_mask)

        limit = int(args.num_episodes)
        sum_rewards = [float(x) for x in sum_rewards[:limit]]
        lengths = [int(x) for x in lengths[:limit]]
        success_once = [bool(x) for x in success_once[:limit]]
        success_at_end = [bool(x) for x in success_at_end[:limit]]
        avg_sum_reward = float(np.mean(sum_rewards)) if sum_rewards else 0.0
        avg_episode_length = float(np.mean(lengths)) if lengths else 0.0
        metrics = {
            "task_alias": str(args.task_alias),
            "task_description": str(args.task_description),
            "task_plan_fp": str(plan_fp),
            "spawn_data_fp": str(spawn_fp),
            "n_episodes": float(len(lengths)),
            "success_once_rate": float(np.mean(success_once)) if success_once else 0.0,
            "success_at_end_rate": float(np.mean(success_at_end))
            if success_at_end
            else 0.0,
            "avg_episode_length": avg_episode_length,
            "avg_sum_reward": avg_sum_reward,
            "avg_return_per_step": (
                avg_sum_reward / float(env_cfg.max_episode_steps)
                if env_cfg.max_episode_steps > 0
                else 0.0
            ),
            "sum_rewards": sum_rewards,
            "lengths": lengths,
            "success_once": success_once,
            "success_at_end": success_at_end,
            "eval_s": float(time.time() - start),
        }
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_output_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        if client is not None:
            client.close()
        if envs is not None:
            envs.close()


if __name__ == "__main__":
    main()

"""Run MS-HAB subtask evaluation against a remote Praxis policy server."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mshab.runtime_bootstrap import load_env_factory
from praxis_remote import PolicyClient

EXPECTED_STATE_DIM = 42
EXPECTED_ACTION_DIM = 13
EXPECTED_RGB_SHAPE = (3, 128, 128)
_PROGRESS_LOG_INTERVAL_SEC = 15.0


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
    parser.add_argument("--policy-task", required=True)
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--ms-asset-dir", default=None)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--stationary-base", action="store_true")
    parser.add_argument("--stationary-torso", action="store_true")
    parser.add_argument(
        "--stationary-head", dest="stationary_head", action="store_true"
    )
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
    parser.add_argument(
        "--debug-max-control-steps",
        type=int,
        default=None,
        help=(
            "Stop after this many env control steps even if no episode has "
            "finished. Intended for contract diagnostics only."
        ),
    )
    return parser.parse_args()


def rearrange_root(ms_asset_dir: Path) -> Path:
    return (
        ms_asset_dir / "data" / "scene_datasets" / "replica_cad_dataset" / "rearrange"
    )


def task_plan_path(
    *, root: Path, task: str, subtask: str, split: str, target: str
) -> Path:
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


def _stat_summary(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    return {
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def summarize_remote_observation_batch(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    if not observations:
        return {}
    summary: dict[str, Any] = {
        "batch_size": int(len(observations)),
        "tasks": sorted({str(item.get("task")) for item in observations}),
    }
    for key in (
        "observation.state",
        "observation.images.fetch_head",
        "observation.images.fetch_hand",
    ):
        values = [np.asarray(item[key]) for item in observations if key in item]
        if values:
            summary[key] = _stat_summary(np.stack(values, axis=0))
    return summary


def prepare_policy_action(
    action: Any,
    *,
    num_envs: int,
    action_dim: int = EXPECTED_ACTION_DIM,
) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    expected = (int(num_envs), int(action_dim))
    if tuple(array.shape) != expected:
        raise ValueError(
            f"Expected policy action shape {expected}, got {array.shape}. "
            "This must match the MS-HAB Fetch action contract."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Policy returned non-finite MS-HAB actions.")
    return array


def _space_vector(value: Any, *, action_dim: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (action_dim,):
        return array
    if array.ndim == 2 and array.shape[1] == action_dim:
        return array[0]
    return None


def action_bounds_from_env(
    envs: Any, *, action_dim: int = EXPECTED_ACTION_DIM
) -> tuple[np.ndarray, np.ndarray] | None:
    for attr in ("single_action_space", "action_space"):
        space = getattr(envs, attr, None)
        if space is None:
            continue
        low = _space_vector(getattr(space, "low", None), action_dim=action_dim)
        high = _space_vector(getattr(space, "high", None), action_dim=action_dim)
        if low is not None and high is not None:
            return low, high
    return None


def clip_action_to_bounds(
    action: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if bounds is None:
        return action.astype(np.float32, copy=False)
    low, high = bounds
    return np.clip(action, low[None, :], high[None, :]).astype(np.float32, copy=False)


class ActionClipStatsAccumulator:
    def __init__(
        self,
        *,
        action_dim: int = EXPECTED_ACTION_DIM,
        enabled: bool,
    ) -> None:
        self.action_dim = int(action_dim)
        self.enabled = bool(enabled)
        self.count = 0
        self.clipped = np.zeros((self.action_dim,), dtype=np.int64)
        self.rows_clipped = 0
        self.max_abs_delta = np.zeros((self.action_dim,), dtype=np.float64)

    def update(self, before: np.ndarray, after: np.ndarray) -> None:
        before = prepare_policy_action(
            before, num_envs=int(before.shape[0]), action_dim=self.action_dim
        ).astype(np.float64, copy=False)
        after = prepare_policy_action(
            after, num_envs=int(after.shape[0]), action_dim=self.action_dim
        ).astype(np.float64, copy=False)
        delta = np.abs(after - before)
        clipped = delta > 1e-6
        self.count += int(before.shape[0])
        self.clipped += np.sum(clipped, axis=0)
        self.rows_clipped += int(np.any(clipped, axis=1).sum())
        self.max_abs_delta = np.maximum(self.max_abs_delta, np.max(delta, axis=0))

    def as_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "enabled": self.enabled,
                "count": 0,
                "action_dim": self.action_dim,
            }
        return {
            "enabled": self.enabled,
            "count": int(self.count),
            "action_dim": self.action_dim,
            "clipped_count": self.clipped.astype(int).tolist(),
            "clipped_fraction": (self.clipped / float(self.count))
            .astype(float)
            .tolist(),
            "row_fraction_any_clipped": float(self.rows_clipped / float(self.count)),
            "max_abs_delta": self.max_abs_delta.astype(float).tolist(),
        }


class ActionStatsAccumulator:
    def __init__(
        self,
        *,
        action_dim: int = EXPECTED_ACTION_DIM,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.action_dim = int(action_dim)
        self.bounds = bounds
        self.count = 0
        self.sum = np.zeros((self.action_dim,), dtype=np.float64)
        self.sum_sq = np.zeros((self.action_dim,), dtype=np.float64)
        self.min = np.full((self.action_dim,), np.inf, dtype=np.float64)
        self.max = np.full((self.action_dim,), -np.inf, dtype=np.float64)
        self.outside_unit = np.zeros((self.action_dim,), dtype=np.int64)
        self.rows_outside_unit = 0
        self.outside_bounds = np.zeros((self.action_dim,), dtype=np.int64)
        self.rows_outside_bounds = 0

    def update(self, action: np.ndarray) -> None:
        action = prepare_policy_action(
            action, num_envs=int(action.shape[0]), action_dim=self.action_dim
        ).astype(np.float64, copy=False)
        self.count += int(action.shape[0])
        self.sum += np.sum(action, axis=0)
        self.sum_sq += np.sum(np.square(action), axis=0)
        self.min = np.minimum(self.min, np.min(action, axis=0))
        self.max = np.maximum(self.max, np.max(action, axis=0))

        outside_unit = np.abs(action) > 1.0
        self.outside_unit += np.sum(outside_unit, axis=0)
        self.rows_outside_unit += int(np.any(outside_unit, axis=1).sum())

        if self.bounds is not None:
            low, high = self.bounds
            outside_bounds = (action < low[None, :]) | (action > high[None, :])
            self.outside_bounds += np.sum(outside_bounds, axis=0)
            self.rows_outside_bounds += int(np.any(outside_bounds, axis=1).sum())

    def as_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "action_dim": self.action_dim}
        mean = self.sum / float(self.count)
        var = np.maximum(self.sum_sq / float(self.count) - np.square(mean), 0.0)
        out: dict[str, Any] = {
            "count": int(self.count),
            "action_dim": self.action_dim,
            "min": self.min.astype(float).tolist(),
            "max": self.max.astype(float).tolist(),
            "mean": mean.astype(float).tolist(),
            "std": np.sqrt(var).astype(float).tolist(),
            "max_abs": np.maximum(np.abs(self.min), np.abs(self.max))
            .astype(float)
            .tolist(),
            "outside_unit_count": self.outside_unit.astype(int).tolist(),
            "outside_unit_fraction": (self.outside_unit / float(self.count))
            .astype(float)
            .tolist(),
            "row_fraction_any_outside_unit": float(
                self.rows_outside_unit / float(self.count)
            ),
        }
        if self.bounds is not None:
            low, high = self.bounds
            out.update(
                {
                    "bounds_low": low.astype(float).tolist(),
                    "bounds_high": high.astype(float).tolist(),
                    "outside_bounds_count": self.outside_bounds.astype(int).tolist(),
                    "outside_bounds_fraction": (self.outside_bounds / float(self.count))
                    .astype(float)
                    .tolist(),
                    "row_fraction_any_outside_bounds": float(
                        self.rows_outside_bounds / float(self.count)
                    ),
                }
            )
        return out


def latest_rgb_batch(value: Any, *, key: str) -> np.ndarray | None:
    if value is None:
        return None
    array = to_numpy(value)
    if array.ndim == 5:
        array = array[:, -1]
    if array.ndim != 4:
        raise ValueError(
            f"Expected batched RGB {key} with 4 or 5 dims, got {array.shape}"
        )
    if array.shape[1] == 3:
        pass
    elif array.shape[-1] == 3:
        array = np.moveaxis(array, -1, 1)
    else:
        raise ValueError(
            f"Expected batched RGB {key} as BCHW or BHWC, got {array.shape}"
        )
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32, copy=False)
    if tuple(array.shape[1:]) != EXPECTED_RGB_SHAPE:
        raise ValueError(
            f"Expected {key} RGB shape (B, {EXPECTED_RGB_SHAPE[0]}, "
            f"{EXPECTED_RGB_SHAPE[1]}, {EXPECTED_RGB_SHAPE[2]}) after conversion, "
            f"got {array.shape}. This must match the MS-HAB PolicyIO image shape."
        )
    return array


def _required_rgb_batch(
    obs: dict[str, Any],
    pixels: dict[str, Any],
    key: str,
) -> np.ndarray:
    batch = latest_rgb_batch(pixels.get(key, obs.get(key)), key=key)
    if batch is None:
        available = sorted(
            str(obs_key)
            for obs_key in set(obs) | {f"pixels.{pixels_key}" for pixels_key in pixels}
        )
        raise KeyError(
            "MS-HAB Praxis eval requires RGB keys 'fetch_head' and 'fetch_hand'. "
            "Use --obs-mode rgb or --obs-mode rgbd. "
            f"Missing {key!r}; available keys: {available}"
        )
    return batch


def build_remote_observations(
    obs: dict[str, Any], *, policy_task: str
) -> list[dict[str, Any]]:
    state_batch = to_numpy(obs["state"]).astype(np.float32, copy=False)
    if state_batch.ndim != 2 or int(state_batch.shape[1]) != EXPECTED_STATE_DIM:
        raise ValueError(
            f"Expected MS-HAB state shape (B, {EXPECTED_STATE_DIM}), "
            f"got {state_batch.shape}."
        )
    pixels = obs.get("pixels", {})
    if not isinstance(pixels, dict):
        pixels = {}
    head_batch = _required_rgb_batch(obs, pixels, "fetch_head")
    hand_batch = _required_rgb_batch(obs, pixels, "fetch_hand")

    batch_size = int(state_batch.shape[0])
    observations: list[dict[str, Any]] = []
    for index in range(batch_size):
        item: dict[str, Any] = {
            "observation.state": state_batch[index],
            "observation.images.fetch_head": head_batch[index],
            "observation.images.fetch_hand": hand_batch[index],
            "task": policy_task,
        }
        observations.append(item)
    return observations


def extend_done_values(dest: list[Any], value: Any, done_mask: np.ndarray) -> None:
    array = to_numpy(value)
    dest.extend(array[done_mask].tolist())


def _running_success_rate(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _emit_progress_line(
    *,
    task_alias: str,
    done_count: int,
    total_episodes: int,
    success_once: list[bool],
    success_at_end: list[bool],
    loop_start_time: float,
) -> None:
    elapsed_s = max(0.0, float(time.time() - loop_start_time))
    parts = [
        "MSHAB_EVAL",
        f"task_alias={task_alias}",
        f"done={done_count}/{total_episodes}",
        f"succ_rate={100.0 * _running_success_rate(success_once):.1f}%",
        f"succ_end={100.0 * _running_success_rate(success_at_end):.1f}%",
        f"elapsed_s={elapsed_s:.1f}",
    ]
    if done_count > 0 and elapsed_s > 0:
        parts.append(f"{elapsed_s / float(done_count):.2f}s/ep")
    print(" ".join(parts), flush=True)


def main() -> None:
    args = parse_args()
    ms_asset_dir, EnvConfig, make_env = load_env_factory(args.ms_asset_dir)
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
        lane_generations = [0 for _ in range(int(args.num_envs))]
        episode_ids = [
            f"{args.task_alias}:lane{lane_idx}:episode0"
            for lane_idx in range(int(args.num_envs))
        ]
        client.reset(episode_ids=episode_ids)
        action_bounds = action_bounds_from_env(envs, action_dim=EXPECTED_ACTION_DIM)
        action_stats = ActionStatsAccumulator(
            action_dim=EXPECTED_ACTION_DIM, bounds=action_bounds
        )
        policy_action_stats = ActionStatsAccumulator(
            action_dim=EXPECTED_ACTION_DIM, bounds=action_bounds
        )
        action_clip_stats = ActionClipStatsAccumulator(
            action_dim=EXPECTED_ACTION_DIM,
            enabled=action_bounds is not None,
        )
        first_observation_summary: dict[str, Any] | None = None

        sum_rewards: list[float] = []
        success_once: list[bool] = []
        success_at_end: list[bool] = []
        lengths: list[int] = []
        control_steps = 0
        loop_start_time = time.time()
        last_progress_log_time = loop_start_time
        _emit_progress_line(
            task_alias=str(args.task_alias),
            done_count=0,
            total_episodes=int(args.num_episodes),
            success_once=success_once,
            success_at_end=success_at_end,
            loop_start_time=loop_start_time,
        )

        while len(lengths) < int(args.num_episodes):
            observations = build_remote_observations(
                obs, policy_task=str(args.policy_task)
            )
            if first_observation_summary is None:
                first_observation_summary = summarize_remote_observation_batch(
                    observations
                )
            policy_action = prepare_policy_action(
                client.predict_action(
                    observations,
                    policy_kwargs=policy_kwargs,
                    episode_ids=episode_ids,
                ),
                num_envs=int(args.num_envs),
                action_dim=EXPECTED_ACTION_DIM,
            )
            policy_action_stats.update(policy_action)
            action = clip_action_to_bounds(policy_action, action_bounds)
            action_clip_stats.update(policy_action, action)
            action_stats.update(action)
            obs, _reward, _term, _trunc, infos = envs.step(
                torch.as_tensor(action, device=device)
            )
            control_steps += 1
            done_mask = to_numpy(
                infos.get("_episode", np.zeros((args.num_envs,), dtype=bool))
            ).astype(bool)
            if np.any(done_mask):
                episode_info = infos["episode"]
                extend_done_values(sum_rewards, episode_info["r"], done_mask)
                extend_done_values(lengths, episode_info["l"], done_mask)
                extend_done_values(success_once, episode_info["s_o"], done_mask)
                extend_done_values(success_at_end, episode_info["s_e"], done_mask)
                done_indices = np.flatnonzero(done_mask).astype(int).tolist()
                done_episode_ids = [episode_ids[index] for index in done_indices]
                client.reset(episode_ids=done_episode_ids)
                for index in done_indices:
                    lane_generations[index] += 1
                    episode_ids[index] = (
                        f"{args.task_alias}:lane{index}:"
                        f"episode{lane_generations[index]}"
                    )
            now = time.time()
            if np.any(done_mask) or (
                now - last_progress_log_time >= _PROGRESS_LOG_INTERVAL_SEC
            ):
                _emit_progress_line(
                    task_alias=str(args.task_alias),
                    done_count=min(len(lengths), int(args.num_episodes)),
                    total_episodes=int(args.num_episodes),
                    success_once=success_once,
                    success_at_end=success_at_end,
                    loop_start_time=loop_start_time,
                )
                last_progress_log_time = now
            if args.debug_max_control_steps is not None and control_steps >= int(
                args.debug_max_control_steps
            ):
                print(
                    "MSHAB_EVAL_DEBUG_MAX_STEPS "
                    f"task_alias={args.task_alias} "
                    f"control_steps={control_steps} "
                    f"episodes_done={len(lengths)}/{args.num_episodes}",
                    flush=True,
                )
                break

        limit = int(args.num_episodes)
        sum_rewards = [float(x) for x in sum_rewards[:limit]]
        lengths = [int(x) for x in lengths[:limit]]
        success_once = [bool(x) for x in success_once[:limit]]
        success_at_end = [bool(x) for x in success_at_end[:limit]]
        _emit_progress_line(
            task_alias=str(args.task_alias),
            done_count=len(lengths),
            total_episodes=limit,
            success_once=success_once,
            success_at_end=success_at_end,
            loop_start_time=loop_start_time,
        )
        avg_sum_reward = float(np.mean(sum_rewards)) if sum_rewards else 0.0
        avg_episode_length = float(np.mean(lengths)) if lengths else 0.0
        metrics = {
            "task_alias": str(args.task_alias),
            "policy_task": str(args.policy_task),
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
            "control_steps": int(control_steps),
            "debug_max_control_steps": (
                int(args.debug_max_control_steps)
                if args.debug_max_control_steps is not None
                else None
            ),
            "debug_truncated_before_episodes": bool(len(lengths) < limit),
            "action_stats": action_stats.as_dict(),
            "policy_action_stats": policy_action_stats.as_dict(),
            "action_clip_stats": action_clip_stats.as_dict(),
            "first_observation_summary": first_observation_summary or {},
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

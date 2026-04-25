from collections import deque
from typing import Dict, List, Optional

import gymnasium as gym

import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.common import flatten_state_dict


class FetchDepthObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, cat_state=True, cat_pixels=False) -> None:
        super().__init__(env)

        self.cat_pixels = cat_pixels
        self.cat_state = cat_state
        self._stack_fn = torch.stack
        self._cat_fn = torch.cat

        self._base_env: BaseEnv = env.unwrapped
        init_raw_obs = common.to_tensor(self._base_env._init_raw_obs)

        self._base_env.update_obs_space(common.to_numpy(self.observation(init_raw_obs)))

    def observation(self, observation):
        agent_obs = observation["agent"]
        extra_obs = observation["extra"]
        fetch_head_rgb = self._camera_rgb(observation, "fetch_head")
        fetch_hand_rgb = self._camera_rgb(observation, "fetch_hand")
        fetch_head_depth = self._camera_depth(observation, "fetch_head")
        fetch_hand_depth = self._camera_depth(observation, "fetch_hand")

        pixel_obs = {}
        if self.cat_pixels:
            if fetch_head_depth is not None and fetch_hand_depth is not None:
                pixel_obs["all_depth"] = self._stack_fn(
                    [fetch_head_depth, fetch_hand_depth], axis=-3
                )
        else:
            if fetch_head_rgb is not None:
                pixel_obs["fetch_head"] = fetch_head_rgb
            if fetch_hand_rgb is not None:
                pixel_obs["fetch_hand"] = fetch_hand_rgb
            if fetch_head_depth is not None:
                pixel_obs["fetch_head_depth"] = fetch_head_depth
            if fetch_hand_depth is not None:
                pixel_obs["fetch_hand_depth"] = fetch_hand_depth

        return (
            dict(
                state=self._cat_fn(
                    [
                        flatten_state_dict(agent_obs, use_torch=True),
                        flatten_state_dict(extra_obs, use_torch=True),
                    ],
                    axis=1,
                ),
                **pixel_obs,
            )
            if self.cat_state
            else dict(
                agent=agent_obs,
                extra=extra_obs,
                **pixel_obs,
            )
        )

    def _camera_rgb(self, observation, camera_name):
        camera_obs = observation["sensor_data"][camera_name]
        if "rgb" not in camera_obs:
            return None
        rgb = camera_obs["rgb"]
        if rgb.ndim == 4 and rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)
        elif not (rgb.ndim == 4 and rgb.shape[1] == 3):
            raise ValueError(
                f"Expected {camera_name} rgb as BHWC or BCHW, got {tuple(rgb.shape)}"
            )
        if rgb.is_floating_point():
            return rgb.float()
        return rgb.float() / 255.0

    def _camera_depth(self, observation, camera_name):
        camera_obs = observation["sensor_data"][camera_name]
        if "depth" not in camera_obs:
            return None
        depth = camera_obs["depth"]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            return depth.permute(0, 3, 1, 2)
        if depth.ndim == 3:
            return depth[:, None, ...]
        if depth.ndim == 4 and depth.shape[1] == 1:
            return depth
        raise ValueError(
            f"Expected {camera_name} depth as BHWC, BHW, or BCHW, got {tuple(depth.shape)}"
        )


# TODO (arth): deprecate this in favor of StackedDictObservationWrapper + stacking_keys
#   will need to update rl and bc train scripts to matchs new output (i.e. no "pixels" key)
class FrameStack(gym.Wrapper):
    def __init__(
        self,
        env,
        num_stack: int,
        stacking_keys: List[str] = ["fetch_head_depth", "fetch_hand_depth"],
    ) -> None:
        super().__init__(env)
        self._base_env = env.unwrapped
        self._num_stack = num_stack
        self._stacking_keys = stacking_keys

        assert all([k in env.observation_space.spaces for k in stacking_keys])

        self._frames: Dict[str, deque] = dict()

        init_raw_obs: dict = self._base_env._init_raw_obs
        pixel_init_raw_obs = dict()
        self._stack_dim = dict()
        for sk in self._stacking_keys:
            obs_space = self.observation_space.spaces[sk]
            init_raw_obs_sk_replace = init_raw_obs.pop(sk)[:, None, ...]
            stack_dim = -len(obs_space.shape[1:]) - 1

            pixel_init_raw_obs[sk] = np.repeat(
                init_raw_obs_sk_replace, num_stack, axis=stack_dim
            )
            self._frames[sk] = deque(maxlen=num_stack)
            self._stack_dim[sk] = stack_dim
        init_raw_obs["pixels"] = pixel_init_raw_obs
        self._base_env.update_obs_space(init_raw_obs)

        self._stack_fn = torch.stack

    def _get_stacked_frames(self):
        return dict(
            (sk, self._stack_fn(tuple(self._frames[sk]), axis=self._stack_dim[sk]))
            for sk in self._stacking_keys
        )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            frame = obs.pop(sk)
            for _ in range(self._num_stack):
                self._frames[sk].append(frame)
        obs["pixels"] = self._get_stacked_frames()
        return obs, info

    def step(self, *args, **kwargs):
        obs, rew, term, trunc, info = super().step(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            self._frames[sk].append(obs.pop(sk))
        obs["pixels"] = self._get_stacked_frames()
        return obs, rew, term, trunc, info


class StackedDictObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        num_stack: int,
        stacking_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(env)
        self._base_env: BaseEnv = env.unwrapped
        self._num_stack = num_stack

        if stacking_keys is None:
            assert isinstance(env.single_observation_space, gym.spaces.Dict)
            self._stacking_keys = env.single_observation_space.keys()
        else:
            assert all([k in env.observation_space.spaces for k in stacking_keys])
            self._stacking_keys = stacking_keys

        self._running_stacks: Dict[str, deque] = dict()

        init_raw_obs: dict = self._base_env._init_raw_obs
        stacked_init_raw_obs = dict()
        self._stack_dim = dict()
        for sk in self._stacking_keys:
            obs_space = self.observation_space.spaces[sk]
            init_raw_obs_sk_replace = init_raw_obs.pop(sk)[:, None, ...]
            stack_dim = -len(obs_space.shape[1:]) - 1

            stacked_init_raw_obs[sk] = np.repeat(
                init_raw_obs_sk_replace, num_stack, axis=stack_dim
            )
            self._running_stacks[sk] = deque(maxlen=num_stack)
            self._stack_dim[sk] = stack_dim
        init_raw_obs.update(**stacked_init_raw_obs)
        self._base_env.update_obs_space(init_raw_obs)

        self._stack_fn = torch.stack

    def _get_stacked_obs(self):
        return dict(
            (
                sk,
                self._stack_fn(
                    tuple(self._running_stacks[sk]), axis=self._stack_dim[sk]
                ),
            )
            for sk in self._stacking_keys
        )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            frame = obs.pop(sk)
            for _ in range(self._num_stack):
                self._running_stacks[sk].append(frame)
        return self._get_stacked_obs(), info

    def step(self, *args, **kwargs):
        obs, rew, term, trunc, info = super().step(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            self._running_stacks[sk].append(obs.pop(sk))
        return self._get_stacked_obs(), rew, term, trunc, info

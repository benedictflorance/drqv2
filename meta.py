# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple
from dm_env import specs
import numpy as np


class ActionRepeatWrapper():
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._num_repeats):
            obs, reward, done, info = self._env.step(action)
            time_step = {"observation": obs, "reward": reward, "done": done, "info": info, "discount": 1.0, 'is_last': self._env.curr_path_length == self._env.max_path_length}
            total_reward += time_step["reward"]
            if self._env.curr_path_length == self._env.max_path_length: #https://github.com/rlworkgroup/metaworld/issues/236
                break
        time_step["reward"] = total_reward
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper():
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[3 * num_frames], (84,84)], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        time_step["observation"] = obs
        return time_step

    def reset(self):
        obs = self._env.reset()
        time_step = {"observation": obs, "reward": 0, "done": False, "info": {}, "discount": 1.0, "is_last": False}
        pixels = self._env.render(offscreen=True, resolution=(84, 84), camera_name='topview') 
        for _ in range(self._num_frames):
            self._frames.append(pixels.transpose(2, 0, 1))
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._env.render(offscreen=True, resolution=(84, 84), camera_name='topview') 
        for _ in range(self._num_frames):
            self._frames.append(pixels.transpose(2, 0, 1))
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper():
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_space
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.low,
                                               wrapped_action_spec.high,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_space.dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_space

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ExtendedTimeStep(NamedTuple):
    is_last: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def last(self):
        return self.is_last

    def __getitem__(self, attr):
        return getattr(self, attr)
        
class ExtendedTimeStepWrapper():
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step["observation"],
                                action=action,
                                is_last=time_step["is_last"],
                                reward=time_step["reward"] or 0.0,
                                discount=time_step["discount"] or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

def make(name, frame_stack, action_repeat, seed):
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    task = name.replace("_", "-") + "-v2-goal-observable"
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]()
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env

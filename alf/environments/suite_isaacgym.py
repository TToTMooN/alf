# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from absl import logging
import math
import numpy as np
import os
import random
import scipy.interpolate
import subprocess
import sys
import time
import torch
from unittest.mock import Mock
import weakref

try:
    import isaacgymenvs
    from omegaconf import DictConfig, OmegaConf
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
except ImportError:
    issacgymenvs = None

import numpy as np
import copy
import gym

import alf
from alf.environments import suite_gym
from alf.environments import alf_wrappers


def is_available():
    """Check if isaacgym is installed"""
    return isaacgymenvs is not None


@alf.configurable
class TensorAction(gym.ActionWrapper):
    """Convert action to tensor if it is a numpy, required by IsaacGym.
    """

    def __init__(self, env):
        """Create an ContinuousActionClip gym wrapper.

        Args:
            env (gym.Env): A Gym env instance to wrap
        """
        super(TensorAction, self).__init__(env)

    def action(self, action):
        action = torch.from_numpy(action)
        return action


@alf.configurable
class IsaacGymWrapper(alf_wrappers.AlfEnvironmentBaseWrapper):
    def __init__(self, env):
        super().__init(env)

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._gym_env.num_envs


@alf.configurable
def load(cfg,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         alf_env_wrappers=()):

    env = isaacgymenvs.make(
        cfg.seed,
        cfg.task_name,
        cfg.task.env.numEnvs,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
    )

    env = TensorAction(env)

    if not max_episode_steps:  # None or 0
        max_episode_steps = env.max_episode_length

    return suite_gym.wrap_isaac_gym_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        image_channel_first=False)

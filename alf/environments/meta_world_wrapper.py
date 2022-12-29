# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import abc
from alf.utils.dist_utils import extract_distribution_parameters
from collections import OrderedDict
import copy
import cProfile
import math
import numpy as np
import random
import six

import torch
import torch.nn.functional as F
import gym
import alf
from alf.data_structures import StepType, TimeStep, _is_numpy_array
import alf.nest as nest
import alf.tensor_specs as ts
from alf.utils import spec_utils
from alf.utils.tensor_utils import to_tensor
from alf.environments.alf_wrappers import AlfEnvironmentBaseWrapper, MultitaskWrapper


def round_robin_strategy(num_tasks, last_task=None):
    """A function for sampling tasks in round robin fashion.
    Args:
        num_tasks (int): Total number of tasks.
        last_task (int): Previously sampled task.
    Returns:
        int: task id.
    """
    if last_task is None:
        return 0
    return (last_task + 1) % num_tasks


def uniform_random_strategy(num_tasks, _):
    """A function for sampling tasks uniformly at random.
    Args:
        num_tasks (int): Total number of tasks.
        _ (object): Ignored by this sampling strategy.
    Returns:
        int: task id.
    """
    return random.randint(0, num_tasks - 1)


def weighted_random_strategy(num_tasks, task_weight):
    """A function for sampling tasks with weighted prob at random.
    Args:
        num_tasks (int): Total number of tasks.
        task weight: prob weight of each task.
    Returns:
        int: task id.
    """
    assert (num_tasks == len(task_weight))
    return random.choices(range(num_tasks), weights=task_weight)[0]


@alf.configurable
class SuccessRateWrapper(gym.Wrapper):
    """Add a info on success or not of the whole episode rather than number of success steps.
       Only one step will be True if the whole episode is a success one.
    """

    def __init__(self, env):
        super().__init__(env)
        self._exist_success_in_episode = False

    def reset(self, **kwargs):
        self._exist_success_in_episode = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # count success when it's first time success in episode

        if (not self.exist_success()) and info["success"]:
            env_info = {'episode_success': 1.0, 'exist_success': True}
            self._exist_success_in_episode = True
        else:
            env_info = {
                'episode_success': 0.0,
                'exist_success': self.exist_success()
            }
        env_info.update(info)
        return obs, reward, done, env_info

    def exist_success(self):
        return self._exist_success_in_episode


@alf.configurable
class MultitaskMetaworldWrapper(MultitaskWrapper):
    def __init__(self,
                 envs,
                 task_names,
                 goal_dict=None,
                 env_id=None,
                 mode='add-onehot',
                 sample_strategy='uniform',
                 separate_task_info=False,
                 task_sample_weight=None):
        """
        Args:
            envs (list[AlfEnvironment]): a list of environments. Each one
                represents a different task.
            task_names (list[str]): the names of each task.
            env_id (int): (optional) ID of the environment.
            mode(str): 
        """
        assert len(envs) > 0, "`envs should not be empty"
        assert len(set(task_names)) == len(task_names), (
            "task_names should "
            "not contain duplicated names: %s" % str(task_names))
        assert mode in ['vanilla', 'add-onehot', 'onehot-dict']
        assert sample_strategy in ['uniform', 'robin', 'weighted']

        self._task_sample_weight = None
        if sample_strategy == 'uniform':
            self._sample_strategy = uniform_random_strategy
        elif sample_strategy == 'robin':
            self._sample_strategy = round_robin_strategy
        elif sample_strategy == 'weighted' and task_sample_weight is not None:
            self._sample_strategy = weighted_random_strategy
            self._task_sample_weight = task_sample_weight
        else:
            raise NotImplementedError

        self._mode = mode
        self._envs = envs
        self._action_spec = envs[0].action_spec()
        self._reward_spec = envs[0].reward_spec()

        self._task_names = task_names
        self._sample_goal = goal_dict is not None
        self._goal_dict = goal_dict
        self._separate_task_info = separate_task_info
        self._env_info_spec = copy.copy(envs[0].env_info_spec())
        self._env_info_spec.update(
            self._add_task_names({
                'task_count': [alf.TensorSpec(())] * self.num_tasks,
                'task_success': [alf.TensorSpec(())] * self.num_tasks
            }))
        if self._separate_task_info:
            self._env_info_spec.update(
                self._add_task_names({
                    'ret': [alf.TensorSpec(())] * self.num_tasks,
                    'grasp_succ': [alf.TensorSpec(())] * self.num_tasks,
                    'in_place_rew': [alf.TensorSpec(())] * self.num_tasks,
                    'near_obj': [alf.TensorSpec(())] * self.num_tasks,
                    'obj_to_target': [alf.TensorSpec(())] * self.num_tasks,
                    'task_count': [alf.TensorSpec(())] * self.num_tasks,
                    'task_success': [alf.TensorSpec(())] * self.num_tasks,
                }))
        self.current_task_succ = False

        if env_id is None:
            env_id = 0
        self._env_id = np.int32(env_id)

        for env in self._envs:
            # assert env obs and action space shape
            assert env.observation_spec(
            ).shape == self._envs[0].observation_spec().shape, (
                "All environement should have same observation shape."
                "Got %s vs %s" % (self._envs[0].observation_spec().shape,
                                  env.observation_spec().shape))
            assert env.action_spec().shape == self._envs[0].action_spec(
            ).shape, ("All environement should have same action shape."
                      "Got %s vs %s" % (self._envs[0].action_spec().shape,
                                        env.action_spec().shape))
            env.reset()

        # set observation spec according to mode of observation space
        if self._mode == 'vanilla':
            self._observation_spec = envs[0].observation_spec()
        elif self._mode == 'add-onehot':
            task_lb, task_ub = np.zeros(self.num_tasks), np.ones(
                self.num_tasks)
            env_lb, env_ub = envs[0].observation_spec(
            ).minimum, envs[0].observation_spec().maximum
            env_shape = np.array(envs[0].observation_spec().shape)
            env_shape[0] += self.num_tasks
            self._observation_spec = alf.BoundedTensorSpec(
                shape=env_shape,
                dtype="float32",
                minimum=np.concatenate([env_lb, task_lb]),
                maximum=np.concatenate([env_ub, task_ub]))
        elif self._mode == 'onehot-dict':
            self._observation_spec = dict(
                observation=envs[0].observation_spec(),
                task_id=alf.TensorSpec([self.num_tasks]))

        self._task_counts = [np.array(0.0)] * self.num_tasks
        self._task_succ_rates = [np.array(0.0)] * self.num_tasks
        if self._separate_task_info:
            self._grasp_success = [np.array(0.0)] * self.num_tasks
            self._in_place_reward = [np.array(0.0)] * self.num_tasks
            self._near_object = [np.array(0.0)] * self.num_tasks
            self._object_to_target = [np.array(0.0)] * self.num_tasks
            self._average_return = [np.array(0.0)] * self.num_tasks

        # metadata for rendering
        self._metadata = copy.copy(self._envs[0].metadata)
        self._metadata.update({'render.modes': ["rgb_array", "human"]})
        # reset and sample initial task
        self._current_env_id = np.int64(-1)
        self.reset()

    def _reset(self):
        # record counts before reset if not the first time
        if self._current_env_id != -1:
            self._task_counts[self._current_env_id] = self._task_counts[
                self._current_env_id] + 1
            self._task_succ_rates[
                self._current_env_id] = self._task_succ_rates[
                    self._current_env_id] + self.current_task_succ
        if self._task_sample_weight is not None:
            self._current_env_id = self._sample_strategy(
                self.num_tasks, self._task_sample_weight)
        else:
            self._current_env_id = self._sample_strategy(
                self.num_tasks, self._current_env_id)
        if self._sample_goal:
            self._envs[self._current_env_id].set_task(
                random.choice(self._goal_dict[self.current_task_name]))
        time_step = self._envs[self._current_env_id].reset()

        return self._complete_obs_and_info(time_step)

    def _step(self, action):
        # step the selected current env
        time_step = self._envs[self._current_env_id].step(action)
        return self._complete_obs_and_info(time_step)

    def _complete_obs_and_info(self, time_step):
        obs, info = time_step.observation, copy.copy(time_step.env_info)
        # Add task id into observation via concat or dict
        if self._mode == 'vanilla':
            pass
        elif self._mode == 'add-onehot':
            obs = np.concatenate([obs, self._active_task_one_hot()])
        elif self._mode == 'onehot-dict':
            obs = dict(observation=obs, task_id=self._active_task_one_hot())
        # Add a record for current task success to update task succ rate
        self.current_task_succ = time_step.env_info['exist_success']
        # Add information of each tasks, task counts, task_succ_rate
        info.update(
            self._add_task_names({
                'task_count': self._task_counts,
                'task_success': self._task_succ_rates
            }))
        if self._separate_task_info:
            self._grasp_success[self._current_env_id] = info['grasp_success']
            self._in_place_reward[self.
                                  _current_env_id] = info['in_place_reward']
            self._near_object[self._current_env_id] = info['near_object']
            self._object_to_target[self.
                                   _current_env_id] = info['obj_to_target']
            self._average_return[self.
                                 _current_env_id] = info['unscaled_reward']
            info.update(
                self._add_task_names({
                    'ret': self._average_return,
                    'grasp_succ': self._grasp_success,
                    'in_place_rew': self._in_place_reward,
                    'near_obj': self._near_object,
                    'obj_to_target': self._object_to_target
                }))
        return time_step._replace(observation=obs, env_info=info)

    def _add_task_names(self, info):
        for k, v in info.items():
            info[k] = dict(zip(self._task_names, v))
        return info

    def _active_task_one_hot(self):
        """One-hot representation of active task.
        Returns:
            numpy.ndarray: one-hot representation of active task
        """
        one_hot = np.zeros(self.num_tasks, dtype='float32')
        index = self._current_env_id or 0
        one_hot[index] = 1
        return one_hot

    def render(self, mode='rgb_array'):
        return self._envs[self._current_env_id].render(mode)

    @property
    def metadata(self):
        return self._metadata

    @property
    def current_task_name(self):
        return self._task_names[self._current_env_id]

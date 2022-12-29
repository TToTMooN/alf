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

from os import environ
import numpy as np
import copy
import gym
import metaworld
import random
import gin
import alf
from alf.environments import alf_wrappers, suite_gym
from alf.environments.meta_world_wrapper import SuccessRateWrapper, MultitaskMetaworldWrapper


@alf.configurable
def load(environment_name,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         seed=0,
         gym_env_wrappers=(),
         alf_env_wrappers=()):
    """
    Load selected metaworld environment or benchmark environments with specifed wrappers.
    Note that metaworld env itself does not have a time limit in environment but a maximum
    of 500 steps in simulation. Default limit length is set to 150.

    Args:
        environment_name:
        env_id:
        discount:
        max_episode_steps:
        gym_env_wrappers:
        alf_env_wrappers:
    """
    if environment_name == 'mt1-reach':
        task_name = 'reach-v2'
    elif environment_name == 'mt1-push':
        task_name = 'push-v2'
    elif environment_name == 'mt1-pickplace':
        task_name = 'pick-place-v2'
    elif environment_name == 'mt1-peginsert':
        task_name = 'peg-insert-side-v2'
    elif environment_name == 'mt1-drawerclose':
        task_name = 'drawer-close-v2'

    mt = metaworld.MT1(task_name, seed=seed)
    env = mt.train_classes[task_name]()
    task = mt.train_tasks[0]
    env.set_task(task)

    if not max_episode_steps:  # None or 0
        max_episode_steps = env.max_path_length
    max_episode_steps = min(env.max_path_length, max_episode_steps)

    return suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers)


@alf.configurable
def load_mt(environment_name,
            env_id=None,
            discount=1.0,
            max_episode_steps=None,
            sample_strategy='uniform',
            sample_goal=False,
            seed=0,
            gym_env_wrappers=(),
            alf_env_wrappers=()):
    if environment_name == "mt1-reach-peg":
        environment_name_list = ['reach-v2', 'peg-insert-side-v2']
    elif environment_name == "mt1-transfer-base":
        environment_name_list = [
            'reach-v2', 'door-open-v2', 'drawer-open-v2', 'window-close-v2'
        ]
    elif environment_name == "mt1-transfer-window-open-set":
        environment_name_list = ['window-open-v2']
    elif environment_name == "mt1-transfer-button-press-set":
        environment_name_list = ['button-press-v2']
    elif environment_name == "mt1-transfer-drawer-close-set":
        environment_name_list = ['drawer-close-v2']
    elif environment_name == "mt1-transfer-door-close-set":
        environment_name_list = ['door-close-v2']
    elif environment_name == "mt1-transfer-set":
        environment_name_list = [
            'button-press-v2', 'drawer-close-v2', "window-open-v2",
            "door-close-v2"
        ]
    elif environment_name == "mt1-reach-medium":
        environment_name_list = ['reach-v2', 'door-open-v2', 'drawer-open-v2']
    elif environment_name == "mt1-simple":
        environment_name_list = ['reach-v2', 'drawer-close-v2']
    elif environment_name == "mt1-reach-pick":
        environment_name_list = ["reach-v2", 'pick-place-v2']
    elif environment_name == 'mt1-reach-push-peg':
        environment_name_list = ['reach-v2', 'push-v2', 'peg-insert-side-v2']
    elif environment_name == 'mt1-pick-peg':
        environment_name_list = ['pick-place-v2', 'peg-insert-side-v2']
    elif environment_name == 'mt1-peg':
        environment_name_list = ['peg-insert-side-v2']
    elif environment_name == 'mt1-pick':
        environment_name_list = ['pick-place-v2']
    else:
        environment_name_list = [environment_name]
        # return NotImplementedError
    envs = []
    full_env_name_list = []
    goal_dict = {} if sample_goal else None

    exist_duplicate = len(environment_name_list) != len(
        set(environment_name_list))
    for i, env_name in enumerate(environment_name_list):
        mt = metaworld.MT1(env_name, seed=seed)
        env = mt.train_classes[env_name]()
        # policy conditioned on goals, sample goal at every reset
        task = mt.train_tasks[0]
        env.set_task(task)

        if exist_duplicate:
            full_env_name = str(i) + '-' + env_name
        else:
            full_env_name = env_name
        full_env_name_list.append(full_env_name)

        if goal_dict is not None:
            goal_dict[full_env_name] = mt.train_tasks

        if not max_episode_steps:  # None or 0
            max_episode_steps = env.max_path_length
        max_episode_steps = min(env.max_path_length, max_episode_steps)

        env = suite_gym.wrap_env(
            env,
            env_id=env_id,
            discount=discount,
            gym_env_wrappers=gym_env_wrappers,
            alf_env_wrappers=alf_env_wrappers)
        envs.append(env)

    mtenv = MultitaskMetaworldWrapper(
        envs,
        task_names=full_env_name_list,
        goal_dict=goal_dict,
        sample_strategy=sample_strategy,
        env_id=env_id)
    mtenv = alf_wrappers.TimeLimit(mtenv, duration=max_episode_steps)

    return mtenv


@alf.configurable
def load_mt_benchmark(environment_name,
                      env_id=None,
                      discount=1.0,
                      max_episode_steps=150,
                      seed=0,
                      sample_strategy='uniform',
                      sample_goal=False,
                      gym_env_wrappers=(),
                      alf_env_wrappers=()):
    assert environment_name in ['mt10', 'mt50']
    if environment_name == 'mt10':
        mt_benchmark = metaworld.MT10(seed=seed)
    else:
        mt_benchmark = metaworld.MT50(seed=seed)
    envs = []
    goal_dict = {} if sample_goal else None
    random.seed(seed)
    for env_name, env_cls in mt_benchmark.train_classes.items():
        print(env_name)
        env = env_cls()
        env_tasks = [
            task for task in mt_benchmark.train_tasks
            if task.env_name == env_name
        ]
        task = random.choice(env_tasks)
        env.set_task(task)
        if goal_dict is not None:
            goal_dict[env_name] = env_tasks
        if not max_episode_steps:  # None or 0
            max_episode_steps = env.max_path_length
        max_episode_steps = min(env.max_path_length, max_episode_steps)
        env = suite_gym.wrap_env(
            env,
            env_id=env_id,
            discount=discount,
            gym_env_wrappers=gym_env_wrappers,
            alf_env_wrappers=alf_env_wrappers)
        envs.append(env)
    environment_name_list = [t[0] for t in mt_benchmark.train_classes.items()]
    mtenv = MultitaskMetaworldWrapper(
        envs,
        task_names=environment_name_list,
        goal_dict=goal_dict,
        sample_strategy=sample_strategy,
        env_id=env_id)
    mtenv = alf_wrappers.TimeLimit(mtenv, duration=max_episode_steps)
    return mtenv


@alf.configurable
def load_test_pickplace_duplicate(env_name,
                                  env_id=None,
                                  discount=1.0,
                                  max_episode_steps=None,
                                  sample_strategy='uniform',
                                  seed=0,
                                  gym_env_wrappers=(),
                                  alf_env_wrappers=()):
    task_name = 'pick-place-v2'
    environment_name_list = ['pick-place-v2']
    mt = metaworld.MT1(task_name, seed=seed)
    env = mt.train_classes[task_name]()
    task = mt.train_tasks[0]
    env.set_task(task)

    if not max_episode_steps:  # None or 0
        max_episode_steps = env.max_path_length
    max_episode_steps = min(env.max_path_length, max_episode_steps)
    mt_env = [
        suite_gym.wrap_env(
            env,
            discount=discount,
            gym_env_wrappers=gym_env_wrappers,
            alf_env_wrappers=alf_env_wrappers)
    ]
    mt_env = MultitaskMetaworldWrapper(
        mt_env,
        task_names=environment_name_list,
        sample_strategy=sample_strategy,
        env_id=env_id,
        mode='onehot-dict')
    mt_env = alf_wrappers.TimeLimit(mt_env, duration=max_episode_steps)

    single_env = suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers)
    if env_name == 'single':
        return single_env
    else:
        return mt_env

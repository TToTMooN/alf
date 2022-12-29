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
"""A common metaworld task configuration independent of algortihms. This file
defines some basic experiment protocol (e.g., parallel envs, hidden layers,
learning rate, etc) to be shared by different algorithms to be evaluted.
"""

from functools import partial

import alf
from alf.algorithms.data_transformer import RewardNormalizer
from alf.environments import suite_robotics
from alf.utils.math_ops import clipped_exp
from alf.environments.meta_world_wrapper import SuccessRateWrapper
from alf.nest.utils import NestConcat
from alf.environments import suite_metaworld
from alf.environments import meta_world_wrapper
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.mtsac_algorithm import MTSacAlgorithm
from alf.networks import ActorDistributionNetwork, ActorDistributionCompositionalNetwork
from alf.networks import CriticNetwork, CriticCompositionalNetwork
from alf.networks import EncodingNetwork, TaskEncodingNetwork
from alf.networks.new_projection_networks import CompositionalNormalProjectionNetwork
from alf.utils import math_ops
import metaworld
# environment config
alf.config(
    'suite_metaworld.load_mt',
    max_episode_steps=150,
    gym_env_wrappers=[SuccessRateWrapper])

alf.config(
    'suite_metaworld.load',
    max_episode_steps=150,
    gym_env_wrappers=[SuccessRateWrapper])

alf.config(
    'suite_metaworld.load_mt_benchmark',
    max_episode_steps=150,
    gym_env_wrappers=[SuccessRateWrapper])

alf.config('meta_world_wrapper.MultitaskMetaworldWrapper', mode='onehot-dict')
obs_mask = dict(observation=1, task_id=0)
task_mask = dict(observation=0, task_id=1)
obs_task_mask = dict(observation=1, task_id=1)

# layer hypermeter configs
fc_hidden_layers = (400, ) * 3

# training config
alf.config(
    'TrainerConfig',
    temporally_independent_train_step=True,
    initial_collect_steps=1500,
    mini_batch_length=2,
    unroll_length=1,
    mini_batch_size=1280,
    num_updates_per_train_iter=1,
    num_iterations=0,
    num_env_steps=20e6,
    num_checkpoints=50,
    evaluate=False,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summarize_output=True,
    summary_interval=500,
    replay_buffer_length=1000000)

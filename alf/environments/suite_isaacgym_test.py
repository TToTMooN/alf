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

try:
    import isaacgymenvs
    import isaacgym
    import os
    from hydra import compose, initialize
    import yaml
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import to_absolute_path
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
except:
    raise ImportError

from absl import flags
from absl import logging
import pprint
import sys
import torch
import datetime

import alf
from alf.environments import alf_environment, suite_isaacgym


class SuiteIsaacGymTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_isaacgym.is_available():
            self.skipTest('isaac_gym is not available')

    def get_cfg_hydra(self):
        with initialize(config_path="./isaac_gym_envs/cfg"):
            cfg = compose(config_name="config", overrides=["task=Ant"])
        cfg_dict = omegaconf_to_dict(cfg)
        print_dict(cfg_dict)
        set_np_formatting()
        rank = int(os.getenv("LOCAL_RANK", "0"))
        if cfg.multi_gpu:
            # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
            cfg.sim_device = f'cuda:{rank}'
            cfg.rl_device = f'cuda:{rank}'

        # sets seed. if seed is -1 will pick a random one
        cfg.seed += rank
        cfg.seed = set_seed(
            cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)
        return cfg

    def test_unwrapped_env(self):
        cfg = self.get_cfg_hydra()
        env = suite_isaacgym.load(cfg=cfg)
        self._env = env
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        logging.info(
            "observation_spec: %s" % pprint.pformat(env.observation_spec()))
        logging.info("action_spec: %s" % pprint.pformat(env.action_spec()))

        action_spec = env.action_spec()

        try:
            for _ in range(10):
                action = action_spec.sample([env.num_envs])
                logging.info("action: %s" % action)
                for _ in range(10):
                    time_step = env.step(action)
                    # TODO: change logging
                    logging.debug(
                        "observation state: %s, reward=%s" %
                        (time_step.observation['obs'], time_step.reward))
        except:
            raise NotImplementedError


if __name__ == '__main__':
    alf.test.main()

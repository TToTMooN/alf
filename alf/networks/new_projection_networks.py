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

import math
import numpy as np
from typing import Callable

import torch
import torch.nn as nn
import torch.distributions as td

import alf
import alf.layers as layers
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks.network import Network
from alf.networks.projection_networks import _get_transformer
from alf.utils import dist_utils
import alf.utils.math_ops as math_ops


@alf.configurable
class CompositionalNormalProjectionNetwork(Network):
    def __init__(self,
                 input_size,
                 action_spec,
                 num_of_param_set,
                 activation=math_ops.identity,
                 disable_compositional=False,
                 projection_output_init_gain=0.3,
                 std_bias_initializer_value=0.0,
                 squash_mean=True,
                 state_dependent_std=False,
                 std_transform=nn.functional.softplus,
                 scale_distribution=False,
                 dist_squashing_transform=dist_utils.StableTanh(),
                 name="CompositionalNormalProjectionNetwork"):
        """Creates an instance of CompositionalNormalProjectionNetwork.
        
        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            n (int): the size of the paramster set
            activation (Callable): activation function to use in
                dense layers.
            projection_output_init_gain (float): Output gain for initializing
                action means and std weights.
            std_bias_initializer_value (float): Initial value for the bias of the
                ``std_projection_layer``.
            squash_mean (bool): If True, squash the output mean to fit the
                action spec. If ``scale_distribution`` is also True, this value
                will be ignored.
            state_dependent_std (bool): If True, std will be generated depending
                on the current state; otherwise a global std will be generated
                regardless of the current state.
            std_transform (Callable): Transform to apply to the std, on top of
                `activation`.
            scale_distribution (bool): Whether or not to scale the output
                distribution to ensure that the output aciton fits within the
                `action_spec`. Note that this is different from `mean_transform`
                which merely squashes the mean to fit within the spec.
            dist_squashing_transform (td.Transform):  A distribution Transform
                which transforms values into :math:`(-1, 1)`. Default to ``dist_utils.StableTanh()``
            name (str): name of this network.
        """
        super().__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(action_spec, TensorSpec)
        assert len(action_spec.shape) == 1, "Only support 1D action spec!"

        self._action_spec = action_spec
        self._num_param_set = num_of_param_set
        self._mean_transform = math_ops.identity
        self._scale_distribution = scale_distribution
        self._disable_compositional = disable_compositional

        if squash_mean or scale_distribution:
            assert isinstance(action_spec, BoundedTensorSpec), \
                ("When squashing the mean or scaling the distribution, bounds "
                 + "are required for the action spec!")

            action_high = torch.as_tensor(action_spec.maximum)
            action_low = torch.as_tensor(action_spec.minimum)
            self._action_means = (action_high + action_low) / 2
            self._action_magnitudes = (action_high - action_low) / 2
            # Do not transform mean if scaling distribution
            if not scale_distribution:
                self._mean_transform = (
                    lambda inputs: self._action_means + self._action_magnitudes
                    * inputs.tanh())
            else:
                self._transforms = [
                    dist_squashing_transform,
                    dist_utils.AffineTransform(
                        loc=self._action_means, scale=self._action_magnitudes)
                ]

        self._std_transform = math_ops.identity
        if std_transform is not None:
            self._std_transform = std_transform
        if self._disable_compositional:
            self._means_projection_layer = layers.FC(
                input_size,
                action_spec.shape[0],
                activation=activation,
                kernel_init_gain=projection_output_init_gain)
        else:
            self._means_projection_layer = layers.CompositionalFC(
                input_size,
                action_spec.shape[0],
                n=num_of_param_set,
                activation=activation,
                output_comp_weight=False,
                kernel_init_gain=projection_output_init_gain)

        if state_dependent_std:
            if self._disable_compositional:
                self._std_projection_layer = layers.FC(
                    input_size,
                    action_spec.shape[0],
                    activation=activation,
                    kernel_init_gain=projection_output_init_gain,
                    bias_init_value=std_bias_initializer_value)
            else:
                self._std_projection_layer = layers.CompositionalFC(
                    input_size,
                    action_spec.shape[0],
                    n=num_of_param_set,
                    activation=activation,
                    output_comp_weight=False,
                    kernel_init_gain=projection_output_init_gain,
                    bias_init_value=std_bias_initializer_value)
        else:
            if self._disable_compositional:
                self._std = nn.Parameter(
                    action_spec.constant(std_bias_initializer_value),
                    requires_grad=True)
            else:
                self._std = nn.Parameter(
                    action_spec.constant(std_bias_initializer_value).unsqueeze(
                        0).repeat(num_of_param_set, 1),
                    requires_grad=True)
            self._std_projection_layer = None

    def _normal_dist(self, means, stds):
        normal_dist = dist_utils.DiagMultivariateNormal(loc=means, scale=stds)
        if self._scale_distribution:
            # The transformed distribution can also do reparameterized sampling
            # i.e., `.has_rsample=True`
            # Note that in some cases kl_divergence might no longer work for this
            # distribution! Assuming the same `transforms`, below will work:
            # ````
            # kl_divergence(Independent, Independent)
            #
            # kl_divergence(TransformedDistribution(Independent, transforms),
            #               TransformedDistribution(Independent, transforms))
            # ````
            squashed_dist = td.TransformedDistribution(
                base_distribution=normal_dist, transforms=self._transforms)
            return squashed_dist
        else:
            return normal_dist

    def forward(self, inputs, state=()):
        assert type(inputs) == tuple, (
            "Expecting tuple inputs of (input, comp_weight)")
        if self._disable_compositional:
            # disable compositional, use NormalProjectionNetwork forward
            inputs = inputs[0]
        means = self._means_projection_layer(inputs)
        means = self._mean_transform(means)
        if self._std_projection_layer:
            stds = self._std_projection_layer(inputs)
            stds = self._std_transform(stds)
        else:
            if self._disable_compositional:
                stds = self._std
            else:
                stds = torch.matmul(inputs[1], self._std)
            stds = self._std_transform(stds)
        return self._normal_dist(means, stds), state


@alf.configurable
class CompositionalBetaProjectionNetwork(Network):
    """Beta projection network. projection layer use compFC layer.

    Its output is a distribution with independent beta distribution for each
    action dimension. Since the support of beta distribution is [0, 1], we also
    apply an affine transformation so the support fill the range specified by
    ``action_spec``.
    """

    def __init__(self,
                 input_size,
                 action_spec,
                 num_of_param_set,
                 activation=nn.functional.softplus,
                 disable_compositional=False,
                 min_concentration=0.,
                 max_concentration=None,
                 projection_output_init_gain=0.0,
                 bias_init_value=0.541324854612918,
                 grad_clip=0.01,
                 name="BetaProjectionNetwork"):
        """
        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            activation (Callable): activation function to use in
                dense layers.
            bias_init_value (float): the default value is chosen so that, for softplus
                activation, the initial concentration will be close 1, which
                corresponds to uniform distribution.
            grad_clip (float): if provided, the L2-norm of the gradient of concentration
                will be clipped to be no more than ``grad_clip``.
            min_concentration (float): there may be issue of numerical stability
                if the calculated concentration is very close to 0. A positive
                value of this may help to alleviate it.
        """
        super().__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)
        assert action_spec.ndim == 1, "Only support 1D action spec!"

        self._transformer = _get_transformer(action_spec)
        self._disable_compositional = disable_compositional
        self._num_param_set = num_of_param_set

        if self._disable_compositional:
            self._concentration_projection_layer = layers.FC(
                input_size,
                2 * action_spec.shape[0],
                activation=activation,
                bias_init_value=bias_init_value,
                kernel_init_gain=projection_output_init_gain)
        else:
            self._concentration_projection_layer = layers.CompositionalFC(
                input_size,
                2 * action_spec.shape[0],
                n=num_of_param_set,
                activation=activation,
                output_comp_weight=False,
                kernel_init_gain=projection_output_init_gain)
        self._grad_clip = grad_clip
        self._min_concentration = min_concentration
        self._max_concentration = max_concentration

    def forward(self, inputs, state=()):
        assert type(inputs) == tuple, (
            "Expecting tuple inputs of (input, comp_weight)")
        if self._disable_compositional:
            # disable compositional, use NormalProjectionNetwork forward
            inputs = inputs[0]
        concentration = self._concentration_projection_layer(inputs)
        if self._max_concentration is not None:
            concentration = torch.clamp(
                concentration, max=self._max_concentration)
        if self._min_concentration != 0:
            concentration = concentration + self._min_concentration
        if self._grad_clip is not None and inputs[0].requires_grad:
            concentration.register_hook(lambda x: x / (x.norm(
                dim=1, keepdim=True) * (1 / self._grad_clip)).clamp(1.))
        concentration10 = concentration.split(
            concentration.shape[-1] // 2, dim=-1)
        return self._transformer(
            dist_utils.DiagMultivariateBeta(*concentration10)), state


@alf.configurable
class CompositionalTruncatedProjectionNetwork(Network):
    def __init__(self,
                 input_size,
                 action_spec,
                 num_of_param_set,
                 activation=math_ops.identity,
                 disable_compositional=False,
                 projection_output_init_gain=0.3,
                 scale_bias_initializer_value=0.0,
                 state_dependent_scale=False,
                 scale_transform=nn.functional.softplus,
                 dist_ctor=dist_utils.TruncatedNormal,
                 name="TruncatedProjectionNetwork"):
        """Creates an instance of TruncatedProjectionNetwork.

        Its output is a TruncatedDistribution with bounds given by the action
        bounds specified in ``action_spec``.

        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            activation (Callable): activation function to use in
                dense layers.
            projection_output_init_gain (float): Output gain for initializing
                action means and std weights.
            std_bias_initializer_value (float): Initial value for the bias of the
                ``std_projection_layer``.
            state_dependent_scale (bool): If True, std will be generated depending
                on the current state (i.e. inputs); otherwise a global scale will
                be generated regardless of the current state.
            scale_transform (Callable): Transform to apply to the std, on top of
                `activation`.
            dist_ctor(Callable): constructor for the distribution called as:
                `dist_ctor(loc=loc, scale=scale, lower_bound=lower_bound, upper_bound=upper_bound)`.
            name (str): name of this network.
        """
        super().__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(action_spec, TensorSpec)
        assert len(action_spec.shape) == 1, "Only support 1D action spec!"

        self._scale_transform = math_ops.identity
        if scale_transform is not None:
            self._scale_transform = scale_transform

        self._disable_compositional = disable_compositional
        self._num_param_set = num_of_param_set

        if self._disable_compositional:
            self._loc_projection_layer = layers.FC(
                input_size,
                action_spec.shape[0],
                activation=activation,
                kernel_init_gain=projection_output_init_gain)
        else:
            self._loc_projection_layer = layers.CompositionalFC(
                input_size,
                action_spec.shape[0],
                n=num_of_param_set,
                activation=activation,
                output_comp_weight=False,
                kernel_init_gain=projection_output_init_gain)

        if state_dependent_scale:
            if self._disable_compositional:
                self._scale_projection_layer = layers.FC(
                    input_size,
                    action_spec.shape[0],
                    activation=activation,
                    kernel_init_gain=projection_output_init_gain,
                    bias_init_value=scale_bias_initializer_value)
            else:
                self._scale_projection_layer = layers.CompositionalFC(
                    input_size,
                    action_spec.shape[0],
                    n=num_of_param_set,
                    activation=activation,
                    output_comp_weight=False,
                    kernel_init_gain=projection_output_init_gain,
                    bias_init_value=scale_bias_initializer_value)
        else:
            if self._disable_compositional:
                self._scale = nn.Parameter(
                    action_spec.constant(scale_bias_initializer_value),
                    requires_grad=True)
                self._scale_projection_layer = lambda _: self._scale
            else:
                self._scale = nn.Parameter(
                    action_spec.constant(scale_bias_initializer_value).
                    unsqueeze(0).repeat(num_of_param_set, 1),
                    requires_grad=True)
                self._scale_projection_layer = None

        self._action_high = torch.as_tensor(action_spec.maximum).broadcast_to(
            action_spec.shape)
        self._action_low = torch.as_tensor(action_spec.minimum).broadcast_to(
            action_spec.shape)
        self._dist_ctor = dist_ctor

        action_means = (self._action_high + self._action_low) / 2
        action_magnitudes = (self._action_high - self._action_low) / 2

        # Although the TruncatedDistribution will ensure the actions are within
        # the bound, we still make sure the loc parameter to be within the bound
        # for better numerical stability
        self._loc_transform = (
            lambda inputs: action_means + action_magnitudes * inputs.tanh())

    def forward(self, inputs, state=()):
        assert type(inputs) == tuple, (
            "Expecting tuple inputs of (input, comp_weight)")
        if self._disable_compositional:
            # disable compositional, use NormalProjectionNetwork forward
            inputs = inputs[0]
        loc = self._loc_transform(self._loc_projection_layer(inputs))
        if self._scale_projection_layer:
            scale = self._scale_transform(self._scale_projection_layer(inputs))
        else:
            scale = self._scale_transform(torch.matmul(inputs[1], self._scale))
        dist = self._dist_ctor(
            loc=loc,
            scale=scale,
            lower_bound=self._action_low,
            upper_bound=self._action_high)

        return dist, state

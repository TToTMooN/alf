# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""ActorDistributionNetwork and ActorRNNDistributionNetwork."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import alf
import alf.nest as nest
from .encoding_networks import EncodingNetwork, LSTMEncodingNetwork, CompositionalEncodingNetwork, TaskEncodingNetwork
from .projection_networks import NormalProjectionNetwork, CategoricalProjectionNetwork
from .new_projection_networks import CompositionalNormalProjectionNetwork
from .preprocessor_networks import PreprocessorNetwork
from alf.data_structures import namedtuple
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.math_ops import identity
from alf.networks.network import Network


@alf.configurable
class ActorDistributionNetwork(Network):
    """Network which outputs temporally uncorrelated action distributions."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_fc_bn=False,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork,
                 name="ActorDistributionNetwork"):
        """

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the action spec
            input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with ``input_tensor_spec`` (after reshaping).
                If any element is None, then it will be treated as math_ops.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layer sizes.
            activation (nn.functional): activation used for hidden layers.
            kernel_initializer (Callable): initializer for all the layers
                excluding the projection net. If none is provided a default
                xavier_uniform will be used.
            use_fc_bn (bool): whether use Batch Normalization for the internal
                FC layers (i.e. FC layers except the last one).
            discrete_projection_net_ctor (ProjectionNetwork): constructor that
                generates a discrete projection network that outputs discrete
                actions.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                continuous actions.
            name (str):
        """
        super().__init__(input_tensor_spec, name=name)

        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_

        self._action_spec = action_spec
        self._encoding_net = EncodingNetwork(
            input_tensor_spec=input_tensor_spec,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn)
        self._create_projection_net(discrete_projection_net_ctor,
                                    continuous_projection_net_ctor)

    def _create_projection_net(self, discrete_projection_net_ctor,
                               continuous_projection_net_ctor):
        """If there are :math:`N` action specs, then create :math:`N` projection
        networks which can be a mixture of categoricals and normals.
        """

        def _create(spec):
            if spec.is_discrete:
                net = discrete_projection_net_ctor(
                    input_size=self._encoding_net.output_spec.shape[0],
                    action_spec=spec)
            else:
                net = continuous_projection_net_ctor(
                    input_size=self._encoding_net.output_spec.shape[0],
                    action_spec=spec)
            return net

        self._projection_net = nest.map_structure(_create, self._action_spec)
        if nest.is_nested(self._projection_net):
            # need this for torch to pickup the parameters of all the modules
            self._projection_net_module_list = nn.ModuleList(
                nest.flatten(self._projection_net))

    def forward(self, observation, state=()):
        """Computes an action distribution given an observation.

        Args:
            observation (torch.Tensor): consistent with ``input_tensor_spec``
            state: empty for API consistent with ``ActorRNNDistributionNetwork``

        Returns:
            act_dist (torch.distributions): action distribution
            state: empty
        """
        encoding, state = self._encoding_net(observation, state)
        act_dist = nest.map_structure(lambda proj: proj(encoding)[0],
                                      self._projection_net)
        return act_dist, state


@alf.configurable
class ActorDistributionRNNNetwork(ActorDistributionNetwork):
    """Network which outputs temporally correlated action distributions."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 actor_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork,
                 name="ActorRNNDistributionNetwork"):
        """

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the action spec
            input_preprocessors (nested InputPreprocessor): a nest of
                ``InputPreprocessor``, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with ``input_tensor_spec`` (after reshaping).
                If any element is None, then it will be treated as math_ops.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                ``input_tensor_spec`` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layers for encoding the observation.
            lstm_hidden_size (int or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            actor_fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers.
            kernel_initializer (Callable): initializer for all the layers
                excluding the projection net. If none is provided a default
                xavier_uniform will be used.
            discrete_projection_net_ctor (ProjectionNetwork): constructor that
                generates a discrete projection network that outputs discrete
                actions.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                continuous actions.
            name (str):
        """
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            name=name)

        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_

        self._encoding_net = LSTMEncodingNetwork(
            input_tensor_spec=input_tensor_spec,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            pre_fc_layer_params=fc_layer_params,
            hidden_size=lstm_hidden_size,
            post_fc_layer_params=actor_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)
        self._create_projection_net(discrete_projection_net_ctor,
                                    continuous_projection_net_ctor)

    @property
    def state_spec(self):
        return self._encoding_net.state_spec


@alf.configurable
class ActorDistributionMultiTaskNetwork(ActorDistributionNetwork):
    def __init__(
            self,
            input_tensor_spec,
            action_spec,
            input_preprocessors=None,
            observation_preprocessing_combiner=None,
            task_preprocessing_combiner=None,
            conv_layer_params=None,
            fc_layer_params=None,
            activation=torch.relu_,
            kernel_initializer=None,
            use_fc_bn=False,
            discrete_projection_net_ctor=CategoricalProjectionNetwork,
            continuous_projection_net_ctor=CompositionalNormalProjectionNetwork,
            name="ActorDistributionMuitiTaskNetwork"):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            name=name)
        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_

        self._obs_encoding_net = EncodingNetwork(
            input_tensor_spec=input_tensor_spec,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn)
        #current hard code to be identity network of onehot
        self._task_encoding_net = EncodingNetwork(
            input_tensor_spec=input_tensor_spec,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=task_preprocessing_combiner,
            conv_layer_params=None,
            fc_layer_params=None,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn)
        self._create_projection_net(discrete_projection_net_ctor,
                                    continuous_projection_net_ctor)

    def forward(self, observation, state):
        obs_encoding, state = self._obs_encoding_net(observation, state)
        task_encoding, _ = self._task_encoding_net(observation, state)
        inputs = (obs_encoding, task_encoding)
        act_dist = nest.map_structure(lambda proj: proj(inputs)[0],
                                      self._projection_net)
        return act_dist, state


CompositionalWeightState = namedtuple(
    "CompositionalWeightState", ["batch_task", "batch_weight"],
    default_value=())


@alf.configurable
class ActorDistributionCompositionalNetwork(Network):
    def __init__(
            self,
            input_tensor_spec,
            action_spec,
            num_of_param_set,
            task_encoding_net=None,
            use_compositional_encoding=False,
            use_task_input=False,
            input_preprocessors=None,
            observation_preprocessing_combiner=None,
            task_preprocessing_combiner=None,
            conv_layer_params=None,
            fc_layer_params=None,
            task_fc_layer_params=None,
            activation=torch.relu_,
            kernel_initializer=None,
            use_fc_bn=False,
            compositional_initializer=nn.init.orthogonal_,
            discrete_projection_net_ctor=CategoricalProjectionNetwork,
            continuous_projection_net_ctor=CompositionalNormalProjectionNetwork,
            name="ActorDistributionCompositionalNetwork"):

        super().__init__(input_tensor_spec, name=name)

        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_

        self._action_spec = action_spec
        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_
        self._use_compositional_encoding = use_compositional_encoding
        self._use_task_input = use_task_input
        self._num_of_param_set = num_of_param_set
        # task encoding net
        if task_encoding_net is not None:
            self._task_encoding_net = task_encoding_net
        else:
            self._task_encoding_net = TaskEncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                input_preprocessors=input_preprocessors,
                preprocessing_combiner=task_preprocessing_combiner,
                fc_layer_params=task_fc_layer_params,
                activation=activation,
                last_layer_size=num_of_param_set,
                last_activation=F.softmax,
                kernel_initializer=kernel_initializer,
                last_kernel_initializer=compositional_initializer,
                use_fc_bn=use_fc_bn)
        # obs encoding + joint encoding net
        if use_compositional_encoding:
            self._obs_encoding_net = CompositionalEncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                num_of_param_set=num_of_param_set,
                input_preprocessors=input_preprocessors,
                preprocessing_combiner=observation_preprocessing_combiner,
                conv_layer_params=None,
                fc_layer_params=None,
                activation=activation,
                kernel_initializer=kernel_initializer,
                use_fc_bn=use_fc_bn)
        else:
            self._obs_encoding_net = EncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                input_preprocessors=input_preprocessors,
                preprocessing_combiner=observation_preprocessing_combiner,
                conv_layer_params=None,
                fc_layer_params=None,
                activation=activation,
                kernel_initializer=kernel_initializer,
                use_fc_bn=use_fc_bn)
        if self._use_task_input:
            joint_encoder_input_spec = TensorSpec(
                (self._obs_encoding_net.output_spec.shape[0] +
                 self._task_encoding_net.output_spec.shape[0], ))
        else:
            joint_encoder_input_spec = TensorSpec(
                (self._obs_encoding_net.output_spec.shape[0], ))

        if use_compositional_encoding:
            self._joint_encoding_net = CompositionalEncodingNetwork(
                joint_encoder_input_spec,
                num_of_param_set,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
                use_fc_bn=use_fc_bn,
                name=self.name + ".joint_encoder")
        else:
            self._joint_encoding_net = EncodingNetwork(
                joint_encoder_input_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
                use_fc_bn=use_fc_bn,
                name=self.name + ".joint_encoder")

        self._create_projection_net(discrete_projection_net_ctor,
                                    continuous_projection_net_ctor)
        self._reduced_modules = nn.ModuleList([
            self._obs_encoding_net, self._joint_encoding_net,
            self._projection_net
        ])

    def _create_projection_net(self, discrete_projection_net_ctor,
                               continuous_projection_net_ctor):
        """If there are :math:`N` action specs, then create :math:`N` projection
        networks which can be a mixture of categoricals and normals.
        """

        def _create(spec):
            if spec.is_discrete:
                net = discrete_projection_net_ctor(
                    input_size=self._joint_encoding_net.output_spec.shape[0],
                    action_spec=spec)
            else:
                net = continuous_projection_net_ctor(
                    input_size=self._joint_encoding_net.output_spec.shape[0],
                    action_spec=spec)
            return net

        self._projection_net = nest.map_structure(_create, self._action_spec)
        if nest.is_nested(self._projection_net):
            # need this for torch to pickup the parameters of all the modules
            self._projection_net_module_list = nn.ModuleList(
                nest.flatten(self._projection_net))

    def forward(self, observation, state):
        task_encoding, task_state = self._task_encoding_net(observation, state)
        # process observation dict
        if self._use_compositional_encoding:
            inputs = (observation, task_encoding)
            obs_encoding, _ = self._obs_encoding_net(inputs, state)
        else:
            obs_encoding, _ = self._obs_encoding_net(observation, state)
        # concat task embedding if use task as input for policy
        if self._use_task_input:
            joint = torch.cat([obs_encoding, task_encoding], -1)
        else:
            joint = obs_encoding
        # encode observation and (task embedding if use task as input)
        if self._use_compositional_encoding:
            inputs = (joint, task_encoding)
            joint_encoding, _ = self._joint_encoding_net(inputs, state)
        else:
            joint_encoding, _ = self._joint_encoding_net(joint, state)

        # summarize param weight
        if alf.summary.should_summarize_output():
            fc_weight = self._task_encoding_net._fc_layers[0]._weight
            if self._task_encoding_net._last_activation is not None:
                param_weight = self._task_encoding_net._last_activation(
                    fc_weight)
                alf.summary.histogram(
                    name='weight_output/param_weight', data=param_weight)
            alf.summary.histogram(
                name='weight_output/fc_weight', data=fc_weight)

        inputs = (joint_encoding, task_encoding)
        act_dist = nest.map_structure(lambda proj: proj(inputs)[0],
                                      self._projection_net)
        return act_dist, state

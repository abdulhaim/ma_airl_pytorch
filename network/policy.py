
#!/usr/bin/env python
# Created at 2020/3/12


from misc.replay_memory import Memory


from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal

from utils.network_utils import MultiOneHotCategorical, MultiSoftMax
from utils.torch_utils import resolve_activate_function


class Actor(nn.Module):
    def __init__(self, num_states, num_actions, num_discrete_actions=0, discrete_actions_sections: Tuple = (0,),
                 action_log_std=0, use_multivariate_distribution=False,
                 num_hiddens: Tuple = (64, 64), activation: str = "relu",
                 drop_rate=None):
        """
        Deal with hybrid of discrete actions and continuous actions,
        if there's discrete actions, we put discrete actions at left, and continuous actions on the right.
        That's say, each action is arranged by the follow form :
            action = [discrete_action, continuous_action]
        :param num_states:
        :param num_actions:
        :param num_discrete_actions: OneHot encoded actions
        :param discrete_actions_sections:
        :param action_log_std:
        :param num_hiddens:
        :param activation:
        :param drop_rate:
        """
        super(Actor, self).__init__()
        # set up state space and action space
        self.num_states = num_states
        self.num_actions = num_actions
        self.drop_rate = drop_rate
        self.use_multivariate_distribution = use_multivariate_distribution
        # set up discrete action info
        self.num_discrete_actions = num_discrete_actions
        assert sum(discrete_actions_sections) == num_discrete_actions, f"Expected sum of discrete actions's " \
                                                                       f"dimension =  {num_discrete_actions}"
        self.discrete_action_sections = discrete_actions_sections

        # set up continuous action log_std
        self.action_log_std = nn.Parameter(torch.ones(1, self.num_actions - self.num_discrete_actions) * action_log_std,
                                           requires_grad=True)

        # set up module units
        _module_units = [num_states]
        _module_units.extend(num_hiddens)
        _module_units += num_actions,

        self._layers_units = [(_module_units[i], _module_units[i + 1]) for i in range(len(_module_units) - 1)]
        activation = resolve_activate_function(activation)

        # set up module layers
        self._module_list = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units):
            n_units_in, n_units_out = module_unit
            self._module_list.add_module(f"Layer_{idx + 1}_Linear", nn.Linear(n_units_in, n_units_out))
            if idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Activation", activation())
                self._module_list.add_module(f"Layer_{idx + 1}_LayerNorm", nn.LayerNorm(n_units_out))
            if self.drop_rate and idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Dropout", nn.Dropout(self.drop_rate))

        self._module_list.add_module(f"Layer_{idx + 1}_Activation", nn.Tanh())
        # if there's discrete actions, add custom Soft Max layer
        if self.num_discrete_actions:
            self._module_list.add_module(f"Layer_{idx + 1}_Custom_Softmax",
                                         MultiSoftMax(0, self.num_discrete_actions, self.discrete_action_sections))

    def forward(self, x):
        """
        give states, calculate the distribution of actions
        :param x: unsqueezed states
        :return: xxx
        """
        for module in self._module_list:
            x = module(x)
        # note that x include [discrete_action_softmax probability, continuous_action_mean]
        # extract discrete_action probs
        dist_discrete = None
        if self.num_discrete_actions:
            dist_discrete = MultiOneHotCategorical(x[..., :self.num_discrete_actions],
                                                   sections=self.discrete_action_sections)
        continuous_action_mean = x[..., self.num_discrete_actions:]
        continuous_action_log_std = self.action_log_std.expand_as(x[..., self.num_discrete_actions:])
        continuous_action_std = torch.exp(continuous_action_log_std)

        if self.use_multivariate_distribution:
            dist_continuous = MultivariateNormal(continuous_action_mean, torch.diag_embed(continuous_action_std))
        else:
            dist_continuous = Normal(continuous_action_mean, continuous_action_std)

        return dist_discrete, dist_continuous

    def get_action_log_prob(self, states):
        """
        give states, select actions based on the distribution
        and calculate the log probability of actions
        :param states: unsqueezed states
        :param actions: unsqueezed actions
        :return: actions and log probablities
        """
        dist_discrete, dist_continuous = self.forward(states)
        action = dist_continuous.sample()  # [batch_size, num_actions]
        if self.use_multivariate_distribution:
            log_prob = dist_continuous.log_prob(action)  # use multivariate normal distribution
        else:
            log_prob = dist_continuous.log_prob(action).sum(dim=-1)  # [batch_size]

        if dist_discrete:
            discrete_action = dist_discrete.sample()
            discrete_log_prob = dist_discrete.log_prob(discrete_action)  # [batch_size]
            action = torch.cat([discrete_action, action], dim=-1)

            """
            How to deal with log prob?

            1. Add discrete log_prob and continuous log_prob, consider their dependency;
            2. Concat them together
            """
            log_prob = (log_prob + discrete_log_prob)  # add log prob [batch_size, 1]
            # log_prob = torch.cat([discrete_log_prob, log_prob], dim=-1)  # concat [batch_size, 2]

        log_prob.unsqueeze_(-1)
        return action, log_prob  # log_prob [batch_size, 1/2]

    def get_log_prob(self, states, actions):
        """
        give states and actions, calculate the log probability
        :param states: unsqueezed states
        :param actions: unsqueezed actions
        :return: log probabilities
        """
        dist_discrete, dist_continuous = self.forward(states)
        if self.use_multivariate_distribution:
            log_prob = dist_continuous.log_prob(actions[..., self.num_discrete_actions:])
        else:
            log_prob = dist_continuous.log_prob(actions[..., self.num_discrete_actions:]).sum(dim=-1)
        if dist_discrete:
            discrete_log_prob = dist_discrete.log_prob(actions[..., :self.num_discrete_actions])
            log_prob = log_prob + discrete_log_prob
        return log_prob.unsqueeze(-1)

    def get_entropy(self, states):
        """
        give states, calculate the entropy of actions' distribution
        :param states: unsqueezed states
        :return: mean entropy
        """
        dist_discrete, dist_continuous = self.forward(states)
        ent_discrete = dist_discrete.entropy()
        ent_continuous = dist_continuous.entropy()

        ent = torch.cat([ent_discrete, ent_continuous], dim=-1).unsqueeze_(-1)  # [batch_size, 2]
        return ent

    def get_kl(self, states):
        """
        give states, calculate the KL_Divergence of actions' distribution
        :param states: unsqueezed states
        :return: mean kl divergence
        """
        pass


class JointPolicy(nn.Module):
    """
    Joint Policy include:
    agent policy: (agent_state,) -> agent_action
    env policy: (agent_state, agent_action) -> agent_next_state
    """

    def __init__(self, initial_state, config=None):
        super(JointPolicy, self).__init__()
        self.config = config
        self.trajectory_length = config["trajectory_length"]
        self.agent_policy = Actor(num_states=self.config["agent"]["num_states"],
                                  num_actions=self.config["agent"]["num_actions"],
                                  num_discrete_actions=self.config["agent"]["num_discrete_actions"],
                                  discrete_actions_sections=self.config["agent"]["discrete_actions_sections"],
                                  action_log_std=self.config["agent"]["action_log_std"],
                                  use_multivariate_distribution=self.config["agent"]["use_multivariate_distribution"],
                                  num_hiddens=self.config["agent"]["num_hiddens"],
                                  drop_rate=self.config["agent"]["drop_rate"],
                                  activation=self.config["agent"]["activation"])

        self.env_policy = Actor(num_states=self.config["env"]["num_states"],
                                num_actions=self.config["env"]["num_actions"],
                                num_discrete_actions=self.config["env"]["num_discrete_actions"],
                                discrete_actions_sections=self.config["env"]["discrete_actions_sections"],
                                action_log_std=self.config["env"]["action_log_std"],
                                use_multivariate_distribution=self.config["env"]["use_multivariate_distribution"],
                                num_hiddens=self.config["env"]["num_hiddens"],
                                drop_rate=self.config["env"]["drop_rate"],
                                activation=self.config["env"]["activation"])

        # Joint policy generate trajectories sampling initial state from expert data
        self.initial_agent_state = initial_state

    def collect_samples(self, batch_size):
        """
        generate trajectories following current policy
        accelerate by parallel the process
        :param batch_size:
        :return:
        """
        memory = Memory()
        parallelize_size = (batch_size + self.trajectory_length - 1) // self.trajectory_length
        agent_state = self.initial_agent_state[torch.randint(self.initial_agent_state.shape[0], (
            parallelize_size,))]  # agent_state [parallelize_size, num_states]
        for i in range(1, self.trajectory_length + 1):
            with torch.no_grad():
                agent_action, agent_action_log_prob = self.agent_policy.get_action_log_prob(
                    agent_state if len(agent_state.shape) > 1 else agent_state.unsqueeze(-1))
                # agent_action [parallelize_size, num_actions], agent_action_log_prob [parallelize_size, 1]
                env_state = torch.cat([agent_state, agent_action],
                                      dim=-1)  # env_state [parallelize_size, num_states + num_actions]
                env_action, env_action_log_prob = self.env_policy.get_action_log_prob(
                    env_state if len(env_state.shape) > 1 else env_state.unsqueeze(
                        -1))  # env_action [parallelize_size, num_states], env_action_log_prob [parallelize_size, 1]

            assert agent_action_log_prob.shape == env_action_log_prob.shape, "Expected agent_policy log_prob and env_" \
                                                                             "policy log_prob with same size!!!"

            mask = torch.ones_like(env_action_log_prob) if i % self.trajectory_length == 0 else torch.zeros_like(
                env_action_log_prob)

            memory.push(agent_state, agent_action, env_action, agent_action_log_prob + env_action_log_prob, mask)

        return memory.sample()

    def get_log_prob(self, states, actions, next_states):
        agent_action_log_prob = self.agent_policy.get_log_prob(states, actions)
        env_states = torch.cat([states, actions], dim=1)
        env_action_log_prob = self.env_policy.get_log_prob(env_states, next_states)

        return agent_action_log_prob + env_action_log_prob

    def get_next_state(self, states, actions):
        state_actions = torch.cat([states, actions], dim=-1)
        next_state, _ = self.env_policy.get_action_log_prob(state_actions)
        return next_state


class Value(nn.Module):
    def __init__(self, num_states, num_hiddens: Tuple = (64, 64), activation: str = "relu",
                 drop_rate=None):
        super(Value, self).__init__()
        # set up state space and action space
        self.num_states = num_states
        self.drop_rate = drop_rate
        self.num_value = 1
        # set up module units
        _module_units = [num_states]
        _module_units.extend(num_hiddens)
        _module_units += self.num_value,

        self._layers_units = [(_module_units[i], _module_units[i + 1]) for i in range(len(_module_units) - 1)]
        activation = resolve_activate_function(activation)

        # set up module layers
        self._module_list = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units):
            n_units_in, n_units_out = module_unit
            self._module_list.add_module(f"Layer_{idx + 1}_Linear", nn.Linear(n_units_in, n_units_out))
            if idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Activation", activation())
                self._module_list.add_module(f"Layer_{idx + 1}_LayerNorm", nn.LayerNorm(n_units_out))
            if self.drop_rate and idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Dropout", nn.Dropout(self.drop_rate))

    def forward(self, *args):
        """
        give states, calculate the estimated values
        :param x: unsqueezed states
        :return: values
        """
        x = torch.cat(args, -1)
        for module in self._module_list:
            x = module(x)
        return x




class Discriminator(nn.Module):
    def __init__(self, num_states, num_actions, num_hiddens: Tuple = (64, 64), activation: str = "relu",
                 drop_rate=None, use_noise=False, noise_std=0.1):
        super(Discriminator, self).__init__()
        # set up state space and action space
        self.num_states = num_states
        self.num_actions = num_actions
        self.drop_rate = drop_rate
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.num_value = 1
        # set up module units
        _module_units = [num_states + num_actions]
        _module_units.extend(num_hiddens)
        _module_units += self.num_value,

        self._layers_units = [(_module_units[i], _module_units[i + 1]) for i in range(len(_module_units) - 1)]
        activation = resolve_activate_function(activation)

        # set up module layers
        self._module_list = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units):
            self._module_list.add_module(f"Layer_{idx + 1}_Linear", nn.Linear(*module_unit))
            if idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Activation", activation())
            if self.drop_rate and idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Dropout", nn.Dropout(self.drop_rate))
        self._module_list.add_module(f"Layer_{idx + 1}_Activation", nn.Sigmoid())

    def forward(self, states, actions):
        """
        give states, calculate the estimated values
        :param states: unsqueezed states
        :param actions: unsqueezed actions
        :return: values
        """
        x = torch.cat([states, actions], dim=-1)
        if self.use_noise:  # trick: add gaussian noise to discriminator
            x += torch.normal(0, self.noise_std, size=x.shape, device=device)
        for module in self._module_list:
            x = module(x)
        return x
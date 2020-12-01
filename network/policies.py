import torch.nn as nn
from misc.utils import weight_init

class Policy(nn.Module):
    def __init__(self, ob_shape, act_shape):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(ob_shape, 128),
            nn.Linear(128, 128),
            nn.Linear(128, act_shape))

    def forward(self, obs):
        return self.policy(obs)

class ValueFunction(nn.Module):
    def __init__(self, all_ob_shape):
        super(ValueFunction, self).__init__()
        self.value_function = nn.Sequential(
            nn.Linear(all_ob_shape, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 1))

    def forward(self, obs):
        return self.value_function(obs)

class CategoricalPolicy(object):
    def __init__(self, ob_space, ac_space, ob_spaces, ac_spaces, nenv, nsteps, nstack):
        super(CategoricalPolicy, self).__init__()
        nbatch = nenv * nsteps
        nact = ac_space.n
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        all_ac_shape = (nbatch, (sum([ac.n for ac in ac_spaces]) - nact) * nstack)
        value_function = ValueFunction(all_ob_shape)
        policy = Policy(ob_shape, nact)
        self.apply(weight_init)




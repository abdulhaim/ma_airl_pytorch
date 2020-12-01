class ReplayMemory():
    def __init__(self, args):
        self.args = args

        self.obs = []
        self.logprobs = [[] for _ in range(args.n_agent)]
        self.reward = []
        self.entropies = [[] for _ in range(args.n_agent)]

    def add(self, obs, logprobs, reward, entropies):
        self.obs.append(obs)

        for i_agent in range(self.args.n_agent):
            self.logprobs[i_agent].append(logprobs[i_agent])

        self.reward.append(reward)

        for i_agent in range(self.args.n_agent):
            self.entropies[i_agent].append(entropies[i_agent])

    def sample(self):
        return self.obs, self.logprobs, self.reward, self.entropies


#!/usr/bin/env python
# Created at 2020/3/12
from collections import namedtuple
from random import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'log_prob', 'mask'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    def clear(self):
        del self.memory[:]
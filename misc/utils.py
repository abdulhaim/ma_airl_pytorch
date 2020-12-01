import git
import logging
import torch
import numpy as np
import torch.nn as nn
from misc.replay_memory import ReplayMemory
import make_env

#################################################################################
# LOGGING
#################################################################################
def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args, path="."):
    log = {}
    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    repo = git.Repo(path)
    log[args.log_name].info("Branch: {}".format(repo.active_branch))
    log[args.log_name].info("Commit: {}".format(repo.head.commit))

    return log


#################################################################################
# ENV
#################################################################################
def create_env(args):
    def _create_env():
        env = make_env.make_env(args.env_name)
        return env
    return _create_env


def save_checkpoint(agent, args, iteration):
    path = \
        "./pytorch_models/" + \
        "network::" + agent.args.network_type + "_" + \
        "iteration::" + str(iteration) + ".pth"
    checkpoint = {'actor_state_dict': agent.actor.state_dict()}
    torch.save(checkpoint, path)


def load_checkpoint(agent, opponent_model):
    path = "./pytorch_models/" + opponent_model
    checkpoint = torch.load(path)["actor_state_dict"]
    weight = {}
    for name, param in checkpoint.items():
        name = name.replace("training_agent0", "random_agent1")
        weight[name] = param
    agent.actor.load_state_dict(weight)


def collect_trajectory(agents, actors, envs, args, vis=False):
    memory = ReplayMemory(args)
    score = 0.
    observations = envs.reset()  # Shape: (traj_batch_size, 2, 120)

    if args.network_type == "lstm":
        for agent in agents:
            agent.reset_lstm_state()

    for timestep in range(args.ep_max_timesteps):
        if vis:
            envs.render()

        # Get actions
        observations = torch.FloatTensor(observations)

        actions, logprobs, entropies = [], [], []
        for i_agent, agent, actor in zip(range(len(agents)), agents, actors):
            obs = observations[:, i_agent, :]
            action, logprob, entropy = agent.act(obs, actor)
            actions.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)

        # Take step in the environment
        actions = np.stack(actions, axis=1)
        next_observations, rewards, done, _ = envs.step(actions)
        # Add to memory
        memory.add(
            obs=observations[:, 0],
            logprob=logprobs[0],
            reward=torch.FloatTensor(rewards)[:, 0],
            entropy=entropies[0],
            done=done[:, 0])

        # For next timestep
        observations = next_observations

    return memory


def get_return(reward, mask, args):
    reward = torch.stack(reward, dim=1) * mask

    R, return_ = 0., []
    for timestep in reversed(range(reward.shape[-1])):
        R = reward[:, timestep] + args.discount * R
        return_.insert(0, R)
    return_ = torch.stack(return_, dim=1)
    assert reward.shape == return_.shape

    return return_


#################################################################################
# TORCH
#################################################################################
def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        module.bias.data.zero_()

    if isinstance(module, nn.LSTMCell):
        nn.init.xavier_normal_(module.weight_ih)
        nn.init.xavier_normal_(module.weight_hh)
        module.bias_ih.data.zero_()
        module.bias_hh.data.zero_()

#################################################################################
# NUMPY
#################################################################################
def onehot(value, depth):
    a = np.zeros([depth])
    a[value] = 1
    return a

#################################################################################
# RL UTILS
#################################################################################
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


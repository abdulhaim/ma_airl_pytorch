#!/usr/bin/env python
import math
import multiprocessing
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.torch_utils import FLOAT, device

from network.policy import JointPolicy, Value, Discriminator
from misc.dataset import ExpertDataSet
from utils.torch_utils import device, to_device

trans_shape_func = lambda x: x.reshape(x.shape[0] * x.shape[1], -1)

class MAIRL:
    def __init__(self, args, env, policy_fn, tb_writer, log):
        self.exp_name = args.log_name

        self._load_expert_data()
        self._init_model()

    def _load_expert_data(self):
        num_expert_states = self.config["general"]["num_states"]
        num_expert_actions = self.config["general"]["num_actions"]
        expert_batch_size = self.config["general"]["expert_batch_size"]

        self.expert_dataset = ExpertDataSet(data_set_path=self.config["general"]["expert_data_path"],
                                            num_states=num_expert_states,
                                            num_actions=num_expert_actions)

        self.expert_data_loader = DataLoader(dataset=self.expert_dataset,
                                             batch_size=expert_batch_size,
                                             shuffle=True,
                                             num_workers=multiprocessing.cpu_count() // 2)

    def _init_model(self):
        self.V = Value(num_states=self.config["value"]["num_states"],
                       num_hiddens=self.config["value"]["num_hiddens"],
                       drop_rate=self.config["value"]["drop_rate"],
                       activation=self.config["value"]["activation"])
        self.P = JointPolicy(initial_state=self.expert_dataset.state.to(device),
                             config=self.config["jointpolicy"])
        self.D = Discriminator(num_states=self.config["discriminator"]["num_states"],
                               num_actions=self.config["discriminator"]["num_actions"],
                               num_hiddens=self.config["discriminator"]["num_hiddens"],
                               drop_rate=self.config["discriminator"]["drop_rate"],
                               use_noise=self.config["discriminator"]["use_noise"],
                               noise_std=self.config["discriminator"]["noise_std"],
                               activation=self.config["discriminator"]["activation"])

        print("Model Structure")
        print(self.P)
        print(self.V)
        print(self.D)
        print()

        self.optimizer_policy = optim.Adam(self.P.parameters(), lr=self.config["jointpolicy"]["learning_rate"])
        self.optimizer_value = optim.Adam(self.V.parameters(), lr=self.config["value"]["learning_rate"])
        self.optimizer_discriminator = optim.Adam(self.D.parameters(), lr=self.config["discriminator"]["learning_rate"])
        self.scheduler_discriminator = optim.lr_scheduler.StepLR(self.optimizer_discriminator,
                                                                 step_size=2000,
                                                                 gamma=0.95)

        self.discriminator_func = nn.BCELoss()

        to_device(self.V, self.P, self.D, self.D, self.discriminator_func)

    def estimate_advantages(self, rewards, masks, values, gamma, tau, trajectory_length):
        """
        General advantage estimate
        :param rewards: [trajectory length * parallel size, 1]
        :param masks: [trajectory length * parallel size, 1]
        :param values: [trajectory length * parallel size, 1]
        :param gamma:
        :param tau:
        :param trajectory_length: the length of trajectory
        :return:
        """
        trans_shape_func = lambda x: x.reshape(trajectory_length, -1, 1)
        rewards = trans_shape_func(rewards)  # [trajectory length, parallel size, 1]
        masks = trans_shape_func(masks)  # [trajectory length, parallel size, 1]
        values = trans_shape_func(values)  # [trajectory length, parallel size, 1]

        deltas = FLOAT(rewards.size()).to(device)
        advantages = FLOAT(rewards.size()).to(device)

        # calculate advantages in parallel
        prev_value = torch.zeros((rewards.size(1), 1), device=device)
        prev_advantage = torch.zeros((rewards.size(1), 1), device=device)

        for i in reversed(range(rewards.size(0))):
            deltas[i, ...] = rewards[i, ...] + gamma * prev_value * masks[i, ...] - values[i, ...]
            advantages[i, ...] = deltas[i, ...] + gamma * tau * prev_advantage * masks[i, ...]

            prev_value = values[i, ...]
            prev_advantage = advantages[i, ...]

        returns = values + advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # reverse shape for ppo
        return advantages.reshape(-1, 1), returns.reshape(-1, 1)  # [trajectory length * parallel size, 1]

    def ppo_step(self, policy_net, value_net, optimizer_p, optimizer_v, states, actions, next_states, returns, old_log_probs,
                 advantages, ppo_clip_ratio, value_l2_reg):
        # update value net
        values_pred = value_net(states)
        value_loss = nn.MSELoss()(values_pred, returns)

        for param in value_net.parameters():
            value_loss += value_l2_reg * param.pow(2).sum()

        optimizer_v.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 40)
        optimizer_v.step()

        # update policy net
        new_log_probs = policy_net.get_log_prob(states, actions, next_states)
        ratio = torch.exp(new_log_probs - old_log_probs)

        sur_loss1 = ratio * advantages
        sur_loss2 = torch.clamp(ratio, 1 - ppo_clip_ratio, 1 + ppo_clip_ratio) * advantages
        policy_loss = -torch.min(sur_loss1, sur_loss2).mean()

        optimizer_p.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
        optimizer_p.step()

        return value_loss, policy_loss

    def train(self, epoch):
        self.P.train()
        self.D.train()
        self.V.train()

        # collect generated batch
        gen_batch = self.P.collect_samples(self.config["ppo"]["sample_batch_size"])
        # batch: ('state', 'action', 'next_state', 'log_prob', 'mask')
        gen_batch_state = trans_shape_func(
            torch.stack(gen_batch.state))  # [trajectory length * parallel size, state size]
        gen_batch_action = trans_shape_func(
            torch.stack(gen_batch.action))  # [trajectory length * parallel size, action size]
        gen_batch_next_state = trans_shape_func(
            torch.stack(gen_batch.next_state))  # [trajectory length * parallel size, state size]
        gen_batch_old_log_prob = trans_shape_func(
            torch.stack(gen_batch.log_prob))  # [trajectory length * parallel size, 1]
        gen_batch_mask = trans_shape_func(torch.stack(gen_batch.mask))  # [trajectory length * parallel size, 1]

        # grad_collect_func = lambda d: torch.cat([grad.view(-1) for grad in torch.autograd.grad(d, self.D.parameters(), retain_graph=True)]).unsqueeze(0)
        ####################################################
        # update discriminator
        ####################################################
        for expert_batch_state, expert_batch_action in self.expert_data_loader:
            gen_r = self.D(gen_batch_state, gen_batch_action)
            expert_r = self.D(expert_batch_state.to(device), expert_batch_action.to(device))

            # label smoothing for discriminator
            expert_labels = torch.ones_like(expert_r)
            gen_labels = torch.zeros_like(gen_r)

            if self.config["discriminator"]["use_label_smoothing"]:
                smoothing_rate = self.config["discriminator"]["label_smooth_rate"]
                expert_labels *= (1 - smoothing_rate)
                gen_labels += torch.ones_like(gen_r) * smoothing_rate

            e_loss = self.discriminator_func(expert_r, expert_labels)
            g_loss = self.discriminator_func(gen_r, gen_labels)
            d_loss = e_loss + g_loss

            # """ WGAN with Gradient Penalty"""
            # d_loss = gen_r.mean() - expert_r.mean()
            # differences_batch_state = gen_batch_state[:expert_batch_state.size(0)] - expert_batch_state
            # differences_batch_action = gen_batch_action[:expert_batch_action.size(0)] - expert_batch_action
            # alpha = torch.rand(expert_batch_state.size(0), 1)
            # interpolates_batch_state = gen_batch_state[:expert_batch_state.size(0)] + (alpha * differences_batch_state)
            # interpolates_batch_action = gen_batch_action[:expert_batch_action.size(0)] + (alpha * differences_batch_action)
            # gradients = torch.cat([x for x in map(grad_collect_func, self.D(interpolates_batch_state, interpolates_batch_action))])
            # slopes = torch.norm(gradients, p=2, dim=-1)
            # gradient_penalty = torch.mean((slopes - 1.) ** 2)
            # d_loss += 10 * gradient_penalty

            self.optimizer_discriminator.zero_grad()
            d_loss.backward()
            self.optimizer_discriminator.step()

            self.scheduler_discriminator.step()

        self.writer.add_scalar('train/loss/d_loss', d_loss.item(), epoch)
        self.writer.add_scalar("train/loss/e_loss", e_loss.item(), epoch)
        self.writer.add_scalar("train/loss/g_loss", g_loss.item(), epoch)
        self.writer.add_scalar('train/reward/expert_r', expert_r.mean().item(), epoch)
        self.writer.add_scalar('train/reward/gen_r', gen_r.mean().item(), epoch)

        with torch.no_grad():
            gen_batch_value = self.V(gen_batch_state)
            gen_batch_reward = self.D(gen_batch_state, gen_batch_action)

        gen_batch_advantage, gen_batch_return = estimate_advantages(gen_batch_reward, gen_batch_mask,
                                                                    gen_batch_value, self.config["gae"]["gamma"],
                                                                    self.config["gae"]["tau"],
                                                                    self.config["jointpolicy"]["trajectory_length"])

        ####################################################
        # update policy by ppo [mini_batch]
        ####################################################
        ppo_optim_epochs = self.config["ppo"]["ppo_optim_epochs"]
        ppo_mini_batch_size = self.config["ppo"]["ppo_mini_batch_size"]
        gen_batch_size = gen_batch_state.shape[0]
        optim_iter_num = int(math.ceil(gen_batch_size / ppo_mini_batch_size))

        for _ in range(ppo_optim_epochs):
            perm = torch.randperm(gen_batch_size)

            for i in range(optim_iter_num):
                ind = perm[slice(i * ppo_mini_batch_size,
                                 min((i + 1) * ppo_mini_batch_size, gen_batch_size))]
                mini_batch_state, mini_batch_action, mini_batch_next_state, mini_batch_advantage, mini_batch_return, \
                mini_batch_old_log_prob = gen_batch_state[ind], gen_batch_action[ind], gen_batch_next_state[ind], \
                                          gen_batch_advantage[ind], gen_batch_return[ind], gen_batch_old_log_prob[ind]

                v_loss, p_loss = ppo_step(self.P, self.V, self.optimizer_policy, self.optimizer_value,
                                          states=mini_batch_state,
                                          actions=mini_batch_action,
                                          next_states=mini_batch_next_state,
                                          returns=mini_batch_return,
                                          old_log_probs=mini_batch_old_log_prob,
                                          advantages=mini_batch_advantage,
                                          ppo_clip_ratio=self.config["ppo"]["clip_ratio"],
                                          value_l2_reg=self.config["value"]["l2_reg"])

                self.writer.add_scalar('train/loss/p_loss', p_loss, epoch)
                self.writer.add_scalar('train/loss/v_loss', v_loss, epoch)

        print(f" Training episode:{epoch} ".center(80, "#"))
        print('gen_r:', gen_r.mean().item())
        print('expert_r:', expert_r.mean().item())
        print('d_loss', d_loss.item())

    def eval(self, epoch):
        self.P.eval()
        self.D.eval()
        self.V.eval()

        gen_batch = self.P.collect_samples(self.config["ppo"]["sample_batch_size"])
        gen_batch_state = torch.stack(gen_batch.state)
        gen_batch_action = torch.stack(gen_batch.action)

        gen_r = self.D(gen_batch_state, gen_batch_action)
        for expert_batch_state, expert_batch_action in self.expert_data_loader:
            expert_r = self.D(expert_batch_state.to(device), expert_batch_action.to(device))

            print(f" Evaluating episode:{epoch} ".center(80, "-"))
            print('validate_gen_r:', gen_r.mean().item())
            print('validate_expert_r:', expert_r.mean().item())

        self.writer.add_scalar("validate/reward/gen_r", gen_r.mean().item(), epoch)
        self.writer.add_scalar("validate/reward/expert_r", expert_r.mean().item(), epoch)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # dump model from pkl file
        # torch.save((self.D, self.P, self.V), f"{save_path}/{self.exp_name}.pt")
        torch.save(self.D, f"{save_path}/{self.exp_name}_Discriminator.pt")
        torch.save(self.P, f"{save_path}/{self.exp_name}_JointPolicy.pt")
        torch.save(self.V, f"{save_path}/{self.exp_name}_Value.pt")

    def load_model(self, model_path):
        # load entire model
        # self.D, self.P, self.V = torch.load((self.D, self.P, self.V), f"{save_path}/{self.exp_name}.pt")
        self.D = torch.load(f"{model_path}_Discriminator.pt", map_location=device)
        self.P = torch.load(f"{model_path}_JointPolicy.pt", map_location=device)
        self.V = torch.load(f"{model_path}_Value.pt", map_location=device)

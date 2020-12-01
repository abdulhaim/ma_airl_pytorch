import torch
from meta.dice import get_dice_loss
from meta.base import Base
import numpy as np

class Agent(Base):
    def __init__(self, env, log, tb_writer, args, name, i_agent):
        super(Agent, self).__init__(env, log, tb_writer, args, name, i_agent)

        self._set_dim()
        self._set_linear_baseline()
        self._set_policy()

    def inner_update(self, memory, iteration):
        # Sample experiences from memory
        obs_features, logprobs, reward, mask = memory.sample()

        # Compute value for baseline
        value = self.linear_baseline(obs_features, reward, mask)

        # Get masked reward
        reward_stacked = np.stack(reward, axis=1)
        masked_reward = reward_stacked * mask.numpy()

        # Log return
        return_ = 0.
        ep_max_timestep = masked_reward.shape[1]

        for timestep in reversed(range(ep_max_timestep)):
            return_ = masked_reward[:, timestep] + self.args.discount * return_

        return_ = np.average(return_)

        # Get actor grad and update
        actor_loss = get_dice_loss(logprobs, reward, value, mask, self.args)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Log performance
        self.log[self.args.log_name].info("Returns: {:.3f} at iteration {}".format(return_, iteration))

        self.tb_writer.add_scalar("/train_reward", return_, iteration)
        self.tb_writer.add_scalars(
            "/loss/inner_actor_loss", {str(self.i_agent): actor_loss.data.numpy()}, iteration)

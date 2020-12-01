import os
import argparse
import torch
import random
import numpy as np
from misc.utils import set_log, create_env
from tensorboardX import SummaryWriter
from gym_env.multiprocessing_env import SubprocVecEnv
from network.policies import CategoricalPolicy


def main(args):
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logs
    log = set_log(args)
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))

    # Create env
    env = SubprocVecEnv([create_env(args) for _ in range(args.traj_batch_size)])
    policy_fn = CategoricalPolicy

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    if args.generate_expert:
        from generate_expert import generate_expert
        generate_expert(env, policy_fn, log, tb_writer, args)
    else:
        from trainer import MAIRL
        mail = MAIRL(args, env, policy_fn, tb_writer, log)
        mail.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm
    parser.add_argument("--lr", type=int, default=0.1, help="lrs")
    parser.add_argument("--traj-batch-size", type=int, default=64, help="Batch size for both actor and critic")
    parser.add_argument("--batch-size", type=int, default=1000, help="batch-size")
    parser.add_argument("--traj-limitation", type=int, default=200, help="limitation")
    parser.add_argument("--ret-threshold", type=int, default=10, help="threshold")
    parser.add_argument("--dis_lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--disc_type", type=str, default="decentralized", help="disc_type")
    parser.add_argument("--bc_iters", type=int, default=500, help="bc_iters")
    parser.add_argument("--l2", type=float, default=0.1, help="l2")
    parser.add_argument("--d_iters", type=float, default=1, help="d_iters")
    parser.add_argument("--rew_scale", type=float, default=0, help="rew_scale")
    parser.add_argument("--total-timesteps", type=int, default=5e7, help="timesteps")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
    parser.add_argument("--lam", type=float, default=0.95, help="lam")
    parser.add_argument("--nprocs", type=int, default=32, help="nprocs")
    parser.add_argument("--nsteps", type=int, default=20, help="nsteps")
    parser.add_argument("--nstack", type=int, default=1, help="nstack")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="ent_coef")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="vf_coef")
    parser.add_argument("--vf_fisher_coef", type=float, default=1.0, help="vf_fisher_coef")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max_grad_norm")
    parser.add_argument("--kfac_clip", type=float, default=0.001, help="kfac_clip")
    parser.add_argument("--save_interval", type=float, default=100, help="save_interval")
    parser.add_argument("--lrschedule", type=str, default="linear", help="save_interval")
    parser.add_argument("--timesteps_per_batch", type=int, default=1000, help="timesteps_per_batch")
    parser.add_argument("--ncpus", type=int, default=4, help="ncpus")

    # Env
    parser.add_argument("--env-name", type=str, default="simple", help="OpenAI gym environment name")
    parser.add_argument("--expert-path", type=str, default="path-needed", help="simple-spread checkpoint path")

    # Misc
    parser.add_argument("--seed", type=int, default=1, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--log-name", type=str, default="trial_1", help="Prefix for tb_writer and logging")
    parser.add_argument("--generate-expert", type=bool, default=False, help="whether to run ail or generate trajectories")

    args = parser.parse_args()
    main(args=args)

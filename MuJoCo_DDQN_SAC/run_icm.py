import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac_icm import *
from torch.utils.tensorboard import SummaryWriter
from memory import ReplayMemory
from utils import create_log_dir, multi_step_reward
from collections import deque


import random
#plot figure
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
import sparseMuJoCo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--noisy', type=bool, default=False,
                        help='whether add noise to the parameter of Q network (default: False)')
    parser.add_argument('--multi-step', type=int, default=1,
                        help='N-Step Learning (default: 1)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--sigma-init', type=float, default=0.4,
                        help='Sigma initialization value for NoisyNet')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=22, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--logdir', default='runs/',
                        help='the folder that store log info')
    parser.add_argument('--repeat', type=int,  default=0,
                        help='the index of same parameter')

    parser.add_argument('--riclamp', type=float, default='5',
                         help='the predictor ri_clamp')
    parser.add_argument('--filename', default="z.csv",
                        help='save csv files')

    args = parser.parse_args()
    if args.env_name == "Hopper-v2":
        args.num_steps = 1000000
    return args


def set_seed(seed=22):
    """Set the random seed or get one if it is not given"""
    # if not seed:
    #     files = os.listdir('runs/')
    #     if not files:
    #         seed = 0
    #     else:
    #         seed = max([int(f.split('seed=')[1][0]) for f in files]) + 1
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def train():
    args = get_args()
    print(args.seed, ' ', args.env_name, ' ', args.filename, ' ', args.automatic_entropy_tuning)
    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args.env_name)
    
#seed 
    cur_seed = set_seed(args.seed)
    env.seed(cur_seed)
    count_reward = 0
    # Agent

    agent = SAC(env.observation_space.shape[0], env.action_space, args)
   # optimizer = optim.Adam(policy_net.parameters(), lr=lr)


    # Tesnorboard
    writer = SummaryWriter(create_log_dir(args))
    # Memory
    memory = ReplayMemory(args.replay_size)
    # Training Loop
    total_numsteps = 0
    updates = 0
    state_deque = deque(maxlen=args.multi_step)
    reward_deque = deque(maxlen=args.multi_step)
    action_deque = deque(maxlen=args.multi_step)

    reward_list = []
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        if args.noisy:
            agent.critic.sample_noise()
            agent.critic_target.sample_noise()
        while not done:
            if args.start_steps > total_numsteps:
               # action = env.action_space.sample()  # Sample random action
               action = np.random.rand(action_dim) * max_action * 2 - max_action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         args.batch_size,
                                                                                                         updates,)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            if reward != 0:
                count_reward = count_reward + 1
            state_deque.append(state)
            reward_deque.append(reward)
            action_deque.append(action)

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            if (total_numsteps % 5000 == 0):
                eval_reward = eval_policy(agent, args.env_name, args.seed, eval_episodes=10)
                reward_list.append(eval_reward)
                pd.DataFrame(reward_list).to_csv(args.filename, index=False)
                print("count_reward = ", count_reward)
                # plt.plot(reward_list, c='r', alpha=0.3)
                # plt.plot(gaussian_filter1d(reward_list, sigma=5), c='r', label='Rewards')
                # plt.ylabel('Score')
                # plt.xlabel('Episode ')
                # plt.savefig('reward.png')
                # plt.show()

            if len(state_deque) == args.multi_step or done:
                n_reward = multi_step_reward(reward_deque, args.gamma)
                n_state = state_deque[0]
                n_action = action_deque[0]
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == env._max_episode_steps else float(not done)
                memory.push(n_state, n_action, n_reward, next_state, mask)  # Append transition to memory
            state = next_state

        state_deque.clear()
        reward_deque.clear()
        action_deque.clear()
        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        writer.add_scalar('train reward/step', episode_reward, updates)
        print("Train -- Total steps: {}, Episode: {}, Episode length: {}, Episode Reward: {}".format(total_numsteps, i_episode,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))

        # test
        # if i_episode % 10 == 0 and args.eval is True:
        #     avg_reward = 0.
        #     episodes = 10
        #     total_len = 0
        #     for _ in range(episodes):
        #         state = env.reset()
        #         episode_reward = 0
        #         done = False
        #         while not done:
        #             action = agent.select_action(state, evaluate=True)
        #
        #             next_state, reward, done, _ = env.step(action)
        #             episode_reward += reward
        #
        #             state = next_state
        #             total_len += 1
        #         avg_reward += episode_reward
        #     avg_reward /= episodes
        #
        #     writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        #     writer.add_scalar('test avg_reward/step', avg_reward, updates)
        #
        #     print("-----------------------------Test, ", args.env_name, "------------------------------")
        #     print("Test number of episodes: {}, Average episode length: {}, Average episode reward: {}".format(episodes,
        #                                                                      total_len/episodes, round(avg_reward, 2)))
        #     print("--------------------------------------------------------------------------")
        #     reward_list.append(avg_reward)
        #
        #     if i_episode % 10 == 0:
        #         pd.DataFrame(reward_list, columns=['Reward']).to_csv("reward.csv", index=False)
        #         # plt.plot(reward_list, c='r', alpha=0.3)
        #         # plt.plot(gaussian_filter1d(reward_list, sigma=5), c='r', label='Rewards')
        #         # plt.ylabel('Score')
        #         # plt.xlabel('Episode ')
        #         # plt.savefig('reward.png')
        #         # plt.show()

        ##test



    env.close()


def eval_policy(policy, env_name, seed, eval_episodes=10):
   # policy.eval()
    eval_env = gym.make(env_name)
    eval_env.seed(seed)
    avg_reward = 0.

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        t = 0
        while not done:

            action = policy.select_action(state, evaluate=True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            t = t + 1
            if t > 10000:
                break
    avg_reward /= eval_episodes
    print("----------------------------------------------")
    print("ten episodes ava_reward = ", avg_reward)
    print("----------------------------------------------")
    return avg_reward

if __name__ == '__main__':
    torch.set_num_threads(5)
    train()

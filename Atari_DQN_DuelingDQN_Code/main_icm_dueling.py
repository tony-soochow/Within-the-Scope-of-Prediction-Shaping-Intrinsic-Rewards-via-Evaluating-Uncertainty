# coding = gbk
import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time

import gym

from wrappers import *
from memory import ReplayMemory
from model_icm import DQN, ICMAgent, Dueling


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib.pyplot as plt

plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import argparse


Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))


# cling seed
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


def eval_policy(policy, env_name, seed, eval_episodes=10):
    policy.eval()
    eval_env = gym.make(env_name)
    eval_env = make_env(eval_env)
    eval_env.seed(seed)
    avg_reward = 0.

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        t = 0
        while not done:
            state = get_state(state)
            action = policy(state.to(args.device)).max(1)[1].view(1, 1)

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

def plot_figure(avg_reward_list):

     pd.DataFrame(avg_reward_list).to_csv(args.filename, index=False)
     # plt.plot(avg_reward_list, c='r', alpha=0.3)
     # plt.plot(gaussian_filter1d(avg_reward_list, sigma=5), c='r', label='Rewards')
     #
     # plt.ylabel('Average Score')
     # plt.xlabel('Step ')
     # plt.title("KungMaster")
     # plt.savefig("kungmaster_1.png")
     # plt.show()


def select_action(state):
    global steps_done

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(args.device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_size)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))

    #训练策略模型

    compute_reward_batch = batch
    ri = agent.compute_reward(compute_reward_batch)



    actions = tuple((map(lambda a: torch.tensor([[a]], device=args.device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=args.device), batch.reward)))



    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(args.device)

    state_batch = torch.cat(batch.state).to(args.device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    wait_cat = torch.tensor([0.]).to(args.device)
    ri = ri.reshape(31)


    ri = torch.cat([ri, wait_cat], dim = 0)
    reward_batch = reward_batch + ri  #放入训练的reward


    state_action_values = policy_net(state_batch).gather(1, action_batch)


    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def train(env, total_count, render=False):
    episode = 0
    obs = env.reset()
    state = get_state(obs)
    total_reward = 0.0
    t = 0
    for count in range(total_count):

            t += 1
            action = select_action(state)
            if render:
                env.render()

            obs, reward, done, info = env.step(action)

            total_reward += reward
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)
            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:  # > 1w开始优化模�?
              # 因此我们选择在这里加入我们的额外奖励
                #获取批量数据
                # 优化Policy and 计算好奇心奖励放入训�?
                optimize_model()
              #训练好奇心模�?
                transitions = memory.sample(BATCH_SIZE * 2)
                batch = Transition(*zip(*transitions))
                agent.train(batch)



                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            if done:
                if(episode % 20 == 0):
                  print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t,
                                                                                     total_reward))
                episode += 1
                obs = env.reset()
                state = get_state(obs)
                total_reward = 0.0
                t = 0
            if (count + 1) % eval_freq == 0:  #5000step plot figure
               avg_reward = eval_policy(policy_net, env_name, seed, eval_episodes=10)
               avg_reward_list.append(avg_reward)
               plot_figure(avg_reward_list)


    env.close()
    return



if __name__ == '__main__':

    #args
    parser = argparse.ArgumentParser(description='Pytorch value model Args')
    parser.add_argument('--riclimp', default=5, type=float)
    parser.add_argument('--filename', default='q.csv')
    parser.add_argument('--seed', type=int, default="1")
    parser.add_argument('--device', default="cpu")
    parser.add_argument("--Tcount", type=int, default="1000001")
    args = parser.parse_args()

    print(args.riclimp, " ", args.filename, " ", args.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    # create environment
    env_name = "PitfallNoFrameskip-v4"
    print(env_name)
    env = gym.make(env_name)
    action_size = env.action_space.n
    print("action_size = ", action_size)
    env = make_env(env)
    # cling seed
    seed_size = args.seed
    seed = set_seed(seed_size)
    env.seed(seed)

    # hyperparameters
    TOTAL_COUNT = args.Tcount
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 100000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY  #10w
    learning_rate = 0.01
    eval_freq = 5000

    # create networks

    avg_reward_list = []
    policy_net = Dueling(n_actions=action_size).to(device)
    target_net = Dueling(n_actions=action_size).to(device)

    # policy_net = DQN(n_actions=action_size).to(device)
    # target_net = DQN(n_actions=action_size).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    agent = ICMAgent(act_size=1, ri_climap=args.riclimp).to(device)

    #setup rewardmodel




    # extra model

    steps_done = 0

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)  # 10 * 开始使用更新的步数 = 10 * 1w = 10w

    # train model
    train(env, TOTAL_COUNT)
    torch.save(policy_net, "dqn_pong_model")
    policy_net = torch.load("dqn_pong_model")
# test(env, 1, policy_net, render=False)


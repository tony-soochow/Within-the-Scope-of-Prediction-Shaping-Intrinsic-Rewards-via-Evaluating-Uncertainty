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
from models import DQN, PredictorAgent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd


Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

#cling seed
def set_seed(seed = 22):
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


def eval_policy(policy, env_name, seed, eval_episodes = 10):

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
            action = policy(state.to('cuda')).max(1)[1].view(1,1)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            t = t + 1
            # if t > 1000:
            #     break
    avg_reward /= eval_episodes

    print("----------------------------------------------")
    print("ten episodes ava_reward = ", avg_reward)
    print("----------------------------------------------")
    return avg_reward


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

    
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
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
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

def train(env, n_episodes, render=False):
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
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

            if steps_done > INITIAL_MEMORY:  # > 1w开始优化模型
                #因此我们选择在这里加入我们的额外奖励


                optimize_model()
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            if done: #

                if( (episode + 1) % 5 == 0 ):
                   avg_reward = eval_policy(policy_net, env_name, 0, eval_episodes = 10)
                   for _ in range(5):
                     avg_reward_list.append(avg_reward)

                   #画图了
                   pd.DataFrame(avg_reward_list, columns=['Reward']).to_csv("test.csv", index=False)
                   plt.plot(avg_reward_list, c='r', alpha=0.3)
                   plt.plot(gaussian_filter1d(avg_reward_list, sigma=5), c='r', label='Rewards')
                   plt.ylabel('Average Score')
                   plt.xlabel('Episode ')
                   plt.title("test_Sparse Reward Environment")
                   plt.savefig("test.png")
                   plt.show()
                break
        if episode % 20 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
    env.close()
    return

# def test(env, n_episodes, policy, render=True):
#     env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
#     for episode in range(n_episodes):
#         obs = env.reset()
#         state = get_state(obs)
#         total_reward = 0.0
#         for t in count():
#             action = policy(state.to('cuda')).max(1)[1].view(1,1)
#
#             if render:
#                 env.render()
#                 time.sleep(0.02)
#
#             obs, reward, done, info = env.step(action)
#
#             total_reward += reward
#
#             if not done:
#                 next_state = get_state(obs)
#             else:
#                 next_state = None
#
#             state = next_state
#
#             if done:
#                 print("Finished Episode {} with reward {}".format(episode, total_reward))
#                 break
#
#     env.close()
#     return

if __name__ == '__main__':
    # set device



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create environment
    env_name = "BoxingNoFrameskip-v4"
    env = gym.make(env_name)
    action_size = env.action_space.n
    print("action_size = ", action_size)
    env = make_env(env)
  #cling seed
    seed = set_seed()
    env.seed(seed)

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    learning_rate = 0.01
    # create networks

    avg_reward_list = []
    policy_net = DQN(n_actions=action_size).to(device)
    target_net = DQN(n_actions=action_size).to(device)


    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # extra model


    steps_done = 0



    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)  #10 * 开始使用更新的步数 = 10 * 1w = 10w
    
    # train model
    train(env, 4000)
    torch.save(policy_net, "dqn_pong_model")
    policy_net = torch.load("dqn_pong_model")
   # test(env, 1, policy_net, render=False)


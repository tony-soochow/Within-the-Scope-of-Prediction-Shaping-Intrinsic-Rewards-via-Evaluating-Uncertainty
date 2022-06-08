import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import torch.nn as nn
import sparseMuJoCo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        # SAC(env.observation_space.shape[0], env.action_space, args)
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, args).to(device=device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.predictor = PredictorAgent(state_size=num_inputs, action_size=action_space.shape[0],
                                        ri_climp=args.riclamp).to(device)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, args).to(device)
        hard_update(self.critic_target, self.critic)

        if args.noisy:
            self.critic.update_noisy_modules()
            self.critic_target.update_noisy_modules()

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(device).unsqueeze(1)

        # 这是我需要改动的地方
        tmp_state_bathch = state_batch
        tmp_action_batch = action_batch
        tmp_reward_batch = reward_batch

        wait_cat = torch.tensor([[0.]]).to(device)
        ri = self.predictor.compute_reward(tmp_state_bathch, tmp_action_batch)

        ri = torch.cat([ri, wait_cat], dim=0)
        reward_batch = reward_batch + ri  # 放入训练的reward

        self.predictor.train(tmp_state_bathch, tmp_action_batch, tmp_reward_batch)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((
                                   self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


class ConvFeatureExtract(nn.Module):
    def __init__(self, state_size=4, hidden_size=256):
        super(ConvFeatureExtract, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DynamicsModel(nn.Module):
    def __init__(self, encoded_state_size=32, action_size=10):
        super(DynamicsModel, self).__init__()
        self.state_head = nn.Linear(encoded_state_size + action_size, encoded_state_size)
        self.fc1 = nn.Linear(encoded_state_size, encoded_state_size)
        self.fc2 = nn.Linear(encoded_state_size, encoded_state_size)

    def forward(self, state, action):
        next_state_pred = F.relu(self.state_head(torch.cat([state, action], 1)))
        x = F.relu(self.fc1(next_state_pred))

        return self.fc2(x)


class RewardModel(nn.Module):
    def __init__(self, state_size=32, hidden_size=32, action_size=1):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # 这里看看是1 还是 0
        hidden = F.relu(self.fc1(x))
        hidden = F.relu(self.fc2(hidden))
        reward = self.fc3(hidden)
        return reward

    def train(self):
        pass


class PredictorAgent(nn.Module):
    def __init__(self, state_size=4, action_size=10, encoded_state_size=32, ri_climp=10):
        super(PredictorAgent, self).__init__()
        # self.model = PredictorModel(num_input_channel, encoded_state_size)
        # print(state_size, action_size, "zzz")
        self.encoder = ConvFeatureExtract(state_size)
        self.prediction = DynamicsModel(action_size=action_size)
        self.rewardmodel = RewardModel(action_size=action_size)
        # 虽然没有直接训练编码器 但是间接训练了他的
        self.optimpre = torch.optim.Adam(self.prediction.parameters(), lr=1e-4)  # 先不去训练看看
        self.rewardoptim = torch.optim.Adam(self.rewardmodel.parameters(), lr=1e-4)

        self.ri_mean = self.ri_std = None
        self.ri_clamp = ri_climp
        self.ri_scale = 1

        self.beta = 0.01
        self.alpha = 0.3

    def compute_reward(self, state_batch, action_batch):  # batch 是整个经验池的数据 需要用来重新计算奖励

        # 这里放入的就是batch
        # batch = Transition(*zip(*batch)) #得到想要的数据

        state_batch = state_batch.float().to(device)  # 获取数据没问题  返回的总是 当前经 验池的len 4 84 84

        action_batch = action_batch.float().to(device)

        z = self.encoder(state_batch)  # 得到左右状态的潜在编码 接下来构造差值

        z_pred = self.prediction(z[: -1], action_batch[: -1])
        err = (z[1:] - z_pred).pow(2).mean(1)  # 得到好奇心奖励均值
        ri = err.detach()

        pre_reward = self.rewardmodel(z[: -1], action_batch[: -1])  # 这个好像一直都是正数有点奇怪啊

        pre_reward = pre_reward.squeeze(1)
        #  ri = ri + pre_reward

        self.ri_mean = ri.mean()
        self.ri_std = ri.std()
        ri = (ri[..., None] - self.ri_mean) / self.ri_std
        ri.clamp_(-self.ri_clamp, self.ri_clamp)

        self.ri_mean = pre_reward.mean()
        self.ri_std = pre_reward.std()
        pre_reward = (pre_reward[..., None] - self.ri_mean) / self.ri_std
        pre_reward.clamp_(-self.ri_clamp, self.ri_clamp)

        R = self.alpha * ri + (1 - self.alpha) * pre_reward

        return self.beta * R

    def train(self, state_batch, action_batch, reward_batch):
        # 这里直接用batch好像不可以

        # 90w - 100w在8000-8500之间

        state_batch = state_batch.to(device).float()
        action_batch = action_batch.to(device).float()
        reward_batch = reward_batch.to(device).float()

        z = self.encoder(state_batch)  # 得到左右状态的潜在编码 接下来构造差值

        z_pred = self.prediction(z[: -1], action_batch[: -1])

        # err = (z[1:] - z_pred).pow(2).mean(1) # 得到好奇心奖励均值
        # err = err.sum(0).mean()  #这里感觉需要看看了~~
        err = 0.5 * F.mse_loss(z[1:], z_pred)  # 这里改了 成了0.5

        # errloss
        self.optimpre.zero_grad()
        err.backward(retain_graph=True)  #
        self.optimpre.step()

        # rewardloss
        pre_reward = self.rewardmodel(z, action_batch)
        rewardloss = F.mse_loss(pre_reward, reward_batch)

        self.rewardoptim.zero_grad()
        rewardloss.backward()
        self.rewardoptim.step()



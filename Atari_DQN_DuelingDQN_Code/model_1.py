# coding = gbk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))


class DQNbn(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQNbn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)



class Dueling(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(Dueling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        # self.head = nn.Linear(512, n_actions)
        self.n_actions = n_actions

        self.value_stream = nn.Sequential(
            nn.Linear(512, 512),  # 128
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),  # 128
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
     ##   qvals = values + (advantages - advantages.mean())
        qvals = values + advantages - advantages.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)

        return qvals



class ConvFeatureExtract(nn.Module):
    def __init__(self, num_input_channel=4):
        super(ConvFeatureExtract, self).__init__()
        # self.conv1 = nn.Conv2d(num_input_channel, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)
        # # self.head = nn.Linear(448, 40 * 80 * 3) # todo change to dim 20 or so and embed everything
        # # self.head = nn.Linear(448, 128)
        # self.head = nn.Linear(2048, 512)
        # self.head2 = nn.Linear(512, 128) # todo maybe?

        self.conv1 = nn.Conv2d(num_input_channel, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        # 5184   7 * 7 * 64 = 3136
        self.head = nn.Linear(3136, 512)

        self.final = nn.Linear(512, 32)  # 这样绝对太簇五了

    def forward(self, x):
        # import pdb;pdb.set_trace()

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.head(x.view(x.size(0), -1)))
        return self.final(x.view(x.size(0), -1))


class DynamicsModel(nn.Module):
    def __init__(self, encoded_state_size):
        super(DynamicsModel, self).__init__()
        self.state_head = nn.Linear(encoded_state_size + 1, encoded_state_size)
        self.fc1 = nn.Linear(encoded_state_size, encoded_state_size)
        self.fc2 = nn.Linear(encoded_state_size, encoded_state_size)

    def forward(self, state, action):
        next_state_pred = F.relu(self.state_head(torch.cat([state, action], 1)))
        x = F.relu(self.fc1(next_state_pred))
        return self.fc2(x)


class PredictorModel(nn.Module):
    def __init__(self, num_input_channel=4, encoded_state_size=32):
        super(PredictorModel, self).__init__()
        self.featureextract = ConvFeatureExtract(num_input_channel)
        self.dynamicsmodel = DynamicsModel(encoded_state_size)

    def forward(self, state, action):
        z = self.featureextract(state)
        z_pred = self.dynamicsmodel(z, action)
        return z_pred


# reward prediction model -----> 作用增加奖励预测
class RewardModel(nn.Module):
    def __init__(self, state_size=32, hidden_size=32, action_size=1):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # 这里看看�? 还是 0
        hidden = F.relu(self.fc1(x))
        hidden = F.relu(self.fc2(hidden))
        reward = self.fc3(hidden)
        return reward

    def train(self):
        pass


class PredictorAgent(nn.Module):
    def __init__(self, num_input_channel=4, encoded_state_size=32, act_size=10, ri_climap=10):
        super(PredictorAgent, self).__init__()
        # self.model = PredictorModel(num_input_channel, encoded_state_size)

        self.encoder = ConvFeatureExtract(num_input_channel)
        self.prediction = DynamicsModel(encoded_state_size)
        self.rewardmodel = RewardModel(action_size=act_size)

        # 虽然没有直接训练编码�?但是间接训练了他�?
        self.optimpre = torch.optim.Adam(self.prediction.parameters(), lr=1e-4)  # 先不去训练看�?
        self.rewardoptim = torch.optim.Adam(self.rewardmodel.parameters(), lr=1e-4)

        self.ri_mean = self.ri_std = None
        self.ri_clamp = ri_climap
        print(self.ri_clamp)
        self.ri_scale = 1
        self.beta = 0.1
        self.alpha = 0.3


    def compute_reward(self, batch):  # batch 是整个经验池的数�?需要用来重新计算奖�?

        # 这里放入的就是batch
        # batch = Transition(*zip(*batch)) #得到想要的数�?

        state_batch = torch.cat(batch.state).to(device).float()  # 获取数据没问�? 返回的总是 当前�?验池的len 4 84 84

        action_batch = torch.cat(batch.action).to(device).float()

        # 这里用不用还要考虑一下呢�?
        # state_batch -> 32 4 84 84
        z = self.encoder(state_batch)  # 得到左右状态的潜在编码 接下来构造差�?

        z_pred = self.prediction(z[: -1], action_batch[: -1])
        err = (z[1:] - z_pred).pow(2).mean(1)  # 得到好奇心奖励均�?
        ri = err.detach()

        pre_reward = self.rewardmodel(z[: -1], action_batch[: -1])  # 这个好像一直都是正数有点奇怪啊
        pre_reward = pre_reward.squeeze(1)


        self.ri_mean = ri.mean()
        self.ri_std = ri.std()
        ri = (ri[..., None] - self.ri_mean) / self.ri_std
        ri.clamp_(-self.ri_clamp, self.ri_clamp)



        # ri = ri + pre_reward

        #修改后的版本
        self.ri_mean = pre_reward.mean()
        self.ri_std = pre_reward.std()
        pre_reward = (pre_reward[..., None] - self.ri_mean) / self.ri_std
        pre_reward.clamp_(-self.ri_clamp, self.ri_clamp)


        R = self.alpha * ri + (1 - self.alpha) * pre_reward # 0.4 0.6

        return self.beta * R             # 1.25

    def train(self, batch):
        # 这里直接用batch好像不可�?

        state_batch = torch.cat(batch.state).to(device).float()
        action_batch = torch.cat(batch.action).to(device).float()
        reward_batch = torch.cat(batch.reward).to(device).float()

        z = self.encoder(state_batch)  # 得到左右状态的潜在编码 接下来构造差�?

        z_pred = self.prediction(z[: -1], action_batch[: -1])

        # err = (z[1:] - z_pred).pow(2).mean(1) # 得到好奇心奖励均�?
        # err = err.sum(0).mean()  #这里感觉需要看看了~~
        err = F.mse_loss(z[1:], z_pred)

        # errloss
        self.optimpre.zero_grad()
        err.backward(retain_graph=True)  #
        self.optimpre.step()

        # rewardloss
        pre_reward = self.rewardmodel(z, action_batch)
        pre_reward = pre_reward.reshape(64)
        rewardloss = F.mse_loss(pre_reward, reward_batch)

        self.rewardoptim.zero_grad()
        rewardloss.backward()
        self.rewardoptim.step()

        pass




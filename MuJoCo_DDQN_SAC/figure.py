import  torch
import  numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


rew1, rew2 = [], []
rew3 = []
plt.style.use('ggplot')
real1 = []
real2 = []

with open('reward_halfcheetah.csv', 'r') as myFile:
    lines = csv.reader(myFile)
    header_row = next(lines)
    reward = 0
    count = 0
    for line in lines:
        line = np.array(line, float)
        line = torch.tensor(line)
        count += 1
        reward += line
        rew1.append(line)
        if count % 1 == 0:
            reward /= 1
            for i in range(1):
                real1.append(reward)
            reward=0
with open('10.4.csv', 'r') as myFile:
    lines = csv.reader(myFile)
    header_row = next(lines)
    reward = 0
    count = 0
    for line in lines:
        line = np.array(line, float)
        line = torch.tensor(line)
        count += 1
        reward += line
        rew1.append(line)
        if count % 1 == 0:
            reward /= 1
            for i in range(1):
                real2.append(reward)
            reward=0

plt.plot(real1, c = 'r', alpha = 0.2)
plt.plot(gaussian_filter1d(real1, sigma = 5), c = 'r', label = 'DuDQN')

plt.plot(real2, c = 'b', alpha = 0.2)
plt.plot(gaussian_filter1d(real2, sigma = 5), c = 'b', label = 'DuDQN-SPE')

plt.tick_params(labelsize=10)
plt.xticks([0., 40., 80., 120., 160., 200.], ['0.0', '0.4', '0.8', '1.2', '1.6', '2.0'])
#plt.xticks([0., 40., 80., 120., 160., 200.], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

plt.xlabel('Timesteps (1e6)')
plt.ylabel('Average Return')
plt.legend(loc='upper left')
plt.title('Seaquest')
plt.savefig('Seaquest.png')
plt.show()




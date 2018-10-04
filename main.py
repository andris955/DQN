import gym
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import math
import random
from collections import namedtuple
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


btrain = True
show_render = True
img_show_bool = False

game = 'Breakout-v0'

BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
M = 100
TARGET_UPDATE = 10
GAMMA = 0.999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple('Experience',
                        ('state', 'next_state', 'action', 'reward', 'done'))

class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        # (84 - (8 - 1) - 1) / 4 + 1 = 20 Output: 32 * 20 * 20
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # (20 - (4 -1) -1) / 2 + 1 = 9 Output: 64 * 9 * 9
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # (9 - (3-1) -1)/1 + 1 = 7 Output: 64 * 7 * 7 =  3136
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Myenv():

    def __init__(self, gym_env, state_size, capacity):
        self.env = gym_env
        self.possible_actions = gym_env.action_space.n
        self.state_buffer = []
        self.state_size = state_size
        self.exp_buffer = []
        self.exp_buffer_capacity = capacity
        self.i = 0
        self.steps_done = 0

    def preprocess(self, observation):
        screen = resize(rgb2gray(observation), (110, 84))[16:110 - 10, :]
        screen = img_as_ubyte(screen)
        return screen

    def get_initial_state(self):
        self.state_buffer = []
        observation = self.env.reset()
        x_t = self.preprocess(observation)
        for i in range(self.state_size):
            self.state_buffer.append(x_t)
        s_t = self.state_buffer
        return s_t


    def game_step(self, action):
        if show_render:
            self.env.render()

        observation, reward, done, info = self.env.step(action)

        x_t1 = self.preprocess(observation)

        if self.i % 4 == 0 and img_show_bool:
            plt.imshow(x_t1, cmap='gray')
            plt.show()
            self.i = 0

        s_t = self.state_buffer
        self.state_buffer.pop(0)
        self.state_buffer.append(x_t1)
        s_t1 = self.state_buffer
        self.push_experience(s_t, s_t1, action, reward, done)

        if img_show_bool:
            self.i += 1

        return done

    def push_experience(self, s_t, s_t1, action, reward, done):
        if(len(self.exp_buffer)) < self.exp_buffer_capacity:
            exp = [s_t, s_t1, action, reward, done]
            self.exp_buffer.append(exp)
        else:
            pass

    def epsz_greedy(self, policy_net):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        rand_num = np.random.random()
        if rand_num < eps_threshold:
            action = np.random.randint(0, self.possible_actions)
        else:
             with torch.no_grad():
                action = policy_net(torch.from_numpy(np.array([self.state_buffer], dtype=np.float32))).max(1)[1].view(1,1)
        return action

    def sample(self, batch_size):
        return random.sample(self.exp_buffer, batch_size)

def optimize(myenv, policy_net, target_net, optimizer):

    if len(myenv.exp_buffer) < BATCH_SIZE:
        return

    experiences = myenv.sample(BATCH_SIZE)
    batch = Experience(*zip(*experiences))

     # Compute a mask of non-final states and concatenate the batch elements
    #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], device=device, dtype=torch.float).to(device)
    state_batch = torch.tensor(batch.state, device=device, dtype=torch.float).to(device)
    action_batch = torch.tensor(batch.action).to(device)
    reward_batch = torch.tensor(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch)#.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    #next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    next_state = target_net(non_final_next_states).max(1)
    next_state_values = next_state[0].detach()
    next_state_indexes = next_state[1].detach()

    # Compute the expected Q values
    expected_state_action_values_1d = (next_state_values * GAMMA) + reward_batch
    expected_state_action_values_4d = torch.zeros(state_action_values.size())
    for i in range(len(next_state_indexes)):
        expected_state_action_values_4d[i][next_state_indexes[i]] = expected_state_action_values_1d[i]


    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values_4d)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



def train(myenv, policy_net, target_net, optimizer):
    for i_episode in range(M):
        myenv.get_initial_state()
        done = False
        while done is not True:
            action = myenv.epsz_greedy(policy_net)
            done = myenv.game_step(action)
            optimize(myenv, policy_net, target_net, optimizer)
        if i_episode % TARGET_UPDATE == 0: # befagyasztott háló frissítése
            target_net.load_state_dict(policy_net.state_dict())
    print("Training complete")


def main():
    capacity = 10000

    env = gym.make(game).unwrapped
    myenv = Myenv(env, 4, capacity)

    policy_net = DQN(myenv.possible_actions).to(device)
    target_net = DQN(myenv.possible_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() #?

    optimizer = optim.RMSprop(policy_net.parameters())

    if btrain:
        train(myenv, policy_net, target_net, optimizer)
    else:
        pass


main()

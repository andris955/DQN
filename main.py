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

import utilities as u

btrain = True
show_render = True
img_show_bool = False

game = 'Breakout-v0'

PATH = "" #Path to NN parameters

BATCH_SIZE = 128
REPLAY_START_SIZE = 5000
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 20000
M = 1500000
TARGET_UPDATE = 1000
GAMMA = 0.99
CAPACITY = 1000000
K = 4

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
        self.score = 0

    def preprocess(self, observation):
        screen = resize(rgb2gray(observation), (110, 84))[16:110 - 10, :]
        screen = img_as_ubyte(screen)
        return screen

    def get_initial_state(self):
        self.state_buffer = []
        self.score = 0
        observation = self.env.reset()
        x_t = self.preprocess(observation)
        for i in range(self.state_size):
            self.state_buffer.append(x_t)
        s_t = self.state_buffer
        return s_t


    def game_step(self, action, k):
        if show_render:
            self.env.render()

        for i in range(k):
            observation, reward, done, info = self.env.step(action)
            self.score += reward

        x_t1 = self.preprocess(observation)

        if self.i % 4 == 0 and img_show_bool:
            plt.imshow(x_t1, cmap='gray')
            plt.show()
            self.i = 0

        s_t = self.state_buffer
        self.state_buffer.pop(0)
        self.state_buffer.append(x_t1)
        s_t1 = self.state_buffer
        if btrain is True:
            self.push_experience(s_t, s_t1, action, reward, done)

        if img_show_bool:
            self.i += 1

        return done, reward

    def push_experience(self, s_t, s_t1, action, reward, done):
        if(len(self.exp_buffer)) < self.exp_buffer_capacity:
            exp = [s_t, s_t1, action, reward, done]
            self.exp_buffer.append(exp)
        else:
            pass

    def eps_greedy(self, policy_net):
        if self.steps_done > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * ((self.steps_done-REPLAY_START_SIZE) / EPS_DECAY))
        else:
            eps_threshold = 0.9
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

    if len(myenv.exp_buffer) < REPLAY_START_SIZE:
        return

    experiences = myenv.sample(BATCH_SIZE)
    batch = Experience(*zip(*experiences))

     # Compute a mask of non-final states and concatenate the batch elements
    #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], device=device, dtype=torch.float).to(device)
    state_batch = torch.tensor(batch.state, device=device, dtype=torch.float).to(device)
    action_batch = torch.tensor(batch.action).to(device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state = target_net(non_final_next_states).max(1)
    next_state_values = next_state[0].detach()
    next_state_indexes = next_state[1].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1)
    #expected_state_action_values_4d = torch.zeros(state_action_values.size())
    #for i in range(len(next_state_indexes)):
     #   expected_state_action_values_4d[i][next_state_indexes[i]] = expected_state_action_values_1d[i]


    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



def train(myenv, policy_net, target_net, optimizer):
    i_steps = 0
    while i_steps < M:
        myenv.get_initial_state()
        done = False
        while done is not True:
            action = myenv.eps_greedy(policy_net)
            done, reward = myenv.game_step(action, K)
            optimize(myenv, policy_net, target_net, optimizer)
            i_steps += 1
        if i_steps % TARGET_UPDATE == 0:  # befagyasztott háló frissítése és logolás
            u.save_log(i_steps, myenv.score)
            target_net.load_state_dict(policy_net.state_dict())
    u.save_model_params(policy_net)
    u.save_hyperparams(BATCH_SIZE, REPLAY_START_SIZE, EPS_START, EPS_END, EPS_DECAY, M, TARGET_UPDATE, GAMMA, CAPACITY, K)
    print("Training completed")

def eval(myenv, policy_net):
    myenv.get_initial_state()
    done = False
    policy_net.load_state_dict(torch.load(PATH))
    while done is not True:
        with torch.no_grad():
            action = policy_net(torch.from_numpy(np.array([myenv.state_buffer], dtype=np.float32))).max(1)[1].view(1, 1)
        print(action)
        done, reward = myenv.game_step(action, K)
    print(myenv.score)


def main():
    env = gym.make(game).unwrapped
    num_actions = env.action_space.n
    myenv = Myenv(env, num_actions, CAPACITY)

    policy_net = DQN(myenv.possible_actions).to(device)
    target_net = DQN(myenv.possible_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())

    if btrain is True:
        train(myenv, policy_net, target_net, optimizer)
    else:
        eval(myenv, policy_net)


if __name__ == '__main__':
    main()

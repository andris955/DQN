import gym
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import random
from collections import namedtuple
from matplotlib import pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utilities as u

btrain = True
show_render = False

game = 'Breakout-v0'

PATH = "" #Path to NN parameters

BATCH_SIZE = 32
STATE_SIZE = 4
REPLAY_START_SIZE = 50000
EPS_START = 1.
EPS_END = 0.1
M = 4000000
TARGET_UPDATE = 10000
GAMMA = 0.99
EXP_BUFF_CAPACITY = 1000000
K = 1
NO_REP_ACTION = 30
LAST_GAME = 5
LEARNING_RATE = 0.00025
MOMENTUM = 0.95

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple('Experience',
                        ('state', 'next_state', 'action', 'reward', 'done'))


class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # (84 - (8 - 1) - 1) / 4 + 1 = 20 Output: 32 * 20 * 20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # (20 - (4 -1) -1) / 2 + 1 = 9 Output: 64 * 9 * 9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
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


class Myenv:

    def __init__(self, gym_env):
        self.env = gym_env
        self.possible_actions = gym_env.action_space.n
        self.state_buffer = []
        self.state_size = STATE_SIZE+1
        self.exp_buffer = []
        self.exp_buffer_capacity = EXP_BUFF_CAPACITY
        self.steps_done = 0
        self.score = 0
        self.games_done = 0
        self.avg_score = 0
        self.last_actions = []
        self.eps_threshold = 1
        self.done = False
        self.policy_net = DQN(self.possible_actions).to(device)
        self.target_net = DQN(self.possible_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

    @staticmethod
    def preprocess(observation):
        screen = resize(rgb2gray(observation), (110, 84))[16:110 - 10, :]
        screen = img_as_ubyte(screen)
        return screen

    def get_initial_state(self):
        self.done = False
        self.state_buffer = []
        if self.games_done % LAST_GAME == 0:
            self.games_done = 0
        self.games_done += 1
        self.avg_score = ((self.games_done - 1)/self.games_done) * self.avg_score + (1/self.games_done) * self.score
        self.score = 0
        observation = self.env.reset()
        x_t = self.preprocess(observation)
        for i in range(self.state_size):
            self.state_buffer.append(x_t)

    def game_step(self, action):
        #start_time = time.time()
        if show_render:
            self.env.render()

        if self.done is False:
            observation, reward, done, _ = self.env.step(action.item())
            self.score += reward
            self.done = done

        self.steps_done += 1
        x_t1 = self.preprocess(observation)
        self.state_buffer.pop(0)
        self.state_buffer.append(x_t1)

        if btrain is True:
            self.push_experience(self.state_buffer[0:self.state_size-1], self.state_buffer[1:self.state_size], action.item(), reward, done)
            
        #if len(self.exp_buffer) > REPLAY_START_SIZE:
         #   print("Game step: %s" %(time.time()-start_time))
            
        return self.done

    def push_experience(self, s_t, s_t1, action, reward, done):
        #start_time = time.time()
        exp = [s_t, s_t1, action, reward, done]
        if(len(self.exp_buffer)) < self.exp_buffer_capacity:
            self.exp_buffer.append(exp)
        else:
            self.exp_buffer.pop(0)
            self.exp_buffer.append(exp)
            
        #  if len(self.exp_buffer) > REPLAY_START_SIZE:
        #     print("Push experience: %s" %(time.time()-start_time))

    def eps_greedy(self, policy_net):
        # start_time = time.time()
        if self.steps_done < REPLAY_START_SIZE:
            action = torch.tensor([[random.randrange(self.possible_actions)]], device=device, dtype=torch.int32)
            return action

        if self.eps_threshold > EPS_END:
            self.eps_threshold = EPS_END - (EPS_START - EPS_END) * (self.steps_done-(REPLAY_START_SIZE + EXP_BUFF_CAPACITY))/EXP_BUFF_CAPACITY
        else:
            self.eps_threshold = EPS_END

        rand_num = np.random.random()
        if rand_num < self.eps_threshold:
            action = torch.tensor([[random.randrange(self.possible_actions)]], device=device, dtype=torch.int32)
        else:
            with torch.no_grad():
                action = policy_net(torch.from_numpy(np.array([self.state_buffer[1:self.state_size]], dtype=np.float32)).to(device)).max(1)[1]

        #    print("Eps greedy: %s" %(time.time()-start_time))
        return action

    def sample(self, batch_size):
        return random.sample(self.exp_buffer, batch_size)


def optimize(myenv):

    if len(myenv.exp_buffer) < REPLAY_START_SIZE:
        return 0, 0
    
    # start_time = time.time()
    experiences = myenv.sample(BATCH_SIZE)
    batch = Experience(*zip(*experiences))

     # Compute a mask of non-final states and concatenate the batch elements
    next_state_batch = torch.tensor(np.array(batch.next_state, dtype=np.float32), device=device, dtype=torch.float32).to(device) #requires grad false
    next_state_batch = next_state_batch / 255.
    state_batch = torch.tensor(np.array(batch.state, dtype=np.float32), device=device, dtype=torch.float32).to(device) #requires grad false
    state_batch = state_batch / 255.
    action_batch = torch.tensor(batch.action).to(device).unsqueeze(1) #requires grad false
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device) #requires grad false
    done_list = [not i for i in batch.done]
    done_batch = torch.tensor(np.multiply(done_list, 1), dtype=torch.float32).to(device) #requires grad false

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = myenv.policy_net(state_batch).gather(1, action_batch) #requires grad true
    avg_qscore = torch.sum(state_action_values.detach()) / BATCH_SIZE #requires grad false

    # Compute max Q(s_{t+1},a') for all next states.
    next_state_action_values = myenv.target_net(next_state_batch).max(1) #requires grad true
    next_state_values = next_state_action_values[0].detach() #requires grad false

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * done_batch * GAMMA) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1) #requires grad true

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values) #grad_fn object at -> state_action_values grad_fn

    # Optimize the model
    myenv.optimizer.zero_grad()
    loss.backward()
    for param in myenv.policy_net.parameters():
        param.grad.data.clamp_(-1, 1) #32*4*8*8
    myenv.optimizer.step()
    #print("Optimize: %s" %(time.time()-start_time))
    return loss.item(), avg_qscore.item()


def train(myenv):
    running_loss = 0
    while myenv.steps_done < M:
        myenv.get_initial_state()
        done = False
        while done is not True:
            action = myenv.eps_greedy(myenv.policy_net)
            done = myenv.game_step(action)
            if myenv.steps_done % 4 == 0:
                loss, avg_qscore = optimize(myenv)
            running_loss += loss
            if myenv.steps_done % 500 == 0:
                running_loss = running_loss / 250
            if myenv.steps_done % TARGET_UPDATE == 0:
                u.save_log(myenv.steps_done, myenv.avg_score, running_loss, avg_qscore)
                myenv.target_net.load_state_dict(myenv.policy_net.state_dict())
            if myenv.steps_done % 500 == 0:
                running_loss = 0
            if myenv.steps_done % 5000 == 0:
                print(u.datetime.now())
                print([myenv.steps_done, myenv.score])
            if myenv.steps_done % 500000 == 0:
                u.save_model_params(myenv.policy_net)
    u.save_model_params(myenv.policy_net)
    print("Training completed")


def eval_dqn(myenv):
    myenv.get_initial_state()
    done = False
    scores = []
    myenv.policy_net.load_state_dict(torch.load(PATH))
    for i in range(10):
        while done is not True:
            with torch.no_grad():
                action = myenv.policy_net(torch.from_numpy(np.array([myenv.state_buffer[1:myenv.state_size]], dtype=np.float32))).max(1)[1]
            #print(action)
            done = myenv.game_step(action)
            scores.append(myenv.score)
    print(scores)


def main():
    env = gym.make(game)
    myenv = Myenv(env)

    myenv.target_net.load_state_dict(myenv.policy_net.state_dict())
    myenv.target_net.eval()
    myenv.policy_net.eval()

    u.save_hyperparams(BATCH_SIZE, STATE_SIZE, REPLAY_START_SIZE, EPS_START, EPS_END, M, TARGET_UPDATE,
                       GAMMA, EXP_BUFF_CAPACITY, LEARNING_RATE, NO_REP_ACTION, LAST_GAME)
    if btrain is True:
        train(myenv)
    else:
        eval_dqn(myenv)


if __name__ == '__main__':
    main()

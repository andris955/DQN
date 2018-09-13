import gym
import numpy as np
import neural_network as n_n
from PIL import Image

btrain = False
game = 'Breakout-v0'


class Myenv():

    def __init__(self, gym_env, state_size):

        self.env = gym_env
        self.possible_actions = range(gym_env.action_space.n)
        self.state_buffer = []
        self.state_size = state_size
        self.exp_buffer = []

    def get_initial_state(self):

        self.state_buffer = []
        observation = self.env.reset()
        x_t = preprocess(observation)

        for i in range(self.state_size):
            self.state_buffer.append(x_t)

        s_t = self.state_buffer

        return s_t

    def game_step(self, action):

        observation, reward, done, info = self.env.step(action)
        x_t1 = preprocess(observation)

        s_t = self.state_buffer

        self.state_buffer.pop(0)
        self.state_buffer.append(x_t1)

        s_t1 = self.state_buffer
        self.make_experience(s_t, s_t1, action, reward)

        return s_t, s_t1, reward, done, info

    def make_experience(self, s_t, s_t1, action, reward):

        exp = [s_t, action, reward, s_t1]
        self.exp_buffer.append(exp)

    def epsz_greedy(self):

        np.random.seed()
        rand_num = np.random.random()
        if rand_num < epsz:
            action = np.random.sample(self.possible_actions,1)
        else:
            action = 0 #TODO
        return action



def evaluate(env):
    observation = env.reset()
    x_t = preprocess(observation)
    for t in range(100):
        action = n_n.evaluate_nn() # Rossz TODO
        observation, reward, done, info = env.step(action)
        x_t = preprocess(observation)
        print(reward)
		
		
def preprocess(observation):
	
	im = Image.fromarray(observation)
	im = im.convert("L")
	im = im.resize((110, 84), 5)
	up = 13
	left = 0
	down = up + 84
	right = 84
	im = im.crop((up,left,down,right))
	return im
		

def train(myenv):

    for episode in range(M):
        myenv.get_initial_state()
        for t in range(T):
            action = myenv.epsz_greedy()
            myenv.game_step(action)
            random_exp = np.random.sample(myenv.exp_buffer, 1)
            if random_exp[3] == random_exp[3]: #TODO
                y = random_exp[3]
            else:
                y = random_exp[3] + gamma*n_n.evaluate_nn(X) #TODO

            n_n.train_nn(X,y) #TODO


def main():
    env = gym.make(game)
    myenv = Myenv(env, 4)
    if btrain:
        train(myenv)
    else:
        evaluate(env)


main()

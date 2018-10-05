from datetime import datetime
import re
import os
import torch


def date_in_string():
    time = str(datetime.now())
    time = re.sub(' ', '_', time)
    time = re.sub(':', '', time)
    time = time[0:15]
    return time

def create_folder():
    base_path = r'c:\Users\Tokaji András\Documents\BME\MSc\Önálló laboratórium 2\Reinforcment_learning'
    new_folder_path = base_path + '\RL_' + date_in_string()
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

class save():
    def __init___(self):
        self.i_episodes_l = []
        self.action_steps_l = []
        self.score_l = []


    def save_log(self, i_episode, action_steps, score, save_bool):
        if save_bool is False:
            self.i_episodes_l.append(i_episode)
            self.action_steps_l.append(action_steps)
            self.score_l.append(score)
        else:
            path = create_folder()
            outfile = open(path + '\log.txt', 'w')
            for i in range(len(self.score_l)):
                outfile.write('\t' + self.i_episodes_l[i] + '\t' + self.action_steps_l[i] + '\t' + self.score_l[i] + '\n')

def save_hyperparams(batch_size,replay_start_size,eps_start,eps_end,eps_decay,m,target_update,gamma,capacity,k):
    path = create_folder()
    outfile = open(path + '\hyperparams.txt', 'w')
    outfile.write('Batch size: ' + batch_size + '\n' + 'Replay Start Size: ' + replay_start_size + '\n' + 'Eps start: ' + eps_start + '\n'
                  + 'Eps end: ' + eps_end + '\n' + 'Eps decay: ' + eps_decay + '\n' + 'M: ' + m + '\n' + 'Target update: ' + target_update + '\n'
                  + 'Gamma: ' + gamma + '\n' + 'Capacity: ' + capacity + '\n' + 'Frame skipping: ' + k + '\n')

def save_model_params(model):
    PATH = create_folder()
    torch.save(model.state_dict(), PATH)
    return
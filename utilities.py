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
    base_path = os.getcwd()
    new_folder_path = base_path + '\RL_' + date_in_string()
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path


def save_log(i_steps, score):
    path = create_folder()
    with open(path + '\log.txt', 'a') as outfile:
        outfile.write('\t' + i_steps + '\t' + score + '\n')
    return


def save_hyperparams(batch_size, replay_start_size, eps_start, eps_end, eps_decay, m, target_update, gamma, capacity, k):
    path = create_folder()
    with open(path + '\hyperparams.txt', 'w') as outfile:
        outfile.write('Batch size: ' + batch_size + '\n' + 'Replay Start Size: ' + replay_start_size + '\n' + 'Eps start: ' + eps_start + '\n'
                    + 'Eps end: ' + eps_end + '\n' + 'Eps decay: ' + eps_decay + '\n' + 'M: ' + m + '\n' + 'Target update: ' + target_update + '\n'
                    + 'Gamma: ' + gamma + '\n' + 'Capacity: ' + capacity + '\n' + 'Frame skipping: ' + k + '\n')
    return

def save_model_params(model):
    path = create_folder()
    torch.save(model.state_dict(), path)
    return

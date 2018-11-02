from datetime import datetime
import re
import os
import torch


def date_in_string():
    time = str(datetime.now())
    time = re.sub(' ', '_', time)
    time = re.sub(':', '', time)
    time = re.sub('-', '_', time)
    time = time[0:15]
    return time

def create_folder():
    base_path = os.getcwd()
    new_folder_path = base_path + '\RL_' + date_in_string()[0:10]
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path


def save_log(i_steps, avg_score, loss, avg_qscore):
    path = create_folder()
    with open(path + '\log.txt', 'a') as outfile:
        outfile.write(date_in_string() + '\t' + str(i_steps) + '\t' + str(avg_qscore) + '\t' + str(loss) + '\t' + str(avg_score) + '\n')
    return


def save_hyperparams(batch_size, state_size, replay_start_size, eps_start, eps_end, eps_decay, m, target_update, gamma, capacity, k, no_rep_action, last_game_avg):
    path = create_folder()
    with open(path + '\hyperparams_' + date_in_string()[0:10] + '.txt', 'w') as outfile:
        outfile.write('Batch size: ' + str(batch_size) + '\n' + 'State size: ' + str(state_size) + '\n' + 'Replay Start Size: ' + str(replay_start_size) + '\n' + 'Eps start: ' + str(eps_start) + '\n'
                    + 'Eps end: ' + str(eps_end) + '\n' + 'Eps decay: ' + str(eps_decay) + '\n' + 'M: ' + str(m) + '\n' + 'Target update: ' + str(target_update) + '\n'
                    + 'Gamma: ' + str(gamma) + '\n' + 'Capacity: ' + str(capacity) + '\n' + 'Frame skipping: ' + str(k) + '\n'+ 'No more consecutive action: ' + str(no_rep_action) + '\n'+ 'Last N game avg: ' + str(last_game_avg) + '\n')
    return

def save_model_params(model):
    path = create_folder()
    torch.save(model.state_dict(), path + r'\\model_' + date_in_string() + '.pth')
    return

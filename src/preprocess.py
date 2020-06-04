import os
import sys
import json
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

class MyData(Data.Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return {
            'data': self.data_seq[idx],
        }

def preprocess(data_seq, label):
    new_label = []
    for i in range(len(label)):
        label_of_one_song = []
        cur_note = 0
        
        cur_note_onset = label[i][cur_note][0]
        cur_note_offset = label[i][cur_note][1]
        cur_note_pitch = label[i][cur_note][2]

        for j in range(len(data_seq[i])):
            cur_time = j * 0.032 + 0.016

            if abs(cur_time - cur_note_onset) < 0.017:
                label_of_one_song.append(np.array([1, 0, cur_note_pitch]))

            elif cur_time < cur_note_onset or cur_note >= len(label[i]):
                label_of_one_song.append(np.array([0, 0, 0.0]))

            elif abs(cur_time - cur_note_offset) < 0.017:
                label_of_one_song.append(np.array([0, 1, cur_note_pitch]))
                cur_note = cur_note + 1
                if cur_note < len(label[i]):
                    cur_note_onset = label[i][cur_note][0]
                    cur_note_offset = label[i][cur_note][1]
                    cur_note_pitch = label[i][cur_note][2]
            else:
                label_of_one_song.append(np.array([0, 0, cur_note_pitch]))

        new_label.append(label_of_one_song)

    return new_label

if __name__ == '__main__':
    data_path = sys.argv[1]
    pickle_path = sys.argv[2]
    num_song = int(sys.argv[3])
    
    data_seq = []
    for idx in range(1, num_song + 1):
        json_path = os.path.join(data_path, f'{idx}', f'{idx}_feature.json')

        with open(json_path, 'r') as json_file:
            temp = json.loads(json_file.read())

        data = []
        for key, value in temp.items():
            data.append(value)
        
        data = np.array(data).T
        data_seq.append(data)
    
    test_data = MyData(data_seq)

    with open(pickle_path, 'wb') as pkl:
        pickle.dump(test_data, pkl)

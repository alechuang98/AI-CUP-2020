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
        return {'data': self.data_seq[idx]}

class Myrnn(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(Myrnn, self).__init__()
        self.hidden_size = hidden_size

        self.Linear1 = nn.Linear(input_dim, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers = 5, bidirectional = True)
        self.Linear2 = nn.Linear(hidden_size * 2, 2)
        self.Linear3 = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_data):
        out = F.relu(self.Linear1(input_data))
        out, hidden = self.rnn(out)
        # out1 is for onset & offset
        out1 = torch.sigmoid(self.Linear2(out))
        # out2 is for pitch
        out2 = self.Linear3(out)
        return out1, out2

def post_processing(output1, output2, idx):
    output2 = output2.squeeze().cpu().detach().numpy()
    # print(np.count_nonzero(output1.cpu() > 0.1))
    threshold = 0.1
    notes = []
    
    this_onset = None
    this_offset = None
    this_pitch = None

    for i in range(len(output1)):
        if output1[i][0][0] > threshold and this_onset == None:
            this_onset= i
        
        elif output1[i][0][1] > threshold and this_onset != None and this_offset == None:
            this_offset = i
            this_pitch = int(round(np.mean(output2[this_onset : this_offset + 1])))
            notes.append([this_onset * 0.032 + 0.016, this_offset * 0.032 + 0.016, this_pitch])
            
            this_onset = None
            this_offset = None
            this_pitch = None

    predict[f'{idx + 1}'] = notes

def collate_fn(samples):
    batch = {}
    temp = [torch.from_numpy(np.array(sample['data'], dtype = np.float32)) for sample in samples]
    padded_data = rnn_utils.pad_sequence(temp, batch_first = True, padding_value = 0)
    
    batch['data'] = padded_data
    return batch

def testing(model, loader, device):
    model.eval()

    for idx, sample in enumerate(loader):
        data = torch.Tensor(sample['data']).permute(1, 0, 2)
        data = data.to(device, dtype = torch.float)

        output1, output2 = model(data)
        # print(output1.shape, output2.shape)
        post_processing(output1, output2, idx)

if __name__ == '__main__':
    pickle_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]

    test_data = None
    with open(pickle_path, 'rb') as pkl_file:
        test_data = pickle.load(pkl_file)

    input_dim = 23
    hidden_size = 50

    predict = {}
    batch_size = 1
    loader = Data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    model = Myrnn(input_dim, hidden_size)
    model.to(device)

    model.load_state_dict(torch.load(model_path))
    testing(model, loader, device)

    with open(output_path, 'w') as output:
        json.dump(predict, output)
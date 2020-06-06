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
    def __init__(self, data_seq, label):
        self.data_seq = data_seq
        self.label= label

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return {
            'data': self.data_seq[idx],
            'label': self.label[idx]
        }

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

def collate_fn(samples):
    batch = {}
    temp = [torch.from_numpy(np.array(sample['data'], dtype = np.float32)) for sample in samples]
    padded_data = rnn_utils.pad_sequence(temp, batch_first = True, padding_value = 0)
    
    batch['data'] = padded_data
    batch['label'] = [np.array(sample['label'], dtype = np.float32) for sample in samples]
    return batch

def training(model, loader, optimizer, device):
    criterion_onset = nn.BCELoss(reduction = 'mean')
    criterion_pitch = nn.L1Loss(reduction = 'mean')

    nepoch = 50
    model.train()
    best_loss = np.inf
    print('start training ...')

    for epoch in range(nepoch):
        train_loss = 0.0
        total_length = 0.0
        
        for batch_idx, sample in enumerate(loader):
            data = torch.Tensor(sample['data']).permute(1, 0, 2)
            target = torch.Tensor(sample['label']).permute(1, 0, 2)
            # print(data.shape, target.shape)
            
            data_length = list(data.shape)[0]
            data = data.to(device, dtype = torch.float)
            target = target.to(device, dtype = torch.float)

            output1, output2 = model(data)
            # print(output1.shape, output2.shape)

            total_loss = 10 * criterion_onset(output1, torch.narrow(target, dim = 2, start = 0, length = 2))
            total_loss += criterion_pitch(output2, torch.narrow(target, dim = 2, start = 2, length = 1))
            
            train_loss += total_loss.item()
            total_length += 1
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print ('epoch [{:2d}/{}]: sample {}, loss {:.6f}'.format(epoch + 1, nepoch, batch_idx + 1, total_loss), end = '\r')

        print('\nepoch [{:2d}/{}]: avg loss: {:.6f}'.format(epoch + 1, nepoch, train_loss / total_length))

        if train_loss / total_length < best_loss:
            best_loss = train_loss / total_length
            torch.save(model.state_dict(), model_path)

    return model

if __name__ == '__main__':
    pickle_path = sys.argv[1]
    model_path = sys.argv[2]

    train_data = None
    with open(pickle_path, 'rb') as pkl_file:
        train_data = pickle.load(pkl_file)

    input_dim = 23
    hidden_size = 50

    batch_size = 1
    loader = Data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)

    model = Myrnn(input_dim, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    model.to(device)
    model = training(model, loader, optimizer, device)

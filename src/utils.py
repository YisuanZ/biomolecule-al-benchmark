import numpy as np
import torch
import re
import pandas as pd
from gpro.utils.utils_predictor import seq2onehot
from torch.utils.data import DataLoader
from gpro.predictor.attnbilstm.attnbilstm import TestData

DEVICE, = [torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), ]
ALL_AAS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")

def seq2onehot_protein(seqs:list):
    onehot_dict = {aa: i for i, aa in enumerate(ALL_AAS)}
    onehot_array = torch.zeros([len(seqs), len(seqs[0]), len(ALL_AAS)])
    
    for i, seq in enumerate(seqs):
        for j, character in enumerate(seq):
            onehot_idx = onehot_dict[character]
            onehot_array[i, j, onehot_idx] = 1
            
    return onehot_array.tolist()

def get_data(dataset_name, seq_len):

    index = re.search(r'D(\d+)', dataset_name)
    n = int(index.group(1))

    df = pd.read_csv(f'./data/{dataset_name}.csv')
    seqs = df.iloc[:, 1].tolist()
    if n in [11, 12, 14]:
        onehot_seqs = seq2onehot_protein(seqs)
    else:
        onehot_seqs = seq2onehot(seqs, seq_len)
    labels = df.iloc[:, 2].tolist()
    data = list(zip(onehot_seqs, labels))
    
    return data

def get_data_vanilla(dataset_name):

    df = pd.read_csv(f'./data/{dataset_name}.csv')
    seqs = df.iloc[:, 1].tolist()
    labels = df.iloc[:, 2].tolist()
    data = list(zip(seqs, labels))
    
    return data

def EnPredict(x, model, prefix, model_num, rs):
    model = model.to(DEVICE)
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    dataset = TestData(x)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)

    mu_list = []
    for tag_idx in range(model_num):
        mu = []
        path_check = f"{prefix}/best_n{tag_idx+1}_rs{rs}.pth"
        
        model.load_state_dict(torch.load(path_check))
        model.eval()
        for _, x in enumerate(dataloader,0):
            x = x.to(torch.float32).to(DEVICE)
            pred = model(x)
            mu += pred.flatten().tolist()
        mu_list.append(mu)
    
    mu = np.mean(mu_list, axis=0)
    sigma = np.std(mu_list, axis=0)
    return mu, sigma

def McPredict(x, model, prefix, model_num, rs):
    model = model.to(DEVICE)
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    dataset = TestData(x)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)

    mu_list = []
    for _ in range(model_num):
        mu = []
        path_check = f"{prefix}/best_n1_rs{rs}.pth"
        model.load_state_dict(torch.load(path_check))
        model.eval()
        model.enable_dropout()
        for _, x in enumerate(dataloader,0):
            x = x.to(torch.float32).to(DEVICE)
            pred = model(x)
            mu += pred.flatten().tolist()
        mu_list.append(mu)
    
    mu = np.mean(mu_list, axis=0)
    sigma = np.std(mu_list, axis=0)
        
    return mu, sigma

def DKLPredict(x, model, likelihood, prefix, rs):
    model = model.to(DEVICE)
    likelihood = likelihood.to(DEVICE)
    
    path_check = f"{prefix}/best_n1_rs{rs}.pth"
    model.load_state_dict(torch.load(path_check, weights_only=True))
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(x))
        mu = pred.mean.detach().cpu().numpy()
        sigma = pred.stddev.detach().cpu().numpy()

    return mu, sigma
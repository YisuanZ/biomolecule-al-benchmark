import torch
import numpy as np
import pandas as pd
import random
import os
import gpytorch

import utils
import model_zoo_DKL
import train_DKL
import metrics
import configs

random_seed = [137, 419, 823, 1607, 2753]

def DKLUQ(UQalgo, dataset, model_type, save_dir):
    config = configs.get_config(dataset)
    checkpoint_prefix = './checkpoints/UQ/{}/{}_{}'.format(UQalgo, dataset, model_type)
    train_size = 3000
    valid_size = 500
    test_size  = 500

    for rs in random_seed:
        
        data_base = utils.get_data(dataset, config['seq_len'])
        np.random.seed(rs)
        random.seed(rs)
        random.shuffle(data_base)

        test_data  = data_base[:test_size]
        valid_data = data_base[test_size:test_size+valid_size]
        train_data = data_base[test_size+valid_size:test_size+valid_size+train_size]

        train_seqs   = torch.tensor([item[0] for item in train_data])
        train_labels = torch.tensor([item[1] for item in train_data])
        valid_seqs   = torch.tensor([item[0] for item in valid_data])
        valid_labels = torch.tensor([item[1] for item in valid_data])
        test_seqs    = torch.tensor([item[0] for item in test_data])
        test_labels  = torch.tensor([item[1] for item in test_data])

        train_seqs = train_seqs.reshape(train_seqs.size(0), -1)
        valid_seqs = valid_seqs.reshape(valid_seqs.size(0), -1)
        test_seqs  = test_seqs.reshape(test_seqs.size(0), -1)
            
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = model_zoo_DKL.DKLModel(
            train_x=train_seqs, train_y=train_labels, 
            likelihood=likelihood, onehot_dim=config['onehot_dim'],
            seq_len=config['seq_len']
            )

        tags = ['i1n{}'.format(num + 1) for num in range(1)]
        train_DKL.DKLTrain(
            model=model, likelihood=likelihood, 
            valid_x=valid_seqs, valid_y=valid_labels, 
            prefix=checkpoint_prefix, tags=tags, rs=rs
            )

        preds, stds = utils.DKLPredict(test_seqs, model, likelihood, checkpoint_prefix, rs)
        preds, stds, test_labels = np.array(preds), np.array(stds), np.array(test_labels)

        coverages = []
        confidence_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 1.0]
        for cl in confidence_levels:
            coverage = metrics.get_coverage(preds, stds, test_labels, confidence_level=cl)
            coverages.append(coverage)
        pcc, _ = metrics.get_regression_PCC(preds, test_labels)
        ence   = metrics.get_ENCE(preds, stds, test_labels)
        
        df = pd.DataFrame({
            'Dataset': [dataset] * len(confidence_levels),
            'UQalgo': [UQalgo] * len(confidence_levels),
            'Model': [None] * len(confidence_levels),
            'train_DKL Size': [train_size] * len(confidence_levels),
            'Random Seed': rs,
            'Dropout Rate': None,
            'Confidence Level': confidence_levels,
            'Coverage': coverages,
            'PCC': pcc,
            'ENCE': ence,
        })

        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'{UQalgo}_matrics.csv')
        header = not os.path.exists(file_path)
        df.to_csv(file_path, mode='a', index=False, header=header)

    return

if __name__ == "__main__":
    dataset_list = [
        'D1_MPRALegNet_HepG2',
        'D2_MPRALegNet_K562',
        'D3_MPRALegNet_WTC11',
        'D4_Malinois_HepG2',
        'D5_Malinois_K562',
        'D6_Malinois_SKNSH',
        'D7_Ecoli_Wang_2020',
        'D8_Ecoli_Wang_2023',
        'D9_Yeast_Aviv_2022',
        'D10_Yeast_Zelezniak_2022',
        'D11_Gb1_Arnold_2024',
        'D12_TrpB_Arnold_2024',
        'D13_folA_Wagner_2023',
        'D14_CreiLOV_Tong_2023'
    ]
    for dataset in dataset_list:  
        DKLUQ(UQalgo='DKL', dataset=dataset, model_type='MLP', save_dir='./result')
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

import train
import model_zoo
import utils
import configs

random_seed = [137, 419, 823, 1607, 2753]

def SampleNext(mu, sigma, strategy, selected_mask, K, rng):

    candidates = np.where(~selected_mask)[0]
    if strategy == 'UCB':
        score = mu[candidates] + 2 * sigma[candidates]
    elif strategy == 'TS':
        sampled = rng.normal(mu[candidates], sigma[candidates])
        score = sampled
    elif strategy == 'Greedy':
        score = mu[candidates]
    else:
        raise ValueError('Unknown strategy')

    top_indices = candidates[np.argsort(score)[-K:]]
    return top_indices

def ActiveLearning(dataset, model_type, init_state, model_num, save_dir, M=96, K=96, ROUND=4, N=70):
    
    config = configs.get_config(dataset)
    checkpoint_prefix = f'./checkpoints/Sim/{dataset}/{model_type}'
    os.makedirs(save_dir, exist_ok=True)

    data_base   = utils.get_data(dataset, config['seq_len'])
    base_seqs   = np.array([item[0] for item in data_base])
    base_labels = np.array([item[1] for item in data_base])
    n_base      = len(data_base)

    results = []

    for rs in tqdm(range(N), desc=f"Simulations on {dataset} {model_type} {init_state}"):
        rng = np.random.default_rng(rs)

        if init_state == 'low':
            threshold_idx = int(0.2 * n_base)
            low20_indices = np.argsort(base_labels)[:threshold_idx]
            init_indices = rng.choice(low20_indices, M, replace=False)
        else:
            init_indices = rng.choice(n_base, M, replace=False)

        for strategy in ['UCB', 'TS', 'Greedy']:
            
            sampled = set(init_indices)
            max_hit = [max([base_labels[i] for i in sampled])]

            for t in range(ROUND):
                train_data   = [data_base[i] for i in sampled]
                train_seqs   = np.array([item[0] for item in train_data])
                train_labels = np.array([item[1] for item in train_data])

                model_map = {
                    'MLP': model_zoo.MLPEnModel,
                    'CNN': model_zoo.CNNEnModel,
                }
                model = model_map[model_type](
                    onehot_dim=config['onehot_dim'],
                    seq_len=config['seq_len'],
                )

                tags = [f'i1n{k+1}' for k in range(model_num)]
                train.Train(
                    train_seqs, train_labels,
                    train_seqs, train_labels,
                    model,
                    checkpoint_prefix,
                    tags,
                    rs=rs,
                    rate=0.6
                )

                preds, stds = utils.EnPredict(
                    base_seqs, model,
                    checkpoint_prefix,
                    model_num,
                    rs
                )
                preds, stds = np.array(preds), np.array(stds)

                sampled_mask = np.zeros(n_base, dtype=bool)
                sampled_mask[list(sampled)] = True
                new_idx = SampleNext(preds, stds, strategy, sampled_mask, K, rng)
                sampled.update(new_idx)
                max_hit.append(base_labels[list(sampled)].max())

            results.append({
                'Dataset': dataset,
                'Model': model_type,
                'Strategy': strategy,
                'Random Seed': rs,
                'M': M,
                'K': K,
                'Init State': init_state,
                'R4 Max Hit': max_hit[4],
            })

    df = pd.DataFrame(results)
    file_path = os.path.join(save_dir, f'AL_{dataset}_{model_type}.csv')
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
    model_list = ['MLP']
    UQalgo = 'Ensemble'
    init_list = ['rdm', 'low']
    MK_config = {
        'D1_MPRALegNet_HepG2': [96, 192, 384], 
        'D2_MPRALegNet_K562': [96, 192, 384], 
        'D3_MPRALegNet_WTC11': [96, 192, 384],
        'D4_Malinois_HepG2': [96, 192, 384], 
        'D5_Malinois_K562': [96, 192, 384], 
        'D6_Malinois_SKNSH': [96, 192, 384],
        'D7_Ecoli_Wang_2020': [10, 20, 40], 
        'D8_Ecoli_Wang_2023': [10, 20, 40],
        'D9_Yeast_Aviv_2022': [480, 960, 1920], 
        'D10_Yeast_Zelezniak_2022': [10, 20, 40],
        'D11_Gb1_Arnold_2024': [96, 192, 384], 
        'D12_TrpB_Arnold_2024': [96, 192, 384],
        'D13_folA_Wagner_2023': [96, 192, 384],
        'D14_CreiLOV_Tong_2023': [96, 192, 384],
    }

    for dataset, model_type, init_state in product(dataset_list, model_list, init_list):
        for MK in MK_config[dataset]:
            print(f'Dataset: {dataset}, Model: {model_type}, MK: {MK}')
            M = MK  # Initial samples
            K = MK  # Number of samples added per round
            ActiveLearning(
                dataset=dataset, 
                model_type=model_type, 
                init_state=init_state, 
                model_num=5, 
                save_dir='./result', 
                M=M, K=K
                )
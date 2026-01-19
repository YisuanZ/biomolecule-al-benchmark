import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import os
import Levenshtein

import configs, utils

def CaliCurvePlot(result_dir, save_dir, model_num):

    dataset_groups = {
        'Sequential': [
            'D1_MPRALegNet_HepG2', 'D2_MPRALegNet_K562', 'D3_MPRALegNet_WTC11',
            'D4_Malinois_HepG2', 'D5_Malinois_K562', 'D6_Malinois_SKNSH',
            'D7_Ecoli_Wang_2020', 'D8_Ecoli_Wang_2023', 
            'D9_Yeast_Aviv_2022', 'D10_Yeast_Zelezniak_2022'
        ],
        'Mutational': [
            'D11_Gb1_Arnold_2024', 'D12_TrpB_Arnold_2024',
            'D13_folA_Wagner_2023', 'D14_CreiLOV_Tong_2023'
        ]
    }
    
    dp_rate_list = [0.1, 0.2, 0.3, 0.5]
    model_type_list = ['CNN', 'MLP']

    for (group_name, datasets), dp_rate, model_type in product(dataset_groups.items(), dp_rate_list, model_type_list):
        file_list = [f"{result_dir}/Ensemble_{model_num}_matrics.csv",
                    f"{result_dir}/MCDP_{model_num}_matrics.csv",
                    f"{result_dir}/DKL_matrics.csv"]
        dfs = [pd.read_csv(f) for f in file_list]
        
        df = pd.concat(dfs, ignore_index=True)
        df = df[(df['Dropout Rate'] == dp_rate) | (df['Dropout Rate'].isna())]
        df = df[(df['Model'] == model_type) | (df['Model'].isna())]
        df = df[df['Dataset'].isin(datasets)]

        df = (
            df.groupby(['Confidence Level', 'Dataset', 'UQalgo'], as_index=False)
            .agg({'Coverage': ['mean', 'std']})
        )
        df.columns = ['Confidence Level', 'Dataset', 'UQalgo', 'Coverage_mean', 'Coverage_std']
        df['Dataset'] = df['Dataset'].str.extract(r'(D\d+)')
        df['Dataset'] = pd.Categorical(
            df['Dataset'],
            categories=[f'D{i}' for i in range(1, 15)],
            ordered=True
        )

        plt.figure(figsize=(8, 8))
        sns.lineplot(
            data=df,
            x='Confidence Level',
            y='Coverage_mean',
            hue='UQalgo',
            hue_order=['Ensemble', 'MCDP', 'DKL'],
            style='Dataset',
            palette=['#c42536', '#0176bb', '#dcb582'],
            linewidth=2,
        )

        for UQalgo in ['Ensemble', 'MCDP', 'DKL']:
            for dataset in df['Dataset'].unique():
                subset = df[(df['UQalgo']==UQalgo) & (df['Dataset']==dataset)]
                plt.fill_between(
                    subset['Confidence Level'],
                    subset['Coverage_mean'] - subset['Coverage_std'],
                    subset['Coverage_mean'] + subset['Coverage_std'],
                    color=sns.color_palette(['#c42536', '#0176bb', '#dcb582'])[['Ensemble', 'MCDP', 'DKL'].index(UQalgo)],  #['#f65150', '#54d5c7', '#edba38']
                    alpha=0.15
                )

        plt.plot([0, 1], [0, 1], color='grey', linestyle=':', label='Ideal')

        plt.xlabel('Expected Confidence', fontsize=20)
        plt.ylabel('Empirical Proportion Correct', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend([], [], frameon=False)
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir+f'/{group_name}_{model_type}_num{model_num}_dp{dp_rate}_calibration_curves.png', dpi=400)
        plt.close()

def MiscalibrationBarplot(result_dir, save_dir, model_num):
    
    dataset_groups = {
        'Sequential': [
            'D1_MPRALegNet_HepG2', 'D2_MPRALegNet_K562', 'D3_MPRALegNet_WTC11',
            'D4_Malinois_HepG2', 'D5_Malinois_K562', 'D6_Malinois_SKNSH',
            'D7_Ecoli_Wang_2020', 'D8_Ecoli_Wang_2023', 
            'D9_Yeast_Aviv_2022', 'D10_Yeast_Zelezniak_2022'
        ],
        'Mutational': [
            'D11_Gb1_Arnold_2024', 'D12_TrpB_Arnold_2024',
            'D13_folA_Wagner_2023', 'D14_CreiLOV_Tong_2023'
        ]
    }

    file_list = [
        f"{result_dir}/Ensemble_{model_num}_matrics.csv",
        f"{result_dir}/MCDP_{model_num}_matrics.csv",
        f"{result_dir}/DKL_matrics.csv"
    ]
    dfs = [pd.read_csv(f) for f in file_list]
    df = pd.concat(dfs, ignore_index=True)

    dp_rate_list = [0.1, 0.2, 0.3, 0.5]
    model_type_list = ['CNN', 'MLP']

    records = []

    for dp_rate, model_type in product(dp_rate_list, model_type_list):
        df_filtered = df.copy()
        df_filtered = df_filtered[(df_filtered['Dropout Rate'] == dp_rate) | (df_filtered['Dropout Rate'].isna())]
        df_filtered = df_filtered[(df_filtered['Model'] == model_type) | (df_filtered['Model'].isna())]

        for group_name, datasets in dataset_groups.items():
            df_group = df_filtered[df_filtered['Dataset'].isin(datasets)]
            for uq_algo in ['Ensemble', 'MCDP', 'DKL']:
                df_uq = df_group[df_group['UQalgo'] == uq_algo]
                for dataset in df_uq['Dataset'].unique():
                    df_ds = df_uq[df_uq['Dataset'] == dataset].sort_values('Confidence Level')
                    x = df_ds['Confidence Level'].values
                    y = df_ds['Coverage'].values
                    miscal_area = np.trapz(np.abs(y - x), x)
                    records.append({
                        'UQalgo': uq_algo,
                        'Group': group_name,
                        'MiscalibrationArea': miscal_area
                    })

        for uq_algo in ['Ensemble','MCDP','DKL']:
            df_seq = df_filtered[(df_filtered['UQalgo']==uq_algo) & 
                                (df_filtered['Dataset'].isin(dataset_groups['DNA']))]
            dna_areas = []
            for dataset in df_seq['Dataset'].unique():
                df_ds = df_seq[df_seq['Dataset']==dataset].sort_values('Confidence Level')
                x = df_ds['Confidence Level'].values
                y = df_ds['Coverage'].values
                dna_areas.append(np.trapz(np.abs(y-x), x))
            dna_mean = np.mean(dna_areas)

            df_mut = df_filtered[(df_filtered['UQalgo']==uq_algo) & 
                                    (df_filtered['Dataset'].isin(dataset_groups['Protein']))]
            protein_areas = []
            for dataset in df_mut['Dataset'].unique():
                df_ds = df_mut[df_mut['Dataset']==dataset].sort_values('Confidence Level')
                x = df_ds['Confidence Level'].values
                y = df_ds['Coverage'].values
                protein_areas.append(np.trapz(np.abs(y-x), x))
            protein_mean = np.mean(protein_areas)

            all_mean = (dna_mean + protein_mean)/2
            records.append({
                'UQalgo': uq_algo,
                'Group': 'Average',
                'MiscalibrationArea': all_mean
            })

    df_area = pd.DataFrame(records)
    df_plot = df_area.groupby(['UQalgo','Group'], as_index=False)['MiscalibrationArea'].mean()

    plt.figure(figsize=(8,6))
    sns.barplot(
        data=df_plot,
        x='Group',
        y='MiscalibrationArea',
        hue='UQalgo',
        hue_order=['Ensemble','MCDP','DKL'],
        order=['DNA', 'Protein', 'Average'],
        palette=['#c42536','#0176bb','#dcb582'],
    )
    plt.ylabel('Miscalibration Area', fontsize=16)
    plt.xlabel('')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend([], [], frameon=False)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + f'/miscalibration_bar_dp{dp_rate}_{model_type}.png', dpi=400)
    plt.close()

def SteepnessCompute(y, quantiles=np.linspace(0, 1, 1000)):

    top80_thresh = np.quantile(y, 0.80)
    y_max = y.max()
    y = np.array(y, dtype=float)
    y_norm = (y - top80_thresh) / (y_max - top80_thresh)
    curve = np.quantile(y_norm, quantiles)

    return quantiles, curve

def SteepnessViz(results, group, save_dir):
    plt.figure(figsize=(12, 8))

    for i, (name, (q, curve)) in enumerate(results.items()):
        curve = np.clip(curve, a_min=None, a_max=1.0)
        valid_x_mask = (q >= 0.8) & (q <= 1.0)
        q_filtered = q[valid_x_mask]
        curve_filtered = curve[valid_x_mask]
        curve_filtered = np.clip(curve_filtered, a_min=0.0, a_max=1.0)
        
        if group == 'protein':
            colors = sns.color_palette('tab20', len(results))
            plt.plot(q_filtered, curve_filtered, label=f"D{i+11}", color=colors[i], linestyle='-', linewidth=3.8)
        else:
            colors = sns.color_palette('Paired', len(results))
            plt.plot(q_filtered, curve_filtered, label=f"D{i+1}", color=colors[i], linestyle='-', linewidth=3.8)

    plt.xlim(0.79, 1.01)
    plt.ylim(-0.05, 1.05)
    plt.xticks([0.8, 0.85, 0.9, 0.95, 1.0], fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Quantile", fontsize=20)
    plt.ylabel("Relative Fitness", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16, loc='upper left')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/ValueDistributeSharpness_{group}.png", dpi=400, bbox_inches="tight")
    plt.close()

def RuggednessCompute(seqs, values, window_size=1000, stride=500):
    
    seqs = np.array(seqs)
    values = np.array(values, dtype=float)
    order = np.argsort(values)
    seqs = seqs[order]
    seq_len = len(seqs[0])
    n = len(seqs)
    records = []

    for start in tqdm(range(0, n - window_size + 1, stride)):
        end = start + window_size
        window = seqs[start:end]

        min_dists = []
        for i in range(len(window)):
            dmin = min(
                Levenshtein.distance(window[i], window[j])
                for j in range(len(window)) if i != j
            )
            min_dists.append(dmin)

        avg_dist = np.mean(min_dists)
        center = (start + end - 1) / 2
        records.append({
            "Rank_Quantile_Center": center / n,
            "Avg_Levenshtein_Distance": avg_dist,
            "Normalized_Distance": avg_dist / seq_len,
        })

    return pd.DataFrame(records)

def RuggednessViz(df, group, save_dir):

    group_df = df[df['Group'] == group]
    palette = 'Paired'

    group_df['Dataset'] = group_df['Dataset'].str.extract(r'(D\d+)')

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=group_df,
        x="Rank_Quantile_Center",
        y="Normalized_Distance",
        hue="Dataset",
        palette=palette,
        linewidth=3.8
    )
    
    plt.xlabel("Quantile", fontsize=20)
    plt.ylabel("Levenshtein Distance", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend()

    plt.grid(False)
    plt.ylim(0.1, 0.55)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sliding_window_levenshtein_{group}.png", dpi=400, bbox_inches="tight")
    plt.close()

def HSICompute(dataset):

    config = configs.get_config(dataset)
    data_base = utils.get_data(dataset, config['dataset'], config['seq_len'])
    y = np.array([item[1] for item in data_base], dtype=float)

    y_norm = (y - y.min()) / (y.max() - y.min())

    q_low  = 0.80
    q_high = 1.00
    n_bins = 100
    qs = np.linspace(q_low, q_high, n_bins)
    values = np.quantile(y_norm, qs)

    slopes = np.diff(values) / np.diff(qs)
    hsi = slopes.mean()

    return hsi

def HGICompute(dataset, eps_ratio=0.2, min_neighbors=1):
    
    def compute_distance_matrix(seqs):
        n = len(seqs)
        dist_matrix = np.zeros((n, n))
        for i in tqdm(range(n), desc="Computing Levenshtein distances"):
            for j in range(i + 1, n):
                d = Levenshtein.distance(seqs[i], seqs[j])
                dist_matrix[i, j] = dist_matrix[j, i] = d
        return dist_matrix
    
    config = configs.get_config(dataset)
    data  = utils.get_data_vanilla(dataset, config['dataset'])
    seqs = [item[0] for item in data]
    vals = [item[1] for item in data]
    
    topk = int(len(vals) * 0.05)
    vals = np.array(vals, dtype=float)
    idx = np.argsort(vals)[-topk:]
    top_seqs = [str(seqs[i]) for i in idx]
    top_vals = vals[idx]

    seq_len = len(top_seqs[0])
    eps = int(seq_len * eps_ratio)
    
    dist_matrix = compute_distance_matrix(top_seqs)
    hgi_list = []

    for i in range(len(top_vals)):
        neighbors = np.where((dist_matrix[i] <= eps) & (dist_matrix[i] > 0))[0]
        
        if len(neighbors) < min_neighbors:
            hgi_list.append(eps_ratio)
        else:
            neigh_mean = np.mean(dist_matrix[i, neighbors])
            hgi_list.append(neigh_mean / seq_len)
    
    hgi = np.mean(hgi_list)

    return hgi

if __name__ == "__main__":
    CaliCurvePlot(
        result_dir='./result', 
        save_dir='/result/calibration', 
        model_num=5
        )
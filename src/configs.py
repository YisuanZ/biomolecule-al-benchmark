configs = {
    
    'D1_MPRALegNet_HepG2': {
        'onehot_dim': 4,
        'seq_len': 200,
        'total_size': 100000
    },

    'D2_MPRALegNet_K562': {
        'onehot_dim': 4,
        'seq_len': 200,
        'total_size': 100000
    },

    'D3_MPRALegNet_WTC11': {
        'onehot_dim': 4,
        'seq_len': 200,
        'total_size': 92370
    },

    'D4_Malinois_HepG2': {
        'onehot_dim': 4,
        'seq_len': 200,
        'total_size': 100000
    },

    'D5_Malinois_K562': {
        'onehot_dim': 4,
        'seq_len': 200,
        'total_size': 100000
    },

    'D6_Malinois_SKNSH': {
        'onehot_dim': 4,
        'seq_len': 200,
        'total_size': 100000
    },    

    'D7_Ecoli_Wang_2020': { 
        'onehot_dim': 4,
        'seq_len': 50,
        'total_size': 11884
    },

    'D8_Ecoli_Wang_2023': {
        'onehot_dim': 4,
        'seq_len': 165,
        'total_size': 13972
    },

    'D9_Yeast_Aviv_2022': {
        'onehot_dim': 4,
        'seq_len': 80,
        'total_size': 100000
    },

    'D10_Yeast_Zelezniak_2022': {
        'onehot_dim': 4,
        'seq_len': 1000,
        'total_size': 4238
    },
    
    'D11_Gb1_Arnold_2024': {
        'onehot_dim': 20,
        'seq_len': 4,
        'total_size': 149360
    },

    'D12_TrpB_Arnold_2024': {
        'onehot_dim': 20,
        'seq_len': 4,
        'total_size': 159128
    },

    'D13_folA_Wagner_2023': {
        'onehot_dim': 4,
        'seq_len': 9,
        'total_size': 261333
    },

    'D14_CreiLOV_Tong_2023': {
        'onehot_dim': 20,
        'seq_len': 15,
        'total_size': 165428
    },
}

def get_config(dataset):
    return configs[dataset]
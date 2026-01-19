import torch
import torch.nn as nn

def ModelReinit(model):
    onehot_dim = model.onehot_dim
    seq_len = model.seq_len
    model.__init__(onehot_dim = onehot_dim, seq_len = seq_len)
    return model

class MLPEnModel(nn.Module):
    def __init__(self, onehot_dim, seq_len):
        super(MLPEnModel, self).__init__()
        self.seq_len = seq_len
        self.onehot_dim = onehot_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
        self.mlp_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(onehot_dim*seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        mu = self.mlp_net(x)
        return mu
    
class CNNEnModel(nn.Module):
    def __init__(self, onehot_dim, seq_len):
        super(CNNEnModel, self).__init__()
        self.seq_len = seq_len
        self.onehot_dim = onehot_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(onehot_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        mu = self.conv_net(x)
        return mu

class MLPMcModel(nn.Module):
    def __init__(self, onehot_dim, seq_len, dropout_rate=0.1):
        super(MLPMcModel, self).__init__()
        self.seq_len = seq_len
        self.onehot_dim = onehot_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        self.mlp_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(onehot_dim*seq_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        mu = self.mlp_net(x)
        return mu
    
    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

class CNNMcModel(nn.Module):
    def __init__(self, onehot_dim, seq_len, dropout_rate=0.1):
        super(CNNMcModel, self).__init__()
        self.seq_len = seq_len
        self.onehot_dim = onehot_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(onehot_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        mu = self.conv_net(x)
        return mu

    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

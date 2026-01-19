import torch
import torch.nn as nn
import gpytorch

class DKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, onehot_dim=20, seq_len=4):
        super(DKLModel, self).__init__(train_x, train_y, likelihood)

        self.seq_len = seq_len
        self.onehot_dim = onehot_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(onehot_dim*seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
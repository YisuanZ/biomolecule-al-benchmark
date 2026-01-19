import torch
import numpy as np
from scipy.special import erfinv
from scipy.stats import pearsonr, spearmanr

def get_ENCE(preds, stds, labels, n_bins=10):
    
    preds, stds, labels = torch.tensor(preds), torch.tensor(stds), torch.tensor(labels)
    errors = torch.abs(preds - labels)
    uncertainties = stds
    normalized_unc = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())

    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = torch.bucketize(normalized_unc, bins[1:-1])
    ence = 0.0
    for bin_idx in range(n_bins):
        bin_mask = (bin_indices == bin_idx)
        bin_size = bin_mask.sum().item()
        if bin_size == 0:
            continue
        
        bin_uncertainty = (uncertainties[bin_mask] ** 2).mean().sqrt().item()
        bin_error = (errors[bin_mask] ** 2).mean().sqrt().item()
        ence += (bin_size / len(labels)) * abs(bin_uncertainty - bin_error) / bin_uncertainty
    
    return ence

def get_coverage(preds, stds, labels, confidence_level=0.95):  # 接近置信水平最好
    alpha = np.sqrt(2) * erfinv(confidence_level)

    lbd = [p - alpha * s for p, s in zip(preds, stds)]
    ubd = [p + alpha * s for p, s in zip(preds, stds)]
    in_interval = [lower <= label <= upper for lower, upper, label in zip(lbd, ubd, labels)]  # bool列表
    coverage_rate = sum(in_interval) / len(labels)
    
    return coverage_rate

def get_regression_PCC(preds, labels):
    pcc, p_value = pearsonr(preds, labels)
    return pcc, p_value

def get_regression_SCC(preds, labels):
    scc, p_value = spearmanr(preds, labels)
    return scc, p_value

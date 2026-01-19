import torch
import gpytorch
import os
from tqdm import tqdm

DEVICE, = [torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), ]

def DKLTrain(model, likelihood, valid_x, valid_y, prefix, tags, rs, epochs=100):
    subdir_checks = '{}'.format(prefix)
    os.makedirs(subdir_checks, exist_ok=True)

    for tag_idx, tag in enumerate(tags):

        model = model.to(DEVICE)
        likelihood = likelihood.to(DEVICE)
        optimizer = torch.optim.Adam([
            {'params': model.feature_extractor.parameters()},
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': likelihood.parameters()}
        ], lr=1e-3)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        best_model_path = f"{subdir_checks}/best_n{tag_idx + 1}_rs{rs}.pth"

        for epoch in tqdm(range(0, epochs), desc=tag):

            model.train()
            likelihood.train()
            optimizer.zero_grad()
            train_pred = likelihood(model(model.train_inputs[0]))  # 直接使用已经传入的数据
            loss = -mll(train_pred, model.train_targets)
            loss.backward()
            optimizer.step()

            model.eval()
            likelihood.eval()
            with torch.no_grad():
                valid_pred = likelihood(model(valid_x))
                valid_loss = -mll(valid_pred, valid_y).item()

            tqdm.write(f"[{tag}] Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.6f} | Valid Loss: {valid_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        torch.save(model.state_dict(), best_model_path)

    return
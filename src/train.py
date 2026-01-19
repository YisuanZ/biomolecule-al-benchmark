import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from tqdm import tqdm
from gpro.predictor.attnbilstm.attnbilstm import SequenceData
import model_zoo

DEVICE, = [torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), ]

def Train(train_seqs, train_labels, valid_seqs, valid_labels, model, prefix, tags, rs, epochs=100, rate=0.6):
    subdir_checks = '{}'.format(prefix)
    os.makedirs(subdir_checks, exist_ok=True)
    
    valid_x = torch.tensor(valid_seqs, dtype=torch.float32).to(DEVICE)
    valid_y = torch.tensor(valid_labels, dtype=torch.float32).to(DEVICE)

    for tag_idx, tag in enumerate(tags):
        
        np.random.seed(rs)
        random.seed(rs)
        idx = random.sample(range(len(train_seqs)), k=int(rate*len(train_seqs)))
        train_seqs_local = np.array(train_seqs)[idx]
        train_label_local = np.array(train_labels)[idx]

        train_x = torch.tensor(train_seqs_local, dtype=torch.float32).to(DEVICE)
        train_y = torch.tensor(train_label_local, dtype=torch.float32).to(DEVICE)
        train_dataset = SequenceData(train_x, train_y)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

        model = model_zoo.ModelReinit(model).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        best_model_path = f"{subdir_checks}/best_n{tag_idx + 1}_rs{rs}.pth"

        for epoch in range(epochs):
            
            model.train()
            train_epoch_loss = []

            for x, y in train_dataloader:
                x = x.to(torch.float32).to(DEVICE)
                y = y.to(torch.float32).to(DEVICE)
                
                pred = model(x)
                loss = criterion(pred.flatten(), y.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())
            
            avg_train_loss = np.mean(train_epoch_loss)

            model.eval()
            with torch.no_grad():
                valid_pred = model(valid_x)
                valid_loss = criterion(valid_pred.flatten(), valid_y).item()

            tqdm.write(f"[{tag}] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {valid_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        torch.save(model.state_dict(), best_model_path)

    return

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import os

class Luna3DCNN(nn.Module):
    def __init__(self):
        super(Luna3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # [B, 32, 32, 32, 32]
        x = self.pool1(x)            # [B, 32, 16, 16, 16]

        x = F.relu(self.conv2(x))    # [B, 64, 16, 16, 16]
        x = self.pool2(x)            # [B, 64, 8, 8, 8]

        x = F.relu(self.conv3(x))    # [B, 128, 8, 8, 8]
        x = self.global_pool(x)      # [B, 128, 1, 1, 1]
        x = x.view(x.size(0), -1)    # [B, 128]
        x = self.fc(x)               # [B, 1]

        return torch.sigmoid(x)      # Sigmoid for binary classification

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    all_preds, all_labels = [], []

    for x, y in tqdm(loader, desc="Training"):
        x = x.to(device).float()
        y = y.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_preds.extend(output.detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds > 0.5)
    auc = roc_auc_score(all_labels, all_preds)
    return np.mean(losses), acc, auc

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x = x.to(device).float()
            y = y.to(device).float().view(-1, 1)

            output = model(x)
            loss = criterion(output, y)

            losses.append(loss.item())
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds > 0.5)
    auc = roc_auc_score(all_labels, all_preds)
    return np.mean(losses), acc, auc

def run_training(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, checkpoint_path="model.pt"):
    best_auc = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Saved new best model (AUC: {best_auc:.4f})")

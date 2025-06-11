import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import numpy as np

class Luna3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  # BCEWithLogitsLoss expects raw logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    all_preds, all_labels = [], []

    for x, y in tqdm(loader, desc="Training"):
        x = x.to(device).float()
        y = y.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        output = model(x)
        output = torch.clamp(output, -10, 10)  # Clamp logits
        loss = criterion(output, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
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
            output = torch.clamp(output, -10, 10)  # Clamp logits
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
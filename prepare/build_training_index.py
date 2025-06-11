import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_balanced_training_csv(patch_folder, output_csv, oversample_pos=True, downsample_neg=True, seed=42):
    np.random.seed(seed)
    
    all_files = os.listdir(patch_folder)
    data = []
    
    for fname in all_files:
        if not fname.endswith(".npy"):
            continue
        label = 1 if "_pos" in fname else 0
        path = os.path.join(patch_folder, fname)
        data.append((path, label))

    df = pd.DataFrame(data, columns=["path", "label"])
    
    # Separate
    df_pos = df[df["label"] == 1]
    df_neg = df[df["label"] == 0]

    print(f"Original: {len(df_pos)} positive, {len(df_neg)} negative")

    if downsample_neg and len(df_neg) > len(df_pos):
        df_neg = df_neg.sample(n=len(df_pos), replace=False, random_state=seed)

    if oversample_pos:
        df_pos = df_pos.sample(n=len(df_neg), replace=True, random_state=seed)

    df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df_balanced.to_csv(output_csv, index=False)
    print(f"Balanced dataset saved: {output_csv}")
    print(f"Final counts → Positive: {len(df_pos)}, Negative: {len(df_neg)}")

def collect_positive_patches(patch_folder):
    data = []
    for fname in os.listdir(patch_folder):
        if "_pos" in fname and fname.endswith(".npy"):
            path = os.path.join(patch_folder, fname)
            data.append((path, 1))
    return pd.DataFrame(data, columns=["path", "label"])

def collect_negative_patches(patch_folder):
    data = []
    for fname in os.listdir(patch_folder):
        if "_neg" in fname and fname.endswith(".npy"):
            path = os.path.join(patch_folder, fname)
            data.append((path, 0))
    return pd.DataFrame(data, columns=["path", "label"])

def create_train_val_csvs(pos_patch_folder, neg_patch_folder,train_csv, val_csv, seed=42, ratio=1):
    if ratio < 1:
        raise ValueError("Ratio must be >= 1. Use ratio=1 for balanced training.")
    np.random.seed(seed)
    
    df_pos = collect_positive_patches(pos_patch_folder)
    df_neg = collect_negative_patches(neg_patch_folder)
    print(f"Original: {len(df_pos)} positive, {len(df_neg)} negative")

    # --- Split positives and negatives for validation ---
    val_pos = df_pos.sample(frac=0.2, random_state=seed)
    val_neg = df_neg.sample(n=len(val_pos), replace=False, random_state=seed)  # 1:1 for validation

    train_pos = df_pos.drop(val_pos.index)
    train_neg = df_neg.drop(val_neg.index)

    # --- Create custom:1 train ratio ---
    max_neg = min(len(train_neg), len(train_pos) // ratio)
    train_neg = train_neg.sample(n=max_neg, replace=False, random_state=seed)
    train_pos = train_pos.sample(n=ratio * max_neg, replace=True, random_state=seed)

    # --- Combine and shuffle ---
    train_df = pd.concat([train_pos, train_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat([val_pos, val_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"[✓] Training CSV saved: {train_csv} (Pos: {len(train_pos)}, Neg: {len(train_neg)})")
    print(f"[✓] Validation CSV saved: {val_csv} (Pos: {len(val_pos)}, Neg: {len(val_neg)})")

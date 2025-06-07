import os
import pandas as pd
import numpy as np

def build_training_index(patch_folder, output_csv):
    data = []

    for fname in os.listdir(patch_folder):
        if not fname.endswith('.npy'):
            continue
        label = 1 if "_pos" in fname else 0
        full_path = os.path.join(patch_folder, fname)
        data.append((full_path, label))

    df = pd.DataFrame(data, columns=["path", "label"])
    df.to_csv(output_csv, index=False)
    print(f"Saved training index with {len(df)} samples to {output_csv}")

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

    if downsample_neg and len(df_neg) > 2 * len(df_pos):
        df_neg = df_neg.sample(n=len(df_pos), replace=False, random_state=seed)

    if oversample_pos:
        df_pos = df_pos.sample(n=len(df_neg), replace=True, random_state=seed)

    df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df_balanced.to_csv(output_csv, index=False)
    print(f"Balanced dataset saved: {output_csv}")
    print(f"Final counts → Positive: {len(df_pos)}, Negative: {len(df_neg)}")

import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from .utils import world_to_voxel, extract_patch
from .classes import Advanced3DAugment

def normalize_path(path):
    return path.replace("\\", "/")

def extract_negative_patches_from_candidates(
    candidates_csv,
    annotations_csv,
    metadata_csv,
    output_folder,
    patch_size=32,
    max_negatives_per_scan=5,
    intensity_threshold=0.05  # Mean intensity filter
):
    os.makedirs(output_folder, exist_ok=True)
    
    candidates_df = pd.read_csv(candidates_csv)
    annotations_df = pd.read_csv(annotations_csv)
    metadata_df = pd.read_csv(metadata_csv)

    # Convert annotation list for fast lookup
    def is_nodule(candidate, annotations_for_scan, distance_threshold=6):
        for _, row in annotations_for_scan.iterrows():
            distance = np.linalg.norm(np.array(candidate) - np.array([row["coordX"], row["coordY"], row["coordZ"]]))
            if distance < distance_threshold:
                return True
        return False

    for _, meta in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting Negatives"):
        seriesuid = meta["case_id"]
        scan_path = meta["path"]
        origin = np.array([meta["origin_x"], meta["origin_y"], meta["origin_z"]])
        spacing = np.array([meta["spacing_x"], meta["spacing_y"], meta["spacing_z"]])

        volume = np.load(scan_path)
        scan_candidates = candidates_df[candidates_df["seriesuid"] == seriesuid]
        scan_annotations = annotations_df[annotations_df["seriesuid"] == seriesuid]

        # Remove candidates near actual nodules
        negative_candidates = []
        for _, cand in scan_candidates.iterrows():
            if not is_nodule([cand["coordX"], cand["coordY"], cand["coordZ"]], scan_annotations):
                negative_candidates.append([cand["coordX"], cand["coordY"], cand["coordZ"]])

        # Limit to a few negatives per scan
        sampled_candidates = random.sample(negative_candidates, min(len(negative_candidates), max_negatives_per_scan))

        for idx, world_coord in enumerate(sampled_candidates):
            voxel_coord = world_to_voxel(world_coord, origin, spacing)
            patch = extract_patch(volume, voxel_coord, patch_size)

            # Skip mostly empty patches
            if patch.mean() < intensity_threshold:
                continue

            # Sanity check on shape
            if patch.shape != (patch_size, patch_size, patch_size):
                print(f"[!] BAD SHAPE in {seriesuid} at idx {idx} | Voxel coord: {voxel_coord} | Shape: {patch.shape}")
                
                # Optionally save to debug later
                bad_path = os.path.join(output_folder, f"{seriesuid}_neg_{idx}_BADSHAPE.npy")
                np.save(bad_path, patch)
                continue  # Skip saving this malformed patch


            # Save
            
            filename = f"{seriesuid}_neg_{idx}.npy"
            np.save(os.path.join(output_folder, filename), patch)

def _extract_patches_from_annotations(
    annotation_csv,
    metadata_csv,
    output_folder,
    patch_size=32,
    intensity_threshold=0.05,  # Consistent with negatives
    intensity_filter=False
):
    os.makedirs(output_folder, exist_ok=True)

    annotations_df = pd.read_csv(annotation_csv)
    metadata_df = pd.read_csv(metadata_csv)

    
    # !Debugging
    # Check intersection of UIDs
    annotation_uids = set(annotations_df["seriesuid"].unique())
    metadata_uids = set(metadata_df["case_id"].unique())

    missing = annotation_uids - metadata_uids
    print(f"\n[INFO] Missing {len(missing)} out of {len(annotation_uids)} UIDs in metadata.")
    """ 
    # Show a few examples
    for uid in list(missing)[:10]:
        print(f"Missing UID in metadata: {uid}")
        example_uid = uid
        print(f"Annotation UID match: {example_uid in annotation_uids}")
        print(f"Metadata UID match: {example_uid in metadata_uids}")
    return 0
    # ! 
    """

    for idx, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Extracting Positives"):
        
        seriesuid = row['seriesuid']
        scan_meta = metadata_df[metadata_df["case_id"] == seriesuid]

        if scan_meta.empty:
            print(f"[!] No metadata found for {seriesuid}, skipping.")
            continue

        meta = scan_meta.iloc[0]
        origin = np.array([meta["origin_x"], meta["origin_y"], meta["origin_z"]])
        spacing = np.array([meta["spacing_x"], meta["spacing_y"], meta["spacing_z"]])
        scan_path = normalize_path(meta["path"])

        try:
            volume = np.load(scan_path)
        except Exception as e:
            print(f"[!] Failed to load scan at {scan_path}: {e}")
            continue

        world_coord = np.array([row["coordX"], row["coordY"], row["coordZ"]])
        voxel_coord = world_to_voxel(world_coord, origin, spacing)

        patch = extract_patch(volume, voxel_coord, patch_size)

        # Skip low-density patches
        if intensity_filter:
            if patch.mean() < intensity_threshold:
                print(f"[!] Skipping low-intensity patch for {seriesuid} at idx {idx} (mean={patch.mean():.4f})")
                continue

        # Shape check
        if patch.shape != (patch_size, patch_size, patch_size):
            print(f"[!] BAD SHAPE in {seriesuid} at idx {idx} | Voxel coord: {voxel_coord} | Shape: {patch.shape}")
            bad_path = os.path.join(output_folder, f"{seriesuid}_pos_{idx}_BADSHAPE.npy")
            np.save(bad_path, patch)
            continue

        save_path = os.path.join(output_folder, f"{seriesuid}_pos_{idx}.npy")
        np.save(save_path, patch)

def extract_patches_from_annotations(
    annotation_csv, 
    metadata_csv, 
    output_folder, 
    patch_size=32,
    augmentation=True
):
    os.makedirs(output_folder, exist_ok=True)

    annotations = pd.read_csv(annotation_csv)
    metadata = pd.read_csv(metadata_csv)

    augmentor = Advanced3DAugment(target_shape=(patch_size, patch_size, patch_size))

    # Check for missing cases
    annotation_uids = set(annotations["seriesuid"].unique())
    metadata_uids = set(metadata["case_id"].unique())
    missing = annotation_uids - metadata_uids
    print(f"\n[INFO] Missing {len(missing)} out of {len(annotation_uids)} UIDs in metadata.")

    for idx, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Extracting Positive Patches"):
        case_id = row['seriesuid']
        meta = metadata[metadata["case_id"] == case_id]

        if meta.empty:
            continue

        meta = meta.iloc[0]
        origin = np.array([meta["origin_x"], meta["origin_y"], meta["origin_z"]])
        spacing = np.array([meta["spacing_x"], meta["spacing_y"], meta["spacing_z"]])

        raw_path = meta["path"]
        normalized_path = normalize_path(raw_path)
        image = np.load(normalized_path)  # shape: [Z, Y, X]

        world_coord = np.array([row['coordZ'], row['coordY'], row['coordX']])
        voxel_coord = world_to_voxel(world_coord, origin, spacing)
        z, y, x = voxel_coord
        half = patch_size // 2

        # Compute slice bounds, clipped to image
        z_min = max(z - half, 0)
        z_max = min(z + half, image.shape[0])
        y_min = max(y - half, 0)
        y_max = min(y + half, image.shape[1])
        x_min = max(x - half, 0)
        x_max = min(x + half, image.shape[2])

        patch = image[z_min:z_max, y_min:y_max, x_min:x_max]

        # Pad if needed
        pad_z = (patch_size - patch.shape[0])
        pad_y = (patch_size - patch.shape[1])
        pad_x = (patch_size - patch.shape[2])
        patch = np.pad(
            patch,
            [(0, max(0, pad_z)), (0, max(0, pad_y)), (0, max(0, pad_x))],
            mode='constant', constant_values=0
        )

        if patch.shape != (patch_size, patch_size, patch_size):
            print(f"[!] BAD SHAPE in {case_id} at idx {idx} | Voxel coord: {voxel_coord} | Shape: {patch.shape}")
            bad_path = os.path.join(output_folder, f"{case_id}_neg_{idx}_BADSHAPE.npy")
            np.save(bad_path, patch)
            continue

        # Save original
        orig_save_path = os.path.join(output_folder, f"{case_id}_{z}_{y}_{x}_pos.npy")
        np.save(orig_save_path, patch)

        if augmentation:
            # Augment & save
            aug_patch = augmentor(patch)
            aug_save_path = os.path.join(output_folder, f"{case_id}_{z}_{y}_{x}_pos_aug.npy")
            np.save(aug_save_path, aug_patch)

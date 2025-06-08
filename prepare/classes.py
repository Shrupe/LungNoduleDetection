import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates
import os
from scipy.ndimage import zoom

class LunaPatchDataset(Dataset):
    def __init__(self, csv_file, transform=None, hu_min=-1000, hu_max=400, zero_center=True):
        self.data_df = pd.read_csv(csv_file)
        self.transform = transform
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.zero_center = zero_center

    def normalize_hu(self, img):
        # Clip and scale to [0, 1]
        img = np.clip(img, self.hu_min, self.hu_max)
        img = (img - self.hu_min) / (self.hu_max - self.hu_min)

        if self.zero_center:
            img = img - 0.5  # Now in [-0.5, 0.5]

        return img.astype(np.float32)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        patch_path = row['path']
        label = int(row['label'])

        patch = np.load(patch_path)  # Shape: (D, H, W)
        patch = self.normalize_hu(patch)

        if self.transform:
            patch = self.transform(patch)

        # Add channel dimension for CNN input: (1, D, H, W)
        patch = torch.from_numpy(patch).unsqueeze(0)

        # Sanity check
        if patch.shape != (1, 32, 32, 32):
            print(f"[!] Bad shape at idx {idx}: {patch} | Label: {label}")

        return patch, label


class Advanced3DAugment:
    def __init__(self, 
                 target_shape=(32, 32, 32),
                 max_rotate=15,  # degrees
                 max_shift=5,    # voxels
                 max_scale=0.1,  # ±10%
                 elastic_alpha=500, elastic_sigma=20,
                 p_intensity=0.5):
        self.max_rotate = max_rotate
        self.max_shift = max_shift
        self.max_scale = max_scale
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.p_intensity = p_intensity
        self.target_shape = target_shape

    def __call__(self, patch):
        # Accept (D, H, W) or (1, D, H, W) input
        if patch.ndim == 4 and patch.shape[0] == 1:
            vol = patch[0]  # (D, H, W)
        elif patch.ndim == 3:
            vol = patch
        else:
            raise ValueError(f"Unexpected patch shape: {patch.shape}")

        D, H, W = vol.shape

        # 1. Random rotation around each axis
        for axis in [(1,2), (0,2), (0,1)]:
            angle = random.uniform(-self.max_rotate, self.max_rotate)
            vol = self.rotate_3d(vol, angle, axis)

        # 2. Random shift
        # 2. Random shift (rounded to int)
        shifts = [int(round(random.uniform(-self.max_shift, self.max_shift))) for _ in range(3)]
        vol = np.roll(vol, shifts, axis=(0, 1, 2))


        # 3. Random scale (resample & crop/pad)
        scale = 1.0 + random.uniform(-self.max_scale, self.max_scale)
        vol = self.rescale(vol, scale)

        # 4. Elastic deformation
        vol = self.elastic_transform(vol, self.elastic_alpha, self.elastic_sigma)

        # 5. Photometric variation
        if random.random() < self.p_intensity:
            vol = self.intensity_transform(vol)

        # Force crop/resize to target_shape
        vol = self._resize_or_crop(vol, self.target_shape)

        return vol.astype(np.float32)

    def rotate_3d(self, vol, angle, axes):
        from scipy.ndimage import rotate
        return rotate(vol, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0)

    def rescale(self, vol, scale):
        from scipy.ndimage import zoom
        zoomed = zoom(vol, scale, order=1)
        # center crop or pad
        result = np.zeros_like(vol)
        cd = (np.array(zoomed.shape) - np.array(vol.shape)) // 2
        if scale >= 1:
            result = zoomed[
                cd[0]:cd[0]+vol.shape[0],
                cd[1]:cd[1]+vol.shape[1],
                cd[2]:cd[2]+vol.shape[2]
            ]
        else:
            result[
                -cd[0]:-cd[0]+zoomed.shape[0],
                -cd[1]:-cd[1]+zoomed.shape[1],
                -cd[2]:-cd[2]+zoomed.shape[2]
            ] = zoomed
        return result

    def elastic_transform(self, vol, alpha, sigma):
        shape = vol.shape
        dx = gaussian_filter((np.random.rand(*shape)*2 -1), sigma)*alpha
        dy = gaussian_filter((np.random.rand(*shape)*2 -1), sigma)*alpha
        dz = gaussian_filter((np.random.rand(*shape)*2 -1), sigma)*alpha
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]), indexing='xy')
        coords = np.array([y+dy, x+dx, z+dz])
        return map_coordinates(vol, coords, order=1, mode='reflect')

    def intensity_transform(self, vol):
        # Random contrast and brightness
        vol = vol * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
        return np.clip(vol, 0, 1)

    def center_crop(self, vol, target_shape=(32, 32, 32)):
        z, y, x = vol.shape
        tz, ty, tx = target_shape
        startz = (z - tz) // 2
        starty = (y - ty) // 2
        startx = (x - tx) // 2
        return vol[startz:startz+tz, starty:starty+ty, startx:startx+tx]
    
    def _resize_or_crop(self, vol, target_shape):
        """
        Resize or crop a volume to the target shape.
        """
        current_shape = vol.shape
        if current_shape == target_shape:
            return vol

        # If too big, center crop
        if all(cs >= ts for cs, ts in zip(current_shape, target_shape)):
            start = [(cs - ts) // 2 for cs, ts in zip(current_shape, target_shape)]
            return vol[
                start[0]:start[0]+target_shape[0],
                start[1]:start[1]+target_shape[1],
                start[2]:start[2]+target_shape[2]
            ]
        else:
            # Resize if too small or mixed
            zoom_factors = [ts / cs for ts, cs in zip(target_shape, current_shape)]
            return zoom(vol, zoom=zoom_factors, order=1)
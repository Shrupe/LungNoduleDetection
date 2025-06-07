import numpy as np

def world_to_voxel(world_coord, origin, spacing):
    """
    Convert world coordinates (mm) to voxel coordinates.
    
    Parameters:
        world_coord: [x, y, z] in mm
        origin: [x, y, z] of scan origin in mm
        spacing: voxel spacing in mm
        
    Returns:
        Voxel coordinate as a list of ints [z, y, x]
    """
    stretched = np.abs((np.array(world_coord) - origin) / spacing)
    return np.round(stretched).astype(int)[::-1]  # reverse to [z, y, x]

def extract_patch(volume, center_voxel, patch_size=32):
    """
    Extract a centered 3D patch from the volume.
    
    Parameters:
        volume: 3D numpy array (Z, Y, X)
        center_voxel: [z, y, x] center coordinate
        patch_size: Cube edge length
        
    Returns:
        Patch as 3D NumPy array of shape (patch_size, patch_size, patch_size)
    """
    half = patch_size // 2
    z, y, x = center_voxel

    z_min = max(z - half, 0)
    y_min = max(y - half, 0)
    x_min = max(x - half, 0)

    z_max = min(z_min + patch_size, volume.shape[0])
    y_max = min(y_min + patch_size, volume.shape[1])
    x_max = min(x_min + patch_size, volume.shape[2])

    patch = np.zeros((patch_size, patch_size, patch_size), dtype=volume.dtype)
    patch[:z_max - z_min, :y_max - y_min, :x_max - x_min] = volume[z_min:z_max, y_min:y_max, x_min:x_max]

    return patch

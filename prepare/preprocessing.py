import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from skimage import measure, morphology
import pandas as pd

def load_scan(filename):
    itk_img = sitk.ReadImage(filename)
    image = sitk.GetArrayFromImage(itk_img)  # [z, y, x]
    origin = np.array(itk_img.GetOrigin())[::-1]  # Convert to [z, y, x]
    spacing = np.array(itk_img.GetSpacing())[::-1]  # Convert to [z, y, x]
    return image, origin, spacing

def resample(image, spacing, new_spacing=[1, 1, 1]):
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

def get_lung_mask(image):
    binary_image = np.array(image > -320, dtype=np.int8)
    labels = measure.label(binary_image)
    background_label = labels[0, 0, 0]
    binary_image[labels == background_label] = 0
    labels = measure.label(binary_image)
    areas = [(l, np.sum(labels == l)) for l in np.unique(labels) if l != 0]
    areas.sort(key=lambda x: x[1], reverse=True)
    if len(areas) > 2:
        for label, _ in areas[2:]:
            binary_image[labels == label] = 0
    binary_image = morphology.binary_closing(binary_image, morphology.ball(2))
    return binary_image

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return np.clip(image, 0.0, 1.0)

def preprocess_mhd_file(mhd_path, fill_lung_structures=True):
    image, origin, spacing = load_scan(mhd_path)
    image, spacing = resample(image, spacing)
    lung_mask= segment_lung_mask(image, fill_lung_structures)
    image = normalize(image) * lung_mask  # Mask non-lung area
    return image, origin, spacing

def count_subset_directories(directory):
    try:
        return sum(
            1 for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name)) and name.startswith("subset")
        )
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return 0
    except PermissionError:
        print(f"Error: Permission denied to access '{directory}'.")
        return 0
    
def process_all_scans(input_root, output_root, fill_lung_structures=True):

    # adjust output file path and subset folders
    os.makedirs(output_root, exist_ok=True)
    subset_folder_count = count_subset_directories(input_root)
    SUBSET_FOLDERS = [f"subset{i}" for i in range(subset_folder_count)]
    
    metadata = []
    
    for subset in SUBSET_FOLDERS:
        subset_path = os.path.join(input_root, subset)
        if not os.path.isdir(subset_path):
            continue

        mhd_files = [f for f in os.listdir(subset_path) if f.endswith('.mhd')]
        print(f"Processing {len(mhd_files)} files in {subset}...")

        for mhd in mhd_files:
            full_path = os.path.join(subset_path, mhd)
            case_id = mhd.replace(".mhd", "")
            try:
                image, origin, spacing = preprocess_mhd_file(full_path, fill_lung_structures)
                save_path = os.path.join(output_root, f"{case_id}.npy")
                np.save(save_path, image)

                metadata.append({
                    "case_id": case_id,
                    "origin_z": origin[0], "origin_y": origin[1], "origin_x": origin[2],
                    "spacing_z": spacing[0], "spacing_y": spacing[1], "spacing_x": spacing[2],
                    "shape_z": image.shape[0], "shape_y": image.shape[1], "shape_x": image.shape[2],
                    "path": save_path
                })

                print(f"Saved: {save_path} | Shape: {image.shape}")
            except Exception as e:
                print(f"Failed to process {mhd}: {e}")

    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_root, "preprocessed_metadata.csv"), index=False)
    print(f"\nSaved metadata for {len(metadata)} cases.")
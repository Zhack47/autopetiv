import os
import nibabel as nib

import numpy as np
import json
from scipy.ndimage import gaussian_filter


def generate_gaussian_heatmap(coords, shape, sigma=2.0):
    from scipy.ndimage import gaussian_filter
    """
    Generate a 3D Gaussian heatmap for given coordinates.

    Args:
        coords (list): List of [x, y, z] coordinates.
        shape (tuple): Shape of the output volume.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: 3D volume with Gaussian heatmaps at the specified coordinates.
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    for coord in coords:
        if 0 <= coord[0] < shape[0] and 0 <= coord[1] < shape[1] and 0 <= coord[2] < shape[2]:
            heatmap[tuple(coord)] = 1.0

    heatmap = gaussian_filter(heatmap, sigma=sigma)    
    return heatmap

def get_coords(json_path):
    json_file = json.load(open(json_path, "rb"))
    print(json_file)
    tumor_coords = []
    bg_coords = []
    points = json_file["points"]
    for point in points:
        point_type = point["name"]
        if point_type == "tumor":
            tumor_coords.append(point["point"])
        elif point_type == "background":
            bg_coords.append(point["point"])
        else:
            print(f"Unknown point type {point_type}")
    return tumor_coords, bg_coords


def save_heatmap(pet_path, tumor_clicks, bg_clicks, output_path):
    pet_nii = nib.load(pet_path)
    ref = pet_nii.get_fdata()
    ref_shape = ref.shape
    ref_affine = pet_nii.affine
    point_map_pos = np.zeros(ref_shape)
    point_map_neg = np.zeros(ref_shape)
    for t_click in tumor_clicks:
        point_map_pos[(t_click[0],t_click[1],t_click[2])]=1
    for b_click in bg_clicks:
        point_map_neg[(b_click[0],b_click[1],b_click[2])]=1
    heatmap_pos =  gaussian_filter(point_map_pos, 3)
    heatmap_neg =  gaussian_filter(point_map_neg, 6)
    heatmap_pos[ref<=4] = 0
    heatmap_neg[heatmap_pos!=0] = 0
    max_pos = np.max(heatmap_pos)
    max_neg = np.max(heatmap_neg)
    if max_pos:
        heatmap = heatmap_pos/max_pos - heatmap_neg/max_neg
    else:
        heatmap = - heatmap_neg/max_neg
    out = nib.Nifti1Image(heatmap, affine=ref_affine)
    nib.save(out, output_path)


def save_click_heatmaps(click_file, output, input_pet):
    tumor_clicks, bg_clicks = get_coords(click_file)
    save_heatmap(input_pet,
                        tumor_clicks,
                        bg_clicks,
                        os.path.join(output,
                                     f'{input_pet.split("/")[-1].split("_0001.nii.gz")[0]}_0002.nii.gz'))
    
    
    '''pet_img = nib.load(input_pet)
    ref_shape = pet_img.shape
    ref_affine = pet_img.affine
    tumor_coords = clicks['tumor']
    non_tumor_coords = clicks['background']
    
    tumor_heatmap = generate_gaussian_heatmap(tumor_coords, ref_shape, 3)
    non_tumor_heatmap = generate_gaussian_heatmap(non_tumor_coords, ref_shape, 3)

    tumor_nifti = nib.Nifti1Image(tumor_heatmap, ref_affine)
    non_tumor_nifti = nib.Nifti1Image(non_tumor_heatmap, ref_affine)

    os.makedirs(output, exist_ok = True)

    nib.save(tumor_nifti, os.path.join(output, f'{input_pet.split("/")[-1].split("_0001.nii.gz")[0]}_0002.nii.gz')) # foreground clicks
    nib.save(non_tumor_nifti, os.path.join(output, f'{input_pet.split("/")[-1].split("_0001.nii.gz")[0]}_0003.nii.gz')) # background clicks'''

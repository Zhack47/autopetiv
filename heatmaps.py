import os
import json
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def get_coords(json_path):
    json_file = json.load(open(json_path, "rb"))
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
    point_map = np.zeros(ref_shape)
    print(ref_shape)
    for t_click in tumor_clicks:
        point_map[tuple(t_click)]=1
    for b_click in bg_clicks:
        point_map[tuple(b_click)]=-1
    heatmap = gaussian_filter(point_map, 6)
    out = nib.Nifti1Image(heatmap, affine=ref_affine)
    nib.save(out, output_path)


json_root_path = "FDG_PSMA_PETCT_pre-simulated_clicks"
images_root_path = "/mnt/disk_2/Zach/autopetIV/imagesTr"
for json_path in sorted(os.listdir(json_root_path))[:4]:
    tumor_clicks, bg_clicks = get_coords(os.path.join(json_root_path, json_path))
    pet_path = os.path.join(images_root_path, json_path.replace("_clicks.json", "_0001.nii.gz"))
    print(tumor_clicks)
    save_heatmap(pet_path, tumor_clicks, bg_clicks, json_path.replace("_clicks.json", "_0002.nii.gz"))

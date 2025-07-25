import os
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
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
    point_map_pos = np.zeros(ref_shape, dtype=np.float32)
    point_map_neg = np.zeros(ref_shape, dtype=np.float32)
    for t_click in tumor_clicks:
        point_map_pos[(t_click[0],t_click[1],t_click[2])]=1.
    for b_click in bg_clicks:
        point_map_neg[(b_click[0],b_click[1],b_click[2])]=1.
    heatmap_pos =  gaussian_filter(point_map_pos, 3)
    heatmap_neg =  gaussian_filter(point_map_neg, 6)
    
    # Set to zero points where SUV is lower than 4
    heatmap_pos[ref<=4] = 0
    
    # Set to zeros points that are in the positive gaussian map but not a background click
    print(point_map_neg)
    print(heatmap_pos)
    heatmap_neg[heatmap_pos!=0 & point_map_neg==0] = 0  
    
    # Normalizing between 0 and 1
    max_pos = np.max(heatmap_pos)
    max_neg = np.max(heatmap_neg)
    
    if max_pos:
        heatmap = heatmap_pos/max_pos - heatmap_neg/max_neg
    else:  # There are no foreground points
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


if __name__ == "__main__":
    json_root_path = "FDG_PSMA_PETCT_pre-simulated_clicks"
    images_root_path = "/mnt/disk_2/Zach/autopetIV/imagesTr"
    for json_path in tqdm(sorted(os.listdir(json_root_path))):
        tumor_clicks, bg_clicks = get_coords(os.path.join(json_root_path, json_path))
        pet_path = os.path.join(images_root_path, json_path.replace("_clicks.json", "_0001.nii.gz"))
        print(os.path.join("/mnt/disk_2/Zach/Autopet_Heatmaps", json_path.replace("_clicks.json", "_0002.nii.gz")))
        save_heatmap(pet_path, tumor_clicks, bg_clicks, os.path.join("/mnt/disk_2/Zach/Autopet_Heatmaps", json_path.replace("_clicks.json", "_0002.nii.gz")))



# For local testing
'''pet_path = "fdg_e252be4334_07-09-2004-NA-PET-CT Ganzkoerper  primaer mit KM-45425_0001.nii.gz"
json_path = "FDG_PSMA_PETCT_pre-simulated_clicks/fdg_e252be4334_07-09-2004-NA-PET-CT Ganzkoerper  primaer mit KM-45425_clicks.json"
tumor_clicks, bg_clicks = get_coords(json_path)
save_heatmap(pet_path, tumor_clicks, bg_clicks, json_path.replace("_clicks.json", "_0002.nii.gz").replace("FDG_PSMA_PETCT_pre-simulated_clicks/", ""))
'''
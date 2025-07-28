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


def save_empty_heatmap(pet_path, output_path):
    pet_nii = nib.load(pet_path)
    ref = pet_nii.get_fdata()
    ref_shape = ref.shape
    ref_affine = pet_nii.affine
    heatmap = np.zeros(ref_shape, dtype=np.float32)
    out = nib.Nifti1Image(heatmap, affine=ref_affine)
    nib.save(out, output_path)


def save_heatmap(pet_path, tumor_clicks, bg_clicks, output_path):
    pet_nii = nib.load(pet_path)
    ref = pet_nii.get_fdata()
    ref_shape = ref.shape
    ref_affine = pet_nii.affine
    point_map_neg = np.zeros(ref_shape, dtype=np.float32)
    heatmap_pos = np.zeros(ref_shape, dtype=np.float32)
    for t_click in tumor_clicks:
        point_map_pos = np.zeros(ref_shape, dtype=np.float32)
        point_map_pos[(t_click[0],t_click[1],t_click[2])]=1.
        suv_val_at_click = ref[t_click[0],t_click[1],t_click[2]]
        local_gauss = gaussian_filter(point_map_pos, 3)    
        # Set to zero points where SUV is lower than 4 or than SUV click
        local_gauss[ref<min(4, suv_val_at_click)] = 0
        max_pos = np.max(local_gauss)
        print(np.max(local_gauss))
        local_gauss/=max_pos
        print(np.max(local_gauss))
        heatmap_pos +=local_gauss
        heatmap_pos[heatmap_pos>1]=1
    for b_click in bg_clicks:
        point_map_neg[(b_click[0],b_click[1],b_click[2])]=1.
    
    heatmap_neg =  gaussian_filter(point_map_neg, 6)
    
    
    # Set to zeros points that are in the positive gaussian map but not a background click
    heatmap_neg[np.logical_and(heatmap_pos!=0, point_map_neg==0)] = 0  
    
    # Normalizing between 0 and 1
    max_pos = np.max(heatmap_pos)
    max_neg = np.max(heatmap_neg)
    print(max_neg)
    print(max_pos)
    
    if max_pos and max_neg:
        heatmap = heatmap_pos/max_pos - heatmap_neg/max_neg
    elif max_pos:  # There are no background points
        heatmap = heatmap_pos/max_pos
    elif max_neg:  # There are no foreground points
        heatmap = heatmap_neg/max_neg
    else:
        heatmap = heatmap_pos/max_pos
    print(np.max(heatmap))
    print(np.argwhere(heatmap==np.max(heatmap)))
    print(np.min(heatmap))
    out = nib.Nifti1Image(heatmap, affine=ref_affine)
    nib.save(out, output_path)


def save_click_heatmaps(click_file, output, input_pet):
    if os.path.exists(click_file)[0]:
        tumor_clicks, bg_clicks = get_coords(click_file)
        save_heatmap(input_pet,
                            tumor_clicks,
                            bg_clicks,
                            os.path.join(output,
                                         f'{input_pet.split("/")[-1].split("_0001.nii.gz")[0]}_0002.nii.gz'))
    else:
        save_empty_heatmap(input_pet, os.path.join(output,
                                         f'{input_pet.split("/")[-1].split("_0001.nii.gz")[0]}_0002.nii.gz'))


if __name__ == "__main__":
    json_root_path = "../../FDG_PSMA_PETCT_pre-simulated_clicks"
    images_root_path = "/mnt/disk_2/Zach/nnunet2_db/Dataset514_AUTOPETIV/imagesTr"
    for json_path in tqdm(sorted(os.listdir(json_root_path))):
        tumor_clicks, bg_clicks = get_coords(os.path.join(json_root_path, json_path))
        pet_path = os.path.join(images_root_path, json_path.replace("_clicks.json", "_0001.nii.gz"))
        print(os.path.join("/mnt/disk_2/Zach/Autopet_Heatmaps", json_path.replace("_clicks.json", "_0002.nii.gz")))
        save_heatmap(pet_path, tumor_clicks, bg_clicks, os.path.join("/mnt/disk_2/Zach/Autopet_Heatmaps", json_path.replace("_clicks.json", "_0002.nii.gz")))



# For local testing
'''pet_path = "../../fdg_01140d52d8_08-13-2005-NA-PET-CT Ganzkoerper  primaer mit KM-56839.nii.gz"
json_path = "../../FDG_PSMA_PETCT_pre-simulated_clicks/fdg_01140d52d8_08-13-2005-NA-PET-CT Ganzkoerper  primaer mit KM-56839_clicks.json"
tumor_clicks, bg_clicks = get_coords(json_path)
save_heatmap(pet_path, tumor_clicks, bg_clicks, json_path.replace("_clicks.json", "_0002.nii.gz").replace("FDG_PSMA_PETCT_pre-simulated_clicks/", ""))
'''
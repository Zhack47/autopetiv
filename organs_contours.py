if __name__ == "__main__":
    import os
    import nibabel
    import numpy as np
    from tqdm import tqdm
    import SimpleITK as sitk
    from os.path import join
    from totalsegmentator.python_api import totalsegmentator, class_map
    root_path = "/mnt/disk_2/Zach/autopetIV/"
    names = [i.split(".nii.gz")[0] for i in sorted(os.listdir(join(root_path, "labelsTr")))]

    bones = ['sacrum', 'vertebrae_S1', 'vertebrae_L5', 'vertebrae_L4','vertebrae_L3','vertebrae_L2','vertebrae_L1','vertebrae_T12',
             'vertebrae_T11','vertebrae_T10', 'vertebrae_T9','vertebrae_T8','vertebrae_T7', 'vertebrae_T6','vertebrae_T5',
             'vertebrae_T4','vertebrae_T3', 'vertebrae_T2','vertebrae_T1','vertebrae_C7','vertebrae_C6','vertebrae_C5',
             'vertebrae_C4','vertebrae_C3', 'vertebrae_C2','vertebrae_C1', 'humerus_left', 'humerus_right', 'scapula_left',
             'scapula_right', 'clavicula_left', 'clavicula_right', 'femur_left', 'femur_right', 'hip_left', 'hip_right',
             'skull', 'rib_left_1','rib_left_2','rib_left_3','rib_left_4','rib_left_5','rib_left_6','rib_left_7',
             'rib_left_8', 'rib_left_9','rib_left_10','rib_left_11','rib_left_12','rib_right_1','rib_right_2',
             'rib_right_3', 'rib_right_4','rib_right_5','rib_right_6','rib_right_7','rib_right_8','rib_right_9',
             'rib_right_10', 'rib_right_11','rib_right_12','sternum']

    heart_aorta = ['heart', 'aorta']

    lungs = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
             "lung_middle_lobe_right", "lung_lower_lobe_right"]

    labels_autopet = {'background': 0, 'lesion': 1, 'brain': 2, 'liver': 3, 'kidneys': 4, 'urinary_bladder': 5,
                      'spleen': 6, 'digestive_system': 7, 'pancreas': 8, 'prostate': 9, 'skeleton': 10,
                      "heart_aorta": 11, "lungs": 12, "parotid_glands": 13, "submandibular_glands": 14}

    mapping_to_autopet_total = {'brain': ['brain'], 'liver': ['liver'], 'kidneys': ['kidney_left', 'kidney_right'],
                          'urinary_bladder': ['urinary_bladder'], 'spleen': ['spleen'],
                          'digestive_system': ['small_bowel', 'colon', 'duodenum', 'gallbladder', 'stomach'],
                          'pancreas':['pancreas'], 'prostate': ['prostate'], "skeleton": bones,
                          "heart_aorta": heart_aorta, 'lungs': lungs}
    mapping_to_autopet_hgc = {'parotid_gland': ["parotid_gland_left", "parotid_gland_right"],
                              'submandibular_gland': ["submandibular_gland_left", "submandibular_gland_right"]
                              }


    for patient_name in tqdm(names[:2]):
        print(patient_name)

        reverse_hgc_map = {}
        reverse_total_map = {}
        for key, value in class_map["total"].items():
           reverse_total_map[value] = key
        for key, value in class_map["head_glands_cavities"].items():
           reverse_hgc_map[value] = key
        
        ct_path = join(root_path, "imagesTr", f"{patient_name}_0000.nii.gz")
        label_path = join(root_path, "imagesTr", f"{patient_name}_0000.nii.gz")
        new_labels_path = '/mnt/disk_2/Zach/Autopet_supplementary_labels'

        output_seg_path = join(new_labels_path, f"{patient_name}_New_Labels.nii.gz")
        total_organs = totalsegmentator(ct_path, output_seg_path, task='total',
                                  roi_subset=['brain','liver', 'kidney_left', 'kidney_right', 'urinary_bladder',
                                              'prostate', 'spleen', 'pancreas', 'duodenum', 'small_bowel', 'colon',
                                              'stomach', 'gallbladder']+bones+heart_aorta+lungs, skip_saving=True)
        hgc_organs = totalsegmentator(ct_path, output_seg_path, task='head_glands_cavities', skip_saving=True)
        label_nii = nibabel.load(label_path)
        label_np = np.array(label_nii.dataobj)

        total_organs_np = np.array(total_organs.dataobj)
        hgc_organs_np = np.array(hgc_organs.dataobj)
        organs_np_copy = np.zeros_like(hgc_organs_np)

        for label, value in labels_autopet.items():
            for name, names in mapping_to_autopet_hgc.items():
                if label == name:
                    for orig_name in names:
                        src_value = reverse_hgc_map[orig_name]
                        organs_np_copy[hgc_organs_np==src_value] = value
            
            for name, names in mapping_to_autopet_total.items():
                if label == name:
                    for orig_name in names:
                        src_value = reverse_total_map[orig_name]
                        organs_np_copy[total_organs_np==src_value] = value

        # Adding lesions with most priority
        organs_np_copy[label_np == 1] = 1
        out_seg = nibabel.Nifti1Image(organs_np_copy, affine=total_organs.affine)

        nibabel.save(out_seg, join(new_labels_path, f"{patient_name}.nii.gz"))

import cc3d
import numpy as np
import nibabel as nib


def remove_small_lowval_components(nii_seg, nii_tep):
    np_seg = np.array(nii_seg.dataobj)
    np_tep = np.array(nii_tep.dataobj)
    aff = nii_seg.affine
    np_seg[np_seg!=1]=0
    components = cc3d.connected_components(np_seg, connectivity=26)
    for i in range(1, np.max(components)+1):
        component = components==i
        suv_max = np.max(np_tep[component])
        nb_voxels = np.sum(component)
        if nb_voxels < 10 and suv_max < 4:
            np_seg[component]=0
    return nib.Nifti1Image(np_seg, affine=aff)


def remove_small_lowval_components_numpy(np_seg, np_tep):
    np_seg[np_seg!=1]=0
    components = cc3d.connected_components(np_seg, connectivity=26)
    for i in range(1, np.max(components)+1):
        component = components==i
        suv_max = np.max(np_tep[component])
        nb_voxels = np.sum(component)
        if nb_voxels < 10 and suv_max < 4:
            np_seg[component]=0
    return np_seg
import os
import time
import json
import shutil
import subprocess
from pathlib import Path

import SimpleITK
import torch
import numpy as np

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join

from heatmaps import save_click_heatmaps
from smart_tracer_discriminator import SmartTracerDiscriminator, Tracer


class Autopet_baseline:

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        # according to the specified grand-challenge interfaces
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces
        self.output_path = "/output/images/tumor-lesion-segmentation/"
        self.nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        self.lesion_click_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/clicksTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)
    
    def gc_to_swfastedit_format(self, gc_json_path, swfast_json_path):
        with open(gc_json_path, 'r') as f:
            gc_dict = json.load(f)
        swfast_dict = {
            "tumor": [],
            "background": []
        }
        
        for point in gc_dict.get("points", []):
            if point["name"] == "tumor":
                swfast_dict["tumor"].append(point["point"])
            elif point["name"] == "background":
                swfast_dict["background"].append(point["point"])
        with open(swfast_json_path, 'w') as f:
            json.dump(swfast_dict, f)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )
        
        print("Generating heatmaps...")
        time_0_hm = time.time_ns()
        json_file = os.path.join(self.input_path, "lesion-clicks.json")
        save_click_heatmaps(json_file, self.nii_path,
                            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"))
        print(os.listdir(self.nii_path))
        time_1_hm = time.time_ns()
        print(f"Heatmaps generated. Took {(time_1_hm-time_0_hm)/1000000}ms")

        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.result_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        """
        Your algorithm goes here
        """

        print("nnUNet segmentation starting!")

        os.environ['nnUNet_compile'] = 'F'

        maybe_mkdir_p(self.output_path)

        trained_model_path_psma = "nnUNet_results/Dataset516_AUTOPETIVPSMA/nnUNetTrainer_organs_PSMA__nnUNetResEncUNetLPlans__3d_fullres"
        trained_model_path_fdg = "nnUNet_results/Dataset515_AUTOPETIVFDG/nnUNetTrainer_organs_FDG__nnUNetResEncUNetLPlans__3d_fullres"
        #trained_model_path = "nnUNet_results/Dataset514_AUTOPETIV/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"

        ct_mha = subfiles(join(self.input_path, 'images/ct/'), suffix='.mha')[0]
        ct_nii = os.path.join(self.nii_path, "TCIA_001_0000.nii.gz")  # subfiles(join(self.input_path, 'images/ct/'), suffix='.mha')[0]
        pet_nii = os.path.join(self.nii_path, "TCIA_001_0001.nii.gz")  # subfiles(join(self.input_path, 'images/pet/'), suffix='.mha')[0]
        hm_nii = os.path.join(self.nii_path, "TCIA_001_0002.nii.gz")
        uuid = os.path.basename(os.path.splitext(ct_mha)[0])
        output_file_trunc = os.path.join(self.output_path + uuid)


        print("Creating", end="")
        predictor = nnUNetPredictor(
            tile_step_size=0.6,
            use_mirroring=True,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True)
        print("Done")


        print("Reading images", end="")
        images, properties = SimpleITKIO().read_images([ct_nii, pet_nii, hm_nii])
        print("Done")

        ct = images[0]
        pt = images[1]
        hm = images[2]

        src_spacing = properties["sitk_stuff"]["spacing"]
        src_origin = properties["sitk_stuff"]["origin"]
        src_direction = properties["sitk_stuff"]["direction"]
        
        
        tracer = SmartTracerDiscriminator("dd_weights/weights", torch.device("cuda"))(SimpleITK.ReadImage(pet_nii))
        print(f"Found tracer: {tracer}")
        # TODO use final.pth
        if tracer == Tracer.PSMA:
            predictor.initialize_from_trained_model_folder(trained_model_path_psma, use_folds=(0,1,4), checkpoint_name="checkpoint_best.pth")
            target_spacing = tuple(map(float, json.load(open(join(trained_model_path_psma, "plans.json"), "r"))["configurations"][
                    "3d_fullres"]["spacing"]))
        elif tracer == Tracer.FDG:
            predictor.initialize_from_trained_model_folder(trained_model_path_fdg, use_folds=(0,3,4), checkpoint_name="checkpoint_best.pth")
            target_spacing = tuple(map(float, json.load(open(join(trained_model_path_fdg, "plans.json"), "r"))["configurations"][
                    "3d_fullres"]["spacing"]))
        fin_size = ct.shape
        new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(src_spacing, target_spacing[::-1], fin_size)])
        nb_voxels = np.prod(pt.shape)
        print(f"Resampled shape: {new_shape}")
        print(nb_voxels)
        print("Done")
        #predictor.configuration_manager.patch_size = (128,128,128)
        predictor.dataset_json['file_ending'] = '.mha'

        print("Stacking..", end="")
        images = np.stack([ct, pt, hm])
        print("Done")

        if nb_voxels < 3.5e7:
            predictor.predict_single_npy_array(images, properties, None, output_file_trunc, False)
        elif nb_voxels < 6.9e7:
            print("Removing one axis for prediction mirroring")
            predictor.allowed_mirroring_axes = (1, 2)
            predictor.predict_single_npy_array(images, properties, None, output_file_trunc, False)
        else:
            print("Removing all mirroring")
            predictor.allowed_mirroring_axes = None
            predictor.predict_single_npy_array(images, properties, None, output_file_trunc, False)



        out_image = SimpleITK.ReadImage(output_file_trunc+".mha")
        out_np = SimpleITK.GetArrayFromImage(out_image)
        print(out_np.shape)
        print(np.unique(out_np))

        # Keeping only the 'lesion' class
        oneclass_np = np.zeros_like(pt)
        oneclass_np[out_np==1]=1
        oneclass_image = SimpleITK.GetImageFromArray(oneclass_np.astype(np.uint8))
        oneclass_image.SetOrigin(src_origin)
        oneclass_image.SetSpacing(src_spacing)
        oneclass_image.SetDirection(src_direction)

        SimpleITK.WriteImage(oneclass_image, output_file_trunc+".mha")


        print("Prediction finished")

   
    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        #self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()

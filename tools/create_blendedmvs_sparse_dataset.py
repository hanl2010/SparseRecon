import numpy as np
import os
from glob import glob
import shutil
from tqdm import tqdm

scenes = [1,2,3,4,5,6,7,8,9]
data_path = "original BlendedMVS dataset path"
new_data_path = "BlendedMVS dataset new path (sparse images)"

bmvs_train_ids = {
    1: [9, 10, 55],
    2: [59, 9, 52],
    3: [26, 27, 22],
    4: [11, 39, 53],
    5: [32, 42, 47],
    6: [28, 34, 57],
    7: [5, 25, 2],
    8: [16, 21, 33],
    9: [16, 60, 10],
}

for scene in tqdm(scenes):
    scene_path = os.path.join(data_path, f"scan{scene}")
    image_files = glob(os.path.join(scene_path, "image/*.jpg"))
    image_files = sorted(image_files)
    # print(image_files)

    camera_file = os.path.join(scene_path, "cameras.npz")
    camera_dict = np.load(camera_file)
    new_camera_dict = {}
    image_indices = bmvs_train_ids[scene]
    for i, idx in enumerate(image_indices):
        image_file = image_files[idx]
        new_image_name = f"{i:06d}.png"
        new_image_path = os.path.join(new_data_path, f"scan{scene}", "images")
        os.makedirs(new_image_path, exist_ok=True)
        new_image_file = os.path.join(new_image_path, new_image_name)
        shutil.copy(image_file, new_image_file)

        mask_file = os.path.join(data_path, f"scan{scene}", f"mask/{idx:08d}.png")
        new_mask_name = f"{i:06d}.png"
        new_mask_path = os.path.join(new_data_path, f"scan{scene}", f"mask")
        os.makedirs(new_mask_path, exist_ok=True)
        new_mask_file = os.path.join(new_mask_path, new_mask_name)
        shutil.copy(mask_file, new_mask_file)


        new_camera_dict["world_mat_%d" % i] = camera_dict["world_mat_%d" % idx]
        new_camera_dict["scale_mat_%d" % i] = camera_dict["scale_mat_%d" % idx]

    new_camera_file = os.path.join(new_data_path, f"scan{scene}", "cameras.npz")
    np.savez(new_camera_file, **new_camera_dict)
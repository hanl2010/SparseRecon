import numpy as np
import os
from glob import glob
import shutil
from tqdm import tqdm

scenes = [21, 24, 34, 37, 38, 40, 82, 106, 110, 114, 118]
data_path = "original DTU dataset path "
data_type = "large_overlap or small_overlap"

if data_type == "small_overlap":
    image_indices = [22, 25, 28]
    new_data_path = "DTU dataset new path (sparse images of small_overlap)"
else:
    image_indices = [23, 24, 33]
    new_data_path = "DTU dataset new path (sparse images of large_overlap)"

for scene in tqdm(scenes):
    scene_path = os.path.join(data_path, f"scan{scene}")
    image_files = glob(os.path.join(scene_path, "images/*.png"))
    image_files = sorted(image_files)

    camera_file = os.path.join(scene_path, "cameras_sphere.npz")
    camera_dict = np.load(camera_file)
    new_camera_dict = {}
    for i, idx in enumerate(image_indices):
        image_file = image_files[idx]
        new_image_name = f"{i:06d}.png"
        new_image_path = os.path.join(new_data_path, f"scan{scene}", "images")
        os.makedirs(new_image_path, exist_ok=True)
        new_image_file = os.path.join(new_image_path, new_image_name)
        shutil.copy(image_file, new_image_file)

        mask_file = os.path.join(data_path, f"scan{scene}", f"mask/{idx:03d}.png")
        new_mask_name = f"{i:06d}.png"
        new_mask_path = os.path.join(new_data_path, f"scan{scene}", f"mask")
        os.makedirs(new_mask_path, exist_ok=True)
        new_mask_file = os.path.join(new_mask_path, new_mask_name)
        shutil.copy(mask_file, new_mask_file)


        new_camera_dict["world_mat_%d" % i] = camera_dict["world_mat_%d" % idx]
        new_camera_dict["scale_mat_%d" % i] = camera_dict["scale_mat_%d" % idx]

    new_camera_file = os.path.join(new_data_path, f"scan{scene}", "cameras_sphere.npz")
    np.savez(new_camera_file, **new_camera_dict)
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import argparse
import os

from glob import glob
import logging

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from marigold import MarigoldPipeline
from marigold.util.seed_all import seed_all


EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        # default="Bingxin/Marigold",
        default="/data10_1/hanl/.cache/huggingface/hub/models--Bingxin--Marigold/snapshots/0431bc71fafe5b8e86f3e1d9d594e299032d1c5f",
        # default="C:/Users/hanl/.cache/huggingface/hub/models--Bingxin--Marigold/snapshots/0431bc71fafe5b8e86f3e1d9d594e299032d1c5f",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to the input image folder.",)
    parser.add_argument("--dataset_name", dest="dataset_name", default="DTU", help="DTU or BlendedMVS")
    parser.add_argument("--depth_folder", type=str, default="depths_marigold", help="Output directory.")

    # inference setting
    parser.add_argument("--denoise_steps", type=int, default=10, help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",)
    parser.add_argument("--ensemble_size", type=int, default=10, help="Number of predictions to be ensembled, more inference gives better results but runs slower.",)
    parser.add_argument("--half_precision", action="store_true", help="Run with half-precision (16-bit float), might lead to suboptimal result.",)
    # resolution setting
    parser.add_argument("--processing_res", type=int, default=768, help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",)
    parser.add_argument("--output_processing_res", action="store_true", help="When input is resized, out put depth at resized operating resolution. Default: False.",)
    # depth map colormap
    parser.add_argument("--color_map", type=str, default="Spectral", help="Colormap used to render depth predictions.",)
    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=0, help="Inference batch size. Default: 0 (will be set automatically).",)
    parser.add_argument("--apple_silicon", action="store_true", help="Flag of running on Apple Silicon.",)

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size

    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # -------------------- Model --------------------
    if half_precision:
        print("using half precision")
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    pipe = MarigoldPipeline.from_pretrained(checkpoint_path, torch_dtype=dtype)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to(device)


    if args.dataset_name == "DTU":
        scenes = [21, 24, 34, 37, 38, 40, 82, 106, 110, 114, 118]
    elif args.dataset_name == "BlendedMVS":
        scenes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        raise NotImplementedError(args.dataset_name)

    for scene in scenes:
        print("processing scene:", scene)

        image_path = os.path.join(args.data_root, f"scan{scene}", "images")
        output_path = os.path.join(args.data_root, f"scan{scene}", args.depth_folder)
        os.makedirs(output_path, exist_ok=True)

        # Output directories
        output_dir_color = output_path
        output_dir_tif = output_path
        output_dir_npy = output_path
        logging.info(f"output dir = {output_path}")

        # -------------------- Data --------------------
        depth_filename_list = []

        rgb_filename_list = glob(os.path.join(image_path, "*"))
        n_images = len(rgb_filename_list)
        if n_images > 0:
            logging.info(f"Found {n_images} images")
        else:
            logging.error(f"No image found in '{image_path}'")
            exit(1)

        # -------------------- Inference and saving --------------------
        with torch.no_grad():
            for idx, rgb_path in tqdm(enumerate(rgb_filename_list), desc="Estimating depth", leave=True):
                # Read input image
                input_image = Image.open(rgb_path)
                image_w, image_h = input_image.size

                # Predict depth
                pipe_out = pipe(
                    input_image,
                    denoising_steps=denoise_steps,
                    ensemble_size=ensemble_size,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    batch_size=batch_size,
                    color_map=color_map,
                    show_progress_bar=True,
                )

                depth_pred: np.ndarray = pipe_out.depth_np
                depth_colored: Image.Image = pipe_out.depth_colored

                # Save as npy
                rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                depth_name_base = rgb_name_base + "_depth"
                npy_save_path = os.path.join(output_dir_npy, f"{depth_name_base}.npy")
                if os.path.exists(npy_save_path):
                    logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
                np.save(npy_save_path, depth_pred)

                # Colorize
                colored_save_path = os.path.join(output_dir_color, f"{depth_name_base}.png")
                if os.path.exists(colored_save_path):
                    logging.warning(
                        f"Existing file: '{colored_save_path}' will be overwritten"
                    )
                depth_colored.save(colored_save_path)

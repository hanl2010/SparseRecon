# adapted from https://github.com/EPFL-VILAB/omnidata
import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys

from omnidata.modules.midas.dpt_depth import DPTDepthModel
from omnidata.data.transforms import get_transform


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_path, output_file_name, task):
    with torch.no_grad():
        save_path = os.path.join(output_path, f'{output_file_name}_{task}.png')

        print(f'Reading input {img_path} ...')
        img = Image.open(img_path)
        w, h = img.size

        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        # rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
        # trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3, 1)

        # output = model(img_tensor).clamp(min=0, max=1)
        print("input image size:", img_tensor.shape)
        output = model(img_tensor)
        print("output image size:", output.shape)

        if task == 'depth':
            output = F.interpolate(output.unsqueeze(0), (h, w), mode='bicubic').squeeze(0)
            print("interpolated output size:", output.shape)

            np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])
            plt.imsave(save_path, output.detach().cpu().squeeze(), cmap='Spectral')

        else:
            # import pdb; pdb.set_trace()
            np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])
            trans_topil(output[0]).save(save_path)

        print(f'Writing output {save_path} ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')
    parser.add_argument('--pretrained_models', dest='pretrained_models', help="path to pretrained models")
    parser.set_defaults(pretrained_models='tools/omnidata/pretrained_models/')
    parser.add_argument('--task', dest='task', default="depth", help="normal or depth")
    # parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
    parser.add_argument("--data_root", dest="data_root", default="", required=True)
    parser.add_argument("--dataset_name", dest="dataset_name", default="DTU", help="DTU or BlendedMVS")


    args = parser.parse_args()
    weights_dir = args.pretrained_models

    trans_topil = transforms.ToPILImage()
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get target task and model
    if args.task == 'normal':
        image_size = (384, 512)

        pretrained_weights_path = os.path.join(weights_dir, 'omnidata_dpt_normal_v2.ckpt')
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size),
                                            get_transform('rgb', image_size=None)])

    elif args.task == 'depth':
        # image_size = 384
        image_size = (384, 512)
        pretrained_weights_path = os.path.join(weights_dir, 'omnidata_dpt_depth_v2.ckpt')  # 'omnidata_dpt_depth_v1.ckpt'
        # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([transforms.Resize(image_size),
                                            # transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=0.5, std=0.5)])

    else:
        print("task should be one of the following: normal, depth")
        sys.exit()


    if args.dataset_name == "DTU":
        scenes = [21, 24, 34, 37, 38, 40, 82, 106, 110, 114, 118]
    elif args.dataset_name == "BlendedMVS":
        scenes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        raise NotImplementedError(args.dataset_name)

    for scene in scenes:
        # fused_ply_path = os.path.join(args.data_root, f"scan{scene}/dense/fused.ply")
        # if not os.path.exists(fused_ply_path):
        #     print(f"{fused_ply_path} not exist.")
        #     continue

        image_path = os.path.join(args.data_root, f"scan{scene}/images")
        output_path = os.path.join(args.data_root, f"scan{scene}/depths_omnidata")
        os.makedirs(output_path, exist_ok=True)


        img_path = Path(image_path)
        print(img_path)
        if img_path.is_file():
            save_outputs(image_path, output_path, os.path.splitext(os.path.basename(args.img_path))[0], task=args.task)
        elif img_path.is_dir():
            for f in glob.glob(image_path+'/*'):
                save_outputs(f, output_path, os.path.splitext(os.path.basename(f))[0], task=args.task)
        else:
            print("invalid file path!")
            sys.exit()

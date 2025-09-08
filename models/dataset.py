import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import random

from models.utils import build_patch_offset
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from feat_extractor.vismvsnet.feat_utils import FeatExt
from feat_extractor.mvsformer.module import FPNEncoder
from feat_extractor.transmvsnet.module import FeatureNet


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class Camera:
    def __init__(self, image, intrinsic, c2w, image_name, image_index, mask=None, depth=None, depth_mask=None, image_feature=None,
                 src_image_names=None, data_device = "cuda"):
        super(Camera, self).__init__()

        self.data_device = data_device
        self.image_name = image_name
        self.image_index = image_index
        self.src_image_names = src_image_names
        self.intrinsic = intrinsic.cuda()
        self.c2w = c2w.cuda()
        self.intrinsic_inv = torch.inverse(intrinsic).cuda()
        self.w2c = torch.inverse(c2w).cuda()
        self.depth = depth.cuda() if depth is not None else None
        self.depth_mask = depth_mask.cuda() if depth_mask is not None else None


        self.gt_image = image.to(self.data_device)
        assert self.gt_image.shape[-1] == 3
        self.image_width = self.gt_image.shape[1]
        self.image_height = self.gt_image.shape[0]
        if mask is not None:
            self.gt_mask = mask.to(self.data_device)
            self.gt_image *= self.gt_mask
        else:
            self.gt_mask = torch.ones((self.image_height, self.image_width, 1), dtype=torch.float32).to(self.data_device)
            # self.gt_mask = None

        self.image_feature = image_feature
        self.pred_depth = torch.ones((self.image_height, self.image_width, 1)) ## 保存由NeuS渲染的depth，在训练过程中更新

        self.focal_x = intrinsic[0][0]
        self.focal_y = intrinsic[1][1]

    def gen_rays(self, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.image_width - 1, self.image_width // l)
        ty = torch.linspace(0, self.image_height - 1, self.image_height // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        uv = torch.stack([pixels_x, pixels_y], dim=-1)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.c2w[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.c2w[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), uv.transpose(0, 1)

    def gen_random_rays(self, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.image_width, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.image_height, size=[batch_size])
        color = self.gt_image[(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.gt_mask[(pixels_y, pixels_x)]      # batch_size, 1
        depth = self.depth[(pixels_y, pixels_x)]
        depth_mask = self.depth_mask[(pixels_y, pixels_x)]
        uv = torch.stack([pixels_x, pixels_y], dim=-1)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().cuda()  # batch_size, 3
        p = torch.matmul(self.intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_d = torch.matmul(self.c2w[None, :3, :3], rays_d[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.c2w[None, :3, 3].expand(rays_d.shape) # batch_size, 3
        data = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "color": color,
            "mask": mask,
            "uv": uv,
            "depth": depth,
            "depth_mask": depth_mask,
        }
        return data

    def gen_patch_rays(self, patch_num, half_patch_size=5):
        """
            Generate patch rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0+half_patch_size, high=self.image_width-half_patch_size, size=[patch_num])
        pixels_y = torch.randint(low=0+half_patch_size, high=self.image_height-half_patch_size, size=[patch_num])
        offset = build_patch_offset(half_patch_size=half_patch_size)
        uv = torch.stack([pixels_x, pixels_y], dim=-1)
        grid = uv.reshape(-1, 1, 2) + offset[None, :, :]
        grid = grid.reshape(-1, 2).to(torch.int64)
        pixels_x = grid[:, 0]
        pixels_y = grid[:, 1]
        uv = grid

        color = self.gt_image[(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.gt_mask[(pixels_y, pixels_x)]  # batch_size, 1
        # depth = self.depth[(pixels_y, pixels_x)]
        # depth_mask = self.depth_mask_all[img_idx][(pixels_y, pixels_x)]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().cuda()  # batch_size, 3
        p = torch.matmul(self.intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_d = torch.matmul(self.c2w[None, :3, :3], rays_d[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.c2w[None, :3, 3].expand(rays_d.shape)  # batch_size, 3
        data = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "color": color,
            "mask": mask,
            "uv": uv,
        }
        return data

    def get_image_numpy(self, resolution_level):
        # img = cv.imread(self.images_lis[idx])
        img = self.gt_image.cpu().numpy() * 255
        return (cv2.resize(img, (self.image_width // resolution_level, self.image_height // resolution_level))).clip(0, 255)



class Dataset:
    def __init__(self, conf,
                 data_path,
                 case_name,
                 feature_extractor=None,
                 ):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.conf = conf

        # self.data_dir = conf.get_string('data_dir')
        self.data_dir = data_path
        dataset_name = conf.get_string('dataset_name')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')
        depth_folder = conf.get_string("depth_folder", default="omnidata_depth")


        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        image_list = os.listdir(os.path.join(self.data_dir, case_name, "images"))
        image_list = sorted(image_list)
        self.n_images = len(image_list)

        angle_max = conf.get("angle_max")
        use_pairs_file = conf.get("use_pairs_file")
        self.half_patch_size = conf.get("half_patch_size")


        camera_dict = np.load(os.path.join(self.data_dir, case_name, self.render_cameras_name))
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        intrinsics_all = []
        pose_all = []
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose_all.append(torch.from_numpy(pose).float())


        print("loading images ...")
        self.cameras = []
        for i in range(self.n_images):
            image_path = os.path.join(self.data_dir, case_name, f"images/{image_list[i]}")
            image = cv2.imread(image_path)
            self.image_height, self.image_width = image.shape[:2]
            image_name = os.path.basename(image_path)
            image_extension = "."+ image_name.split(".")[-1]
            intrinsic = intrinsics_all[i]
            pose = pose_all[i]
            camera = Camera(image=torch.from_numpy(image).float()/256.0,
                            intrinsic=intrinsic[:3, :3],
                            c2w=pose,
                            image_name=image_name,
                            image_index=i,
                            mask=None,
                            )
            self.cameras.append(camera)


        self.src_indices = []
        if use_pairs_file and os.path.exists(os.path.join(self.data_dir, case_name, "pairs.txt")):
            with open(os.path.join(self.data_dir, case_name, "pairs.txt"), "r") as f:
                pairs = f.readlines()
            for pair in pairs:
                pair_split = pair.split()[1:]  # Exclude the first image (reference image)
                fun = lambda s: int(s.split(".")[0])
                self.src_indices.append(torch.tensor(list(map(fun, pair_split))))
        else:
            self.src_indices, min_image_num = self.get_neighbor_image(poses_src=torch.stack(pose_all, dim=0),
                                                                      poses_target=torch.stack(pose_all, dim=0),
                                                                      angle_max=angle_max)

        main_src_image_map = {}
        for index, src_index in enumerate(self.src_indices):
            assert len(src_index) > 0
            main_image_name = f"{index:06d}{image_extension}"
            src_image_names = [f"{idx:06d}{image_extension}" for idx in src_index]
            main_src_image_map[main_image_name] = src_image_names
        print(main_src_image_map)

        for camera in self.cameras:
            camera.src_image_names = main_src_image_map[camera.image_name]

        print("loading aligned depth and depth mask ...")
        depth_dir = os.path.join(self.data_dir, case_name, depth_folder)
        self.depth_all = []
        self.depth_mask_all = []
        for idx in range(self.n_images):
            depth_name = os.path.join(depth_dir, f"depth_prior_scaled_{idx:03d}.npy")
            if os.path.exists(depth_name):
                depth_scaled = np.load(depth_name, allow_pickle=True)
                depth_scaled = depth_scaled.item()
                depth = torch.from_numpy(depth_scaled["depth"]).cuda()
                depth_mask = torch.from_numpy(depth_scaled["mask"]).cuda() ### When calibrating the depth, we performed "remove_outlier", so a depth mask is required.
            else:
                print(f"#### {case_name} has no depth file ####")
                depth = torch.ones(self.image_height, self.image_width, 1).cuda()
                depth_mask = torch.zeros(self.image_height, self.image_width, 1).cuda()
            self.cameras[idx].depth = depth
            self.cameras[idx].depth_mask = depth_mask

        print("get image features ...")
        if feature_extractor == "vismvsnet":
            self.feat_ext = FeatExt()
            feat_ext_dict = {k[16:]: v for k, v in torch.load('feat_extractor/vismvsnet/vismvsnet.pt')['state_dict'].items() if
                             k.startswith('module.feat_ext')}
            self.feat_ext.load_state_dict(feat_ext_dict)
            for camera in self.cameras:
                image = camera.gt_image
                with torch.no_grad():
                    image_feature = self.feat_ext.forward(image.permute(2,0,1).unsqueeze(0))
                    camera.image_feature = image_feature
        elif feature_extractor == "mvsformer":
            self.feat_ext = FPNEncoder(feat_chs=[8, 16, 32, 64])
            self.feat_ext.load_state_dict(torch.load("feat_extractor/mvsformer/mvsformer_encoder.pth"))
            for camera in self.cameras:
                image = camera.gt_image
                with torch.no_grad():
                    image_feature = self.feat_ext.forward(image.permute(2,0,1).unsqueeze(0))
                    camera.image_feature = image_feature[1]
        elif feature_extractor == "mvsformer_plus":
            self.feat_ext = FPNEncoder(feat_chs=[8, 16, 32, 64])
            self.feat_ext.load_state_dict(torch.load("feat_extractor/mvsformer/mvsformer_plus_encoder.pth"))
            for camera in self.cameras:
                image = camera.gt_image
                with torch.no_grad():
                    image_feature = self.feat_ext.forward(image.permute(2, 0, 1).unsqueeze(0))
                    camera.image_feature = image_feature[1]
        elif feature_extractor == "transmvsnet":
            self.feat_ext = FeatureNet(base_channels=8)
            feat_ext_dict = {k[8:]: v for k, v in torch.load("feat_extractor/transmvsnet/model_dtu.ckpt")['model'].items() if
                             k.startswith('feature')}
            self.feat_ext.load_state_dict(feat_ext_dict)
            self.feat_ext.eval()
            self.feat_ext = self.feat_ext.cuda()
            for camera in self.cameras:
                image = camera.gt_image
                with torch.no_grad():
                    features = self.feat_ext.forward(image.permute(2,0,1).unsqueeze(0))
                    camera.image_feature = features["stage2"]
        else:
            raise NotImplementedError(feature_extractor)

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]

        print('Load data: End')

    def get_random_camera(self):
        camera = random.choice(self.cameras)
        return camera

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def get_neighbor_image(self, poses_src, poses_target, angle_max):
        cameras_center_src = poses_src[:, :3, 3]  # (N, 3)
        cameras_center_target = poses_target[:, :3, 3] # (N, 3)
        direction_normal_src = F.normalize(cameras_center_src, dim=-1)
        direction_normal_target = F.normalize(cameras_center_target, dim =-1)
        direction_cos = torch.sum(direction_normal_src[:, None, :] * direction_normal_target[None, :, :], dim=-1)  # (N, N)
        N = len(cameras_center_src)
        M = len(cameras_center_target)

        cos_max = np.cos(angle_max * np.pi / 180)
        min_neighbor_num = np.inf

        pairs_index = []
        for i in range(N):
            pair_index = []
            for j in range(M):
                if direction_cos[i][j] > cos_max and direction_cos[i][j] < 0.999: # <0.999 Avoid two images being exactly the same
                    pair_index.append(j)
            if len(pair_index) < min_neighbor_num:
                min_neighbor_num = len(pair_index)
            pair_index = torch.tensor(pair_index)
            pairs_index.append(pair_index)
        return pairs_index, min_neighbor_num

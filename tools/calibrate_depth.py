import os
import numpy as np
import cv2
import torch
import trimesh
import open3d as o3d
from tqdm import tqdm
import argparse
from read_write_fused_vis import read_fused

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
    pose[:3, :3] = R.transpose()  # not R but R^-1
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose


def np_gen_rays_at(img_idx, poses_all, intrinsics_all, W, H, resolution_level=1):
    """
    Generate rays at world space from one camera.
    """
    intrinsics_all_inv = np.linalg.inv(intrinsics_all)

    l = resolution_level
    tx = np.linspace(0, W - 1, W // l)
    ty = np.linspace(0, H - 1, H // l)
    pixels_y, pixels_x = np.meshgrid(ty, tx)

    p = np.stack([pixels_x, pixels_y, np.ones_like(pixels_y)], axis=-1) # W, H, 3
    p = np.matmul(intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3

    rays_v = p / np.linalg.norm(p, ord=2, axis=-1, keepdims=True)  # W, H, 3
    # rays_v = p
    rays_v = np.matmul(poses_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = np.broadcast_to(poses_all[img_idx, None, None, :3, 3], rays_v.shape)  # W, H, 3
    return rays_o.transpose((1,0,2)), rays_v.transpose((1,0, 2))


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0.item(), x_1.item()

def read_fused_and_map_to_image(fused_ply, fused_ply_vis, poses, intrinsics):
    '''
    :param fused_ply: ply格式的点云文件
    :param pose:  不同视角的相机位姿c2w (n_view, 4, 4)
    :param intrinsic:  相机内参 (n_view, 4, 4)
    :return:
    '''
    assert os.path.isfile(fused_ply), print(fused_ply)
    base_dir = os.path.dirname(fused_ply)

    # point_cloud = PyntCloud.from_file(fused_ply)
    # xyz_arr = point_cloud.points.loc[:, ["x", "y", "z"]].to_numpy() #(n_p, 3)
    # normal_arr = point_cloud.points.loc[:, ["nx", "ny", "nz"]].to_numpy()
    # color_arr = point_cloud.points.loc[:, ["red", "green", "blue"]].to_numpy()

    mesh_points = read_fused(fused_ply, fused_ply_vis)
    points_all = []
    for i in range(len(poses)):
        points = []
        for mesh_point in mesh_points:
            if i in mesh_point.visible_image_idxs:
                points.append(mesh_point.position)

        if len(points) != 0:
            points = np.stack(points, axis=0)

        points_all.append(points)


    K = intrinsics[0]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    focal = np.array([fx, fy])
    center = np.array([cx, cy])
    pose_inv = np.linalg.inv(poses)

    uv_all = []
    depth_all = []
    for i, points in enumerate(points_all):
        if len(points) != 0:
            points_rot = np.matmul(pose_inv[i, None, :3, :3], points[:, :, None])[..., 0]
            points_trans = points_rot + pose_inv[i, None, :3, 3]
            uv = points_trans[..., :2] / (points_trans[..., 2:]+1e-9)
            uv *= focal[None, :]
            uv += center[None, :]
            depth = np.linalg.norm(points - poses[i, None, :3, 3], axis=-1, keepdims=True) #(n_p, 1)
        else:
            uv = np.array([])
            depth = np.array([])
        uv_all.append(uv)
        depth_all.append(depth)

    data_list = []
    for uv_i, depth_i in zip(uv_all, depth_all):
        depth_collect = []
        uv_collect = []
        if len(uv_i) != 0:
            for i, uv in enumerate(uv_i):
                if uv[0] < 0 or uv[0] > w-1 or uv[1] < 0 or uv[1] > h-1:
                    continue
                depth_collect.append(depth_i[i])
                uv_collect.append(uv)
            depth_i = np.stack(depth_collect, axis=0)
            uv_i = np.stack(uv_collect, axis=0)
        data_list.append({"depth": depth_i, "coord": uv_i})

    data_file = os.path.join(base_dir, 'colmap_fused_depth.npy')
    np.save(data_file, data_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="DTU", help="DTU or BlendedMVS")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--depth_folder", type=str, default="depths_omnidata")
    args = parser.parse_args()


    if args.dataset_name == "DTU":
        scenes = [21, 24, 34, 37, 38, 40, 82, 106, 110, 114, 118]
    elif args.dataset_name == "BlendedMVS":
        scenes = [1,2,3,4,5,6,7,8,9]
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset_name}")


    for scene in tqdm(scenes):
        scene_name = f"scan{scene}"
        if args.dataset_name == "DTU":
            w = 1600
            h = 1200
            camera_dict_path = os.path.join(args.data_root, scene_name, "cameras_sphere.npz")
        elif args.dataset_name == "BlendedMVS":
            h = 576
            w = 768
            camera_dict_path = os.path.join(args.data_root, scene_name, "cameras.npz")


        image_list = sorted(os.listdir(os.path.join(args.data_root, scene_name, "images")))
        n_cams = len(image_list)

        image_wh = np.array([w, h])
        camera_dict = np.load(camera_dict_path)

        world_mats = [camera_dict["world_mat_%d" % idx] for idx in range(n_cams)]
        scale_mats = [camera_dict["scale_mat_%d" % idx] for idx in range(n_cams)]
        intrinsics_all = []
        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = np.matmul(world_mat, scale_mat)
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_all.append(intrinsics)
            pose_all.append(pose)

        intrinsics_all = np.stack(intrinsics_all, axis=0)
        pose_all = np.stack(pose_all, axis=0)


        print("loading colmap fused.ply and generate colmap_fused_depth.npy")
        dense_path = os.path.join(args.data_root, scene_name, "dense")
        depth_path = os.path.join(args.data_root, scene_name, args.depth_folder)
        colmap_depth_file = os.path.join(dense_path, "colmap_fused_depth.npy")
        fused_ply = os.path.join(dense_path, f"fused.ply")
        fused_ply_vis = os.path.join(dense_path, f"fused.ply.vis")
        if not os.path.exists(fused_ply):
            print(f"{fused_ply} not exist.")
            continue

        read_fused_and_map_to_image(fused_ply, fused_ply_vis, poses=pose_all, intrinsics=intrinsics_all)
        colmap_depth = np.load(colmap_depth_file ,allow_pickle=True)

        print("generate scaled depth and point cloud")
        for index in range(n_cams):
            print("loading depth ...")
            image_name = image_list[index].split(".")[0]
            depth_path = os.path.join(args.data_root, scene_name, args.depth_folder)
            depth_file = os.path.join(depth_path, f"{image_name}_depth.npy")
            depth_prior = np.load(depth_file, allow_pickle=True)

            print("scene: {}, index: {}".format(scene_name, index))
            rays_o, rays_d = np_gen_rays_at(img_idx=index,
                                         poses_all=pose_all,
                                         intrinsics_all=intrinsics_all,
                                         W = w,
                                         H = h,
                                         resolution_level=1)

            # depth to point3d
            depth_prior = depth_prior + 1.0
            points_prior = rays_o + rays_d * depth_prior[..., None]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_prior.reshape(-1, 3))
            ## filter point3d prior
            cleaned_pcd, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
            points_prior = np.asarray(cleaned_pcd.points)
            ## get depth prior mask
            mask = np.zeros_like(depth_prior, dtype=np.bool_).reshape(-1, 1)
            mask[idx, :] = True
            mask = mask.reshape(*depth_prior.shape)  # depth prior mask

            uv = np.around(colmap_depth[index]["coord"])
            if len(uv) == 0:
                continue
            pixel_x = uv[:, 0].astype(int)
            pixel_y = uv[:, 1].astype(int)
            rays_o_sample = rays_o[pixel_y, pixel_x]
            rays_d_sample = rays_d[pixel_y, pixel_x]

            mask_sample = mask[pixel_y, pixel_x]
            rays_o_sample = rays_o_sample[mask_sample]
            rays_d_sample = rays_d_sample[mask_sample]

            # get colmap point3d
            depth_gt = colmap_depth[index]["depth"]
            depth_gt = depth_gt[mask_sample]
            points_colmap = rays_o_sample + rays_d_sample * depth_gt

            # get sampled point3d from depth prior
            depth_sample = depth_prior[pixel_y, pixel_x]
            depth_sample = depth_sample[mask_sample]
            points_sample = rays_o_sample + rays_d_sample * depth_sample[:, None]

            ## filter sampled points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_sample.reshape(-1, 3))
            cleaned_pcd, idx = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.3) ## std ratio越低，滤除效果越明显
            points_sample = np.asarray(cleaned_pcd.points)
            points_colmap = points_colmap[idx]
            depth_sample = depth_sample[idx]
            depth_gt = depth_gt[idx]
            rays_o_sample = rays_o_sample[idx]
            rays_d_sample = rays_d_sample[idx]

            # depth_sample = 1.0 / depth_sample ## If the depth prior is inverted
            A = np.stack([depth_sample, np.ones_like(depth_sample)], axis=-1)
            scale, shift = np.linalg.lstsq(A, depth_gt, rcond=None)[0]

            # get scaled depth prior and point3d prior
            depth_sample_scaled = depth_sample * scale + shift
            depth_prior_scaled = depth_prior * scale + shift
            points_sample_scaled = rays_o_sample + rays_d_sample * depth_sample_scaled[:, None]
            points_prior_scaled = rays_o + rays_d * depth_prior_scaled[..., None]

            ## Remove outliers from scaled point3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_prior_scaled.reshape(-1, 3))
            cl, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
            inlier_pcd = pcd.select_by_index(idx)
            ## to numpy
            points_prior_scaled = np.asarray(inlier_pcd.points)

            # save point3d
            point_path = os.path.join(depth_path, f"points_prior_scaled_{index:03d}.npy")
            np.save(point_path, points_prior_scaled)

            # get depth prior mask
            depth_mask = np.zeros(shape=(h, w)).reshape(-1, 1).astype(np.float32)
            depth_mask[idx] = 1.0  # set the mask of retained points to 1
            depth_mask = depth_mask.reshape(h, w, 1)

            # save scaled depth prior
            depth_prior_scaled = depth_prior_scaled.reshape(h, w, 1)
            depth_prior_scaled_file = os.path.join(depth_path, f"depth_prior_scaled_{index:03d}.npy")
            np.save(depth_prior_scaled_file, {"depth":depth_prior_scaled, "mask":depth_mask})

            # save scaled point3d
            mesh_prior_scaled = trimesh.Trimesh(vertices=points_prior_scaled.reshape(-1, 3))
            mesh_sample_scaled = trimesh.Trimesh(vertices=points_sample_scaled.reshape(-1, 3))
            mesh_prior_scaled.export(os.path.join(depth_path, f"points_prior_scaled_{index:03d}.ply"), file_type="ply")
            mesh_sample_scaled.export(os.path.join(depth_path, f"points_sample_scaled_{index:03d}.ply"), file_type="ply")

            # save original point3d
            mesh_prior = trimesh.Trimesh(vertices=points_prior.reshape(-1, 3))
            mesh_colmap = trimesh.Trimesh(vertices=points_colmap.reshape(-1, 3))
            mesh_sample = trimesh.Trimesh(vertices=points_sample.reshape(-1, 3))

            mesh_prior.export(os.path.join(depth_path, f"points_prior_{index:03d}.ply"), file_type="ply")
            mesh_colmap.export(os.path.join(depth_path, f"points_colmap_{index:03d}.ply"), file_type="ply")
            mesh_sample.export(os.path.join(depth_path, f"points_sample_{index:03d}.ply"), file_type="ply")

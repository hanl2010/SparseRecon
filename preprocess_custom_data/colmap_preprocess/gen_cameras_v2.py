import numpy as np
import trimesh
import os
from colmap_read_model import read_images_binary, read_cameras_binary, qvec2rotmat

if __name__ == '__main__':
    root_dir = "mipnerf360"
    scans = ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'stump']
    for scan in scans:
        work_dir = os.path.join(root_dir, scan)
        cam_extrinsics = read_images_binary(os.path.join(work_dir, "sparse/images.bin"))
        cam_intrinsics = read_cameras_binary(os.path.join(work_dir, "sparse/cameras.bin"))

        cam_dict = dict()
        for idx, key in enumerate(cam_extrinsics):
            print(f"reading camera {idx}/{len(cam_extrinsics)}")
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            image_name = extr.name
            cam_index = key - 1
            # assert key == int(image_name.split(".")[0]), f"cam_index={cam_index}, image_name={image_name}"
            print(f"cam_key: {key}, image_name: {image_name}")
            height = intr.height
            width = intr.width
            uid = intr.id
            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = T

            if intr.model == "SIMPLE_PINHOLE":
                fx = intr.params[0]
                fy = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
            elif intr.model == "PINHOLE":
                fx = intr.params[0]
                fy = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            intrinsic = np.eye(4)
            intrinsic[0, 0] = fx
            intrinsic[1, 1] = fy
            intrinsic[0, 2] = cx
            intrinsic[1, 2] = cy

            world_mat = intrinsic @ w2c
            world_mat = world_mat.astype(np.float32)
            cam_dict['camera_mat_{}'.format(cam_index)] = intrinsic
            cam_dict['camera_mat_inv_{}'.format(cam_index)] = np.linalg.inv(intrinsic)
            cam_dict['world_mat_{}'.format(cam_index)] = world_mat
            cam_dict['world_mat_inv_{}'.format(cam_index)] = np.linalg.inv(world_mat)

        n_images = len(cam_extrinsics)
        pcd = trimesh.load(os.path.join(work_dir, 'sparse/points3d_sample.ply'))
        vertices = pcd.vertices
        bbox_max = np.max(vertices, axis=0)
        bbox_min = np.min(vertices, axis=0)
        center = (bbox_max + bbox_min) * 0.5
        radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
        scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
        scale_mat[:3, 3] = center

        for i in range(n_images):
            cam_dict['scale_mat_{}'.format(i)] = scale_mat
            cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

        # np.savez(os.path.join(work_dir, 'dense/cameras_sphere.npz'), **cam_dict)
        np.savez(os.path.join(work_dir, 'cameras_sphere.npz'), **cam_dict)
        print(f'Process done! {work_dir}')

import os
import trimesh
import numpy as np
from colmap_read_model import read_points3d_binary

root_dir = "data_path"
scenes = [1,2,3,4,5,6,7,8,9]
for scene in scenes:
    scene_name = f"scan{scene}"
    print(scene_name)
    scene_dir = os.path.join(root_dir, scene_name)
    points3dfile = os.path.join(scene_dir, 'dense/sparse/points3D.bin')
    pts3d = read_points3d_binary(points3dfile)

    points = []
    for k in pts3d:
        points.append(pts3d[k].xyz)
    points = np.stack(points, axis=0)
    pcd = trimesh.PointCloud(points)
    pcd.export(os.path.join(scene_dir, 'dense/sparse_points.ply'))
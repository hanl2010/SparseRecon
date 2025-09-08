from tqdm import tqdm
import shutil
import os
import subprocess
import numpy as np
import cv2
from read_write_model import read_model, write_model, write_cameras_text, write_images_text, rotmat2qvec
from database import blob_to_array, COLMAPDatabase
from argparse import ArgumentParser


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
def camTodatabase(txtfile, database_path):
    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}

    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()


###############################################################################
parser = ArgumentParser()
parser.add_argument("--cases", nargs="+", type=int, default=[8])
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

data_dir = "E:/data/public_dataset/data_s_volsdf/BlendedMVS/BlendedMVS_sparse"
bmvs_scenes = [1,2,3,4,5,6,7,8,9]


gpu_index = args.gpu
image_w = 768
image_h = 576

if len(args.cases) != 0:
    scenes = [bmvs_scenes[case] for case in args.cases]
else:
    scenes = bmvs_scenes

for scene in scenes:
    scene_name = f"scan{scene}"
    # scene_name = f"scan{scene}_colmap"
    print(f"processing {scene_name} ...")
    scene_path = os.path.join(data_dir, scene_name)
    image_path = os.path.join(scene_path, "images")
    sparse_path = os.path.join(scene_path, "sparse")
    model_path = os.path.join(scene_path, "model")
    dense_path = os.path.join(scene_path, "dense")
    camera_dict_path = os.path.join(data_dir, scene_name, "cameras.npz")

    shutil.rmtree(model_path, ignore_errors=True)
    shutil.rmtree(sparse_path, ignore_errors=True)
    shutil.rmtree(dense_path, ignore_errors=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(dense_path, exist_ok=True)

    ############# read cameras ###############################
    images_list = sorted(os.listdir(image_path))

    n_images = len(images_list)
    camera_dict = np.load(camera_dict_path)
    world_mats = [camera_dict["world_mat_%d" % idx] for idx in range(n_images)]
    scale_mats = [camera_dict["scale_mat_%d" % idx] for idx in range(n_images)]
    intrinsics = []
    c2ws = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = np.matmul(world_mat, scale_mat)
        P = P[:3, :4]
        intrinsic, c2w = load_K_Rt_from_P(None, P)
        intrinsics.append(intrinsic)
        c2ws.append(c2w)

    intrinsics = np.stack(intrinsics, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    # 产生colmap需要的四元数，写入camera.txt 和 images.txt
    sub_images = []
    for idx in range(n_images):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        r = w2c[:3, :3]
        t = w2c[:3, 3]
        qvec = rotmat2qvec(r)
        sub_images.append([idx+1, *qvec, *t, 1, images_list[idx]])

    images_file = os.path.join(model_path, "images.txt")
    with open(images_file, "w") as f:
        for image in sub_images:
            line = " ".join(map(str, image))
            f.write(line + "\n")
            f.write("\n\n")

    focal_x = intrinsics[0][0, 0]
    focal_y = intrinsics[0][1, 1]
    center_x = intrinsics[0][0, 2]
    center_y = intrinsics[0][1, 2]
    # center_x = image_w / 2.0
    # center_y = image_h / 2.0
    sub_cameras = [[1, "PINHOLE", image_w, image_h, focal_x, focal_y, center_x, center_y]]
    camera_file = os.path.join(model_path, "cameras.txt")
    with open(camera_file, "w") as f:
        for cam in sub_cameras:
            line = " ".join([str(elem) for elem in cam])
            f.write(line + "\n")

    ####### 使用全部视图colmap重建时产生的相机参数 #############
    # cameras, images, point3D = read_model(path=sparse_path, ext=".bin")
    # sub_cameras = []
    # sub_images = []
    # idx = 0
    # for _, image in images.items():
    #     if image.name in train_image_list:
    #         idx += 1
    #         sub_images.append([idx, *image.qvec, *image.tvec, idx, image.name])
    #         camera = cameras[image.camera_id]
    #         sub_cameras.append([idx, camera.model, camera.width, camera.height, *camera.params])
    #
    # sub_images = sorted(sub_images, key=lambda x: x[-1])
    # for index in range(len(sub_images)):
    #     sub_images[index][0] = index + 1
    #
    # point3d_file = os.path.join(sub_model_path, "points3D.txt")
    # with open(point3d_file, "w") as f:
    #     f.write("")
    #
    # camera_file = os.path.join(sub_model_path, "cameras.txt")
    # with open(camera_file, "w") as f:
    #     for cam in sub_cameras:
    #         line = " ".join([str(elem) for elem in cam])
    #         f.write(line + "\n")
    # # write_cameras_text(cameras, camera_file)
    #
    # images_file = os.path.join(sub_model_path, "images.txt")
    # with open(images_file, "w") as f:
    #     for image in sub_images:
    #         line = " ".join(map(str, image))
    #         f.write(line + "\n")
    #         f.write("\n\n")

    #############################################################

    point3d_file = os.path.join(model_path, "points3D.txt")
    with open(point3d_file, "w") as f:
        f.write("")

    database_file = os.path.join(model_path, "database.db")

    logfile_name = os.path.join(sparse_path, "colmap_output.txt")
    logfile = open(logfile_name, "w")


    feature_extractor_args = [
        "colmap", "feature_extractor",
        "--database_path", database_file,
        "--image_path", image_path,
        "--ImageReader.camera_model", "PINHOLE",
        # "--ImageReader.mask_path", mask_path
        "--SiftExtraction.gpu_index", gpu_index
    ]
    feature_output = subprocess.check_output(feature_extractor_args, universal_newlines=True)
    logfile.write(feature_output)
    print(feature_output)
    print("Features extracted")

    ### update camera intrinsics in db ###
    camTodatabase(txtfile=camera_file, database_path=database_file)

    exhaustive_matcher_args = [
        'colmap', "exhaustive_matcher",
        '--database_path', database_file,
        "--SiftMatching.gpu_index", gpu_index
    ]
    match_output = subprocess.check_output(exhaustive_matcher_args, universal_newlines=True)
    logfile.write(match_output)
    print(match_output)
    print("feature matched")

    point_triangulator_args = [
        "colmap", "point_triangulator",
        "--database_path", database_file,
        "--image_path", image_path,
        "--input_path", model_path,
        "--output_path", sparse_path
    ]
    triangulator_output = subprocess.check_output(point_triangulator_args, universal_newlines=True)
    logfile.write(triangulator_output)
    print(triangulator_output)
    print("triangulator finished")

    image_undistorter_args = [
        "colmap", "image_undistorter",
        "--image_path", image_path,
        "--input_path", sparse_path,
        "--output_path", dense_path,
    ]
    image_undistorter_output = subprocess.check_output(image_undistorter_args, universal_newlines=True)
    logfile.write(image_undistorter_output)
    print(image_undistorter_output)
    print("image undistorter finished")

    patch_match_stereo_args = [
        "colmap", "patch_match_stereo",
        "--workspace_path", dense_path,
        "--PatchMatchStereo.gpu_index", gpu_index
    ]
    patch_match_stereo_output = subprocess.check_output(patch_match_stereo_args, universal_newlines=True)
    logfile.write(patch_match_stereo_output)
    # print(patch_match_stereo_output)
    print("patch match stereo finished")

    stereo_fusion_args = [
        "colmap", "stereo_fusion",
        "--workspace_path", dense_path,
        "--output_path", os.path.join(dense_path, f"fused.ply")
    ]
    stereo_fusion_output = subprocess.check_output(stereo_fusion_args, universal_newlines=True)
    logfile.write(stereo_fusion_output)
    # print(stereo_fusion_output)
    print("stereo fusion finished")

    logfile.close()

    shutil.rmtree(os.path.join(dense_path, "images"))
    shutil.rmtree(os.path.join(dense_path, "sparse"))
    shutil.rmtree(os.path.join(dense_path, "stereo"))
    shutil.rmtree(model_path)





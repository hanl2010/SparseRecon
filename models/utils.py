import torch
import torch.nn.functional as F

# interpolate SDF zero-crossing points
def find_surface_points(sdf, d_all, device='cuda'):
    # shape of sdf and d_all: only inside  [B, N_rays, N_samples+N_importance]
    sdf_bool_1 = sdf[..., 1:] * sdf[..., :-1] < 0  # [B, N_rays, N_samples+N_importance-1]
    # only find backward facing surface points, not forward facing
    sdf_bool_2 = sdf[..., 1:] < sdf[..., :-1]
    sdf_bool = torch.logical_and(sdf_bool_1, sdf_bool_2)  # [B, N_rays, N_samples+N_importance-1]

    # [B, N_rays]
    max, max_indices = torch.max(sdf_bool, dim=2)
    network_mask = max > 0
    d_surface = torch.zeros_like(network_mask, device=device).float()  # [B, N_rays]

    sdf_0 = torch.gather(sdf[network_mask], 1, max_indices[network_mask][..., None]).squeeze()  # [N_masked_rays]
    sdf_1 = torch.gather(sdf[network_mask], 1, max_indices[network_mask][..., None] + 1).squeeze()  # [N_masked_rays]
    d_0 = torch.gather(d_all[network_mask], 1, max_indices[network_mask][..., None]).squeeze()  # [N_masked_rays]
    d_1 = torch.gather(d_all[network_mask], 1, max_indices[network_mask][..., None] + 1).squeeze()  # [N_masked_rays]
    d_surface[network_mask] = (sdf_0 * d_1 - sdf_1 * d_0) / (sdf_0 - sdf_1)  # [N_masked_rays]

    return d_surface, network_mask  # [B, N_rays]


def idx_cam2img(idx_cam_homo, cam):
    """nhw41 -> nhw31"""
    idx_cam = idx_cam_homo[...,:3,:] / (idx_cam_homo[...,3:4,:]+1e-9)  # nhw31
    idx_img_homo = cam[:,1:2,:3,:3].unsqueeze(1) @ idx_cam  # nhw31
    idx_img_homo = idx_img_homo / (idx_img_homo[...,-1:,:]+1e-9)
    return idx_img_homo


def normalize_for_grid_sample(input_, grid):
    size = torch.tensor(input_.size())[2:].flip(0).to(grid.dtype).to(grid.device).view(1,1,1,-1)  # [[[w, h]]]
    grid_n = grid / size
    grid_n = (grid_n * 2 - 1).clamp(-1.1, 1.1)
    return grid_n


def idx_world2cam(idx_world_homo, cam):
    """nhw41 -> nhw41"""
    idx_cam_homo = cam[:,0:1,...].unsqueeze(1) @ idx_world_homo  # nhw41
    idx_cam_homo = idx_cam_homo / (idx_cam_homo[...,-1:,:]+1e-9)   # nhw41
    return idx_cam_homo


def bin_op_reduce(lst, func):
    result = lst[0]
    for i in range(1, len(lst)):
        result = func(result, lst[i])
    return result


def get_in_range(grid):
    """after normalization, keepdim=False"""
    masks = []
    for dim in range(grid.size()[-1]):
        masks += [grid[..., dim]<=1, grid[..., dim]>=-1]
    in_range = bin_op_reduce(masks, torch.min).to(grid.dtype)
    return in_range


def get_feat_loss(diff_surf_pts, uncerts, feat, cam, feat_src, src_cams, size, center, network_object_mask,
                  object_mask):
    mask = network_object_mask & object_mask  # [B * N_rays]
    size = size[:1]
    center = center[:1]
    if (mask).sum() == 0:
        return torch.tensor(0.0).float().cuda()

    # feat.size(): [B, n_channel, h, w], where h, w are down scaled: 384, 512
    sample_mask = mask.view(feat.size()[0], -1)  # [B, N_rays]
    hit_nums = sample_mask.sum(-1)  # [B]
    accu_nums = [0] + hit_nums.cumsum(0).tolist()
    slices = [slice(accu_nums[i], accu_nums[i + 1]) for i in range(len(accu_nums) - 1)]

    loss = []
    ## for each image in minibatch
    for view_i, slice_ in enumerate(slices):
        if slice_.start < slice_.stop:

            ## projection
            diff_surf_pts_slice = diff_surf_pts[slice_]
            pts_world = (diff_surf_pts_slice / 2 * size.view(1, 1) + center.view(1, 3)).view(1, -1, 1, 3,
                                                                                             1)  # 1m131, where m == n_masked_rays
            pts_world = torch.cat([pts_world, torch.ones_like(pts_world[..., -1:, :])], dim=-2)  # 1m141
            # rgb_pack = torch.cat([rgb[view_i:view_i+1], rgb_src[view_i]], dim=0)  # v3hw
            cam_pack = torch.cat([cam[view_i:view_i + 1], src_cams[view_i]],
                                 dim=0)  # v244, v == 1 + n_src; here cam is depth/feature cam upscaled by 2
            pts_img = idx_cam2img(idx_world2cam(pts_world, cam_pack), cam_pack)  # vm131

            ## gathering
            grid = pts_img[..., :2, 0]  # vm12
            # feat2_pack = self.feat_ext(rgb_pack)[2]  # vchw
            feat2_pack = torch.cat([feat[view_i:view_i + 1], feat_src[view_i]], dim=0)  # [v, n_channel, h, w]
            grid_n = normalize_for_grid_sample(feat2_pack, grid / 2)  # [v, m, 1, 2]
            grid_in_range = get_in_range(grid_n)  # [v, m, 1]
            valid_mask = (grid_in_range[:1, ...] * grid_in_range[1:, ...]).unsqueeze(1) > 0.5  # [n_src, 1, m, 1]
            gathered_feat = F.grid_sample(feat2_pack, grid_n, mode='bilinear', padding_mode='zeros',
                                          align_corners=False)  # vcm1

            ## calculation
            gathered_norm = gathered_feat.norm(dim=1, keepdim=True)  # v1m1
            corr = (gathered_feat[:1] * gathered_feat[1:]).sum(dim=1, keepdim=True) \
                   / gathered_norm[:1].clamp(min=1e-9) / gathered_norm[1:].clamp(min=1e-9)  # (v-1)1m1
            corr_loss = (1 - corr).abs()
            if uncerts is None:
                diff_mask = corr_loss < 0.5
                # print('feat loss mask', (valid_mask & diff_mask).sum().item(), '/',
                # valid_mask.size()[0] * valid_mask.size()[2])
                sample_loss = (corr_loss * valid_mask * diff_mask).mean()
            else:
                uncert = uncerts[view_i].unsqueeze(1).unsqueeze(3)  # (v-1)1m1
                # print(f'uncert: {uncert.min():.4f}, {uncert.median():.4f}, {uncert.max():.4f}')
                sample_loss = ((corr_loss * (-uncert).exp() + uncert) * valid_mask).mean()
        else:
            sample_loss = torch.zeros(1).float().cuda()
        loss.append(sample_loss)
    loss = sum(loss) / len(loss)
    return loss


def build_patch_offset(half_patch_size=5):
    assert half_patch_size > 0, "half_patch_size error: {}".format(half_patch_size)
    offset_range = torch.arange(-half_patch_size, half_patch_size+1)
    offset_y, offset_x = torch.meshgrid(offset_range, offset_range)
    offset = torch.stack([offset_x, offset_y], dim=-1).reshape(-1, 2)
    return offset.float()


def depth2point_world(depthmap, intrinsic_matrix, c2w, normal_rays_d=False):
    assert depthmap.shape[-1] == 1
    H, W = depthmap.shape[:2]
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x, device=grid_x.device)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrinsic_matrix.inverse().T @ c2w[:3,:3].T
    if normal_rays_d:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def point_world2depth(points, intrinsic_matrix, w2c):
    points_rot = torch.matmul(w2c[None, :3, :3], points[:, :, None])[..., 0]
    points_trans = points_rot + w2c[None, :3, 3]  # (n_p, 3)
    points_image = torch.matmul(intrinsic_matrix, points_trans[..., None])[..., 0]
    uv = points_image[:, :2] / points_image[:, 2:]
    depth = points_image[:, 2:]
    return uv, depth

def depth_sample2point_world(depth_sample, uv, intrinsic_matrix, c2w, normal_rays_d=False):
    points = torch.stack([uv[:, 0], uv[:, 1], torch.ones_like(uv[:, 0])], dim=-1).reshape(-1, 3).float()
    rays_d = points @ intrinsic_matrix.inverse().T @ c2w[:3,:3].T
    if normal_rays_d:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:3,3]
    points = depth_sample.reshape(-1, 1) * rays_d + rays_o
    return points

def get_projected_pixel_color(points, camera, src_camera, main_uv):
    """
    Project the points on the ray onto image, and obtain the corresponding colors on the image.
    """
    sample = {}

    src_image = src_camera.gt_image
    src_pose_inv = src_camera.w2c
    src_mask = src_camera.gt_mask
    assert src_image.shape[-1] == 3
    n_view = 1

    image_h, image_w = camera.image_height, camera.image_width
    N, n_p = points.shape[:2]
    image_wh = torch.tensor([image_w-1, image_h-1])

    src_uv, _ = point_world2depth(points.reshape(-1, 3), src_camera.intrinsic, src_pose_inv)
    src_uv = src_uv.reshape(n_view, N * n_p, 1, 2)
    src_uv_normal = src_uv * 2 / image_wh - 1.0  # Rescale the coordinate values to the range of [-1, 1]
    src_image = src_image.permute(2, 0, 1).unsqueeze(0)
    src_mask = src_mask.permute(2, 0, 1).unsqueeze(0)

    colors_pixel = F.grid_sample(src_image, src_uv_normal, align_corners=False)
    masks_pixel = F.grid_sample(src_mask, src_uv_normal, align_corners=False)
    colors_pixel = colors_pixel.permute(0, 2, 3, 1).reshape(n_view, N, n_p, 3)
    masks_pixel = masks_pixel.permute(0, 2, 3, 1).reshape(n_view, N, n_p, 1)
    sample["rgb_pixel"] = colors_pixel
    sample["mask_pixel"] = masks_pixel

    main_image = camera.gt_image.permute(2, 0, 1).unsqueeze(0)
    main_uv_normal = main_uv * 2 / image_wh - 1.0
    colors_pixel_gt = F.grid_sample(main_image, main_uv_normal.reshape(1, N, 1, 2), align_corners=False)
    colors_pixel_gt = colors_pixel_gt.permute(0, 2, 3, 1).reshape(N, 3)
    sample["rgb_pixel_gt"] = colors_pixel_gt

    return sample


def get_projected_patch_color(points, camera, src_camera, main_uv):
    """
    Project the points on the ray onto image, and obtain the corresponding colors on the image.
    """
    sample = {}

    src_image = src_camera.gt_image
    src_pose_inv = src_camera.w2c
    src_mask = src_camera.gt_mask
    assert src_image.shape[-1] == 3
    n_view = 1

    image_h, image_w = camera.image_height, camera.image_width
    N, n_p = points.shape[:2]
    image_wh = torch.tensor([image_w-1, image_h-1])

    src_uv, _ = point_world2depth(points.reshape(-1, 3), src_camera.intrinsic, src_pose_inv)
    src_uv = src_uv.reshape(n_view, N * n_p, 1, 2)
    src_image = src_image.permute(2, 0 ,1).unsqueeze(0)
    src_mask = src_mask.permute(2, 0, 1).unsqueeze(0)

    # Generate the coordinates of the patch based on the point coordinates
    offset = build_patch_offset(half_patch_size=5)
    # (n_view, N*n_p, 1, 2) (1, 1, patch, 2) -> (n_view, N*n_p, patch, 2)
    grid = src_uv.reshape(n_view, N*n_p, 1, 2) + offset[None, None, :, :]
    grid_normal = grid * 2 / image_wh - 1.0 # Rescale the coordinate values to the range of [-1, 1]

    colors_patch = F.grid_sample(src_image, grid_normal, align_corners=False)
    masks_patch = F.grid_sample(src_mask, grid_normal, align_corners=False)
    colors_patch = colors_patch.permute(0, 2, 3, 1).reshape(n_view, N, n_p, -1, 3)
    masks_patch = masks_patch.permute(0, 2, 3, 1).reshape(n_view, N, n_p, -1, 1)
    sample["rgb_patch"] = colors_patch
    sample["mask_patch"] = masks_patch

    main_image = camera.gt_image.permute(2, 0, 1).unsqueeze(0)
    uv = main_uv
    grid = uv[:, None, :] + offset[None, :, :] #(N, patch_size, 2)
    grid_normal = grid * 2 / image_wh - 1.0
    colors_patch_gt = F.grid_sample(main_image, grid_normal.reshape(1, N, -1, 2), align_corners=False)
    colors_patch_gt = colors_patch_gt.permute(0, 2, 3, 1).reshape(N, -1, 3)
    sample["rgb_patch_gt"] = colors_patch_gt

    return sample


def sample_image_features(points, camera, src_camera, main_uv):
    n_view = 1
    N, n_p = points.shape[:2]
    src_pose_inv = src_camera.w2c

    src_image_feature = src_camera.image_feature
    src_uv, _ = point_world2depth(points.reshape(-1, 3), src_camera.intrinsic, src_pose_inv)

    image_wh = torch.tensor([camera.image_width-1, camera.image_height-1])

    src_uv_reshape = src_uv.reshape(n_view, N * n_p, 1, 2)  # (n_view, Hc, Wc, 2)
    src_uv_normal = (src_uv_reshape * 2 / image_wh) - 1.0

    # Obtain the image features at the corresponding pixel coordinates.
    src_latents = F.grid_sample(src_image_feature, src_uv_normal, align_corners=False)  # (1, C, Hc, Wc)
    src_latents = src_latents.permute(0, 2, 3, 1).reshape(n_view, N, n_p, -1)  # (n_view, N, n_p, C)

    src_latents_normalize = F.normalize(src_latents, dim=-1)

    main_image_feature = camera.image_feature
    main_uv_reshape = main_uv.reshape(1, N, 1, 2).expand(1, N, n_p, 2)  # (1, Hc, Wc, 2)
    main_uv_normal = (main_uv_reshape * 2 / image_wh) - 1.0  #  Rescale the coordinate values to the range of [-1, 1]
    main_latents = F.grid_sample(main_image_feature, main_uv_normal, align_corners=False)  # (1, C, Hc, Wc)
    main_latents = main_latents.permute(0, 2, 3, 1).reshape(1, N, n_p, -1)  # (n_view, N, n_p, C)
    main_latents_normalize = F.normalize(main_latents, dim=-1)

    return main_latents_normalize, src_latents_normalize


def get_projected_patch_feature(self, points, camera, src_camera, main_uv):
    sample = {}

    src_pose_inv = src_camera.w2c
    src_image_feature = src_camera.image_feature

    C = src_image_feature.shape[1]
    n_view = 1
    N, n_p = points.shape[:2]
    image_wh = torch.tensor([camera.image_width-1, camera.image_height-1])

    src_uv, _ = point_world2depth(points.reshape(-1, 3), src_camera.intrinsic, src_pose_inv)
    # src_uv = src_uv.reshape(n_view, N * n_p, 1, 2)

    offset = build_patch_offset()
    # (n_view, N*n_p, 1, 2) (1, 1, patch, 2) -> (n_view, N*n_p, patch, 2)
    grid = src_uv.reshape(n_view, N * n_p, 1, 2) + offset[None, None, :, :]
    grid_normal = grid * 2 / image_wh - 1.0

    src_feat_patch = F.grid_sample(src_image_feature, grid_normal, align_corners=False)
    src_feat_patch = src_feat_patch.permute(0, 2, 3, 1).reshape(n_view, N, n_p, -1, C)
    src_feat_patch_normalize = F.normalize(src_feat_patch, dim=-1)
    sample["feat_patch_src"] = src_feat_patch_normalize

    main_image_feature = camera.image_feature
    uv = main_uv
    grid = uv[:, None, :] + offset[None, :, :]  # (N, patch_size, 2)
    grid_normal = grid * 2 / image_wh - 1.0
    feat_patch_gt = F.grid_sample(main_image_feature, grid_normal.reshape(1, N, -1, 2), align_corners=False)
    feat_patch_gt = feat_patch_gt.permute(0, 2, 3, 1).reshape(N, -1, C)
    feat_patch_gt_normalize = F.normalize(feat_patch_gt, dim=-1)
    sample["feat_patch_gt"] = feat_patch_gt_normalize

    return sample


def get_occ_mask(rays_o, rays_d, depth, camera, src_camera, threshold=0.015):
    surf_points = rays_o + rays_d * depth

    ### Calculate the distance from the center of the source view to the surface point， which is visible from the reference view.

    ### Project the surface points onto the src view, and obtain the pred depth in the source view.

    src_uv, _ = point_world2depth(surf_points, src_camera.intrinsic, src_camera.w2c)
    src_dist = torch.norm(surf_points - src_camera.c2w[None, :3, 3], dim=-1, keepdim=True)
    src_mask = ((src_uv[..., 0] >= 0) & (src_uv[..., 0] < camera.image_width) &
                (src_uv[..., 1] >= 0) & (src_uv[..., 1] < camera.image_height))
    n_view = 1
    n_p = len(surf_points)
    image_wh = torch.tensor([camera.image_width-1, camera.image_height-1])
    src_uv = src_uv.reshape(n_view, n_p, 1, 2)
    src_uv_normal = src_uv * 2 / image_wh - 1.0
    src_depth = src_camera.pred_depth.permute(2,0,1)[None, ...]
    src_dist_pred = F.grid_sample(src_depth, grid=src_uv_normal, align_corners=False)
    src_dist_pred = src_dist_pred.reshape(n_p, 1)
    dist_error = src_dist - src_dist_pred
    occ_mask = (dist_error > threshold).squeeze()
    return (~occ_mask) & src_mask

def get_depth_confidence_map(rays_o, rays_d, depth, main_uv, camera, src_camera, threshold=5.0):
    surf_points = rays_o + rays_d * depth
    n_view = 1
    n_p = len(surf_points)
    image_wh = torch.tensor([camera.image_width-1, camera.image_height-1])
    src_uv, _ = point_world2depth(surf_points, src_camera.intrinsic, src_camera.w2c)
    src_mask = ((src_uv[..., 0] >= 0) & (src_uv[..., 0] < camera.image_width) &
                (src_uv[..., 1] >= 0) & (src_uv[..., 1] < camera.image_height))
    src_uv = src_uv.reshape(n_view, n_p, 1, 2)
    src_uv_normal = src_uv * 2 / image_wh - 1.0
    src_depth = src_camera.pred_depth.permute(2,0,1)[None, ...]
    src_dist_pred = F.grid_sample(src_depth, grid=src_uv_normal, align_corners=False)
    src_dist_pred = src_dist_pred.reshape(n_p, 1)

    src_points = depth_sample2point_world(src_dist_pred, src_uv.reshape(-1, 2), src_camera.intrinsic, src_camera.c2w, normal_rays_d=True)
    ref_uv_projection, _ = point_world2depth(src_points, camera.intrinsic, camera.w2c)
    uv_error = torch.norm(ref_uv_projection - main_uv, dim=-1)
    uv_mask = (uv_error <= threshold)
    uv_mask = uv_mask & src_mask
    confidence = (1.0 / torch.exp(uv_error)).detach()
    confidence[~uv_mask] = 0.0
    return confidence


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

    return x_0, x_1
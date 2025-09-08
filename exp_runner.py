import os
# import time
import logging
import argparse

import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
# from models.dataset import Dataset
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.patch_projector import PatchProjector
from models.ssim import SSIM
from models.utils import (get_projected_pixel_color, get_projected_patch_color, sample_image_features, get_occ_mask,
                          get_depth_confidence_map, compute_scale_and_shift)
import random



def seed_everything(seed):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Runner:
    def __init__(self, args):
        self.device = torch.device('cuda')

        # Configuration
        self.case_name = args.case
        self.conf_path = args.conf
        self.use_rgb_pixel_loss = args.use_rgb_pixel_loss
        self.use_rgb_patch_loss = args.use_rgb_patch_loss
        self.rgb_patch_loss_weight = args.rgb_patch_loss_weight
        self.use_feat_loss = args.use_feat_loss
        self.use_embed_mask = args.use_embed_mask
        self.use_depth_loss = args.use_depth_loss
        self.use_occ_mask = args.use_occ_mask
        self.use_depth_confidence = args.use_depth_confidence



        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', self.case_name)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        # self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', self.case_name)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset = Dataset(self.conf['dataset'],
                               data_path=args.data_path,
                               case_name=self.case_name,
                               feature_extractor=args.feature_extractor,
                              )

        self.camera_map = {}
        for camera in self.dataset.cameras:
            self.camera_map[camera.image_name] = camera

        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.iter_use_patch_warping = self.conf.get_int("train.iter_use_patch_warping", default=0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')

        self.is_continue = args.is_continue
        self.mode = args.mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf'], use_embed_mask=self.use_embed_mask).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], use_embed_mask=self.use_embed_mask).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network'], use_embed_mask=self.use_embed_mask).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'],
                                     )

        self.patch_projector = PatchProjector(patch_size=self.dataset.half_patch_size)
        self.patch_loss_func = SSIM(h_patch_size=self.dataset.half_patch_size)

        # Load checkpoint
        latest_model_name = None
        if self.is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            if len(model_list_raw) != 0:
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]

        if args.ckpt != None:
            # checkpoint_path = args.ckpt
            checkpoint_path = os.path.join(self.base_exp_dir, "checkpoints", f"{args.ckpt}.pth")
            logging.info('Find checkpoint: {}'.format(checkpoint_path))
            self.load_checkpoint(checkpoint_path)

        elif latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            checkpoint_path = os.path.join(self.base_exp_dir, 'checkpoints', latest_model_name)
            self.load_checkpoint(checkpoint_path)

        # Backup codes and configs for debug
        # if self.mode[:5] == 'train':
            # self.file_backup()

    def train(self):
        ## When loading checkpoints, calculate the pred depth in advance.
        if self.is_continue or args.ckpt:
            for i in range(self.dataset.n_images):
                self.validate_image(idx=i)

        # self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.writer = None
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step

        for iter_i in tqdm(range(res_step)):
            camera = self.dataset.get_random_camera()
            src_image_name = random.choice(camera.src_image_names)
            src_camera = self.camera_map[src_image_name]

            data = camera.gen_random_rays(batch_size=self.batch_size)

            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            true_rgb = data["color"]
            mask = data["mask"]
            main_uv = data["uv"]
            depth_gt = data["depth"]
            depth_gt_mask = data["depth_mask"]

            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5

            self.sdf_network.progress.data = torch.tensor(self.iter_step / self.end_iter)
            self.nerf_outside.progress.data = torch.tensor(self.iter_step / self.end_iter)
            self.color_network.progress.data = torch.tensor(self.iter_step / self.end_iter)

            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              )

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            weights_wo_background = render_out["weights_wo_background"]
            points = render_out["points"]
            depth = render_out["depth_mean"]
            # depth = render_out["depth_max"]
            gradients = render_out['gradients']

            batch_size, n_samples, _ = points.shape

            occ_mask = None
            if self.use_occ_mask and self.iter_step > 20000:
                occ_mask = get_occ_mask(rays_o, rays_d, depth, camera, src_camera)
            if self.use_depth_confidence and self.iter_step > 20000:
                depth_confidence = get_depth_confidence_map(rays_o, rays_d, depth, main_uv, camera, src_camera)
                occ_mask = (depth_confidence > 0)

            warped_rgb_pixel_loss = 0.0
            if self.use_rgb_pixel_loss:
                sample = get_projected_pixel_color(points=points,
                                                   camera=camera,
                                                   src_camera=src_camera,
                                                   main_uv=main_uv,
                                                   )
                rgb_pixel = sample["rgb_pixel"]  # (n_view, N, n_p, 3)
                mask_pixel = sample["mask_pixel"]  # (n_view, N, n_p, 1)
                rgb_pixel_gt = sample["rgb_pixel_gt"]  # (N, 3)

                warped_rgb_pixel_vals = (weights_wo_background[None, :, :, None] * rgb_pixel).sum(dim=-2)  # (n_view, N, 3)
                warped_rgb_pixel_vals = torch.mean(warped_rgb_pixel_vals, dim=0)  # (N, 3)
                if occ_mask is not None:
                    warped_rgb_pixel_loss = torch.mean(torch.abs(rgb_pixel_gt - warped_rgb_pixel_vals) * occ_mask.unsqueeze(-1))
                else:
                    warped_rgb_pixel_loss = torch.mean(torch.abs(rgb_pixel_gt - warped_rgb_pixel_vals))

            warped_rgb_patch_loss = 0.0
            if self.use_rgb_patch_loss and self.iter_step > self.iter_use_patch_warping:
                ### naive patch warp ###
                sample = get_projected_patch_color(points=points,
                                                   camera=camera,
                                                   src_camera=src_camera,
                                                   main_uv=main_uv,
                                                   )
                rgb_patch = sample["rgb_patch"]  # (n_view, N, n_p, patch_size, 3)
                mask_patch = sample["mask_patch"]  # (n_view, N, n_p, patch_size, 1)
                rgb_patch_gt = sample["rgb_patch_gt"]  # (N, patch_size, 3)

                ################################################
                ### The following is the standard patch warp ###
                uv_normal = main_uv * 2 / torch.tensor([self.dataset.image_width-1, self.dataset.image_height-1]) - 1.0
                rgb_patch, mask_patch, _ = self.patch_projector.patch_warp(points,
                                                                           uv=uv_normal,
                                                                           normals=gradients.reshape(batch_size, n_samples, 3),
                                                                           src_imgs=src_camera.gt_image.permute(2,0,1).unsqueeze(0),
                                                                           ref_intrinsic=camera.intrinsic,
                                                                           src_intrinsics=src_camera.intrinsic.unsqueeze(0),
                                                                           ref_c2w=camera.c2w,
                                                                           src_c2ws=src_camera.c2w.unsqueeze(0))
                rgb_patch = rgb_patch.permute(2, 0, 1, 3, 4) ## (n_rays, n_p, n_view, patch_size, 3) -> (n_view, n_rays, n_p, patch_size, 3)
                mask_patch = mask_patch.permute(2, 0, 1, 3).unsqueeze(-1)
                ################################################

                warped_rgb_patch_vals = (weights_wo_background[None, :, :, None, None] * rgb_patch).sum( dim=2)  # (n_view, N, patch_size, 3)
                warped_rgb_patch_loss = self.patch_loss_func.forward(img_pred=warped_rgb_patch_vals, img_gt=rgb_patch_gt)

                if occ_mask is not None:
                    warped_rgb_patch_loss = self.rgb_patch_loss_weight * torch.mean(warped_rgb_patch_loss * occ_mask)
                else:
                    warped_rgb_patch_loss = self.rgb_patch_loss_weight * torch.mean(warped_rgb_patch_loss)


            warped_feat_loss = 0.0
            if self.use_feat_loss:
                main_latent, src_latents = sample_image_features(points=points,
                                                                 camera=camera,
                                                                 src_camera=src_camera,
                                                                 main_uv=main_uv
                                                                 )
                ### main loss ###
                correlation = torch.sum(main_latent * src_latents, dim=-1)  # (n_view, N, n_p)
                feature_sim = torch.sum(correlation * weights_wo_background[None, :, :], dim=-1).squeeze()
                if occ_mask is not None:
                    warped_feat_loss = torch.mean((1 - feature_sim)*occ_mask)
                else:
                    warped_feat_loss = torch.mean(1 - feature_sim)

            depth_loss = 0.0
            if self.use_depth_loss and self.iter_step > 20000:
                depth_error = torch.abs(depth_gt - depth) * depth_gt_mask
                if occ_mask is not None:
                    depth_error = depth_error.squeeze() * (~occ_mask)
                depth_loss = 0.5 * torch.mean(depth_error)

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = 0.0
            # mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = (color_fine_loss +
                   eikonal_loss * self.igr_weight +
                   mask_loss * self.mask_weight +
                   warped_rgb_pixel_loss +
                   warped_rgb_patch_loss +
                   warped_feat_loss +
                    depth_loss)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            if self.writer:
                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {:.4} lr={:.4} psnr={:.4}'.format(self.iter_step, loss,
                                                                   self.optimizer.param_groups[0]['lr'],
                                                                   psnr))
                # print("warped_rgb_pixel_loss", warped_rgb_pixel_loss)
                # print("warped_rgb_patch_loss", warped_rgb_patch_loss)
                # print("warped_feat_pixel_loss", warped_feat_loss)
                # print("depth_loss", depth_loss)


            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                if self.dataset.n_images > 3:
                    self.validate_image()
                else:
                    for i in range(self.dataset.n_images):
                        self.validate_image(idx=i)

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1, render_image_only=False):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        camera = self.dataset.cameras[idx]
        src_image_name = random.choice(camera.src_image_names)
        src_camera = self.camera_map[src_image_name]
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, uv = camera.gen_rays(resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        uv = uv.reshape(-1, 2).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_depth_mean = []
        out_depth_max = []
        self.renderer.eval()
        print("data length:", len(rays_o))
        for rays_o_batch, rays_d_batch, uv_batch in tqdm(zip(rays_o, rays_d, uv)):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,
                                              )

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            if feasible("depth_mean"):
                out_depth_mean.append(render_out["depth_mean"].detach().cpu().numpy())
            # if feasible("depth_max"):
            #     out_depth_max.append(render_out["depth_max"].detach().cpu().numpy())

            del render_out

        self.renderer.train()

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = camera.w2c[:3, :3].cpu().numpy()
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        depth_mean_img = None
        depth_mean_array= None
        if len(out_depth_mean) > 0:
            depth_mean_img = np.concatenate(out_depth_mean, axis=0).reshape([H, W, -1])
            depth_mean_array = depth_mean_img
            camera.pred_depth = torch.from_numpy(depth_mean_array).cuda()
            depth_mean_img = (depth_mean_img/depth_mean_img.max() * 255).clip(0, 255)

        # depth_max_img = None
        # if len(out_depth_max) > 0:
        #     depth_max_img = np.concatenate(out_depth_max, axis=0).reshape([H, W, -1])
            # np.save(os.path.join(self.base_exp_dir, "depths", f"depth_pred_max_{idx}.npy"), depth_max_img.squeeze())
            # depth_max_img = (depth_max_img/depth_max_img.max() * 255).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, "depths"), exist_ok=True)
        # os.makedirs(os.path.join(self.base_exp_dir, "feat_error_map"), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if render_image_only:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           img_fine[..., i])
                return img_fine[..., i]

            else:
                if len(out_rgb_fine) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'validations_fine',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                               np.concatenate([img_fine[..., i],
                                               camera.get_image_numpy(resolution_level=resolution_level)]))
                if len(out_normal_fine) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'normals',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                               normal_img[..., i])
                if len(out_depth_mean) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir, "depths", '{:0>8d}_{}_{}_mean.png'.format(self.iter_step, i, idx)),
                               depth_mean_img[..., i])
                # if len(out_depth_max) > 0:
                #     cv.imwrite(os.path.join(self.base_exp_dir, "depths", '{:0>8d}_{}_{}_max.png'.format(self.iter_step, i, idx)),
                #                depth_max_img[..., i])


    def validate_mesh(self, world_space=False, resolution=128, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        if resolution == 512:
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_512.ply'.format(self.iter_step)))
        else:
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.WARNING, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/dtu_small_overlap.conf')
    parser.add_argument('--data_path', type=str, default='DATA_PATH')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='scan24')
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--feature_extractor", type=str, default="vismvsnet")
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--use_rgb_pixel_loss", action="store_true")
    parser.add_argument("--use_rgb_patch_loss", action="store_true")
    parser.add_argument("--rgb_patch_loss_weight", type=float, default=1.0)
    parser.add_argument("--use_feat_loss", action="store_true", default=False)
    parser.add_argument("--use_depth_loss", action="store_true", default=False)
    parser.add_argument("--use_embed_mask", action="store_true", default=False)
    parser.add_argument("--use_depth_confidence", action="store_true", default=False)
    parser.add_argument("--use_occ_mask", action="store_true", default=False)

    args = parser.parse_args()


    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)

    print("#################################")
    for k in args.__dict__.keys():
        print("{}: {}".format(k, args.__dict__[k]))
    print("#################################")

    runner = Runner(args)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == "validate_image":
        for idx in range(runner.dataset.n_images):
            runner.validate_image(idx=idx, resolution_level=2)
    # elif args.mode == "eval_image":
    #     psnr_i, ssim_i, lpips_i = runner.render_novel_image_and_eval()
    else:
        raise NotImplementedError(args.mode)



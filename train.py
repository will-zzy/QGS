#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import torchvision
import random
import imageio
from random import randint
from utils.loss_utils import l1_loss, ssim, get_img_grad_weight, lncc
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.vis_utils import get_grayscale_image
from utils.system_utils import load_config
from omegaconf import OmegaConf
from scene.cameras import Camera
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Borrowed from PGSR
def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam



def training(config, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    dataset = config.dataset
    opt = config.optimizer
    gs_model = config.gs_model
    pipe = config.pipeline
    
    tb_writer = prepare_output_and_logger(config)
    gaussians = GaussianModel(gs_model)
    scene = Scene(config, gaussians)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    dist_loss, normal_loss, geo_loss, ncc_loss = None, None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    reset_opacity_densify_delay = 0
    for iteration in range(first_iter, opt.iterations + 1):        
            
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        gt_image, gt_image_gray = viewpoint_cam.get_image()
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        return_depth = iteration > opt.dist_from_iter
        return_normal = iteration > opt.normal_from_iter
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, opt, bg, 
                            kernel_size=dataset.kernel_size, stop_z_gradient = False, return_depth=return_depth, return_normal=return_normal)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        image = rendering
        ssim_loss = (1.0 - ssim(image, gt_image))
        if dataset.use_alpha:
            weight = viewpoint_cam.get_mask
            gt_image = gt_image * weight
        
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        
        
        # Depth and curvature distortion regularization. The curvature distortion and flatten loss performs poorly.
        rend_dist = render_pkg["render_dist"]
        if iteration > opt.dist_from_iter:
            dist_loss = opt.lambda_dist * (rend_dist).mean()
            loss += dist_loss
        
        rend_normal = render_pkg['render_normal']
        depth_normal = render_pkg['surf_normal']
        
        
        
        
        if iteration > opt.normal_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["render_normal"]
            depth_normal = render_pkg["surf_normal"]
            
            rend_curvature_log = torch.clamp_max(torch.log(torch.clamp_min(torch.nan_to_num(render_pkg["render_curvature"].squeeze(),0.0001), 0.0001)), opt.curvature_clamp_threshold)
            image_weight = 1 - torch.sigmoid(rend_curvature_log)
            
            if not opt.wo_image_weight:
                # normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
                normal_loss = weight * (image_weight.detach() * (1 - (normal * depth_normal)).sum(dim=0)[None]).mean()
            else:
                # normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
                normal_loss = weight * (image_weight.detach() * (1 - (normal * depth_normal)).sum(dim=0)[None]).mean()
                
            loss += normal_loss
        
        
        if dataset.use_alpha:
            alpha_map = viewpoint_cam.get_mask
            loss += 0.05 * torch.abs(alpha_map.permute(1,2,0) - render_pkg["render_alpha"].permute(1,2,0)).mean()
        
        # Borrowed from PGSR
        if iteration > opt.multi_view_weight_from_iter:
            # if iteration > opt.multi_view_weight_from_iter and iteration % 10 == 0:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['surf_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['surf_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, opt, bg, kernel_size=dataset.kernel_size, stop_z_gradient = False, return_depth=True, return_normal=True)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['surf_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['surf_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                # if iteration % 200 == 0:
                #     gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                #     if 'app_image' in render_pkg:
                #         img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                #     else:
                #         img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                #     normal_show = (((rend_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                #     depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                #     d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                #     d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                #     depth = render_pkg['render_depth'].squeeze().detach().cpu().numpy()
                #     depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                #     depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                #     depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                #     image_weight = image_weight.detach().cpu().numpy()
                #     image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                #     image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                #     row0 = np.concatenate([gt_img_show, img_show, depth_normal_show], axis=1)
                #     row1 = np.concatenate([d_mask_show_color, depth_color, image_weight_color], axis=1)
                #     image_to_show = np.concatenate([row0, row1], axis=0)
                #     cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3] 
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3] # source坐标系下，ref相机的位置

                        ## compute Homography
                        # ref_local_n = render_pkg["render_normal"].permute(1,2,0)
                        ref_local_n = render_pkg["surf_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['surf_distance'].squeeze()
                        ref_local_d[torch.abs(ref_local_d) < 1e-5] = 1e-5
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] + \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

        
        
        
        loss.backward()
        
        iter_end.record()
        radii = render_pkg["radii"]
        
        # Output the images from the training process.
        render_curvature_min_max=[]
        with torch.no_grad():
            if (iteration) % 50 == 0:
                image_write = torch.clamp(render_pkg["render"], 0.0, 1.0).permute(1,2,0)
                depth_write = render_pkg["surf_depth"].permute(1,2,0).squeeze()
                alpha_write = render_pkg["render_alpha"].permute(1,2,0).squeeze()
                surf_normal_write = render_pkg["surf_normal"].permute(1,2,0)
                surf_normal_write = surf_normal_write * 0.5 + 0.5
                render_normal_write = render_pkg["render_normal"].permute(1,2,0)
                render_normal_write = render_normal_write * 0.5 + 0.5
                render_curvature_write = render_pkg["render_curvature"].permute(1,2,0)
                render_curvature_log_write = torch.log(render_curvature_write + torch.nan_to_num(render_curvature_write, 0.0).min()+1e-7)
                
                
                render_curvature = render_pkg["render_curvature"].permute(1,2,0)
                render_curvature_log_write = torch.log(render_curvature)
                
                
                # render_curvature_log_write = torch.nan_to_num(render_curvature_log_write, 0.0)
                image_write = (image_write.detach().cpu().numpy()*255).astype(np.uint8)
                depth_write = get_grayscale_image(depth_write, data_range=None,cmap='jet')
                alpha_write = get_grayscale_image(alpha_write, data_range=[0, 1],cmap='jet')
                render_normal_write = (render_normal_write.detach().cpu().numpy()*255).astype(np.uint8)
                surf_normal_write = (surf_normal_write.detach().cpu().numpy()*255).astype(np.uint8)
                render_curvature_write = get_grayscale_image(render_curvature_write, data_range=None,cmap='jet')
                # render_curvature_min_max.append(render_curvature_log_write.min())
                # render_curvature_min_max.append(render_curvature_log_write.max())
                # render_curvature_min_max.append(render_curvature_log_write.median())
                
                # img_grad_write = get_grayscale_image(image_weight,data_range=None,cmap='jet')
                curv_mask = 1 - torch.sigmoid(torch.log(render_pkg["render_curvature"].permute(1,2,0)))
                curv_mask_write = get_grayscale_image(curv_mask, data_range=None,cmap='jet')
                
                render_curvature_log_write = get_grayscale_image(render_curvature_log_write, data_range=None,cmap='jet')
                # s3_s1s2_write = get_grayscale_image(torch.log(render_pkg["s3_s1s2"].permute(1,2,0)), data_range=None,cmap='jet')
                output_list = {
                    "rgb": image_write,
                    "depth": depth_write,
                    "alpha_map":alpha_write,
                    "render_normal": render_normal_write,
                    "surf_normal": surf_normal_write,
                    # "curvature": render_curvature_write,
                    "curvature_log": render_curvature_log_write,
                    # "img_grad":img_grad_write,
                    # "curvature_mask":curv_mask_write,
                    # "s3_s1s2":s3_s1s2_write
                }
                if return_depth:
                    output_list.update({"depth": depth_write,
                                        "surf_normal": surf_normal_write})
                if return_normal:
                    output_list.update({"curvature_log": render_curvature_log_write,
                                        "render_normal": render_normal_write,})
                    output_list.update({"depth": depth_write})
                prefix = f"grad_threshold={opt.densify_grad_threshold}_lambda_dist={opt.lambda_dist}"
                for case in output_list:
                    output_dir = os.path.join(dataset.model_path, prefix, case)
                    os.makedirs(output_dir,exist_ok=True)
                    imageio.imwrite(os.path.join(output_dir,f"{iteration}_{case}.jpg"), output_list[case])
                
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() if dist_loss is not None else 0.0
            ema_normal_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}",
                                        #   "distort": f"{ema_dist_for_log:.{5}f}",
                                        #   "distcurv": f"{ema_curv_dist_for_log:.{5}f}",
                                        #   "normal": f"{ema_normal_for_log:.{5}f}",
                                          "geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                                          "ncc": f"{ema_multi_view_pho_for_log:.{5}f}",
                                          "Points":f"{len(gaussians.get_xyz)}",
                                          })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, opt, background, dataset.kernel_size), dataset)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, radii, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if reset_opacity_densify_delay <= 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                    else:
                        reset_opacity_densify_delay -= 1

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    # For scenes with thousands of images, after each reset of opacity, 
                    # the Gaussians in the scene need to be sufficiently optimized before pruning.
                    reset_opacity_densify_delay = opt.reset_opacity_densify_delay 
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./exp/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    OmegaConf.save(args,os.path.join(args.model_path, 'config.yaml'))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, dataset):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    rendering = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    image = rendering[:3, :, :]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.get_image()[0].to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    output_dir = os.path.join(dataset.model_path, "test")
                    output_dir = os.path.join(output_dir,f"{iteration}")
                    os.makedirs(output_dir,exist_ok=True)
                    imageio.imwrite(os.path.join(output_dir,f"{config['name']}_{idx}_rgb.jpg"),(image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])  
                
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default= [i * 500 for i in range(0, 60)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 45_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--conf_path',default='./config/base.yaml')

    args, extras = parser.parse_known_args()
    args.save_iterations.append(args.iterations)
    config_base = OmegaConf.load("./config/base.yaml")
    config_case = OmegaConf.load(args.conf_path)
    config = OmegaConf.merge(config_base, config_case)
    config.model_path = config.model_path.replace(" ", "")
    config.model_path = config.model_path.replace("\n", "")
    # config = load_config(args.conf_path, cli_args=extras)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    # # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(config, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
    
    model_path = config.model_path
    
    os.system(f"python render.py --conf_path {model_path}/config.yaml")
    
    os.system(f"python ./scripts/eval_tnt/run.py --conf_path {model_path}/config.yaml")
    
    

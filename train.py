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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.vis_utils import get_grayscale_image
from utils.system_utils import load_config
from omegaconf import OmegaConf

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


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

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx

    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
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
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        gt_image = viewpoint_cam.get_image.cuda()
        render_pkg = render(viewpoint_cam, gaussians, pipe, opt, background, kernel_size=dataset.kernel_size, stop_z_gradient = False)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        image = rendering[:3, :, :]
        lambda_normal = opt.lambda_normal if iteration > opt.normal_from_iter else 0.0
        lambda_dist = opt.lambda_dist if iteration >  opt.dist_from_iter else 0.0
        lambda_curv_dist = opt.lambda_curv_dist if iteration > opt.curv_from_iter else 0.0
        lambda_curv_flat = opt.lambda_curv_flat if iteration > opt.curv_from_iter else 0.0
        
        # Depth and curvature distortion regularization. The curvature distortion and flatten loss performs poorly.
        rend_dist = render_pkg["render_dist"]
        render_curvature_dist = render_pkg["render_curvature_dist"]
        dist_loss = lambda_dist * (rend_dist).mean()
        # curvature_dist_loss = lambda_curv_dist * (render_curvature_dist).mean()
        curvature_dist_loss = torch.tensor(0, device = image.device)
        
        rend_normal = render_pkg['render_normal']
        surf_normal = render_pkg['surf_normal']
        
        num_curvature = torch.abs(torch.nan_to_num(render_pkg["render_curvature"],0.00001))
        rend_curvature_log = torch.clamp_max(torch.log(torch.clamp_min(num_curvature, 0.00001)), opt.curvature_clamp_threshold)
        curv_mask = 1 - torch.sigmoid(rend_curvature_log)
        
        curv_flat_mask = num_curvature < 2.0
        # curv_flat_loss = lambda_curv_flat * render_pkg["render_curvature"][curv_flat_mask].mean()
        curv_flat_loss = torch.tensor(0, device = image.device)
        
        alpha_map = viewpoint_cam.get_mask
        normal_error = (alpha_map * curv_mask.detach() * (1 - (rend_normal * surf_normal))).sum(dim=0)[None]
        normal_loss = lambda_normal * (normal_error).mean()
        
        alpha_loss = torch.tensor(0, device = image.device)
        if dataset.use_alpha:
            alpha_loss = 0.05 * torch.abs(alpha_map.permute(1,2,0) - render_pkg["render_alpha"].permute(1,2,0)).mean()
        
        Ll1 = l1_loss(image, gt_image)
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = rgb_loss + dist_loss + normal_loss + curvature_dist_loss + curv_flat_loss + alpha_loss
        loss.backward()
        
        iter_end.record()
        radii = render_pkg["radii"]
        
        # Output the images from the training process.
        with torch.no_grad():
            image_write = torch.clamp(render_pkg["render"], 0.0, 1.0).permute(1,2,0)
            depth_write = render_pkg["render_depth"].permute(1,2,0).squeeze()
            alpha_write = render_pkg["render_alpha"].permute(1,2,0).squeeze()
            surf_normal_write = render_pkg["surf_normal"].permute(1,2,0)
            surf_normal_write = surf_normal_write * 0.5 + 0.5
            render_normal_write = render_pkg["render_normal"].permute(1,2,0)
            render_normal_write = render_normal_write * 0.5 + 0.5
            render_curvature_write = render_pkg["render_curvature"].permute(1,2,0)
            render_curvature_log_write = torch.log(render_curvature_write + render_curvature_write.min() + 1e-7)
            
            image_write = (image_write.detach().cpu().numpy()*255).astype(np.uint8)
            depth_write = get_grayscale_image(depth_write, data_range=None,cmap='jet')
            alpha_write = get_grayscale_image(alpha_write, data_range=[0,1],cmap='jet')
            render_normal_write = (render_normal_write.detach().cpu().numpy()*255).astype(np.uint8)
            surf_normal_write = (surf_normal_write.detach().cpu().numpy()*255).astype(np.uint8)
            render_curvature_write = get_grayscale_image(render_curvature_write, data_range=None,cmap='jet')
            render_curvature_log_write = get_grayscale_image(render_curvature_log_write, data_range=None,cmap='jet')
            
            output_list = {
                "rgb": image_write,
                "depth": depth_write,
                "alpha_map":alpha_write,
                "render_normal": render_normal_write,
                "surf_normal": surf_normal_write,
                "curvature": render_curvature_write,
                "curvature_log": render_curvature_log_write,
            }
            if (iteration) % 50 == 0:
                prefix = f"grad_threshold={opt.densify_grad_threshold}_lambda_dist={opt.lambda_dist}"
                for case in output_list:
                    output_dir = os.path.join(dataset.model_path, prefix, case)
                    os.makedirs(output_dir,exist_ok=True)
                    imageio.imwrite(os.path.join(output_dir,f"{iteration}_{case}.jpg"), output_list[case])
                
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * rgb_loss.item() + 0.6 * ema_loss_for_log
            ema_curv_dist_for_log = 0.4 * curvature_dist_loss.item() + 0.6 * ema_dist_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}",
                                          "distort": f"{ema_dist_for_log:.{5}f}",
                                        #   "distcurv": f"{ema_curv_dist_for_log:.{5}f}",
                                        #   "normal": f"{ema_normal_for_log:.{5}f}",
                                          "radiiMax":f"{radii.max().item():.{5}f}",
                                          "Points":f"{len(gaussians.get_xyz)}",
                                        #   "s":f"{s.item():.{5}f}",
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

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
    
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
                    gt_image = torch.clamp(viewpoint.get_image.to("cuda"), 0.0, 1.0)
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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000, 45_000, 60_000])
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

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

import torch
import math
from diff_quadratic_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.depth_utils import depth_to_normal
from torchvision import transforms

def render(viewpoint_camera, 
           pc : GaussianModel, 
           pipe, opt, 
           bg_color : torch.Tensor, 
           kernel_size = None, 
           scaling_modifier = 1.0, 
           override_color = None, 
           subpixel_offset=None, 
           stop_z_gradient = False,
           app_model=None,
           return_depth=True,
           return_normal=True
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        sigma=pc.sigma,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        cam_intr=viewpoint_camera.cam_intr,
        prefiltered=False,
        debug=pipe.debug,
        # debug=True,
        stop_z_gradient = stop_z_gradient,
        reciprocal_z = pipe.reciprocal_z,
        return_depth=return_depth,
        return_normal=return_normal,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # opacity = pc.get_opacity_with_3D_filter
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = pc.get_scaling
    rotations = pc.get_rotation

    view2gaussian_precomp = None
    if pipe.compute_view2gaussian_python:
        view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # # we local direction
            # cam_pos_local = view2gaussian_precomp[:, 3, :3]
            # cam_pos_local_scaled = cam_pos_local / scales
            # dir_pp = -cam_pos_local_scaled
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, aabb, n_touched = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        view2gaussian_precomp=view2gaussian_precomp)
    
    # bbox_occupied_pixels = (aabb[:, 2:3] - aabb[:, 0:1]) * (aabb[:, 3:4] - aabb[:, 1:2])
    # occupancy_rate = (n_touched.view(-1,1) / (bbox_occupied_pixels + 1)).view(-1)
    # visibility_filter = (radii > 0) & (occupancy_rate > opt.occupancy_rate) if pipe.occlusion_awared_denom else (radii > 0)
    visibility_filter = radii > 0
    rets = {
        "render_all": rendered_image,
        "render": rendered_image[:3, :, :],
        "viewspace_points": screenspace_points,
        "visibility_filter" : visibility_filter,
        "radii": radii,
        "aabb": aabb,
        "n_touched": n_touched
    }
    
    render_dist = rendered_image[8:9]
    render_dist = torch.nan_to_num(render_dist)
    
    render_normal = rendered_image[3:6, ...] # normal in the view space
    # render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1) # 世界坐标系
    
    render_depth = rendered_image[6:7, ...]
    render_depth = torch.nan_to_num(render_depth, 0, 0)
    
    render_alpha = rendered_image[7:8, ...]

    render_depth_median = rendered_image[9:10]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    render_depth_expected = render_depth / render_alpha
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    render_curvature = rendered_image[11:12]
    render_curvature = torch.nan_to_num(render_curvature, 0, 0)
    render_curvature_dist = rendered_image[12:13]
    render_curvature_dist = torch.nan_to_num(render_curvature_dist, 0, 0)
    
    surf_normal, surf_point = depth_to_normal(viewpoint_camera, surf_depth) # (W,H,3)
    surf_distance = torch.nan_to_num(torch.sum(surf_normal * surf_point, dim=-1), 0, 0) # <= 0
    surf_normal = surf_normal.permute(2,0,1)
    surf_point = surf_point.permute(2,0,1)
    alpha_mask = viewpoint_camera.get_mask if viewpoint_camera.get_mask is not None else torch.ones_like(render_alpha)
    surf_normal = surf_normal * (render_alpha).detach() * alpha_mask
    s3_s1s2 = rendered_image[13:14]
    rets.update({
            'render_alpha': render_alpha,
            'render_normal': render_normal,
            'surf_normal': surf_normal,
            'surf_distance': surf_distance,
            'surf_point': surf_point,
            'render_depth': render_depth_expected,
            'surf_depth': surf_depth,
            'render_dist': render_dist,
            'render_curvature': render_curvature,
            'render_curvature_dist': render_curvature_dist,
            's3_s1s2': s3_s1s2
    })
    
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image[:3, :, :] + appear_ab[1]
        rets.update({"app_image": app_image})   
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return rets

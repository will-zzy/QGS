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
from torch import nn
import numpy as np
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import torch.nn.functional as F

def dilate(bin_img, ksize=6):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=12):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def process_image(image_path, resolution, ncc_scale):
    image = Image.open(image_path)
    if len(image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(image.split()[3], resolution)
        gt_image = resized_image_rgb
        if ncc_scale != 1.0: # 需要原图像的分辨率来计算NCC
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = torch.cat([PILtoTorch(im, ncc_resolution) for im in image.split()[:3]], dim=0)
    else:
        resized_image_rgb = PILtoTorch(image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = PILtoTorch(image, ncc_resolution)
    gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
    return gt_image, gray_image, loaded_mask

class Camera(nn.Module):
    def __init__(self, cam_info, resolution, uid,
                 gt_alpha_mask=None, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.R = cam_info.R # W2C.R^T
        self.T = cam_info.T
        self.nearest_id = []
        self.nearest_names = []
        self.FoVx = cam_info.FovX
        self.FoVy = cam_info.FovY
        self.image_name = cam_info.image_name
        self.image_path = cam_info.image_path
        self.cam_intr = torch.tensor(cam_info.cam_intr * cam_info.downsample,dtype=torch.float32,device="cpu")
        self.use_alpha = cam_info.use_alpha
        self.image_type = cam_info.image_type
        self.uid = uid
        self.resolution = resolution
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.gt_alpha_mask = gt_alpha_mask
        self.Fx = self.cam_intr[0]
        self.Fy = self.cam_intr[1]
        self.Cx = self.cam_intr[2]
        self.Cy = self.cam_intr[3]

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
            
        self.original_image, self.image_gray, self.mask = None, None, None
        self.preload_img = self.image_type == "all"
        self.ncc_scale = cam_info.downsample ###
        
        gt_image, gray_image, loaded_mask = process_image(self.image_path, self.resolution, self.ncc_scale)
        if self.preload_img:
            device = self.data_device
        else:
            device = "cpu"
            
        self.original_image = gt_image.to(device)
        self.original_image_gray = gray_image.to(device)
        if loaded_mask is not None:
            self.mask = loaded_mask.to(device)
        else:
            self.mask = torch.ones_like(self.original_image[0], device=device)

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = self.image_height / (2.0 * tan_fovy)
        self.focal_x = self.image_width / (2.0 * tan_fovx)
        
    # @property
    # def get_image(self):
    #     if self.image_type == "all":
    #         return self.image
    #     if self.image_type == "iterable":
    #         image = Image.open(self.image_path)
    #         if len(image.split()) > 3:
    #             if self.use_alpha:
    #                 loaded_mask = PILtoTorch(image.split()[3], self.resolution).to(self.data_device)
    #             else:
    #                 loaded_mask = torch.ones_like(PILtoTorch(image.split()[0], self.resolution)).to(self.data_device)
    #             resized_image_rgb = torch.cat([PILtoTorch(im, self.resolution) for im in image.split()[:3]], dim=0).to(self.data_device)
    #             return_image = (loaded_mask * resized_image_rgb).to(self.data_device)
    #         else:
    #             return_image = PILtoTorch(image, self.resolution).to(self.data_device)
    #     return return_image  
    def get_image(self):
        if self.preload_img:
            return self.original_image, self.original_image_gray
        else:
            return self.original_image.cuda(), self.original_image_gray.cuda()

    @property
    def get_mask(self):
        if self.preload_img:
            return self.mask
        else:
            image = Image.open(self.image_path)
            if len(image.split()) > 3 and self.use_alpha:
                return_image = PILtoTorch(image.split()[3], self.resolution).to(self.data_device)
            else:
                return_image = torch.ones_like(PILtoTorch(image.split()[0], self.resolution)).to(self.data_device)
        return return_image  
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W, device="cuda").float() + 0.5, torch.arange(H, device="cuda").float() + 0.5, indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1)
        return rays_d  
    
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]]).cuda()
        return K
    
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


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

class Camera(nn.Module):
    def __init__(self, cam_info, resolution, uid,
                 gt_alpha_mask=None, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.R = cam_info.R
        self.T = cam_info.T
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

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if self.image_type == "all":
            image = Image.open(self.image_path)
            if len(image.split()) > 3:
                loaded_mask = PILtoTorch(image.split()[3], resolution)
                self.gt_alpha_mask = loaded_mask.to(self.data_device)
                resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image.split()[:3]], dim=0)
                if self.use_alpha:
                    self.image = (loaded_mask * resized_image_rgb).to(self.data_device)
                else:
                    self.image = (resized_image_rgb).to(self.data_device)
            else:
                self.image = PILtoTorch(image, resolution).to(self.data_device)
                self.gt_alpha_mask = torch.ones_like(PILtoTorch(image.split()[0], self.resolution)).to(self.data_device)
        

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
        
    @property
    def get_image(self):
        if self.image_type == "all":
            return self.image
        if self.image_type == "iterable":
            image = Image.open(self.image_path)
            if len(image.split()) > 3:
                if self.use_alpha:
                    loaded_mask = PILtoTorch(image.split()[3], self.resolution).to(self.data_device)
                else:
                    loaded_mask = torch.ones_like(PILtoTorch(image.split()[0], self.resolution)).to(self.data_device)
                resized_image_rgb = torch.cat([PILtoTorch(im, self.resolution) for im in image.split()[:3]], dim=0).to(self.data_device)
                return_image = (loaded_mask * resized_image_rgb).to(self.data_device)
            else:
                return_image = PILtoTorch(image, self.resolution).to(self.data_device)
        return return_image   
     
    @property
    def get_mask(self):
        if self.image_type == "all":
            return self.gt_alpha_mask
        if self.image_type == "iterable":
            image = Image.open(self.image_path)
            if len(image.split()) > 3 and self.use_alpha:
                return_image = PILtoTorch(image.split()[3], self.resolution).to(self.data_device)
            else:
                return_image = torch.ones_like(PILtoTorch(image.split()[0], self.resolution)).to(self.data_device)
        return return_image    

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


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
import random
import json
import cv2
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, read_extrinsics_binary, read_intrinsics_binary
import torch
from PIL import Image
def params_from_models(camera_model: np.ndarray):
    K = np.array([[camera_model[0], 0, camera_model[2]],
                  [0, camera_model[1], camera_model[3]],
                  [0,  0,  1]])
    try:
        distortion = camera_model[4:]
    except:
        distortion = np.array([0,0,0,0])
    return K,distortion
class Scene:

    gaussians : GaussianModel

    def __init__(self, config, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.config = config
        self.root_dir = config.root_dir
        self.model_path = config.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.img_downsample = self.config.dataset.downsample
        self.config.white_background = False
        self.img_dir = os.path.join(self.root_dir, f"images_undistorted_{self.img_downsample}")
        images_prefix = "images" if os.path.exists(os.path.join(self.root_dir, "sparse")) else "train"
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        self.load_images(images_prefix) # clone image
        self.scene_info = self.load_cam() # 包含pointCloud, train/test_cam_infos, nerf_normalization和ply_path
        
        if not self.loaded_iter:
            with open(self.scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        self.cameras_extent = self.scene_info.nerf_normalization["radius"]
        print(f"cameras_extent {self.cameras_extent}")
        
        self.multi_view_num = config.gs_model.multi_view_num
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale, self.config.dataset)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, self.config.dataset)
            
            print("computing nearest_id")
            self.world_view_transforms = []
            camera_centers = []
            center_rays = []
            for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                # 记录所有相机的世界坐标系下的光心、视线朝向和W2C
                self.world_view_transforms.append(cur_cam.world_view_transform)
                camera_centers.append(cur_cam.camera_center)
                R = torch.tensor(cur_cam.R).float().cuda()
                T = torch.tensor(cur_cam.T).float().cuda()
                center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda() # 超前看的方向，colmap坐标系
                center_ray = center_ray@R.transpose(-1,-2) # c2w转到世界坐标系
                center_rays.append(center_ray)
                
            self.world_view_transforms = torch.stack(self.world_view_transforms)
            camera_centers = torch.stack(camera_centers, dim=0)
            center_rays = torch.stack(center_rays, dim=0)
            center_rays = torch.nn.functional.normalize(center_rays, dim=-1)# 归一化朝向
            diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy() # [N_cams, N_cams]，每个相机和每个相机之间的距离
            tmp = torch.sum(center_rays[:,None]*center_rays[None], dim=-1) # 每个相机和每个相机视线的余弦值
            angles = torch.arccos(tmp)*180/3.14159 # 转换为夹角，角度制
            angles = angles.detach().cpu().numpy()
            with open(os.path.join(self.model_path, "multi_view.json"), 'w') as file:
                for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                    sorted_indices = np.lexsort((angles[id], diss[id])) # 先按夹角排序，再按距离排序
                    # sorted_indices = np.lexsort((diss[id], angles[id]))
                    mask = (angles[id][sorted_indices] < self.config.dataset.multi_view_max_angle) & \
                        (diss[id][sorted_indices] > self.config.dataset.multi_view_min_dis) & \
                        (diss[id][sorted_indices] < self.config.dataset.multi_view_max_dis) # 要求夹角在最大夹角以内，距离在最小与最大距离之间
                    sorted_indices = sorted_indices[mask] # 邻域视角索引，按照先夹角后距离排序
                    multi_view_num = min(self.multi_view_num, len(sorted_indices))
                    json_d = {'ref_name' : cur_cam.image_name, 'nearest_name': []}
                    for index in sorted_indices[:multi_view_num]:
                        # 遍历邻域视角，并写入multi_view.json以及train_cameras
                        cur_cam.nearest_id.append(index)
                        cur_cam.nearest_names.append(self.train_cameras[resolution_scale][index].image_name)
                        json_d["nearest_name"].append(self.train_cameras[resolution_scale][index].image_name)
                    json_str = json.dumps(json_d, separators=(',', ':'))
                    file.write(json_str)
                    file.write('\n')
                    # print(f"frame {cur_cam.image_name}, neareast {cur_cam.nearest_names}, \
                    #       angle {angles[id][cur_cam.nearest_id]}, diss {diss[id][cur_cam.nearest_id]}")

        # for resolution_scale in resolution_scales:
        #     print("Loading Training Cameras")
        #     self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale, self.config.dataset)
        #     print("Loading Test Cameras")
        #     self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, self.config.dataset)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    def load_images(self, images):
        if not os.path.exists(self.img_dir):    
            self.undist_and_downsample()
        if len(os.listdir(self.img_dir)) != len(os.listdir(os.path.join(self.root_dir,images))):
            self.undist_and_downsample()
        pass    
    
    def load_cam(self):
        if os.path.exists(os.path.join(self.root_dir, "sparse")):
            print("Found sparse directory, assuming Colmap data set!")
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.root_dir, 
                                                          self.img_dir,
                                                          self.config.dataset)
        elif os.path.exists(os.path.join(self.root_dir, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](self.root_dir, self.config.dataset.white_background, self.config.dataset.eval, self.config.dataset)
        elif os.path.exists(os.path.join(self.config.source_path, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](self.root_dir, self.config.dataset.white_background, self.config.dataset.eval)
        else:
            assert False, "Could not recognize scene type!"
        return scene_info
    
    # For scenes with many images, undistort and downsample all images, then store them on the disk.
    def undist_and_downsample(self):
        PREFIX = f"images_undistorted_{self.img_downsample}"
        output_dir = os.path.join(self.root_dir,PREFIX)
        input_dir = os.path.join(self.root_dir,"images")
        if os.path.exists(os.path.join(self.root_dir, "transforms_train.json")):
            input_dir = os.path.join(self.root_dir,"train")
            
        os.makedirs(output_dir, exist_ok=True)  
        if os.path.exists(os.path.join(self.root_dir, "sparse")):
            models, img_data = self.read_meta()
            for img_idx in tqdm(img_data):
                img_path = os.path.join(input_dir,img_data[img_idx].name)
                output_path = os.path.join(output_dir,img_data[img_idx].name)
                distorted = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                img_wh = int(distorted.shape[1] * self.img_downsample), int(distorted.shape[0] * self.img_downsample)
                if self.config.dataset.undistortion and models[img_data[img_idx].camera_id].model in ["OPENCV"]:
                    try:
                        K,distortion = params_from_models(models[img_data[img_idx].camera_id].params)
                        undistorted = cv2.undistort(distorted,
                                                    K,
                                                    distortion)
                    except:
                        undistorted = distorted
                else:
                    undistorted = distorted

                undistorted = cv2.resize(undistorted,(img_wh[0],img_wh[1]))
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                cv2.imwrite(output_path,undistorted)
        else:
            input_path = os.path.join(input_dir, os.listdir(input_dir)[0])
            img_shape = Image.open(input_path).size
            target_size = (int(self.img_downsample * img_shape[0]), int(self.img_downsample * img_shape[1]))
            for filename in tqdm(os.listdir(input_dir)):
                if filename.lower().endswith(".png"):
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, filename)

                    # 打开图片并 resize
                    with Image.open(input_path) as img:
                        resized_img = img.resize(target_size, Image.LANCZOS)
                        resized_img.save(output_path)
    def read_meta(self):
        
        try:
            img_data = read_extrinsics_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
            cam_data = read_intrinsics_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        except:
            img_data = read_extrinsics_text(os.path.join(self.root_dir, 'sparse/0/images.txt'))
            cam_data = read_intrinsics_text(os.path.join(self.root_dir, 'sparse/0/cameras.txt'))   

        return cam_data,img_data    
    

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.optimize import fsolve
from tqdm import tqdm
import cv2
torch.set_default_dtype(torch.float32)


def build_covariance_2d(
    mean3d, cov3d, viewmatrix, tan_fovx, tan_fovy, focal_x, focal_y
):
    import math
    t = (mean3d @ viewmatrix[:3, :3]) + viewmatrix[-1:, :3]
    tz = t[..., 2]
    tx = t[..., 0]
    ty = t[..., 1]

    # Eq.29 locally affine transform
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    W = viewmatrix[:3, :3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0, 2, 1)

    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.0
    return cov2d[:, :2, :2] + filter[None]

def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R,q

def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r).permute(0, 2, 1)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R ,q= build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L,q

def getProjectionMatrix(znear, zfar, fovX, fovY):
    import math
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def focal2fov(focal, pixels):
    import math
    return 2 * math.atan(pixels / (2 * focal))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim = -1)

def homogeneous_vec(vec):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([vec, torch.zeros_like(vec[..., :1])], dim = -1)

def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask

def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lambda1 = mid + torch.sqrt((mid**2 - det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2 - det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

def alpha_blending(alpha, colors):
  T = torch.cat([torch.ones_like(alpha[-1:]), (1-alpha).cumprod(dim = 0)[:-1]], dim=0)
  omega = T * alpha
  image = (omega * colors).sum(dim=0).reshape(-1, colors.shape[-1])
  alphamap = (T * alpha).sum(dim=0).reshape(-1, 1)
  return image, alphamap, omega

def alpha_blending_with_gaussians(dist2, colors, opacities, H, W):
    colors = colors.reshape(-1, 1, colors.shape[-1])
    gaussians = torch.exp(-dist2 / 2)
    gaussians = gaussians[..., None]
    alpha = opacities.unsqueeze(1) * gaussians
    
    # accumulate gaussians
    image, _, omega = alpha_blending(alpha, colors)
    return image.reshape(H, W, -1), omega


def get_inputs(num_points = 8):
    length = 0.5
    x = np.linspace(-1, 1, num_points) * length
    y = np.linspace(-1, 1, num_points) * length
    x, y = np.meshgrid(x, y)
    means3D = torch.from_numpy(np.stack([x,y, 0 * np.random.rand(*x.shape)], axis=-1).reshape(-1, 3)).cuda().float()
    quats = torch.zeros(1, 4).repeat(len(means3D), 1).cuda()
    quats[..., 0] = 1.
    scale = length / (num_points - 1)
    scales = torch.zeros(1,3).repeat(len(means3D), 1).fill_(scale).cuda() * 1.
    return means3D, scales, quats

from math import cos,sin, pi
def get_Ry(theta):
    R = torch.eye(3)
    R[0,0] = cos(theta)
    R[2,2] = cos(theta)
    R[0,2] = -sin(theta)
    R[2,0] = sin(theta)
    return R
def get_Rx(theta):
    R = torch.eye(3)
    R[1,1] = cos(theta)
    R[2,2] = cos(theta)
    R[1,2] = -sin(theta)
    R[2,1] = sin(theta)
    return R
def get_Rz(theta):
    R = torch.eye(3)
    R[0,0] = cos(theta)
    R[1,1] = cos(theta)
    R[0,1] = -sin(theta)
    R[1,0] = sin(theta)
    return R
def get_cameras(idx):
    intrins = torch.tensor([[711.1111,   0.0000, 256.0000,   0.0000],
               [  0.0000, 711.1111, 256.0000,   0.0000],
               [  0.0000,   0.0000,   1.0000,   0.0000],
               [  0.0000,   0.0000,   0.0000,   1.0000]]).cuda()
    R = torch.tensor([
         [-8.6086e-01,  3.7950e-01, -3.3896e-01],
         [ 5.0884e-01,  6.4205e-01, -5.7346e-01],
         [ 1.0934e-08, -6.6614e-01, -7.4583e-01]]).cuda()
    t =  torch.tensor([
         [6.7791e-01],
         [1.1469e+00],
         [0.3517e+00]
         ]).cuda()

    dRx = get_Rx(-20 / 180 * pi).cuda()
    dRy = get_Ry(-20 / 180 * pi).cuda()
    R = dRy @ dRx @ R

    t[0] = t[0] - 99 * 0.02
    t[1] = t[1] - 99 * 0.005
    dRz = get_Rz(99 / 180 * pi).cuda()
    R = dRz @ R
    
    c2w = torch.cat([R, t],dim = 1).cuda()
    bottom_vector = torch.tensor([[0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).cuda()
    c2w = torch.cat([c2w, bottom_vector],dim = 0)
    
    width, height = intrins[0, 2] * 2, intrins[0, 2] * 2
    focal_x, focal_y = intrins[0, 0], intrins[1, 1]
    viewmat = torch.linalg.inv(c2w).permute(1, 0) # w2c.T
    FoVx = focal2fov(focal_x, width)
    FoVy = focal2fov(focal_y, height)
    projmat = getProjectionMatrix(znear=0.2, zfar=1000, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda() # P.T
    projmat = viewmat @ projmat # (P @ w2c).T
    return intrins, viewmat, projmat, height, width, c2w.T

def QuadraticCurveGeodesicDistance_numpy(x0, a = 1):
    u0 = 2 * a * x0
    sqrt_tmp = np.sqrt(u0**2 + 1)
    first = np.log(sqrt_tmp + u0) / (4 * a)
    first[np.isnan(first)] = 0.0
    s = first + x0 * sqrt_tmp / 2  
    return s

def QuadraticCurveGeodesicDistance_torch(x0, a = 1):
    if torch.is_tensor(a):
        a = a.view(x0.shape)
    u0 = 2 * a * x0
    sqrt_tmp = torch.sqrt(u0**2 + 1)
    first = torch.log(sqrt_tmp + u0) / (4 * a)
    first[first.isnan()] = 0.0
    s = first + x0 * sqrt_tmp / 2  
    return s

def GetRootFromEquation(f, sigma, a=1):
    roots = []
    for i in range(0, sigma.shape[0]):
        inverse_GD = lambda x : f(x, a[i]) - sigma[i]
        root = fsolve(inverse_GD, x0 = np.random.rand(1))
        roots.append(torch.tensor(root))
    roots = torch.stack(roots)
    return roots
def getQuadraticCurveA(cos_theta, sin_theta, s,sign_s1, sign_s2, sign_s3):
    a = s[..., 2] * (sign_s1 * cos_theta**2 / s[..., 0]**2 + sign_s2 * sin_theta**2 / s[..., 1]**2)
    return torch.abs(a)

def setup(means3D, scales_, quats, opacities, colors, viewmat, projmat, sigma):
    scales = scales_.clone()
    rotations, q = build_rotation(quats)
    rotations = rotations.permute(0,2,1)
    p_view = (means3D @ viewmat[:3,:3]) + viewmat[-1:,:3]
    WH_R = (rotations @ viewmat[:3,:3])
    WH = torch.cat([homogeneous_vec(WH_R[:,:3,:]), homogeneous(p_view.unsqueeze(1))], dim = 1)
    T_t = WH @ projmat # (KWH)^T [N, 4, 4]
    device = T_t.device
    
    a = torch.abs(scales[..., 2] / torch.max(torch.abs(scales[..., :2]), dim = 1)[0]**2)
    a = a.cpu().numpy()
    sigma = torch.max(torch.abs(scales[..., :2]), dim = 1)[0].cpu().numpy() * sigma
    
    sign_s1 = torch.sign(scales[0, 0])
    sign_s2 = torch.sign(scales[0, 1])
    sign_s3 = torch.sign(scales[0, 2])
    
    t0 = torch.tensor(GetRootFromEquation(QuadraticCurveGeodesicDistance_numpy, sigma, a=a))
    t0 = t0.to(device).to(torch.float32) 
    t0_2 = t0**2 * torch.tensor(a,device = t0.device).view(-1,1)
    
    scale_up = torch.cat([t0, t0, t0_2,torch.ones_like(t0)], dim = 1)
    saddle = 1 if sign_s1 * sign_s2 < 0 else 0 
    convex = 1 if not saddle and sign_s1 * sign_s3 >= 0 else 0
    concave = 1 if not saddle and sign_s1 * sign_s3 < 0 else 0
    P = torch.tensor([
        [-1, -1, -float(saddle | concave), 1],
        [1, -1, -float(saddle | concave), 1],
        [-1, 1, -float(saddle | concave), 1],
        [1, 1, -float(saddle | concave), 1],
        [-1, -1, float(saddle | convex), 1],
        [1, -1, float(saddle | convex), 1],
        [-1, 1, float(saddle | convex), 1],
        [1, 1, float(saddle | convex), 1], 
    ],device=device)
    P = P.unsqueeze(0) * scale_up.unsqueeze(1)
    P = P.unsqueeze(2) # [N, 8, 1, 4]
    p = P @ T_t.unsqueeze(1) # [N, 8, 1, 4] @ [N, 1, 4, 4] = [N,8,1,4]
    p = p.squeeze()
    uv = p[:, :, :2] / p[:, :, 3:4]
    
    x_min, x_max = uv[..., 0].min(dim=1)[0].view(-1, 1), uv[..., 0].max(dim=1)[0].view(-1, 1) # [N]
    y_min, y_max = uv[..., 1].min(dim=1)[0].view(-1, 1), uv[..., 1].max(dim=1)[0].view(-1, 1) # [N]
    
    depth = p_view[..., 2] # depth is used only for sorting
    index = depth.sort()[1]
    T_t = T_t[index]
    opacities = opacities[index]
    colors = colors[index]
    p = p[index]
    x_min = x_min[index]
    x_max = x_max[index]
    y_min = y_min[index]
    y_max = y_max[index]
    
    return x_min, x_max, y_min, y_max, depth, T_t, p, opacities, colors, WH

def quadratic_splatting(means3D, scales, quats, colors, opacities, intrins, viewmat, projmat, c2w, sigma = 1.0):
    N = means3D.shape[0]
    
    projmat = torch.zeros(4, 4).cuda() 
    projmat[:3, :3] = intrins
    projmat[-1, -2] = 1.0 
    projmat = projmat.T
    x_min, x_max, y_min, y_max, depth, T_t, p, opacities, colors, WH = setup(means3D, scales, quats, opacities, colors, viewmat, projmat, sigma)
    setup_batch = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "T_t": T_t,
        "p": p
    }
    
    device = T_t.device
    V2G = torch.zeros_like(T_t, device = device)
    V2G[:, 3, 3] = 1.0
    V2G[:, :3, :3] = torch.inverse(T_t[:, :3, :3])
    t2 = -T_t[:, 3:4, :3] @ V2G[:, :3, :3]
    V2G[:, 3:4, :3] = t2
    
    H, W = (intrins[0, -1] * 2).long(), (intrins[1, -1] * 2).long()
    H, W = H.item(), W.item()
    pix = torch.stack(torch.meshgrid(torch.arange(H),
        torch.arange(W), indexing='xy'), dim = -1).to('cuda') # [W, H, 2]

    pix = pix.view(-1, 2)
    pix = torch.cat([pix, torch.ones([pix.shape[0], 1],device = device)], dim = -1) # [WxH, 3]
    # Compute ray splat intersection
    
    cam_pos_local = V2G[:, -1, :3].unsqueeze(1).repeat([1, H * W, 1]) #[N, WxH, 3]
    cam_ray_local = pix.unsqueeze(0) @ V2G[:, :3, :3] # [1, WxH, 3] X [N, 3, 3] = [N, WxH, 3]

    scales_ = scales.unsqueeze(1).repeat([1, W * H, 1])
    sign_s1 = torch.sign(scales[0, 0])
    sign_s2 = torch.sign(scales[0, 1])
    sign_s3 = torch.sign(scales[0, 2])
    
    rs1_2 = 1 / scales[..., 0:1]**2 * sign_s1
    rs2_2 = 1 / scales[..., 1:2]**2 * sign_s2
    rs3 = 1 / scales[..., 2:3]
    
    A = rs1_2 * cam_ray_local[..., 0]**2 + rs2_2 * cam_ray_local[..., 1]**2
    B = 2 * (rs1_2 * cam_pos_local[..., 0] * cam_ray_local[..., 0] + \
            rs2_2 * cam_pos_local[..., 1] * cam_ray_local[..., 1]) - rs3 * cam_ray_local[..., 2]
    C = rs1_2 * cam_pos_local[..., 0]**2 + rs2_2 * cam_pos_local[..., 1]**2 - rs3 * cam_pos_local[..., 2]

    
    discriminant = B**2 - 4 * A * C
    intersect_mask = discriminant > 0 
    discriminant_sq_with_intersect = torch.sqrt(discriminant[intersect_mask])
    
    A_with_intersect = A[intersect_mask]
    B_with_intersect = B[intersect_mask]
    scales_ = scales_[intersect_mask]
    
    
    B_2A = B_with_intersect / (2 * A_with_intersect)
    disc_sq_2A =  discriminant_sq_with_intersect / (2 * A_with_intersect)
    root_1 = (-B_2A + disc_sq_2A).view(-1, 1)
    root_2 = (-B_2A - disc_sq_2A).view(-1, 1)
    
    point_local_1 = cam_pos_local[intersect_mask] + root_1 * cam_ray_local[intersect_mask]
    point_local_2 = cam_pos_local[intersect_mask] + root_2 * cam_ray_local[intersect_mask]
    
    proj_point1 = torch.norm(point_local_1[:, :2],dim=1)
    proj_point2 = torch.norm(point_local_2[:, :2],dim=1)
    
    cos_theta_1 = point_local_1[:, 0] / proj_point1 
    sin_theta_1 = point_local_1[:, 1] / proj_point1

    cos_theta_2 = point_local_2[:, 0] / proj_point2 
    sin_theta_2 = point_local_2[:, 1] / proj_point2
    
    r1 = proj_point1
    r2 = proj_point2
     
    a1 = getQuadraticCurveA(cos_theta_1, sin_theta_1, scales_, sign_s1, sign_s2, sign_s3)
    a2 = getQuadraticCurveA(cos_theta_2, sin_theta_2, scales_, sign_s1, sign_s2, sign_s3)
    
    s1 = QuadraticCurveGeodesicDistance_torch(r1, a1)
    s2 = QuadraticCurveGeodesicDistance_torch(r2, a2)

    s_sigma_1 = (sign_s1 * sign_s2 * scales_[..., 0] * scales_[..., 1]) / ((scales_[..., 1] * cos_theta_1)**2 + (scales_[..., 0] * sin_theta_1)**2)**0.5 * sigma
    s_sigma_2 = (sign_s1 * sign_s2 * scales_[..., 0] * scales_[..., 1]) / ((scales_[..., 1] * cos_theta_2)**2 + (scales_[..., 0] * sin_theta_2)**2)**0.5 * sigma

    s = torch.ones_like(s1) * 999
    two_valid_mask = (s1 <= s_sigma_1) & (s2 <= s_sigma_2)
    use_s1 = (root_1 <= root_2).view(-1)
    use_s1[~two_valid_mask] = False
    
    two_invalid_mask = ((s1 > s_sigma_1) & (s2 > s_sigma_2))
    one_valid_mask = ~(two_valid_mask | two_invalid_mask)
    
    only_s1_valid_mask = one_valid_mask & (s1 <= s_sigma_1)
    use_s1[only_s1_valid_mask] = True
    use_s2 = ~(use_s1 | two_invalid_mask)
    
    s1 /= s_sigma_1 / sigma
    s2 /= s_sigma_2 / sigma
    s[use_s1] = s1[use_s1]
    s[use_s2] = s2[use_s2]
    
    s_final = torch.ones([cam_ray_local.shape[0], cam_ray_local.shape[1]], device = device) * 999
    s_final[intersect_mask] = s
    s_final_2 = s_final * s_final
    
    image, omega = alpha_blending_with_gaussians(s_final_2, colors, opacities, H, W)
    return image, omega, setup_batch

def one_quadratic_splatting(idx, means3D, scales, quats, colors, opacities,count, output_folder, sigma = 1.0):
    intrins, viewmat, projmat, height, width, c2w = get_cameras(idx)
    intrins = intrins[:3, :3]
    image, omega, setup_batch = quadratic_splatting(means3D, scales, quats, colors, opacities, intrins, viewmat, projmat, c2w, sigma=sigma)
    x_min, x_max, y_min, y_max = setup_batch['x_min'], setup_batch['x_max'], setup_batch['y_min'], setup_batch['y_max']
    fig1, (ax1) = plt.subplots(1, 1)
    img1 = image.detach().cpu().numpy()
    from matplotlib.patches import Rectangle
    lb = torch.cat([x_min, y_min],dim = 1).detach().cpu().numpy()
    hw = torch.cat([x_max - x_min, y_max - y_min],dim = 1).detach().cpu().numpy()
    for k in range(means3D.shape[0]):
        ax1.add_patch(Rectangle(lb[k], hw[k, 0], hw[k, 1], facecolor='none', edgecolor='white'))
    ax1.imshow(img1)
    plt.savefig(f"{output_folder}/{count:04d}.png")
    return 
    
if __name__ == "__main__":
    torch.set_printoptions(precision=12, sci_mode=False)
    
    num_points=4
    means3D, scales_, quats = get_inputs(num_points=num_points)
    scales = scales_.clone()
    scales_[:,2] *= 0.2
    scales_[:,0] *= 0.5
    scales_[:,1] *= 0.4

    means3D = nn.Parameter(means3D)
    quats = nn.Parameter(quats)
    opacities = nn.Parameter(torch.ones_like(means3D[:, :1])) 
    colors = matplotlib.colormaps['Accent'](np.random.randint(0,num_points**2, num_points**2) / num_points**2)[..., :3]
    colors = nn.Parameter(torch.from_numpy(colors).cuda()).to(torch.float32)
    
    count = 0
    sigma = 1.5
    NUM = 30
    output_folder = "./demo"
    os.makedirs(output_folder, exist_ok=True)
    for i in tqdm(torch.linspace(-1, 1, NUM)):
        scales = scales_.clone()
        scales[:, 2] *= i
        one_quadratic_splatting(8, means3D, scales, quats, colors, opacities, count, output_folder, sigma=sigma)
        count += 1
    for i in tqdm(torch.linspace(1, -1, NUM)):
        scales = scales_.clone()
        scales[:, 2] *= i
        one_quadratic_splatting(8, means3D, scales, quats, colors, opacities, count, output_folder, sigma=sigma)
        count += 1
    for j in tqdm(torch.linspace(1, -1, NUM)):
        scales = scales_.clone()
        scales[:, 1] *= j
        scales[:, 2] *= -j
        one_quadratic_splatting(0, means3D, scales, quats, colors, opacities, count, output_folder, sigma=sigma)
        count += 1
    for j in tqdm(torch.linspace(-1, 1, NUM)):
        scales = scales_.clone()
        scales[:, 1] *= j
        scales[:, 2] *= -j
        one_quadratic_splatting(0, means3D, scales, quats, colors, opacities, count, output_folder, sigma=sigma)
        count += 1
    for j in tqdm(torch.linspace(1, -1, NUM)):
        scales = scales_.clone()
        scales[:, 0] *= j
        scales[:, 2] *= -j
        one_quadratic_splatting(0, means3D, scales, quats, colors, opacities, count, output_folder, sigma=sigma)
        count += 1
    for j in tqdm(torch.linspace(-1, 1, NUM)):
        scales = scales_.clone()
        scales[:, 0] *= j
        scales[:, 2] *= -j
        one_quadratic_splatting(0, means3D, scales, quats, colors, opacities, count, output_folder, sigma=sigma)
        count += 1
    
    images = [img for img in os.listdir(output_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(output_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("QGS_demo.mp4", fourcc, 30, (width, height))
    for image in images:
        img_path = os.path.join(output_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    video.release()
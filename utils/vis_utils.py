# copy from nerfstudio and 2DGS
import torch
from matplotlib import cm
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth,
    accumulation,
    near_plane = 2.0,
    far_plane = 6.0,
    cmap="turbo",
):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image

def save_points(path_save, pts, colors=None, normals=None, BRG2RGB=False):
    """save points to point cloud using open3d"""
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(path_save, cloud)
    

def colormap(img, cmap='jet'):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    if img.shape[1:] != (H, W):
        img = torch.nn.functional.interpolate(img[None], (W, H), mode='bilinear', align_corners=False)[0]
    return img


def get_grayscale_image(img, data_range, cmap):
    img = convert_data(img)
    img = np.nan_to_num(img)
    if data_range is None:
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = img.clip(data_range[0], data_range[1])
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
    assert cmap in [None, 'jet', 'magma']
    if cmap == None:
        img = (img * 255.).astype(np.uint8)
        img = np.repeat(img[...,None], 3, axis=2)
    elif cmap == 'jet':
        img = (img * 255.).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    elif cmap == 'magma':
        img = 1. - img
        base = cm.get_cmap('magma')
        num_bins = 256
        colormap = LinearSegmentedColormap.from_list(
            f"{base.name}{num_bins}",
            base(np.linspace(0, 1, num_bins)),
            num_bins
        )(np.linspace(0, 1, num_bins))[:,:3]
        a = np.floor(img * 255.)
        b = (a + 1).clip(max=255.)
        f = img * 255. - a
        a = a.astype(np.uint16).clip(0, 255)
        b = b.astype(np.uint16).clip(0, 255)
        img = colormap[a] + (colormap[b] - colormap[a]) * f[...,None]
        img = (img * 255.).astype(np.uint8)
    return img
def convert_data(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return [convert_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: convert_data(v) for k, v in data.items()}
    else:
        raise TypeError('Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting', type(data))

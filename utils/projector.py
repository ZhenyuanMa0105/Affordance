import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mitsuba as mi
import torch
from torch import nn

def standardize_bbox(pc):
    mins = np.amin(pc, axis=0)
    maxs = np.amax(pc, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs - mins)
    result = ((pc - center) / scale).astype(np.float32) # [-0.5, 0.5]
    return result

def get_camera_intrinsic(vfov=25, img_width=800, img_height=800):
    hfov = vfov * img_width / img_height

    fy = img_height * 0.5 / (math.tan(vfov * 0.5 * math.pi / 180))
    fx = img_width * 0.5 / (math.tan(hfov * 0.5 * math.pi / 180))

    Ox = img_width / 2.0
    Oy = img_height / 2.0

    K = torch.tensor([
        [fx, 0, Ox],
        [0, fy, Oy],
        [0, 0, 1]
    ], dtype=torch.float32)

    # print("Camera intrinsic matrix:")
    # print(K)
    return K

def look_at(eye, center, up):
    # Normalize input vectors
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    
    # Reconstruct the right vector
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Reconstruct the up vector
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Create the view matrix
    view = np.array([
        [right[0], right[1], right[2], -np.dot(right, eye)],
        [up[0],    up[1],    up[2],    -np.dot(up, eye)],
        [-forward[0], -forward[1], -forward[2], np.dot(forward, eye)],
        [0, 0, 0, 1]
    ])
    R, t = view[:3, :3], view[:3, 3]
    
    return R, t

def get_camera_extrinsics(origin=[3,3,3], target=[0, 0, 0], up=[0, 1, 0]):
    # Compute the camera transformation matrix
    lookat = mi.Transform4f.look_at(origin=origin, target=target, up=up)
    views = np.array(lookat.inverse().matrix)
    views[2, :] = -views[2, :]
    return views[:3, :]

def project_points_to_image(point_cloud, ext_trans, K):
    """
    Projects 3D points to 2D image pixel coordinates.
    Args:
    - point_cloud: (B, 3, 2048)
    - view_matrices: (4, 3, 4)
    - K: (3, 3) 
    Returns:
        Bx4x2xN array of 2D pixel coordinates.
    """
    B, _, N = point_cloud.shape
    device = point_cloud.device

    point_cloud_hom = torch.cat([point_cloud, torch.ones(B, 1, N, device=device)], dim=1)

    # Transform 3D points using camera transformation matrix
    point_cloud_transformed = torch.einsum('bkij,bjn->bkin', ext_trans, point_cloud_hom)

    # Project 3D points to 2D using camera intrinsic matrix
    points_2d_homogeneous = torch.einsum('ij,bkjn->bkin', K, point_cloud_transformed)

    # Convert homogeneous coordinates to pixel coordinates
    pixel_coords = points_2d_homogeneous[:, :, :2, :] / points_2d_homogeneous[:, :, 2:3, :]

    return pixel_coords


def plot_2d_projection(pixel_coords, image_shape, it=0):
    """
    Plots the 2D projected points onto an image and saves it.
    
    Args:
        pixel_coords (np.ndarray): Nx2 array of 2D pixel coordinates.
        image_shape (tuple): Shape of the output image (height, width).
        output_path (str): Path to save the resulting image.
    """
    output_path = f'projection_image_{it}.png'
    # Create a blank white image
    # fig, ax = plt.subplots()
    # ax.set_xlim(0, image_shape[1])  # width
    # ax.set_ylim(image_shape[0], 0)  # height (invert y axis for correct image orientation)
    # ax.set_aspect('equal')

    # # Plot the 2D points
    # ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c='r', s=2)

    # # Save the figure as an image
    # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    # plt.close()
    background_image_path = f'view{it}.jpg'  # 替换为你的背景图片路径

    # 加载背景图片
    background_img = mpimg.imread(background_image_path)

    # 定义图像宽度和高度（与背景图片一致）
    img_height, img_width, _ = background_img.shape

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # 显示背景图片
    ax.imshow(background_img, extent=[0, img_width, img_height, 0])  # 设置背景图片填充整个图像范围

    # 绘制点云，使用 points_image 的 (x, y) 坐标
    ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c='r', s=2)  # 红色点
    
    # 设置坐标轴的范围，与图像的大小一致
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)

    # 垂直翻转 y 轴
    ax.invert_yaxis()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

def init_camera_config(batch_size=32, fov=25, img_width=800, img_height=800):
    mi.set_variant('scalar_rgb')
    K = get_camera_intrinsic(fov, img_width, img_height)
    origin_list = [[3, 3, 3], [3, 3, -3], [-3, 3, -3], [-3, 3, 3]]
    ext_trans = torch.Tensor(np.array([get_camera_extrinsics(origin=i) for i in origin_list]))
    ext_trans = ext_trans.unsqueeze(0).expand(batch_size, *ext_trans.shape)
    return K, ext_trans

def get_2d_masks(pixel_coords, img_seg):
    """
    Get 2D masks from 2D pixel coordinates.
    Args:
    - pixel_coords: (B, 4, 2, N)
    - img_seg: (B, 4, H, W)
    Returns:
        Bx4xN array of 2D masks.
    """
    B, V, _, N = pixel_coords.shape  # B-batch size, V-number of views, N-number of points
    H, W = img_seg.shape[-2], img_seg.shape[-1]

    batch_indices = torch.arange(B).view(B, 1, 1).expand(-1, V, N)  # (B, V, N)
    view_indices = torch.arange(V).view(1, V, 1).expand(B, -1, N)   # (B, V, N)
    
    x_coords = torch.clamp(torch.round(pixel_coords[:, :, 0]).long(), 0, W - 1)
    y_coords = torch.clamp(torch.round(pixel_coords[:, :, 1]).long(), 0, H - 1)

    masks_at_coords = img_seg[batch_indices, view_indices, y_coords, x_coords].bool()

    return masks_at_coords


class Projector(nn.Module):
    def __init__(self, batch_size=32, device='cuda', fov=25, img_width=800, img_height=800):
        super(Projector, self).__init__()
        self.device = device
        self.K, self.ext_trans = init_camera_config(batch_size, fov, img_width, img_height)
        self.K = self.K.to(self.device)
        self.ext_trans = self.ext_trans.to(self.device)
    def forward(self, point_cloud, img_seg):
        with torch.no_grad():
            pixel_coords = project_points_to_image(point_cloud, self.ext_trans, self.K)
            masks_at_coords  = get_2d_masks(pixel_coords, img_seg)
        return masks_at_coords

if __name__=='__main__':
    
    data_root='/storage_fast/ycli/3d_affordance/LASO_dataset'
    with open(os.path.join(data_root, f'objects_train.pkl'), 'rb') as f:
        objects_file = pickle.load(f)
    point_cloud = objects_file['1b67b4bfed6688ba5b22feddf58c05e1']
    point_cloud = standardize_bbox(point_cloud)
    batch_size = 32
    K, ext_trans = init_camera_config(batch_size)
    print(ext_trans.shape)
    point_cloud = torch.tensor(point_cloud).permute(1, 0)
    point_cloud = point_cloud.repeat(32, 1, 1)
    print(point_cloud.shape)
    pixel_coords = project_points_to_image(point_cloud, ext_trans, K)
    pixel_coords = pixel_coords[15].permute(0, 2, 1).cpu().numpy()
    print(pixel_coords.shape)

    image_shape = (800, 800)  # Define your image resolution (height, width)
    for i in range(pixel_coords.shape[0]):
        plot_2d_projection(pixel_coords[i], image_shape, i)

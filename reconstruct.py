import numpy as np
import open3d as o3d
import cv2
import torch
import yaml
import copy
import json
import glob
import os
import argparse
from scipy.spatial.transform import Rotation as Rot

def parse_args():
    parser = argparse.ArgumentParser(description='Solve hand & object poses.')

    # parser.add_argument('--frame', help='Frame number to be optimized', default=0, type=int)

    args = parser.parse_args()

    return args

def depth_image_to_point_cloud(depth, rgb, K, depth_scale = 1000.0):
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    u = torch.tensor(u.astype(float)).to(device)
    v = torch.tensor(v.astype(float)).to(device)
    Z = depth / depth_scale
    X = (u - K[0, 2]) * Z / K[0, 0]  # (u-cx) * Z / fx
    Y = (v - K[1, 2]) * Z / K[1, 1]  # (v-cy) * Z / fy

    img_stack = torch.dstack((X, Y, Z))

    X = torch.ravel(X)
    Y = torch.ravel(Y)
    Z = torch.ravel(Z)
    
    # remove points which is too far
    valid = Z < 1.5
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    dummy = torch.ones_like(Z).to(device)
    R = torch.ravel(rgb[:, :, 2])[valid] / 255.
    G = torch.ravel(rgb[:, :, 1])[valid] / 255.
    B = torch.ravel(rgb[:, :, 0])[valid] / 255.
     
    position = torch.vstack((X, Y, Z, dummy))
    colors = torch.vstack((R, G, B)).transpose(0, 1)

    return position, colors, img_stack

def get_scene_pcd(dpth_img, rgb_img, intrinsic):
    # scene point cloud
    pos, rgb, img_stack = depth_image_to_point_cloud(dpth_img, rgb_img, intrinsic)
    pos, rgb = pos.cpu().detach().numpy().T[:,:3], rgb.cpu().detach().numpy()
    # valid = (np.max(rgb, axis=1) + np.min(rgb, axis=1)) / 2 > 0.4
    # pos, rgb = pos[valid], rgb[valid]
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(pos)
    scene_pcd.colors = o3d.utility.Vector3dVector(rgb)

    return scene_pcd

def compute_camera_intrinsics_matrix(image_width, image_heigth, horizontal_fov):
    vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
    horizontal_fov *= np.pi / 180

    f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

    K = np.array([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    return K

def run(device):
    rgb_pth_list = sorted(glob.glob('data/2023-05-12-15-23-43/env_00000/rgb/side/*.png'))
    depth_pth_list = sorted(glob.glob('data/2023-05-12-15-23-43/env_00000/depth/side/*.npy'))
    semantic_pth_list = sorted(glob.glob('data/2023-05-12-15-23-43/env_00000/semantic/side/*.npy'))

    K = compute_camera_intrinsics_matrix(640, 480, 90)
    K = torch.from_numpy(K).to(device)

    colors = np.array([[0, 0, 0],
                       [100, 100, 100],
                       [255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255],
                       [255, 255, 0]])

    for rgb_pth, depth_pth, semantic_pth in zip(rgb_pth_list, depth_pth_list, semantic_pth_list):
        rgb = cv2.imread(rgb_pth, 3)
        depth = np.load(depth_pth) * (-1)
        semantic = np.load(semantic_pth)

        print(np.unique(depth))

        semantic_rgb = np.zeros((rgb.shape), dtype=np.uint8)
        unique_sem_id = np.unique(semantic)

        for i in range(len(unique_sem_id)):
            mask = (semantic == unique_sem_id[i])
            semantic_rgb[mask] = colors[i]

        rgb = torch.from_numpy(rgb).to(device)
        depth = torch.from_numpy(depth).to(device)
        semantic_rgb = torch.from_numpy(semantic_rgb).to(device)
        
        scene_pcd = get_scene_pcd(depth, semantic_rgb, K)
        
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001)
        o3d.visualization.draw_geometries([scene_pcd])

if __name__ == '__main__':
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device', device)

    run(device)
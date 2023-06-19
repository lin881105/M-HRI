import numpy as np
import open3d as o3d
import cv2
import torch
import yaml
import copy
import json
import os
import argparse
from scipy.spatial.transform import Rotation as Rot

def parse_args():
    parser = argparse.ArgumentParser(description='Solve hand & object poses.')

    parser.add_argument('--frame', help='Frame number to be optimized', default=0, type=int)

    args = parser.parse_args()

    return args

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device', device)

args = parse_args()

def get_transform_from_3kpt(scene_points, obj_points):
    scene_mean = np.mean(scene_points, axis=1).reshape((3, 1))
    obj_mean = np.mean(obj_points, axis=1).reshape((3, 1))

    scene_points_ = scene_points - scene_mean
    obj_points_ = obj_points - obj_mean

    W = scene_points_ @ obj_points_.T

    u, s, vh = np.linalg.svd(W, full_matrices=True)
    R = u @ vh

    if np.linalg.det(R) < 0:
        m = scene_points.shape[1]
        vh[m-1,:] *= -1
        R = np.dot(vh.T, u.T)

    t = scene_mean - R @ obj_mean

    transform = np.identity(4)
    transform[:3,:3] = R
    transform[:3, 3] = t.reshape((1, 3))

    return transform

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

def keypoints_to_point_cloud(img_stack, keypoints):
    X, Y, Z = img_stack[:, :, 0], img_stack[:, :, 1], img_stack[:, :, 2]
    x_list, y_list, z_list = [], [], []

    for i in keypoints:
        x_list.append(X[i[0], i[1]])
        y_list.append(Y[i[0], i[1]])
        z_list.append(Z[i[0], i[1]])

    # colors = np.array([ [255, 0, 0],
    #                     [255, 0, 0],
    #                     [255, 0, 0]])  / 255.

    X = torch.tensor(x_list, device=device)
    Y = torch.tensor(y_list, device=device)
    Z = torch.tensor(z_list, device=device)
    kpts = torch.vstack((X, Y, Z))

    return kpts.cpu().detach().numpy().T

def get_scene_pcd(dpth_img, rgb_img, intrinsic, vis=False):
    # scene point cloud
    pos, rgb, img_stack = depth_image_to_point_cloud(torch.tensor(dpth_img, device=device), torch.tensor(rgb_img, device=device), torch.tensor(intrinsic, device=device))
    pos, rgb = pos.cpu().detach().numpy().T[:,:3], rgb.cpu().detach().numpy()
    valid = (np.max(rgb,axis=1) + np.min(rgb,axis=1)) / 2 > 0.4
    # valid = np.all((valid,(np.max(rgb,axis=1) + np.min(rgb,axis=1)) / 2 > 0.1),axis=0)
    pos, rgb = pos[valid], rgb[valid]
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(pos)
    scene_pcd.colors = o3d.utility.Vector3dVector(rgb)

    # visualize
    # if vis:
    #     edges = []
    
    #     for i in range(len(position)-1):
    #         edges.append([i, i+1])

    #     icp_line_set = o3d.geometry.LineSet(points = o3d.utility.Vector3dVector(position.cpu().detach().numpy().T),
    #                                         lines = o3d.utility.Vector2iVector(edges))
    #     icp_line_set.colors = o3d.utility.Vector3dVector(colors)

    #     o3d.visualization.draw_geometries([scene_pcd, icp_line_set])

    # return position.cpu().detach().numpy().T, scene_pcd
    return scene_pcd

def load_image(root_pth, camera_serials, extrinsic_str):
    rgb_img_dict, dpth_img_dict = {}, {}

    for camera_id in camera_serials:
        rgb_pth = root_pth + camera_id + f'/color_{str(args.frame).zfill(6)}.jpg'
        dpt_pth = root_pth + camera_id + f'/aligned_depth_to_color_{str(args.frame).zfill(6)}.npy'

        rgb_img_dict[camera_id] = cv2.imread(rgb_pth, 3)
        dpth_img_dict[camera_id] = np.load(dpt_pth)

    extrinsic_dict, intrinsic_dict = {}, {}

    # extrinsic
    with open(root_pth + '../calibration/extrinsics_' + extrinsic_str + '/extrinsics.yml', 'r') as f:
        extrinsic_file = yaml.load(f, Loader=yaml.FullLoader)['extrinsics']
        
        for camera_id in camera_serials:
            extrinsic_dict[camera_id] = np.array(extrinsic_file[camera_id]).reshape(3, 4)

    # intrinsic
    for camera_id in camera_serials:
        with open(root_pth + '../calibration/intrinsics/' + camera_id + '_640x480.yml', 'r') as f:
            intrinsic_file = yaml.load(f, Loader=yaml.FullLoader)['color']
            intrinsic_dict[camera_id] = np.array([  [intrinsic_file['fx'], 0, intrinsic_file['ppx']],
                                                    [0, intrinsic_file['fy'], intrinsic_file['ppy']],
                                                    [0, 0, 1]])

    return rgb_img_dict, dpth_img_dict, extrinsic_dict, intrinsic_dict

def get_mesh_kpt():
    obj_points = {}
    obj_points['A'] = np.array([ [-1.4, 3, 0.9],
                                [1.4, 3, 0.9],
                                [1.4, -3, 0.9],
                                [-1.4, -3, 0.9],
                                [-1.4, 3, -0.9],
                                [1.4, 3, -0.9],
                                [1.4, -3, -0.9],
                                [-1.4, -3, -0.9]]) / 100
    obj_points['B'] = np.array([[3, -1.45,-1.45],
                                [3, -1.45, 1.45],
                                [-3, -1.45, 1.45],
                                [-3, -1.45, -1.45],
                                [3, 1.45, -1.45],
                                [3, 1.45, 1.45],
                                [-3, 1.45, 1.45],
                                [-3, 1.45, -1.45]]) / 100
    obj_points['D'] = np.array([[-1.45, -1.45, -3],
                                [1.45, -1.45, -3],
                                [1.45, -1.45, 3],
                                [-1.45, -1.45, 3],
                                [-1.45, 1.45, -3],
                                [1.45, 1.45, -3],
                                [1.45, 1.45, 3],
                                [-1.45, 1.45, 3],
                                [0, 0, -3]]) / 100

    obj_points['E'] = np.array([[1.4, -2.8, 1.4],
                                [1.4, -2.8, -1.4],
                                [1.4, 2.8, -1.4],
                                [1.4, 2.8, 1.4],
                                [-1.4, -2.8, 1.4],
                                [-1.4, -2.8, -1.4]]) / 100

    obj_points['box'] = np.array([[7.15, -4.15,2.39],
                                [7.15, 4.15, 2.39],
                                [-7.15, 4.15, 2.39],
                                [-7.15, -4.15, 2.39],
                                [7.15, -4.15, -2.39],
                                [7.15, 4.15, -2.39],
                                [-7.15, 4.15, -2.39],
                                [-7.15, -4.15, -2.39]]) / 100
    return obj_points

def get_2d_kpt(obj_type, camera_id):
    labels = []
    for t in obj_type:
        tmp_list = []
        for i in range(1, 9):
            tmp_list.append(t + ': keypoint' + str(i))
        labels.append(tmp_list)

    label2pixel = {}
    with open('kpt/' + camera_id + '.txt', 'r') as f:
    # with open('./kpts.txt', 'r') as f:
        json_file =  json.load(f)['keypoints']
    for kpt in json_file:
        label2pixel[kpt["label"]] = [kpt['y'], kpt['x']]

    img_kpts_dict, vis_kpts_dict= {}, {}
    for obj_label, t in zip(labels, obj_type):
        tmp_img_kpts, tmp_vis_kpts = [], []
        for cnt, l in enumerate(obj_label):
            if l in label2pixel.keys():
                tmp_img_kpts.append(label2pixel[l])
                tmp_vis_kpts.append(cnt)

        img_kpts_dict[t], vis_kpts_dict[t] = tmp_img_kpts, tmp_vis_kpts

    return img_kpts_dict, vis_kpts_dict

def loadData():
    root_pth = '/home/hcis-s12/Lego_Assembly/Dataset/DexYCB/dex-ycb/data/20221001_171534/'

    with open(root_pth + 'meta.yml', 'r') as f:
        meta_file = yaml.load(f, Loader=yaml.FullLoader)
        camera_serials = meta_file['serials']
        extrinsic_str = meta_file['extrinsics']

    rgb_img_dict, dpth_img_dict, extrinsic_dict, intrinsic_dict = load_image(root_pth, camera_serials, extrinsic_str)

    # object mesh
    # key frmae = [0, 38, 66, 98, 132]
    # obj_type = ['D','B']
    obj_type = ['A', 'B', 'D', 'E']
    obj_mesh_dict = {}

    color_list = [[0, 0, 1],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]]

    for t, c in zip(obj_type, color_list):
        obj_pth = f'/home/hcis-s12/.local/lib/python3.8/site-packages/pybullet_data/brick/type_{t}.obj'
        obj_mesh = o3d.io.read_triangle_mesh(obj_pth)
        obj_mesh.paint_uniform_color(c)
        # obj_mesh.compute_vertex_normals()
        obj_mesh_dict[t] = obj_mesh

    mesh_kpts = get_mesh_kpt()

    # scene point cloud
    scene_pcd = o3d.geometry.PointCloud()
    for camera_id in camera_serials:
        tmp_pcd = get_scene_pcd(dpth_img_dict[camera_id], rgb_img_dict[camera_id], intrinsic_dict[camera_id])
        tmp_extrinsic = np.vstack([extrinsic_dict[camera_id], np.array([0, 0, 0, 1])])
        tmp_pcd.transform(tmp_extrinsic)
        scene_pcd += tmp_pcd

    # 3D annotation
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(scene_pcd)
    vis.run()
    vis.destroy_window()

    picked_points_list = vis.get_picked_points()

    if len(picked_points_list) == 0:
        print('zero keypoints')
        exit(0)

    # get 3D kpts
    idx = 0
    scene_kpts_dict = {}
    picked_points_list = vis.get_picked_points()
    scene_pcd_position_array=np.asarray(scene_pcd.points)
    for t in obj_type:
        scene_kpts_dict[t] = np.zeros((3, 3))
        tmp = 3
        for i in range(3):
            scene_kpts_dict[t][:, i] = scene_pcd_position_array[picked_points_list[i+idx]].T
        # print(np.sqrt(np.sum(np.square(scene_kpts_dict[t][:, 0] - scene_kpts_dict[t][:, 1]))))
        # print(np.sqrt(np.sum(np.square(scene_kpts_dict[t][:, 1] - scene_kpts_dict[t][:, 2]))))
        idx += tmp

    obj_kpts_dict = {}
    obj_kpts_dict[obj_type[0]] = mesh_kpts[obj_type[0]][[0, 1, 2], :].T
    obj_kpts_dict[obj_type[1]] = mesh_kpts[obj_type[1]][[0, 1, 2], :].T
    obj_kpts_dict[obj_type[2]] = mesh_kpts[obj_type[2]][[0, 1, 2], :].T
    obj_kpts_dict[obj_type[3]] = mesh_kpts[obj_type[3]][[1, 2, 3], :].T

    return scene_kpts_dict, obj_kpts_dict, scene_pcd, intrinsic_dict, extrinsic_dict, dpth_img_dict[camera_serials[0]].shape, obj_mesh_dict, rgb_img_dict, obj_type, camera_serials

def visualize(scene_pcd, transform_obj_mesh_list, obj_mesh_dict, obj_points_dict):
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries(transform_obj_mesh_list)
    o3d.visualization.draw_geometries([scene_pcd])
    o3d.visualization.draw_geometries(transform_obj_mesh_list + [scene_pcd])

def render_cv2(pcd : o3d.geometry.PointCloud, extr : np.ndarray, intr : np.ndarray, img_size : tuple, rgb_img,camera_id,  distorsion = None, rgb = [0, 0, 255],vis = True):
    extr = np.linalg.inv(np.vstack([extr, np.array([0, 0, 0, 1])]))
    
    r_vec = extr[:3, :3]
    t_vec = extr[:3, 3]

    obj_pts = np.asarray(pcd.points)

    imgpts, jac = cv2.projectPoints(obj_pts, r_vec, t_vec, intr, distorsion)
    imgpts = np.rint(np.squeeze(imgpts)).astype(np.int32)
    # print(imgpts)
    H, W = img_size
    cond = np.where((imgpts[:, 1] < 0) | (imgpts[:, 1] >= H) | (imgpts[:, 0] < 0) | (imgpts[:, 0] >= W))
    imgpts = np.delete(imgpts, cond, axis=0)

    render_img = np.zeros((H, W), dtype=np.bool)
    render_img[imgpts[:, 1], imgpts[:, 0]] = True

    # with open('sem_label'+camera_id+'.npy', 'wb') as f:
    #     np.save(f, render_img)
    render_img = np.zeros((H, W, 3), dtype=np.uint8)
    render_img[imgpts[:, 1], imgpts[:, 0]] = rgb

    cv2.addWeighted(render_img, 0.3, rgb_img, 0.7, 0, render_img)
    if vis:
        cv2.imshow('My Image', render_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return render_img

def meshToPcd(mesh):
    pcd = mesh.sample_points_uniformly(number_of_points=100000)
    return pcd

def transform_mesh(scene_pcd, scene_points_dict, obj_points_dict, obj_mesh_dict, camera_serials):
    transform_obj_mesh_list = []
    transform_array = np.zeros((len(obj_mesh_dict), 4, 4))
    for cnt, t in enumerate(obj_type):
        transform = get_transform_from_3kpt(scene_points_dict[t], obj_points_dict[t][:,:3])

        tf_mesh_points = transform @ np.vstack([obj_points_dict[t], np.ones(3)])
        # tf_mesh_points = np.vstack([scene_points_dict[t], np.ones(3)])
        # print(tf_mesh_points)
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(tf_mesh_points.T[:, :3])
        rgb = np.eye(3)
        rgb = np.vstack([rgb, np.zeros(3)])
        rgb = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,1]])
        tmp_pcd.colors = o3d.utility.Vector3dVector(rgb)
        scene_pcd += tmp_pcd

        transform_array[cnt] = transform
        tmp_mesh = copy.deepcopy(obj_mesh_dict[t])
        tmp_mesh.transform(transform)
        tmp_mesh.compute_vertex_normals()
        transform_obj_mesh_list.append(tmp_mesh)

    # with open('dexycb_initpose.npy', 'wb') as f:
    #     np.save(f, transform_array)

    obj_pcd = o3d.geometry.PointCloud()
    for mesh in transform_obj_mesh_list:
        obj_pcd += meshToPcd(mesh)
    for c in camera_serials:
        label = render_cv2(obj_pcd, extrinsic_dict[c], intrinsic_dict[c], img_size, rgb_img_dict[c],c)

    # transform object pose to apriltag
    tmp_tf_obj = []

    pth="/home/hcis-s12/Lego_Assembly/Dataset/DexYCB/dex-ycb/data/20221001_171534/"

    with open(pth + "meta.yml", 'r') as f:
        meta = yaml.safe_load(f)

    extr_file = "/home/hcis-s12/Lego_Assembly/Dataset/DexYCB/dex-ycb/data/calibration/extrinsics_" + meta["extrinsics"]+"/extrinsics.yml"

    with open(extr_file, "r") as f:
        extr = yaml.load(f, Loader=yaml.FullLoader)

    tag_T = np.array(extr["extrinsics"]["apriltag"], dtype=np.float32).reshape(3, 4)
    tag_T = np.vstack([tag_T, np.array([0, 0, 0, 1])])

    for i in range(len(obj_mesh_dict)):
        tmp = np.matmul(np.linalg.inv(tag_T), transform_array[i])
        rot_matrix, t = tmp[:3, :3], tmp[:3, 3]
        q = Rot.from_matrix(rot_matrix).as_quat()
        tmp_pose = np.zeros(7)
        tmp_pose[:3] = t.T
        tmp_pose[3:] = q
        tmp_tf_obj.append(tmp_pose)

        print(tmp_pose)
    
    # output pose file
    with open(f'./obj_pose.npy', 'wb') as f:
        np.save(f, np.array(tmp_tf_obj))

    return transform_obj_mesh_list

scene_points_dict, obj_points_dict, scene_pcd, intrinsic_dict, extrinsic_dict, img_size, obj_mesh_dict, rgb_img_dict, obj_type, camera_serials = loadData()
transform_obj_mesh_list = transform_mesh(scene_pcd, scene_points_dict, obj_points_dict, obj_mesh_dict, camera_serials)
visualize(scene_pcd, transform_obj_mesh_list, obj_mesh_dict, obj_points_dict)
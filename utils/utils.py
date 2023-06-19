
import math
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
import torch
import os


def gymapi_transform2mat(transform):
    mat = np.eye(4)

    mat[:3,3] = np.array((transform.p.x,transform.p.y,transform.p.z))

    quat = R.from_quat((transform.r.x,transform.r.y,transform.r.z,transform.r.w))

    mat[:3,:3] = quat.as_matrix()

    return mat

def mat2gymapi_transform(mat):
    transform = gymapi.Transform()

    p = mat[:3,3]

    transform.p = gymapi.Vec3(p[0],p[1],p[2])
    
    quat = R.from_matrix(mat[:3,:3]).as_quat()
    transform.r.x = quat[0]
    transform.r.y = quat[1]
    transform.r.z = quat[2]
    transform.r.w = quat[3]


    # euler = quat.as_euler("xyz",degrees=False)

    # transform.r = gymapi.Quat.from_euler_zyx(euler[0],euler[1],euler[2])
    # print(euler)

    return transform

def multiply_gymapi_transform(trans_A,trans_B):
    mat_A = gymapi_transform2mat(trans_A)
    mat_B = gymapi_transform2mat(trans_B)

    return mat2gymapi_transform(mat_A @ mat_B)



def tensor_6d_pose2mat(pos,quat):
    pos = pos.cpu().numpy()
    quat = quat.cpu().numpy()

    mat = np.eye(4)

    rot = R.from_quat(quat).as_matrix()
    mat[:3,3] = pos
    mat[:3,:3] = rot

    print(mat)

    return mat
    
def mat2posrot(mat):
    # print(mat.shape)
    
    pos = mat[:,:3,3]
    rot_mat = mat[:,:3,:3]
    # print(rot_mat.shape)
    
    rot = R.from_matrix(rot_mat)
    
    quat = rot.as_quat()
    
    # print(pos)
    # print(quat)
    
    pose = np.hstack([pos.reshape(-1, 3), quat.reshape(-1, 4)])
    
    return pose

def get_dense_waypoints(start_config : list or tuple or np.ndarray, end_config : list or tuple or np.ndarray, resolution : float=0.005):
    assert len(start_config) == 7 and len(end_config) == 7
    d12 = np.asarray(end_config[:3]) - np.asarray(start_config[:3])
    steps = int(np.ceil(np.linalg.norm(np.divide(d12, resolution), ord=2)))
    obj_init_quat = quaternion.as_quat_array(start_config[6,3,4,5])
    obj_tgt_quat = quaternion.as_quat_array(end_config[6,3,4,5])
    ret = []
    # plan trajectory in the same way in collision detection module
    for step in range(steps):
        ratio = (step + 1) / steps
        pos = ratio * d12 + np.asarray(start_config[:3])
        quat = quaternion.slerp_evaluate(obj_init_quat, obj_tgt_quat, ratio)
        quat = quaternion.as_float_array(quat)[4,5,6,3]
        position7d = tuple(pos) + tuple(quat)
        ret.append(position7d)


def slerp(q1, q2, t):
    """Spherical linear interpolation for batches of quaternions."""
    q1_norm = torch.norm(q1, dim=1, keepdim=True)
    q2_norm = torch.norm(q2, dim=1, keepdim=True)

    q1 = q1 / q1_norm  # Normalize quaternion
    q2 = q2 / q2_norm

    dot = torch.sum(q1 * q2, dim=1, keepdim=True)

    q2 *= torch.sign(dot)  # Ensure shortest path
    dot *= torch.sign(dot)

    DOT_THRESHOLD = 0.9995
    lerp_flag = dot > DOT_THRESHOLD

    inv_sin_theta_0 = 1 / torch.sqrt(1 - dot * dot)
    theta_0 = torch.acos(dot)
    theta = theta_0 * t

    s0 = torch.cos(theta) - dot * torch.sin(theta) * inv_sin_theta_0
    s1 = torch.sin(theta) * inv_sin_theta_0
    q_slerp = (s0 * q1 + s1 * q2)

    # Perform lerp for close quaternions
    q_lerp = (1 - t) * q1 + t * q2
    q_lerp = q_lerp / torch.norm(q_lerp, dim=1, keepdim=True)

    q_slerp = torch.where(lerp_flag, q_lerp, q_slerp)

    return q_slerp
def create_gif(img_root_pth):
    import cv2
    import imageio
    import glob

    rgb_pth_dict = {}
    hand_rgb_pth_dict = {}


    rgb_pth_dict = sorted(glob.glob(os.path.join(img_root_pth, 'rgb','side', '*.png')))
    hand_rgb_pth_dict = sorted(glob.glob(os.path.join(img_root_pth, 'hand_rgb', '*.png')))



    hand_img_array = []
    robot_img_array = []

    # for frame in range(len(hand_rgb_pth_dict)):
    #     if frame%skip == 0:
    #         delete.append(frame)
    # del hand_rgb_pth_dict[delete]


    for rgb_pth, hand_rgb_pth in zip(rgb_pth_dict, hand_rgb_pth_dict):
        rgb = cv2.imread(rgb_pth)
        rgb = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        
        hand_rgb = cv2.imread(hand_rgb_pth)
        hand_rgb = cv2.cvtColor(hand_rgb,cv2.COLOR_RGB2BGR)

        
        hand_img_array.append(hand_rgb)
        robot_img_array.append(rgb)


    # img_array_dict[camera_id] = img_array
    imageio.mimsave(os.path.join(img_root_pth,'hand.gif'),hand_img_array,fps=5)
    imageio.mimsave(os.path.join(img_root_pth,'robot.gif'),robot_img_array,fps=5)

if __name__ == "__main__":
    goal_A_pose_1 = gymapi.Transform()
    goal_A_pose_1.p = gymapi.Vec3(0,0.04,0.02)
    goal_A_pose_1.r = gymapi.Quat.from_euler_zyx(math.pi*0.5, 0, 0)

    mat = gymapi_transform2mat(goal_A_pose_1)

    trans = mat2gymapi_transform(mat)

    print(trans.r.w)
    print(goal_A_pose_1.r.w)





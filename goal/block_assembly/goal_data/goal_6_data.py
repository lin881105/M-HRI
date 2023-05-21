from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymutil

import numpy as np
import torch
from utils import utils
import pytorch3d.transforms
import math




class Goal_6():
    def __init__(self):

        self.device = 'cuda:0'
 

        self.goal = [3,0]
        self.goal_pose = []

        goal_pose_1 = gymapi.Transform()
        goal_pos_1 = torch.Tensor((0, 0.0, 0.03)).to(device=self.device)

        goal_rot_1 = torch.Tensor((0,0,  0)).to(device=self.device)
        goal_rot_1 = pytorch3d.transforms.euler_angles_to_matrix(goal_rot_1,"XYZ").to(device=self.device)
        # goal_rot_1 = torch.round(torch.tensor(goal_rot_1), decimals=3)

        goal_pose_1 = torch.eye(4)
        goal_pose_1[:3,:3] = goal_rot_1
        goal_pose_1[:3, 3] = goal_pos_1

        goal_pose_1 = utils.mat2gymapi_transform(goal_pose_1.cpu().numpy())

        goal_pose_2 = gymapi.Transform()
        goal_pos_2 = torch.Tensor((0, 0, 0.0645)).to(device=self.device)

        goal_rot_2 = torch.Tensor((0,0,  0)).to(device=self.device)
        goal_rot_2 = pytorch3d.transforms.euler_angles_to_matrix(goal_rot_2,"XYZ").to(device=self.device)
        # goal_rot_2 = torch.round(torch.tensor(goal_rot_2), decimals=3)
        goal_pose_2 = torch.eye(4)
        goal_pose_2[:3,:3] = goal_rot_2
        goal_pose_2[:3, 3] = goal_pos_2

        

        goal_pose_2 = utils.mat2gymapi_transform(goal_pose_2.cpu().numpy())

        self.block_height = [0.03,0.0045]



        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_1))
        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_2))

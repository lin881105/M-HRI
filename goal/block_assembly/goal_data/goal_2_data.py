from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymutil

import numpy as np
import torch
from utils import utils
import pytorch3d.transforms
import math
from goal.block_assembly.goal_data.goal import Goal



class Goal_2(Goal):
    def __init__(self):

        self.device = 'cuda:0'
 

        self.goal = [3,3]
        self.goal_pose = []

        goal_pose_1 = gymapi.Transform()
        goal_pos_1 = torch.Tensor((0, 0.0, 0.0145)).to(device=self.device)

        goal_rot_1 = torch.Tensor((-0.5*np.pi,0,  0)).to(device=self.device)
        goal_rot_1 = pytorch3d.transforms.euler_angles_to_matrix(goal_rot_1,"XYZ").to(device=self.device)

        goal_pose_1 = torch.eye(4)
        goal_pose_1[:3,:3] = goal_rot_1
        goal_pose_1[:3, 3] = goal_pos_1

        goal_pose_1 = utils.mat2gymapi_transform(goal_pose_1.cpu().numpy())

        goal_pose_2 = gymapi.Transform()
        goal_pos_2 = torch.Tensor((0, 0, 0.0435)).to(device=self.device)

        goal_rot_2 = torch.Tensor((-0.5*np.pi,0,  0)).to(device=self.device)
        goal_rot_2 = pytorch3d.transforms.euler_angles_to_matrix(goal_rot_2,"XYZ").to(device=self.device)
        goal_pose_2 = torch.eye(4)
        goal_pose_2[:3,:3] = goal_rot_2
        goal_pose_2[:3, 3] = goal_pos_2

        

        goal_pose_2 = utils.mat2gymapi_transform(goal_pose_2.cpu().numpy())

        self.block_height = [0.0145,0.0145]



        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_1))
        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_2))

        self.hand_rel_mat_list = []
        self.hand_pose_list = []
        hand_goal_pose = np.array([-2.21341878e-01, -9.16626230e-02,  4.62329611e-02,  7.10443914e-01,
        1.03342474e+00,  6.83313906e-01,  9.92064774e-02, -3.72702628e-01,
        4.46040370e-02, -5.28285541e-02, -5.52620320e-03,  5.32112420e-01,
       -4.80155941e-05,  1.02795474e-01,  5.55250525e-01, -8.04432929e-02,
       -2.32878298e-01,  3.73034596e-01, -1.14381686e-01, -5.27032204e-02,
        8.16306233e-01, -7.38171935e-02,  1.53227076e-02,  5.00212252e-01,
       -2.46633589e-01,  4.64982808e-01,  3.36053818e-01, -5.66753924e-01,
       -1.40147611e-01,  5.69896758e-01, -2.88565457e-01,  1.14357322e-01,
        3.84515792e-01, -5.99057525e-02,  6.84615299e-02,  3.46477389e-01,
       -3.27767521e-01, -9.36669484e-02,  8.54686618e-01, -2.59945124e-01,
        8.57274905e-02,  5.39167941e-01,  8.35047126e-01, -3.57780121e-02,
        1.43417954e-01, -4.95447308e-01, -5.04968353e-02,  5.73221631e-02,
        6.18857801e-01, -1.02387838e-01,  3.12080264e-01], dtype=np.float32)
        obj_init = np.array([-0.19434147, -0.11968146, 0.01548913, -0.50471319, 0.49524648, -0.4952225 , 0.50472785])


        self.hand_rel_mat_list.append(self.get_hand_rel_mat(hand_goal_pose,obj_init))
        self.hand_rel_mat_list.append(self.get_hand_rel_mat(hand_goal_pose,obj_init))
        

        self.hand_pose_list.append(hand_goal_pose.reshape(1,1,51))
        self.hand_pose_list.append(hand_goal_pose.reshape(1,1,51))


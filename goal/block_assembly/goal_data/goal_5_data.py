from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymutil

import numpy as np
import torch
from utils import utils
import pytorch3d.transforms
import math
import pickle
from goal.block_assembly.goal_data.goal import Goal


class Goal_5(Goal):
    def __init__(self):

        self.device = 'cuda:0'

        with open('gen_hand_pose/test_ft.pickle', 'rb') as f:
            self._dict = pickle.load(f)

        self.goal = [1,4]
        self.goal_pose = []

        goal_pose_1 = gymapi.Transform()
        goal_pose_1.p = gymapi.Vec3(0, 0, 0.0145)

        goal_pose_1.r = gymapi.Quat.from_euler_zyx( 0,0, np.pi)

        goal_pose_2 = gymapi.Transform()
        goal_pose_2.p = gymapi.Vec3(-0.005, 0, 0.0318)
        # goal_pose_2.r = gymapi.Quat.from_euler_zyx(0, -np.pi/2, -np.pi)
        # [ 0.6032,  0.3693,  0.3741, -0.5999]
        goal_pose_2.r.x = 0.6032
        goal_pose_2.r.y = 0.3693
        goal_pose_2.r.z = 0.3741
        goal_pose_2.r.w = -0.5999

        self.block_height = [0.0045,0.014]


        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_1))
        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_2))

        self.hand_rel_mat_list = []
        self.hand_pose_list = []
        hand_goal_pose = self._dict[1]["hand_ref_pose"]
        obj_init = self._dict[1]["obj_init"]
        
        self.hand_rel_mat_list.append(self.get_hand_rel_mat(hand_goal_pose,obj_init))

        self.hand_pose_list.append(hand_goal_pose)
        
        hand_goal_pose = self._dict[3]["hand_ref_pose"]
        obj_init = self._dict[3]["obj_init"]
        
        
        self.hand_rel_mat_list.append(self.get_hand_rel_mat(hand_goal_pose,obj_init))
        

        self.hand_pose_list.append(hand_goal_pose)
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time
from utils import utils
import scipy.io as sio
from goal.block_assembly.goal_data.goal import Goal
import copy
import pickle


class Goal_1(Goal):
    def __init__(self):

        self.device = 'cuda:0'

        with open('gen_hand_pose/dexgraspnet_all.pickle', 'rb') as f:
            self._dict = pickle.load(f)

        self.goal = [0,0,1]
        self.goal_pose = []

        goal_pose_1 = gymapi.Transform()
        goal_pose_1.p = gymapi.Vec3(0, 0.0255, 0.03)

        goal_pose_1.r = gymapi.Quat.from_euler_zyx(math.pi*0.5, 0, 0)

        goal_pose_2 = gymapi.Transform()
        goal_pose_2.p = gymapi.Vec3(0, -0.0255, 0.03)
        goal_pose_2.r = gymapi.Quat.from_euler_zyx(math.pi*0.5, 0, 0)

        goal_pose_3 = gymapi.Transform()
        goal_pose_3.p = gymapi.Vec3(0, 0, 0.0745)
        goal_pose_3.r = gymapi.Quat.from_euler_zyx(math.pi*0.5,0,math.pi*0.5)

        self.block_height = [0.03,0.03,0.0145]

        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_1))
        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_2))
        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_3))

        self.hand_rel_mat_list = []
        self.hand_pose_list = []

        self.hand_rel_mat_list.append(self.get_hand_rel_mat(self._dict[0][1]["hand_ref_pose"][0,0,:],self._dict[0][1]["obj_init"]))
        self.hand_rel_mat_list.append(self.get_hand_rel_mat(self._dict[0][1]["hand_ref_pose"][0,0,:],self._dict[0][1]["obj_init"]))
        self.hand_rel_mat_list.append(self.get_hand_rel_mat(self._dict[1][21]["hand_ref_pose"][0,0,:],self._dict[1][21]["obj_init"]))

        self.hand_pose_list.append(self._dict[0][1]["hand_ref_pose"])
        self.hand_pose_list.append(self._dict[0][1]["hand_ref_pose"])
        self.hand_pose_list.append(self._dict[1][21]["hand_ref_pose"])



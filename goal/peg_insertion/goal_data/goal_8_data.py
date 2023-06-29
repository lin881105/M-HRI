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


class Goal_8(Goal):
    def __init__(self):

        self.device = 'cuda:0'

        # with open('gen_hand_pose/test_ft.pickle', 'rb') as f:
        #     self._dict = pickle.load(f)

        self.goal = [3,2]
        self.goal_pose = []

        goal_pose_1 = gymapi.Transform()
        goal_pose_1.p = gymapi.Vec3(0.007, 0.049, 0)
        goal_pose_1.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        goal_pose_2 = gymapi.Transform()
        goal_pose_2.p = gymapi.Vec3(-0.045, -0.049, 0)
        goal_pose_2.r = gymapi.Quat.from_euler_zyx(0, 0, 0)


        self.peg_height = [0.0,0.0]


        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_1))
        self.goal_pose.append(utils.gymapi_transform2mat(goal_pose_2))
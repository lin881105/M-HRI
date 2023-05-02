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


class Goal_1():
    def __init__(self):

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

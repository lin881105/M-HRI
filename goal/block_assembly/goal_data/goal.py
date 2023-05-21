from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R


class Goal():
    def __init__(self):
        self.device = 'cuda:0'

    def get_hand_rel_mat(self,hand_rel_pose,obj_init):

        obj_init_mat = np.eye(4)
        obj_init_mat[:3, :3] = R.from_quat(obj_init[3:7]).as_matrix()
        obj_init_mat[:3, 3] = obj_init[0:3]
        hand_goal_mat = np.eye(4)
        hand_goal_mat[:3, :3] = R.from_euler("XYZ", hand_rel_pose[3:6]).as_matrix()
        hand_goal_mat[:3, 3] = hand_rel_pose[0:3]
        hand_rel_mat = np.linalg.inv(obj_init_mat) @ hand_goal_mat
        
        return hand_rel_mat



        


# if __name__ == "__main__":
#     x = Goal()

#     print(x._dict[0][0]["hand_ref_pose"])
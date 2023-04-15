from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import os
import math
import numpy as np
import torch
import random
import time
import scipy.io as sio
from utils import utils
from scipy.spatial.transform import Rotation as R
from tqdm import trange

# regionx = np.random.uniform(-0.085, 0.085)
# regiony = np.random.uniform(-0.085, 0.085)

# flag = True
# while True:
#     rand_x = np.random.uniform(-0.13,0.13, 3)
#     rand_y = np.random.uniform(-0.23,0.23, 3)
#     print(rand_x, rand_y)
#     for i in range(rand_x.shape[0]):
#         if (np.linalg.norm(np.array((regionx,regiony)) - np.array(rand_x[i],rand_y[i])) < 0.05):
#             flag = False
#             break
#             # randxy = np.random.uniform([-0.13, -0.23], [0.13, 0.23], (3, 2))
#         else:
#             for j in range(i+1, rand_x.shape[0]):
#                 if (np.linalg.norm(np.array(rand_x[i],rand_y[i]) - np.array(rand_x[j],rand_y[j])) < 0.02):
#                     flag = False
#                     break
#     if not flag:
#         continue
#     else:
#         break    

def check_in_region(region_xy, rand_xy):
    for i in range(rand_xy.shape[0]):
        if np.linalg.norm(region_xy - rand_xy[i]) < 0.08:
            return True
    
    return False

def check_contact_block(rand_xy):
    for i in range(rand_xy.shape[0]):
        for j in range(i+1, rand_xy.shape[0]):
            if np.linalg.norm(rand_xy[i] - rand_xy[j]) < 0.05:
                return True
            
    return False

region_xy = np.random.uniform([-0.085, -0.085], [0.085, 0.085], 2)

while True:
    rand_xy = np.random.uniform([-0.13, -0.23], [0.13, 0.23], (3, 2))
    
    if check_in_region(region_xy, rand_xy) or check_contact_block(rand_xy):
        continue
    else:
        break
    
print("success generate initial pos!!!")

class BlockAssembly():
    def __init__(self):
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        self.num_envs = 1
        self.env_spacing = 1.5
        self.max_episode_length = 195

        self.create_sim()
        
        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # Look at the first env
        cam_pos = gymapi.Vec3(1, 0, 1.5)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # create observation buffer
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

    def create_sim(self):
        # parse arguments
        args = gymutil.parse_arguments(description="Joint control Methods Example")

        args.use_gpu = False
        args.use_gpu_pipeline = False
        self.device = 'cpu' if torch.cuda.is_available() else 'cpu'

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 1

        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(spacing, 0.75 * spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        asset_file_mano = "urdf/mano/zeros/mano_addtips.urdf"

        # create mano asset
        asset_path_mano = os.path.join(asset_root, asset_file_mano)
        asset_root_mano = os.path.dirname(asset_path_mano)
        asset_file_mano = os.path.basename(asset_path_mano)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        mano_asset = self.gym.load_asset(self.sim, asset_root_mano, asset_file_mano, asset_options)
        self.num_mano_dofs = self.gym.get_asset_dof_count(mano_asset)
        
        # create table asset
        table_dims = gymapi.Vec3(0.3, 0.5, 0.4)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # create target region
        region_dims = gymapi.Vec3(0.1,0.1,0.0001)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        region_asset = self.gym.create_box(self.sim, region_dims.x,region_dims.y, region_dims.z, asset_options)

        # create block asset
        block_asset_list = []
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        # asset_options.fix_base_link = True
        block_type = ['A.urdf', 'B.urdf', 'C.urdf', 'D.urdf', 'E.urdf']
        for t in block_type:
            block_asset_list.append(self.gym.load_asset(self.sim, asset_root, 'urdf/block_assembly/block_' + t, asset_options))
    

        # set mano dof properties
        mano_dof_props = self.gym.get_asset_dof_properties(mano_asset)
        mano_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        mano_dof_props["stiffness"][:3].fill(500)
        mano_dof_props["stiffness"][3:].fill(50)
        mano_dof_props["damping"][:3].fill(200)
        mano_dof_props["damping"][3:].fill(200)
        mano_dof_props["friction"].fill(1)

        self.mano_dof_lower_limits = mano_dof_props['lower']
        self.mano_dof_upper_limits = mano_dof_props['upper']
        self.mano_dof_lower_limits = to_torch(self.mano_dof_lower_limits, device=self.device)
        self.mano_dof_upper_limits = to_torch(self.mano_dof_upper_limits, device=self.device)

        # set YCB properties
        # ycb_rb_props = self.gym.get_asset_rigid_shape_properties(ycb_asset)
        # ycb_rb_props[0].rolling_friction = 1

        # set default pose
        handobj_start_pose = gymapi.Transform()
        handobj_start_pose.p = gymapi.Vec3(0, 0, 0)
        handobj_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.45, 0.0, 0.5 * table_dims.z)

        region_pose = gymapi.Transform()

        # read block goal pos
        mat_file = "goal/block_assembly/goal_A_data.mat"
        mat_dict = sio.loadmat(mat_file)

        self.goal_list = mat_dict["block_list"][0]
        self.goal_pose = mat_dict["block_pose"]
        # self.rel_pick_pos = mat_dict["pick_pose"]
        # self.rel_place_pos = mat_dict["place_pose"]
        self.block_pos_world = mat_dict["block_world"]

        # cache some common handles for later use
        self.mano_indices, self.table_indices = [], []
        self.block_indices = [[] for _ in range(self.num_envs)]
        self.block_masses = [[] for _ in range(self.num_envs)]
        self.envs = []

        # create and populate the environments
        for i in range(num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            # create table and set properties
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 1, 0) # 001
            table_sim_index = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_sim_index)

            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(60, 33, 0) / 255)

            # add region
            region_pose.p.x = table_pose.p.x + region_xy[0]
            region_pose.p.y = table_pose.p.y + region_xy[1]
            region_pose.p.z = table_dims.z #+ 0.001
            region_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))

            region_handle = self.gym.create_actor(env_ptr, region_asset, region_pose, "target", i, 1, 1) # 001
            self.gym.set_rigid_body_color(env_ptr, region_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0., 0., 0.))

            for cnt, idx in enumerate(self.goal_list):
                block_pose = gymapi.Transform()
                block_pose.p.x = table_pose.p.x + rand_xy[cnt, 0]
                block_pose.p.y = table_pose.p.y + rand_xy[cnt, 1]
                block_pose.p.z = table_dims.z + 0.03

                r1 = R.from_euler('z', np.random.uniform(-math.pi, math.pi))
                r2 = R.from_matrix(self.goal_pose[cnt][:3,:3])
                rot = r1 * r2
                euler = rot.as_euler("xyz", degrees=False)

                block_pose.r = gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2])
                # block_pose=utils.mat2gymapi_transform(block_pos_world[cnt])
                block_handle = self.gym.create_actor(env_ptr, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i, 2 ** (cnt + 1), cnt + 2) # 010
               
                # block_pose = utils.mat2gymapi_transform(utils.gymapi_transform2mat(region_pose)@goal_pose[cnt])
                # block_handle = gym.create_actor(env, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i+1)
                # block_handles.append(block_handle)

                color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                self.gym.set_rigid_body_color(env_ptr, block_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                block_idx = self.gym.get_actor_rigid_body_index(env_ptr, block_handle, 0, gymapi.DOMAIN_SIM)
                self.block_indices[i].append(block_idx)
            
            # create mano and set properties
            mano_handle = self.gym.create_actor(env_ptr, mano_asset, handobj_start_pose, "mano", i, 2 ** (len(self.goal_list)), len(self.goal_list) + 2) # 100
            mano_sim_index = self.gym.get_actor_index(env_ptr, mano_handle, gymapi.DOMAIN_SIM)
            self.mano_indices.append(mano_sim_index)

            self.gym.set_actor_dof_properties(env_ptr, mano_handle, mano_dof_props)

        self.mano_indices = to_torch(self.mano_indices, dtype=torch.long, device=self.device)
        self.block_indices = to_torch(self.block_indices, dtype=torch.long, device=self.device)

    def reset_idx(self):
        # reset hand root pose
        # hand_init_pose = torch.tile(self.data_hand_init, (self.num_envs // self.data_num, 1)).clone()

        # self.root_state_tensor[self.mano_indices, 0:3] = hand_init_pose[:, 0:3]
        # self.root_state_tensor[self.mano_indices, 3:7] = euler_to_quat(hand_init_pose[:, 3:6], "XYZ")
        # self.root_state_tensor[self.mano_indices, 7:13] = torch.zeros_like(self.root_state_tensor[self.mano_indices, 7:13])
        # init_hand_indices = self.mano_indices.to(torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(init_hand_indices), len(init_hand_indices))

        # reset hand pose
        # hand_init_pose[:, :6] = 0
        # self.dof_state[:, 0] = hand_init_pose.reshape(-1)
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(hand_init_pose.reshape(-1)))

        # reset object pose
        # obj_init_pose = torch.tile(self.data_obj_init, (self.num_envs // self.data_num, 1)).clone()
        
        # self.root_state_tensor[self.ycb_indices, 0:3] = obj_init_pose[:, 0:3]
        # self.root_state_tensor[self.ycb_indices, 3:7] = obj_init_pose[:, 3:7]
        # self.root_state_tensor[self.ycb_indices, 7:13] = torch.zeros_like(self.root_state_tensor[self.ycb_indices, 7:13])
        # init_object_indices = self.ycb_indices.to(torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(init_object_indices), len(init_object_indices))
        pass

    def simulate(self):
        torch.set_printoptions(sci_mode=False)

        self.reset_idx()

        cnt = 0
        while not self.gym.query_viewer_has_closed(self.viewer):            
            cnt += 1

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)



            # print('-' * 10)
            # print(self.root_state_tensor[self.ycb_indices, :7])
            # print()

            # process predicted actions
            # actions_tensor = torch.tile(self.data_hand_ref, (self.num_envs // self.data_num, 1))

            # tf_mat = pose7d_to_matrix(self.root_state_tensor[self.mano_indices, :7])
            # cur_wrist_mat = pose6d_to_matrix(actions_tensor[:, :6], "XYZ")
            # new_wrist_mat = torch.bmm(torch.linalg.inv(tf_mat), cur_wrist_mat)
            # new_wrist_tensor = matrix_to_pose_6d(new_wrist_mat, "XYZ")

            # actions_tensor[:, :6] = new_wrist_tensor

            # set position target
            # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor))

            # rigidbody_transforms = self.gym.get_actor_rigid_body_states(env0, mano_hand0, gymapi.STATE_ALL)[rigidbodyid_list]["pose"]["p"]
            # rigidbody_pos = np.vstack([rigidbody_transforms["x"], rigidbody_transforms["y"], rigidbody_transforms["z"]]).T

            # for i in range(rigidbody_pos.shape[0]):
            #     body_states = self.gym.get_actor_rigid_body_states(env0, sphere_list[i], gymapi.STATE_ALL)
            #     body_states["pose"]["p"] = tuple(rigidbody_pos[i])
            #     self.gym.set_actor_rigid_body_states(env0, sphere_list[i], body_states, gymapi.STATE_ALL)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == "__main__":
    issac = BlockAssembly()
    issac.simulate()
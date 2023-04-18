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
import pytorch3d.transforms
import quaternion

# set random seed
np.random.seed(20)
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik", "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
    {"name": "--headless", "action": "store_true", "help": "Run headless"},
]

args = gymutil.parse_arguments(
    description="Joint control Methods Example",
    custom_parameters=custom_parameters,
    )


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

class ManoBlockAssembly():
    def __init__(self):
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        self.num_envs = 10
        self.env_spacing = 1.5
        self.max_episode_length = 195
        
        

        self.create_sim()
        self.get_hand_rel_mat()
        # create viewer using the default camera properties
        use_viewer = not args.headless
        if use_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
        else:
            self.viewer = None

        # Look at the first env
        cam_pos = gymapi.Vec3(1, 0, 1.5)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # create observation buffer
        self.gym.prepare_sim(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_mano_dofs, 2)

        self.set_init_hand_pos()

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        # self.stage_tensor = torch.zeros((self.num_envs),dtype=torch.long).to(self.device)
        self.stage = 0
        

        # self.get_step_size()

    def create_sim(self):

        self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        if args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = args.num_threads
            sim_params.physx.use_gpu = args.use_gpu
        else:
            raise Exception("This exampe can only be used with PhysX")

        # sim_params.physx.solver_type = 1
        # sim_params.physx.num_position_iterations = 4
        # sim_params.physx.num_velocity_iterations = 1
        # sim_params.physx.contact_offset = 0.005
        # sim_params.physx.rest_offset = 0.0
        # sim_params.physx.bounce_threshold_velocity = 0.2
        # sim_params.physx.max_depenetration_velocity = 1
        # sim_params.physx.num_threads = args.num_threads
        # sim_params.physx.use_gpu = args.use_gpu

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
        asset_options.fix_base_link = True
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
        # self.hand_init_pose = np.eye(4)
        # self.hand_init_pose[:3,3] = np.array((0,0,0.5))
        # self.hand_init_pose[:3,:3] = R.from_euler("xyz",(-90,180,0),degrees=True).as_matrix()
        handobj_start_pose = gymapi.Transform()
        handobj_start_pose.p = gymapi.Vec3(0, 0, 0.0)
        handobj_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # self.handobj_start_pose = utils.mat2gymapi_transform(self.hand_init_pose)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.45, 0.0, 0.5 * table_dims.z)

        region_pose = gymapi.Transform()

        # read block goal pos
        mat_file = "goal/block_assembly/goal_B_data.mat"
        mat_dict = sio.loadmat(mat_file)

        self.block_list = mat_dict["block_list"][0]
        self.goal_pose = mat_dict["block_pose"]
        # self.rel_pick_pos = mat_dict["pick_pose"]
        # self.rel_place_pos = mat_dict["place_pose"]
        self.block_pos_world = mat_dict["block_world"]
        self.block_height = mat_dict["block_height"]

        # cache some common handles for later use
        self.mano_indices, self.table_indices = [], []
        self.block_indices = [[] for _ in range(self.num_envs)]
        self.block_masses = [[] for _ in range(self.num_envs)]
        self.envs = []

        _goal_list = []

        # create and populate the environments
        for i in range(num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            # create table and set properties
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 1, 0) # 001
            table_sim_index = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_sim_index)

            # self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(60, 33, 0) / 255)

            # add region
            region_pose.p.x = table_pose.p.x + region_xy[0]
            region_pose.p.y = table_pose.p.y + region_xy[1]
            region_pose.p.z = table_dims.z #+ 0.001
            # region_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))

            region_handle = self.gym.create_actor(env_ptr, region_asset, region_pose, "target", i, 1, 1) # 001
            self.gym.set_rigid_body_color(env_ptr, region_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0., 0., 0.))

            goal = []
            np.random.seed(20)
            for cnt, idx in enumerate(self.block_list):
                block_pose = gymapi.Transform()
                block_pose.p.x = table_pose.p.x + rand_xy[cnt, 0]
                block_pose.p.y = table_pose.p.y + rand_xy[cnt, 1]
                block_pose.p.z = table_dims.z + self.block_height[0][cnt]

                # angle = np.random.uniform(-math.pi, math.pi)
                # r1 = R.from_euler('z', angle)
                r2 = R.from_matrix(self.goal_pose[cnt][:3,:3])
                # rot = r1 * r2
                # euler = rot.as_euler("xyz", degrees=False)
                euler = r2.as_euler("xyz", degrees=False)

                block_pose.r = gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2])
            
                # block_pose=utils.mat2gymapi_transform(block_pos_world[cnt])
                block_handle = self.gym.create_actor(env_ptr, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i, 2 ** (cnt + 1), cnt + 2) # 010
                goal.append(torch.tensor((block_pose.p.x,block_pose.p.y,0.5,block_pose.r.x,block_pose.r.y,block_pose.r.z,block_pose.r.w)).to(self.device))
                goal.append(torch.tensor((block_pose.p.x,block_pose.p.y,block_pose.p.z,block_pose.r.x,block_pose.r.y,block_pose.r.z,block_pose.r.w)).to(self.device))
                goal.append(torch.tensor((block_pose.p.x,block_pose.p.y,0.5,block_pose.r.x,block_pose.r.y,block_pose.r.z,block_pose.r.w)).to(self.device))
                # block_pose = utils.mat2gymapi_transform(utils.gymapi_transform2mat(region_pose)@goal_pose[cnt])
                # block_handle = gym.create_actor(env, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i+1)
                # block_handles.append(block_handle)
                tmp_pose = utils.mat2gymapi_transform(utils.gymapi_transform2mat(region_pose) @ self.goal_pose[cnt])
                goal_place_pose = torch.Tensor((tmp_pose.p.x,tmp_pose.p.y,tmp_pose.p.z,tmp_pose.r.x,tmp_pose.r.y,tmp_pose.r.z,tmp_pose.r.w)).to(self.device)
                goal_preplace_pose = torch.Tensor((tmp_pose.p.x,tmp_pose.p.y,0.5,tmp_pose.r.x,tmp_pose.r.y,tmp_pose.r.z,tmp_pose.r.w)).to(self.device)
                # goal.append(goal_preplace_pose)
                goal.append(goal_place_pose)
                goal.append(goal_preplace_pose)



                color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                self.gym.set_rigid_body_color(env_ptr, block_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                block_idx = self.gym.get_actor_index(env_ptr, block_handle, gymapi.DOMAIN_SIM)
                self.block_indices[i].append(block_idx)

            # save block goal pose
            
            _goal_list.append(torch.stack(goal))

            # create mano and set properties
            mano_handle = self.gym.create_actor(env_ptr, mano_asset, handobj_start_pose, "mano", i, 2 ** (len(self.block_list)) - 1, len(self.block_list) + 2) # 100
            mano_sim_index = self.gym.get_actor_index(env_ptr, mano_handle, gymapi.DOMAIN_SIM)
            self.mano_indices.append(mano_sim_index)

            self.gym.set_actor_dof_properties(env_ptr, mano_handle, mano_dof_props)

        self.mano_indices = to_torch(self.mano_indices, dtype=torch.long, device=self.device)
        self.block_indices = to_torch(self.block_indices, dtype=torch.long, device=self.device) 
        # self.goal_list = torch.Tensor(num_envs,len(self.block_list),6).to(self.device)
        # torch.cat(_goal_list,out=self.goal_list)
        self.goal_list = torch.stack(_goal_list)
        

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

    def get_hand_rel_mat(self):
        obj_init = np.array([-0.19434147, -0.11968146, 0.01548913, -0.50471319, 0.49524648, -0.4952225 , 0.50472785])
        # obj_init = np.array([-0.14587866,  0.20421203,  0.02021903, -0.06949283, -0.103413  ,-0.43897412,  0.88929234]) # 拱門

        obj_init_mat = np.eye(4)
        obj_init_mat[:3, :3] = R.from_quat(obj_init[3:7]).as_matrix()
        obj_init_mat[:3, 3] = obj_init[0:3]

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
    #     hand_goal_pose = np.array([-0.10150228,  0.19490343,  0.07588965,  1.0861137 , -0.84895056,
    #    -0.54786706,  0.22902231, -0.28977403,  0.3116039 ,  0.08973111,
    #     0.00571879,  0.7467749 , -0.11694898,  0.04409688,  0.22905038,
    #    -0.11792404, -0.33667254,  0.7789884 , -0.06663863, -0.14730793,
    #     0.6393313 ,  0.01266811,  0.01322488,  0.34988308, -0.8594047 ,
    #     0.06921063,  0.82026833, -0.15663488,  0.00436937,  0.5340845 ,
    #    -0.32596567,  0.12056006,  0.24855043, -0.23608978, -0.01125655,
    #     0.8863701 , -0.3165149 , -0.05750437,  0.5138516 , -0.19815606,
    #     0.10261149,  0.455346  ,  1.094469  , -0.236023  ,  0.30784672,
    #    -0.59952945,  0.06054716, -0.09722855,  0.57003605,  0.07307385,
    #     0.2632291 ], dtype=np.float32) #拱門
        hand_goal_mat = np.eye(4)
        hand_goal_mat[:3, :3] = R.from_euler("XYZ", hand_goal_pose[3:6]).as_matrix()
        hand_goal_mat[:3, 3] = hand_goal_pose[0:3]
        self.hand_rel_mat = torch.tensor(np.linalg.inv(obj_init_mat) @ hand_goal_mat, dtype=torch.float32).to(self.device)
    
    def get_step_size(self):
        step_size_list = []
        for i in range(self.num_envs):
            step_size = []
            tmp_obj_pose = np.linalg.inv(self.get_hand_rel_mat()@np.linalg.inv(self.hand_init_pose))
            tmp_obj_pose = utils.mat2gymapi_transform(tmp_obj_pose)
            current_pose = torch.tensor((
                tmp_obj_pose.p.x,
                tmp_obj_pose.p.y,
                tmp_obj_pose.p.z,
                tmp_obj_pose.r.x,
                tmp_obj_pose.r.y,
                tmp_obj_pose.r.z,
                tmp_obj_pose.r.w,
            )).to(self.device)
            for j in range(len(self.goal_list[i])):
                step_size.append((self.goal_list[i,j]-current_pose)/1000)
                current_pose = self.goal_list[i,j]
            step_size = torch.stack(step_size).to(self.device)
            step_size_list.append(step_size)
        self.step_size_list = torch.stack(step_size_list).to(self.device)

    
    def set_init_hand_pos(self):
        self.dof_state[:,2,0] += 0.5

        self.dof_state[:,5,0] += np.pi
        
        self.dof_state[:,3,0] += -np.pi/2

        dof_indices = self.mano_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                        gymtorch.unwrap_tensor(self.dof_state),
                                        gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        
        target = self.dof_state[:, :, 0].clone()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(target),
                                                gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
    

    def set_hand_pos(self,new_wrist_mat):
        self.hand_goal_pose = torch.tensor([-2.21341878e-01, -9.16626230e-02,  4.62329611e-02,  7.10443914e-01,
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
        6.18857801e-01, -1.02387838e-01,  3.12080264e-01], dtype=torch.float32).to(self.device)

        new_dof_state = self.dof_state.clone()
        
        new_dof_state[:, 0:3, 0] = new_wrist_mat[:, :3, 3]
        new_dof_state[:, 3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:, :3, :3], "XYZ")
        new_dof_state[:,6:,0] = self.hand_goal_pose[6:]
        # print(self.dof_state[:, :, 0])

        self.dof_state[:,0:6,0] = self.dof_state[:,0:6,0] + (new_dof_state[:,0:6,0] - self.dof_state[:,0:6,0])*0.01
        self.dof_state[:,6:,0] = self.dof_state[:,6:,0] + (new_dof_state[:,6:,0] - self.dof_state[:,6:,0])*0.005
        
        dof_indices = self.mano_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                        gymtorch.unwrap_tensor(self.dof_state),
                                        gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        
        target = self.dof_state[:, :, 0].clone()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(target),
                                                gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        
        return new_dof_state
    
        
    def set_hand_object_pos(self):
        curr_block_pose = self.root_state_tensor[self.block_indices[:,self.stage//6], :7].clone()
        target_block_pose = self.goal_list[:,self.stage,:].clone()
        
        

        self.root_state_tensor[self.block_indices[:,self.stage//6], :6] += (target_block_pose[:,:6] - curr_block_pose[:,:6])*0.008
        
        goal_obj_indices = self.block_indices[:,self.stage//6].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.root_state_tensor),
                                                gymtorch.unwrap_tensor(goal_obj_indices), len(goal_obj_indices))
        

        goal_pose = self.root_state_tensor[self.block_indices[:,self.stage//6],:7]

        cur_obj_mat = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
        cur_obj_mat[:,:3, 3] = goal_pose[:,:3]
        cur_obj_mat[:,:3, :3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:, [6,3,4,5]])

        new_wrist_mat = cur_obj_mat @ self.hand_rel_mat

        new_dof_state = self.dof_state.clone()
        
        new_dof_state[:, 0:3, 0] = new_wrist_mat[:, :3, 3]
        new_dof_state[:, 3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:, :3, :3], "XYZ")
        new_dof_state[:,6:,0] = self.hand_goal_pose[6:]
        # print(self.dof_state[:, :, 0])

        self.dof_state[:,0:6,0] = new_dof_state[:,0:6,0]
        self.dof_state[:,6:,0] = self.dof_state[:,6:,0]
        
        dof_indices = self.mano_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                        gymtorch.unwrap_tensor(self.dof_state),
                                        gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        
        target = self.dof_state[:, :, 0].clone()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(target),
                                                gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        


        return target_block_pose

    def check_block_pos_reach(self,target_block_pose):
        curr_block_pose = self.root_state_tensor[self.block_indices[:,self.stage//6], :7]
        diff = torch.norm(curr_block_pose-target_block_pose)
        if diff < 0.01:
            return True
        else:
            return False
        
    def check_hand_pos_reach(self,new_dof_state):
        diff = torch.norm(self.dof_state[:,:6,0]-new_dof_state[:,:6,0])
        if diff < 0.05:
            return True
        else:
            return False


    def check_hand_reach(self,new_dof_state):

        diff = self.dof_state - new_dof_state
        print(torch.norm(diff))

        if torch.norm(diff) < 0.01:
            return True
        else:
            return False

    def update(self):

        set_object = False


        if self.stage == 2 or self.stage==3 or self.stage==7 or self.stage==8:
            set_object = True

        if not set_object:
            goal_pose = self.goal_list[:,self.stage,:]

            cur_obj_mat = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
            cur_obj_mat[:,:3, 3] = goal_pose[:,:3]
            cur_obj_mat[:,:3, :3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:, [6,3,4,5]])

            target_hand_pose = cur_obj_mat @ self.hand_rel_mat

            new_target_dof = self.set_hand_pos(target_hand_pose)
    
            if self.check_hand_pos_reach(new_target_dof):
                self.stage+=1


        else:
            
            target_block_pose = self.set_hand_object_pos()
            goal_pose = self.root_state_tensor[self.block_indices[:,self.stage//6],:7]

            cur_obj_mat = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
            cur_obj_mat[:,:3, 3] = goal_pose[:,:3]
            cur_obj_mat[:,:3, :3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:, [6,3,4,5]])

            target_hand_pose = cur_obj_mat @ self.hand_rel_mat

            new_target_dof = self.set_hand_pos(target_hand_pose)

            if self.check_block_pos_reach(target_block_pose):
                self.stage+=1




        # exit()
    
        # # set object pose
        # self.root_state_tensor[self.block_indices[:,1], 2] += 0.001
        # goal_obj_indices = torch.tensor(self.block_indices[:,1]).to(torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                         gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                         gymtorch.unwrap_tensor(goal_obj_indices), len(goal_obj_indices))
        

        # print(block_pose)
        
        # print(self.goal_list[:,0,:])

        # print(cur_obj_mat)


        # np.array([-0.19434147, -0.11968146, 0.01548913, -0.50471319, 0.49524648, -0.4952225 , 0.50472785])



        # target_hand_pos = target_hand_pose[:,:3]

        # target_hand_rot = target_hand_pose[:,3:6]

        # cur_hand_pos = self.dof_state[:,0:3,0]

        # cur_hand_rot = self.dof_state[:,3:6,0]

        

        # print(self.dof_state[:,3:6,0])

        # print(cur_obj_mat.shape)

        # print(self.get_hand_rel_mat().shape)
        
        # exit()
        
        

        # if self.stage%2 == 0 or self.stage%6 == 5:
        #     # move hand
        #     cur_obj_pose = init_pose + self.step_size_list[:,self.stage,:]*count
            
        #     cur_obj_mat = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1)
        #     cur_obj_mat[:,:3, 3] = cur_obj_pose[:,:3]
        #     cur_obj_mat[:,:3, :3] = pytorch3d.transforms.quaternion_to_matrix(cur_obj_pose[:, [6, 3, 4, 5]])
        #     new_wrist_mat = cur_obj_mat @ self.get_hand_rel_mat()
            
        #     self.dof_state[:, 0:3, 0] = new_wrist_mat[:, :3, 3]
        #     self.dof_state[:, 3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:, :3, :3], "XYZ")
            
        #     dof_indices = self.mano_indices.to(dtype=torch.int32)
        #     self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                     gymtorch.unwrap_tensor(self.dof_state),
        #                                     gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
            
        #     target = self.dof_state[:, :, 0].clone()
        #     self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                             gymtorch.unwrap_tensor(target),
        #                                             gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        #     count+=1
        #     if count >= 1000:
        #         self.stage_tensor += 1
        #         count = 0
        #     return count
        
        # else:
        #      # set object pose
        #     self.root_state_tensor[self.block_indices[:,self.stage//6], :7] += self.step_size_list[:,self.stage_tensor,:]
        #     goal_obj_indices = torch.tensor([self.block_indices[:,self.stage_tensor/6]]).to(torch.int32)
        #     self.gym.set_actor_root_state_tensor_indexed(
        #         self.sim,
        #         gymtorch.unwrap_tensor(self.root_state_tensor),
        #         gymtorch.unwrap_tensor(goal_obj_indices),
        #         len(goal_obj_indices)
        #     )

        #     # set hand pose
        #     cur_obj_mat = torch.eye(4)
        #     cur_obj_mat[:3, 3] = cur_obj_pose[:3]
        #     cur_obj_mat[:3, :3] = pytorch3d.transforms.quaternion_to_matrix(cur_obj_pose[3:7][[3, 0, 1, 2]])
        #     new_wrist_mat = cur_obj_mat @ self.get_hand_rel_mat()

        #     new_wrist_mat = cur_obj_mat @ self.get_hand_rel_mat()
        #     self.dof_state[0:3, 0] = new_wrist_mat[:3, 3]
        #     self.dof_state[3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:3, :3], "XYZ")
        #     dof_indices = self.mano_indices.to(dtype=torch.int32)
        #     self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                     gymtorch.unwrap_tensor(self.dof_state),
        #                                     gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
            
        #     target = self.dof_state[:, 0].clone()
        #     self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                             gymtorch.unwrap_tensor(target),
        #                                             gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        #     count+=1
        #     if count >= 1000:
        #         self.stage_tensor += 1
        #         count = 0
        #     return count
        

    def simulate(self):
        torch.set_printoptions(sci_mode=False)

        # self.reset_idx()

        step = 0

        cnt = 0
        while not self.gym.query_viewer_has_closed(self.viewer):            

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)

            # print(self.root_state_tensor[self.block_indices[:,0], :7].shape)
            # print(self.block_indices)
            # print(self.step_size_list.shape)

            self.update()


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
    issac = ManoBlockAssembly()
    issac.simulate()
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
from time import sleep
# set random seed
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik", "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
    {"name": "--headless", "action": "store_true", "help": "Run headless"},
    {"name": "--goal", "type":int, "default": 0}
]

args = gymutil.parse_arguments(
    description="Joint control Methods Example",
    custom_parameters=custom_parameters,
    )

class ManoBlockAssembly():
    def __init__(self):#goal,num_envs,init_block_pose,init_region_pose
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        self.num_envs = args.num_envs
        self.env_spacing = 1.5
        self.goal = args.goal
        self.device = "cuda:0"
        # self.init_block_pose = init_block_pose
        # self.init_region_pose = init_region_pose
        self.get_goal_pose()
        self.generate_pose()
        self.create_sim()
        
        

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

        self.stage = torch.zeros((self.num_envs),dtype=torch.long).to(self.device)
        # self.stage = 0
        

        # self.get_step_size()

    def check_in_region(self, region_xy, rand_xy):
        # for j in range(rand_xy.shape[0]):
        #     for i in range(rand_xy.shape[1]):
        #         if np.linalg.norm(region_xy[j] - rand_xy[j][i]) < 0.08:
        #             return True
        reset_idx = torch.where(torch.norm(region_xy.unsqueeze(1).repeat(1,len(self.goal_list),1) - rand_xy,dim=2)<0.08,True,False)
        # reset_idx = torch.logical_not(_diff).to(self.device)
   
        # reset_idx = torch.logical_not(_diff).to(self.device)
        
        return reset_idx

    def check_contact_block(self,rand_xy):
        # for k in range(rand_xy.shape[0]):
        #     for i in range(rand_xy.shape[1]):
        #         for j in range(i+1, rand_xy.shape[2]):
        #             if np.linalg.norm(rand_xy[k][i] - rand_xy[k][j]) < 0.08:
        #                 return True
        _diff=[]
        for i in range(len(self.goal_list)):
            for j in range(i+1, len(self.goal_list)):
                _diff.append(torch.norm(rand_xy[:,i,:] - rand_xy[:,i+1,:],dim=1).unsqueeze(1).to(self.device))

        # for i in range(len(self.goal_list)):
        #     _tmp = rand_xy[:,i,:].repeat(1,3,1)

            
        _diff = torch.cat(_diff,dim=1).to(self.device)
        
        reset_idx = torch.where(_diff<0.08,True,False)
                        
        return reset_idx
    
    def generate_pose(self):

        # self.region_xy = np.random.uniform([-0.085, -0.085], [0.085, 0.085], (self.num_envs,2))
        self.region_xy = torch.FloatTensor(self.num_envs,2).uniform_(-0.085,0.085).to(self.device)
        self.rand_xy = torch.stack((torch.FloatTensor(self.num_envs,len(self.goal_list)).uniform_(-0.13,0.13),
                                    torch.FloatTensor(self.num_envs,len(self.goal_list)).uniform_(-0.23,0.23)),dim=2).to(self.device)

        while True:
            # self.rand_xy = np.random.uniform([-0.13, -0.23], [0.13, 0.23], (self.num_envs,len(self.goal_list), 2))
            region_reset_idx = self.check_in_region(self.region_xy, self.rand_xy)
            block_reset_idx = self.check_contact_block(self.rand_xy)
            # print(region_reset_idx)
            # print(block_reset_idx)

            reset_idx = torch.logical_or(region_reset_idx,block_reset_idx).to(self.device)

            if torch.all(torch.logical_not(reset_idx)):
                break
            else:
                # print(reset_idx)

                # print(self.rand_xy[reset_idx, :].shape)
                # print(torch.sum(torch.any(reset_idx, dim=1)))
                self.rand_xy[torch.any(reset_idx,dim=1),:, :] = torch.stack((torch.FloatTensor(torch.sum(torch.any(reset_idx, dim=1)), len(self.goal_list)).uniform_(-0.13,0.13),
                                                         torch.FloatTensor(torch.sum(torch.any(reset_idx, dim=1)), len(self.goal_list)).uniform_(-0.23,0.23)),dim=2).to(self.device)
            
        print("success generate initial pos!!!")
    
    def get_goal_pose(self):
        # read block goal pos
        mat_file = f"goal/block_assembly/goal_data/goal_{self.goal}_data.mat"
        mat_dict = sio.loadmat(mat_file)

        self.goal_list = mat_dict["block_list"][0]
        self.goal_pose = mat_dict["block_pose"]
        self.block_height = mat_dict["block_height"]
        self.hand_rel_mat = torch.tensor(mat_dict["hand_rel_pose"],dtype=torch.float32).to(self.device)
        self.hand_goal_pose = torch.tensor(mat_dict["hand_pose"],dtype=torch.float32).to(self.device)


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


        # cache some common handles for later use
        self.mano_indices, self.table_indices = [], []
        self.block_indices = [[] for _ in range(self.num_envs)]
        self.block_masses = [[] for _ in range(self.num_envs)]
        self.envs = []

        _goal_list = []

        # init_pose_mat = sio.loadmat("data/2023-05-09-22-32-32/env_00000/init_pose.mat")
        # init_block_pose = init_pose_mat["block_init_pose_world"]
        # init_region_pose = init_pose_mat["region_init_pose_world"][0]

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
            region_pose.p.x = table_pose.p.x + self.region_xy[i][0]
            region_pose.p.y = table_pose.p.y + self.region_xy[i][1]
            region_pose.p.z = table_dims.z #+ 0.001
            region_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            # region_pose.p.x = self.init_region_pose[i][0]
            # region_pose.p.y = self.init_region_pose[i][1]
            # region_pose.p.z = self.init_region_pose[i][2]
            # region_pose.r.x = self.init_region_pose[i][3]
            # region_pose.r.y = self.init_region_pose[i][4]
            # region_pose.r.z = self.init_region_pose[i][5]
            # region_pose.r.w = self.init_region_pose[i][6]

            region_handle = self.gym.create_actor(env_ptr, region_asset, region_pose, "target", i, 1, 1) # 001
            self.gym.set_rigid_body_color(env_ptr, region_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0., 0., 0.))

            goal = []

        
            for cnt, idx in enumerate(self.goal_list):
                # block_pose = gymapi.Transform()
                # block_pose.p.x = self.init_block_pose[i][cnt][0]
                # block_pose.p.y = self.init_block_pose[i][cnt][1]
                # block_pose.p.z = self.init_block_pose[i][cnt][2]
                # block_pose.r.x = self.init_block_pose[i][cnt][3]
                # block_pose.r.y = self.init_block_pose[i][cnt][4]
                # block_pose.r.z = self.init_block_pose[i][cnt][5]
                # block_pose.r.w = self.init_block_pose[i][cnt][6]

                block_pose = gymapi.Transform()
                block_pose.p.x = table_pose.p.x + self.rand_xy[i][cnt, 0]
                block_pose.p.y = table_pose.p.y + self.rand_xy[i][cnt, 1]
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
                goal.append(goal_preplace_pose)
                goal.append(goal_place_pose)
                goal.append(goal_preplace_pose)



                color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                self.gym.set_rigid_body_color(env_ptr, block_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                block_idx = self.gym.get_actor_index(env_ptr, block_handle, gymapi.DOMAIN_SIM)
                self.block_indices[i].append(block_idx)

            # save block goal pose
            
            _goal_list.append(torch.stack(goal))

            # create mano and set properties
            mano_handle = self.gym.create_actor(env_ptr, mano_asset, handobj_start_pose, "mano", i, 2 ** (len(self.goal_list)) - 1, len(self.goal_list) + 2) # 100
            mano_sim_index = self.gym.get_actor_index(env_ptr, mano_handle, gymapi.DOMAIN_SIM)
            self.mano_indices.append(mano_sim_index)

            self.gym.set_actor_dof_properties(env_ptr, mano_handle, mano_dof_props)

        self.mano_indices = to_torch(self.mano_indices, dtype=torch.long, device=self.device)
        self.block_indices = to_torch(self.block_indices, dtype=torch.long, device=self.device) 
        # self.goal_list = torch.Tensor(num_envs,len(self.block_list),6).to(self.device)
        # torch.cat(_goal_list,out=self.goal_list)
        self.goal_list = torch.stack(_goal_list)
        
    
    # def get_step_size(self):
    #     step_size_list = []
    #     for i in range(self.num_envs):
    #         step_size = []
    #         tmp_obj_pose = np.linalg.inv(self.get_hand_rel_mat()@np.linalg.inv(self.hand_init_pose))
    #         tmp_obj_pose = utils.mat2gymapi_transform(tmp_obj_pose)
    #         current_pose = torch.tensor((
    #             tmp_obj_pose.p.x,
    #             tmp_obj_pose.p.y,
    #             tmp_obj_pose.p.z,
    #             tmp_obj_pose.r.x,
    #             tmp_obj_pose.r.y,
    #             tmp_obj_pose.r.z,
    #             tmp_obj_pose.r.w,
    #         )).to(self.device)
    #         for j in range(len(self.goal_list[i])):
    #             step_size.append((self.goal_list[i,j]-current_pose)/1000)
    #             current_pose = self.goal_list[i,j]
    #         step_size = torch.stack(step_size).to(self.device)
    #         step_size_list.append(step_size)
    #     self.step_size_list = torch.stack(step_size_list).to(self.device)

    
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
   
    def set_hand_pos(self,new_wrist_mat,idx):

        new_dof_state = self.dof_state.clone()
        # print(idx[0])
        # print(new_wrist_mat.shape)
        # exit()
  
        new_dof_state[idx[0], 0:3, 0] = new_wrist_mat[:, :3, 3]
        new_dof_state[idx[0], 3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:, :3, :3], "XYZ")
        new_dof_state[idx[0],6:,0] = self.hand_goal_pose[self.stage[idx]//6,0,0,6:]
        # print(self.dof_state[:, :, 0])

        self.dof_state[:,0:6,0] = self.dof_state[:,0:6,0] + (new_dof_state[:,0:6,0] - self.dof_state[:,0:6,0])*0.01
        self.dof_state[:,6:,0] = self.dof_state[:,6:,0] + (new_dof_state[:,6:,0] - self.dof_state[:,6:,0])*0.005

        
        dof_indices = self.mano_indices.to(dtype=torch.int32)
        # print(dof_indices.shape)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                        gymtorch.unwrap_tensor(self.dof_state),
                                        gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        
        target = self.dof_state[:, :, 0].clone()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(target),
                                                gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        
        return new_dof_state
    
        
    def set_hand_object_pos(self,idx):
    
        # print(self.block_indices)
        # print(self.stage[idx]//6)
        # print(self.block_indices[idx[0],self.stage[idx]//6])
        
        # print(self.root_state_tensor[self.block_indices[self.stage//6]].shape)


        curr_block_pose = self.root_state_tensor[self.block_indices[idx[0],self.stage[idx]//6], :7].clone()
        target_block_pose = self.goal_list[idx[0],self.stage[idx],:].clone()
        

        self.root_state_tensor[self.block_indices[idx[0],self.stage[idx]//6], :6] += (target_block_pose[:,:6] - curr_block_pose[:,:6])*0.008
        
        goal_obj_indices = self.block_indices[idx[0],self.stage[idx]//6].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.root_state_tensor),
                                                gymtorch.unwrap_tensor(goal_obj_indices), len(goal_obj_indices))
        

        goal_pose = self.root_state_tensor[self.block_indices[idx[0],self.stage[idx]//6],:7]

        cur_obj_mat = torch.eye(4).unsqueeze(0).repeat(idx[0].shape[0], 1, 1).to(self.device)
        cur_obj_mat[:,:3, 3] = goal_pose[:,:3]
        cur_obj_mat[:,:3, :3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:, [6,3,4,5]])

        new_wrist_mat = cur_obj_mat @ self.hand_rel_mat[self.stage[idx]//6]

        new_dof_state = self.dof_state.clone()
        
        new_dof_state[idx[0], 0:3, 0] = new_wrist_mat[:, :3, 3]
        new_dof_state[idx[0], 3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:, :3, :3], "XYZ")
        new_dof_state[idx[0],6:,0] = self.hand_goal_pose[self.stage[idx]//6,0,0,6:]
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
        


        return self.goal_list[torch.arange(0,self.num_envs,dtype=torch.long).to(self.device),self.stage,:].clone()

    def check_block_pos_reach(self,target_block_pose,block_threshold,idx):
        
        curr_block_pose = self.root_state_tensor[self.block_indices[torch.arange(0,self.num_envs,dtype=torch.long).to(self.device),self.stage//6], :7]
   
        diff = torch.norm(curr_block_pose-target_block_pose,dim=1)
        # print((curr_block_pose-target_block_pose).shape)
        # print(diff.shape)
        # print(block_threshold.shape)
        # if diff < block_threshold:
        #     return True
        # else:
        #     return False
        # print(block_threshold.shape)
        return torch.where(diff<block_threshold,True,False)
        
    def check_hand_pos_reach(self,new_dof_state,threshold,idx):
        # print(self.dof_state[:,:6,0].shape)
        # print(new_dof_state[:,:6,0].shape)
        # exit()
        diff = torch.norm(self.dof_state[:,:6,0]-new_dof_state[:,:6,0],dim=1)
        # print(diff.shape)
        # print(threshold.shape)
        # exit()
        # if diff < threshold:
        #     return True
        # else:
        #     return False
        return torch.where(diff<threshold,True,False)
        
    def reset_grasp_pose(self,envs):
        self.dof_state[envs,6:,0] = self.dof_state[envs,6:,0] + (0 - self.dof_state[envs,6:,0])*0.005
        
        dof_indices = self.mano_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                        gymtorch.unwrap_tensor(self.dof_state),
                                        gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        
        target = self.dof_state[:, :, 0].clone()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(target),
                                                gymtorch.unwrap_tensor(dof_indices), len(dof_indices))



    def update(self):

        # set_object = torch.zeros((self.num_envs)).to(self.device)
        # print(self.stage)


        # if self.stage%6 == 2 or self.stage%6 ==3 or self.stage%6==4 :
        # set_object=True
        set_object = torch.logical_or(torch.logical_or(torch.where(self.stage%6==2,True,False),torch.where(self.stage%6==3,True,False)),torch.where(self.stage%6==4,True,False)).to(self.device)
        # if self.stage%6 == 4:
        #     block_threshold = 0.01
        # else:
        #     block_threshold = 0.1
        block_threshold = torch.where(self.stage%6==4,0.01,0.1).to(self.device)
        
        # if self.stage%6 == 1:
        #     threshold = 0.005
        # else:
        #     threshold = 0.1
        threshold = torch.where(self.stage%6==1,0.005,0.1).to(self.device)

        # if self.stage%6 == 5:
        #     self.reset_grasp_pose()
        reset_idx = torch.where(self.stage%6==5)[0].to(self.device)
        self.reset_grasp_pose(reset_idx)


        # if not set_object:
        #     goal_pose = self.goal_list[:,self.stage,:]

        #     cur_obj_mat = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
        #     cur_obj_mat[:,:3, 3] = goal_pose[:,:3]
        #     cur_obj_mat[:,:3, :3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:, [6,3,4,5]])

        #     target_hand_pose = cur_obj_mat @ self.hand_rel_mat

        #     new_target_dof = self.set_hand_pos(target_hand_pose)
    
        #     if self.check_hand_pos_reach(new_target_dof,threshold):
        #         self.stage+=1
        


        idx = torch.logical_not(set_object)
        idx = torch.where(idx)

       
        goal_pose = self.goal_list[idx[0],self.stage[idx],:]

        curr_obj_mat = torch.eye(4).unsqueeze(0).repeat(idx[0].shape[0],1,1).to(self.device)
        curr_obj_mat[:,:3,3] = goal_pose[:,:3]
        curr_obj_mat[:,:3,:3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:,[6,3,4,5]])
        target_hand_pose = curr_obj_mat @ self.hand_rel_mat[self.stage[idx]//6]
        new_target_dof = self.set_hand_pos(target_hand_pose,idx)
        _reach = self.check_hand_pos_reach(new_target_dof,threshold,idx)
        


        self.stage[_reach]+=1

        ###############################################################################################

        idx = torch.where(set_object)
        target_block_pose = self.set_hand_object_pos(idx)
        goal_pose = self.root_state_tensor[self.block_indices[idx[0],self.stage[idx]//6],:7]

        cur_obj_mat = torch.eye(4).unsqueeze(0).repeat(idx[0].shape[0], 1, 1).to(self.device)
        cur_obj_mat[:,:3, 3] = goal_pose[:,:3]
        cur_obj_mat[:,:3, :3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:, [6,3,4,5]])

        target_hand_pose = cur_obj_mat @ self.hand_rel_mat[self.stage[idx]//6]

        new_target_dof = self.set_hand_pos(target_hand_pose,idx)

        _reach = self.check_block_pos_reach(target_block_pose,block_threshold,idx)
        # print(_reach)
        self.stage[_reach]+=1


        
        # else:
            
        #     target_block_pose = self.set_hand_object_pos()
        #     goal_pose = self.root_state_tensor[self.block_indices[:,self.stage//6],:7]

        #     cur_obj_mat = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
        #     cur_obj_mat[:,:3, 3] = goal_pose[:,:3]
        #     cur_obj_mat[:,:3, :3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:, [6,3,4,5]])

        #     target_hand_pose = cur_obj_mat @ self.hand_rel_mat

        #     new_target_dof = self.set_hand_pos(target_hand_pose)

        #     if self.check_block_pos_reach(target_block_pose,block_threshold):
        #         self.stage+=1
        

    def simulate(self):
        torch.set_printoptions(sci_mode=False)

        # self.reset_idx()
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


            
            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == "__main__":
    issac = ManoBlockAssembly()
    issac.simulate()
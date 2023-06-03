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
from tqdm import trange
import datetime
# set random seed
# custom_parameters = [
#     {"name": "--controller", "type": str, "default": "ik", "help": "Controller to use for Franka. Options are {ik, osc}"},
#     {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
#     {"name": "--headless", "action": "store_true", "help": "Run headless"},
#     {"name": "--goal", "type":int, "default": 0}
# ]

# args = gymutil.parse_arguments(
#     description="Joint control Methods Example",
#     custom_parameters=custom_parameters,
#     )

class ManoBlockAssembly():
    def __init__(self,success_envs,init_block_pose,init_region_pose,img_path_root,args):
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        self.num_envs = len(success_envs)
        self.success_envs = success_envs
        self.init_block_pose = init_block_pose
        self.init_region_pose = init_region_pose
        self.env_spacing = 1.5
        self.goal = args.goal
        self.device = "cuda:0"
        self.img_pth = img_path_root
        self.save=args.save
        # self.init_block_pose = init_block_pose
        # self.init_region_pose = init_region_pose
        self.get_goal_pose()
        # self.generate_pose()
        self._create_image_directories()
        self.create_sim(args)
        args.headless = True
        
        
        

        # create viewer using the default camera properties
        use_viewer = not args.headless
        if use_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
        else:
            self.viewer = None
        self.create_camera()
        # Look at the first env
        cam_pos = gymapi.Vec3(1, 0, 1.5)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # create observation buffer
        self.gym.prepare_sim(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_mano_dofs, 2)

        

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


    def create_sim(self,args):

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
        self.num_per_row = int(math.sqrt(self.num_envs))
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
        table_dims = gymapi.Vec3(0.4, 0.6, 0.4)
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
        for i,env_idx in enumerate(self.success_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            # create table and set properties
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 1, 0) # 001
            table_sim_index = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_sim_index)

            # self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(60, 33, 0) / 255)

            # add region
            # region_pose.p.x = table_pose.p.x + self.region_xy[i][0]
            # region_pose.p.y = table_pose.p.y + self.region_xy[i][1]
            # region_pose.p.z = table_dims.z #+ 0.001
            # region_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi/4, math.pi/4))
            region_pose.p.x = self.init_region_pose[env_idx][0]
            region_pose.p.y = self.init_region_pose[env_idx][1]
            region_pose.p.z = self.init_region_pose[env_idx][2]
            region_pose.r.x = self.init_region_pose[env_idx][3]
            region_pose.r.y = self.init_region_pose[env_idx][4]
            region_pose.r.z = self.init_region_pose[env_idx][5]
            region_pose.r.w = self.init_region_pose[env_idx][6]

            region_handle = self.gym.create_actor(env_ptr, region_asset, region_pose, "target", i, 1, 1) # 001
            self.gym.set_rigid_body_color(env_ptr, region_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0., 0., 0.))

            goal = []

        
            for cnt, idx in enumerate(self.goal_list):
                block_pose = gymapi.Transform()
                block_pose.p.x = self.init_block_pose[env_idx][cnt][0]
                block_pose.p.y = self.init_block_pose[env_idx][cnt][1]
                block_pose.p.z = self.init_block_pose[env_idx][cnt][2]
                block_pose.r.x = self.init_block_pose[env_idx][cnt][3]
                block_pose.r.y = self.init_block_pose[env_idx][cnt][4]
                block_pose.r.z = self.init_block_pose[env_idx][cnt][5]
                block_pose.r.w = self.init_block_pose[env_idx][cnt][6]

                # block_pose = gymapi.Transform()
                # block_pose.p.x = table_pose.p.x + self.rand_xy[i][cnt, 0]
                # block_pose.p.y = table_pose.p.y + self.rand_xy[i][cnt, 1]
                # block_pose.p.z = table_dims.z + self.block_height[0][cnt]


             
                # angle = np.random.uniform(-math.pi, math.pi)
                # r1 = R.from_euler('z', angle)
                # r2 = R.from_matrix(self.goal_pose[cnt][:3,:3])
                # # rot = r1 * r2
                # # euler = rot.as_euler("xyz", degrees=False)
                # euler = r2.as_euler("xyz", degrees=False)

                # block_pose.r = gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2])
            
                # block_pose=utils.mat2gymapi_transform(block_pos_world[cnt])
                block_handle = self.gym.create_actor(env_ptr, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i, 2 ** (cnt + 1), cnt + 2) # 010
                goal.append(torch.tensor((block_pose.p.x,block_pose.p.y,0.7,block_pose.r.x,block_pose.r.y,block_pose.r.z,block_pose.r.w)).to(self.device))
                goal.append(torch.tensor((block_pose.p.x,block_pose.p.y,block_pose.p.z,block_pose.r.x,block_pose.r.y,block_pose.r.z,block_pose.r.w)).to(self.device))
                goal.append(torch.tensor((block_pose.p.x,block_pose.p.y,0.7,block_pose.r.x,block_pose.r.y,block_pose.r.z,block_pose.r.w)).to(self.device))
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
        self.block_goal_list = torch.stack(_goal_list)
        
    
    def create_camera(self):
        # point camera at middle env
        
        side_cam_pos = gymapi.Vec3(4, 3, 2)
        side_cam_target = gymapi.Vec3(-4, -3, 0)

        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, side_cam_pos, side_cam_target)

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 640
        camera_properties.height = 480
        camera_properties.enable_tensors = True


        self.side_camera_handle_list = []

        for i in range(self.num_envs):

            # Set a fixed position and look-target for the first camera
            # position and target location are in the coordinate frame of the environment
            camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_properties)
            camera_position = gymapi.Vec3(0.7, 0.4, 0.8)
            camera_target = gymapi.Vec3(0, 0, 0)
            self.gym.set_camera_location(camera_handle, self.envs[i], camera_position, camera_target)
            self.side_camera_handle_list.append(camera_handle)

    def _create_image_directories(self):
        
        # create root path
        # os.makedirs(os.path.join('data',f'goal_{args.goal}',exist_ok=True))

        env_pth = os.path.join(self.img_pth, 'env_{}')
        
        # create path for each envs
        for i in self.success_envs:
            envid_str = str(i).zfill(5)

            self.img_pth_rgb = os.path.join(env_pth, 'hand_rgb')
            self.img_pth_depth = os.path.join(env_pth, 'hand_depth')
            self.img_pth_semantic = os.path.join(env_pth, 'hand_semantic')

            # create rgb, depth, semantic path
            os.mkdir(self.img_pth_rgb.format(envid_str))
            os.mkdir(self.img_pth_depth.format(envid_str))
            os.mkdir(self.img_pth_semantic.format(envid_str))

    
    def _write_images(self):

        for i,idx in enumerate(self.success_envs):
            if self.done_count[i]<100:
                img_rgb_pth = self.img_pth_rgb.format(str(idx).zfill(5))
                img_depth_pth = self.img_pth_depth.format(str(idx).zfill(5))
                img_semantic_pth = self.img_pth_semantic.format(str(idx).zfill(5))

                frame_id_str = str(self.frame_count//10).zfill(5)

                self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.side_camera_handle_list[i], gymapi.IMAGE_COLOR,
                    os.path.join(img_rgb_pth, 'frame_{}.png'.format(frame_id_str)))
                
                side_depth = self.gym.get_camera_image(self.sim, self.envs[i], self.side_camera_handle_list[i], gymapi.IMAGE_DEPTH)
                side_depth[side_depth == -np.inf] = 0
                np.save(os.path.join(img_depth_pth, 'frame_{}'.format(frame_id_str)), side_depth)

                side_semantic = self.gym.get_camera_image(self.sim, self.envs[i], self.side_camera_handle_list[i], gymapi.IMAGE_SEGMENTATION)

                np.save(os.path.join(img_semantic_pth, 'frame_{}'.format(frame_id_str)), side_semantic)

    
    def _orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:,:, 0:3] * torch.sign(q_r[:,:, 3]).unsqueeze(-1)

    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    
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
   
    def set_hand_prepick_pos(self,new_wrist_mat,idx):

        new_dof_state = self.dof_state.clone()
        # print(idx[0])
        # print(new_wrist_mat.shape)
        # exit()
  
        new_dof_state[idx[0], 0:3, 0] = new_wrist_mat[:, :3, 3]
        new_wrist_rot = pytorch3d.transforms.matrix_to_quaternion(new_wrist_mat[:,:3,:3])
        current_wrist_rot = pytorch3d.transforms.euler_angles_to_matrix(self.dof_state[idx[0],3:6,0],"XYZ")
        current_wrist_rot = pytorch3d.transforms.matrix_to_quaternion(current_wrist_rot)
        # new_dof_state[idx[0], 3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:, :3, :3], "XYZ")
        new_dof_state[idx[0],6:,0] = self.hand_goal_pose[self.stage[idx]//6,0,0,6:]
        # print(self.dof_state[:, :, 0])



        self.dof_state[:,0:3,0] = self.dof_state[:,0:3,0] + (new_dof_state[:,0:3,0] - self.dof_state[:,0:3,0])*0.04
        # self.dof_state[:,3:6,0] = self.dof_state[:,3:6,0] + pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix((new_wrist_rot-current_wrist_rot)*0.04),"XYZ")
        self.dof_state[idx[0],3:6,0] = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(utils.slerp(current_wrist_rot,new_wrist_rot,0.02)),"XYZ")

        self.dof_state[:,6:,0] = self.dof_state[:,6:,0] + (new_dof_state[:,6:,0] - self.dof_state[:,6:,0])*0.02
        # self.dof_state[reset,6:,0] = self.dof_state[reset,6:,0] + (0 - self.dof_state[reset,6:,0])*0.02



        
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
    
    def set_hand_pick_pos(self,new_wrist_mat,idx):

        new_dof_state = self.dof_state.clone()
        # print(idx[0])
        # print(new_wrist_mat.shape)
        # exit()
  
        new_dof_state[idx[0], 0:3, 0] = new_wrist_mat[:, :3, 3]
        new_wrist_rot = pytorch3d.transforms.matrix_to_quaternion(new_wrist_mat[:,:3,:3])
        current_wrist_rot = pytorch3d.transforms.euler_angles_to_matrix(self.dof_state[idx[0],3:6,0],"XYZ")
        current_wrist_rot = pytorch3d.transforms.matrix_to_quaternion(current_wrist_rot)
        # new_dof_state[idx[0], 3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:, :3, :3], "XYZ")
        new_dof_state[idx[0],6:,0] = self.hand_goal_pose[self.stage[idx]//6,0,0,6:]
        # print(self.dof_state[:, :, 0])



        self.dof_state[:,0:3,0] = self.dof_state[:,0:3,0] + (new_dof_state[:,0:3,0] - self.dof_state[:,0:3,0])*0.04
        # self.dof_state[:,3:6,0] = self.dof_state[:,3:6,0] + pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix((new_wrist_rot-current_wrist_rot)*0.04),"XYZ")
        self.dof_state[idx[0],3:6,0] = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(utils.slerp(current_wrist_rot,new_wrist_rot,0.02)),"XYZ")

        # self.dof_state[reset,6:,0] = self.dof_state[reset,6:,0] + (0 - self.dof_state[:,6:,0])*0.02
        self.dof_state[:,6:,0] = self.dof_state[:,6:,0] + (new_dof_state[:,6:,0] - self.dof_state[:,6:,0])*0.02

        
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

    def set_hand_placed_pos(self,new_wrist_mat,idx):

        new_dof_state = self.dof_state.clone()
        # print(idx[0])
        # print(new_wrist_mat.shape)
        # exit()
  
        new_dof_state[idx[0], 0:3, 0] = new_wrist_mat[:, :3, 3]
        new_wrist_rot = pytorch3d.transforms.matrix_to_quaternion(new_wrist_mat[:,:3,:3])
        current_wrist_rot = pytorch3d.transforms.euler_angles_to_matrix(self.dof_state[idx[0],3:6,0],"XYZ")
        current_wrist_rot = pytorch3d.transforms.matrix_to_quaternion(current_wrist_rot)
        # new_dof_state[idx[0], 3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:, :3, :3], "XYZ")
        # print(self.dof_state[:, :, 0])



        self.dof_state[:,0:3,0] = self.dof_state[:,0:3,0] + (new_dof_state[:,0:3,0] - self.dof_state[:,0:3,0])*0.04
        # self.dof_state[:,3:6,0] = self.dof_state[:,3:6,0] + pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix((new_wrist_rot-current_wrist_rot)*0.04),"XYZ")
        self.dof_state[idx[0],3:6,0] = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(utils.slerp(current_wrist_rot,new_wrist_rot,0.02)),"XYZ")

        # self.dof_state[reset,6:,0] = self.dof_state[reset,6:,0] + (0 - self.dof_state[:,6:,0])*0.02
        self.dof_state[:,6:,0] = self.dof_state[:,6:,0] + (0 - self.dof_state[:,6:,0])*0.002

        
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
        target_block_pose = self.block_goal_list[idx[0],self.stage[idx],:].clone()
        

        # self.root_state_tensor[self.block_indices[idx[0],self.stage[idx]//6], :7] += (target_block_pose[:,:7] - curr_block_pose[:,:7])*0.02
        self.root_state_tensor[self.block_indices[idx[0],self.stage[idx]//6], :3] += (target_block_pose[:,:3] - curr_block_pose[:,:3])*0.02
        self.root_state_tensor[self.block_indices[idx[0],self.stage[idx]//6], 3:7] = utils.slerp(curr_block_pose[:,3:7],target_block_pose[:,3:7],0.02)
        # print(target_block_pose[:,3:7].shape)
        # print(curr_block_pose[:,3:7].shape)
        # print(self.root_state_tensor[self.block_indices[idx[0],self.stage[idx]//6], 3:7].shape)
        # print(pytorch3d.transforms.matrix_to_quaternion(pytorch3d.transforms.euler_angles_to_matrix(self.orientation_error(target_block_pose[:,3:7],curr_block_pose[:,3:7]),"XYZ")).shape)
        # print(pytorch3d.transforms.matrix_to_quaternion(pytorch3d.transforms.euler_angles_to_matrix(self.orientation_error(target_block_pose[:,3:7],curr_block_pose[:,3:7]))).shape)
        # self.root_state_tensor[self.block_indices[idx[0],self.stage[idx]//6], 3:7] += pytorch3d.transforms.matrix_to_quaternion(pytorch3d.transforms.euler_angles_to_matrix(self.orientation_error(target_block_pose[:,3:7],curr_block_pose[:,3:7]),"ZYX"))*0.02
        
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
        


        return self.block_goal_list[torch.arange(0,self.num_envs,dtype=torch.long).to(self.device),self.stage,:].clone()

    def check_block_pos_reach(self,target_block_pose,block_threshold,idx):
        
        curr_block_pose = self.root_state_tensor[self.block_indices[torch.arange(0,self.num_envs,dtype=torch.long).to(self.device),self.stage//6], :7]
   
        pos_diff = torch.norm(curr_block_pose[:,:3]-target_block_pose[:,:3],dim=1)
        rot_diff = torch.norm(self.orientation_error(curr_block_pose[:,3:],target_block_pose[:,3:]),dim=1)
        z_diff = curr_block_pose[:,2]-target_block_pose[:,2]
        reach = torch.logical_and(torch.where(pos_diff<block_threshold,True,False),torch.where(rot_diff<0.05,True,False))
        reach = torch.logical_and(reach,idx)
        reach = torch.logical_and(reach,torch.where(z_diff<(block_threshold/5),True,False))
        # print(z_diff)
   
        # print((curr_block_pose-target_block_pose).shape)
        # print(diff.shape)
        # print(block_threshold.shape)
        # if diff < block_threshold:
        #     return True
        # else:
        #     return False
        # print(block_threshold.shape)
        # print(f'pos_diff: {pos_diff}')
        # print(f'rot_diff: {rot_diff}')
        # print(f'idx: {idx}')
        # print(f'z_diff: {torch.where(z_diff<0.005,True,False)}')
        # return torch.logical_and(torch.logical_and(torch.where(pos_diff<block_threshold,True,False),idx),torch.where(z_diff<0.005,True,False))
        # return torch.logical_and(torch.where(diff<block_threshold,True,False),idx)
        return reach
        
    def check_hand_pos_reach(self,new_dof_state,threshold,idx):
        # print(self.dof_state[:,:6,0].shape)
        # print(new_dof_state[:,:6,0].shape)
        diff = torch.norm(self.dof_state[:,:6,0]-new_dof_state[:,:6,0],dim=1)
        # hand_pose_diff = torch.norm(self.dof_state[:,6:,0]-new_dof_state[:,6:,0],dim=1)
        # print(hand_pose_diff)
        # print(diff.shape)
        # print(threshold.shape)
        # exit()
        # if diff < threshold:
        #     return True
        # else:
        #     return False
        return torch.logical_and(torch.where(diff<threshold,True,False),idx)
    

    def check_hand_pose_reach(self,new_dof_state,threshold,idx):
        diff = torch.norm(self.dof_state[:,:6,0]-new_dof_state[:,:6,0],dim=1)
        hand_pose_diff = torch.norm(self.dof_state[:,6:,0]-new_dof_state[:,6:,0],dim=1)
        # print(hand_pose_diff)
        return torch.logical_and(torch.logical_and(torch.where(diff<threshold,True,False),torch.where(hand_pose_diff<threshold,True,False)),idx)



        
    def reset_grasp_pose(self,envs):
        self.dof_state[envs,6:,0] = self.dof_state[envs,6:,0] + (0 - self.dof_state[envs,6:,0])*0.01
        
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
        # set_object = torch.logical_or(torch.logical_or(torch.where(self.stage%6==2,True,False),torch.where(self.stage%6==3,True,False)),torch.where(self.stage%6==4,True,False)).to(self.device)
        set_object = torch.logical_or(torch.where(self.stage%6==2,True,False),torch.where(self.stage%6==3,True,False)).to(self.device)
        set_object = torch.logical_or(set_object, torch.where(self.stage%6==4,True,False))

        set_hand_prepick = torch.where(self.stage%6==0,True,False).to(self.device)

        set_hand_pick = torch.where(self.stage%6==1,True,False).to(self.device)

        set_hand_placed = torch.where(self.stage%6==5,True,False).to(self.device)

        # if self.stage%6 == 4:
        #     block_threshold = 0.01
        # else:
        #     block_threshold = 0.1
        block_threshold = torch.where(self.stage%6==4,0.01,0.1).to(self.device)
        
        # if self.stage%6 == 1:
        #     threshold = 0.005
        # else:
        #     threshold = 0.1
        threshold = torch.logical_or(torch.where(self.stage%6==1,True,False),torch.where(self.stage%6==5,True,False))
        threshold = torch.where(threshold,0.01,0.05).to(self.device)

        # if self.stage%6 == 5:
        #     self.reset_grasp_pose()
        # reset_idx = torch.where(self.stage%6==5,True,False).to(self.device)
        
        # self.reset_grasp_pose(reset_idx)


        # if not set_object:
        #     goal_pose = self.goal_list[:,self.stage,:]

        #     cur_obj_mat = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
        #     cur_obj_mat[:,:3, 3] = goal_pose[:,:3]
        #     cur_obj_mat[:,:3, :3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:, [6,3,4,5]])

        #     target_hand_pose = cur_obj_mat @ self.hand_rel_mat

        #     new_target_dof = self.set_hand_pos(target_hand_pose)
    
        #     if self.check_hand_pos_reach(new_target_dof,threshold):
        #         self.stage+=1


        _idx = torch.logical_and(set_hand_prepick,torch.logical_not(self.done))

        idx = torch.where(_idx)

        if idx[0].shape[0]>0:
            # print(self.stage[idx])
            goal_pose = self.block_goal_list[idx[0],self.stage[idx],:]
            curr_obj_mat = torch.eye(4).unsqueeze(0).repeat(idx[0].shape[0],1,1).to(self.device)
            curr_obj_mat[:,:3,3] = goal_pose[:,:3]
            curr_obj_mat[:,:3,:3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:,[6,3,4,5]])
            target_hand_pose = curr_obj_mat @ self.hand_rel_mat[self.stage[idx]//6]
            new_target_dof = self.set_hand_prepick_pos(target_hand_pose,idx)
            _reach = self.check_hand_pos_reach(new_target_dof,threshold,_idx)
            self.stage[_reach]+=1

        


        # _idx = torch.logical_not(set_object)
        _idx = torch.logical_and(set_hand_pick,torch.logical_not(self.done))

        idx = torch.where(_idx)
        
        if idx[0].shape[0]>0:
            # print(self.stage[idx])
            goal_pose = self.block_goal_list[idx[0],self.stage[idx],:]
            curr_obj_mat = torch.eye(4).unsqueeze(0).repeat(idx[0].shape[0],1,1).to(self.device)
            curr_obj_mat[:,:3,3] = goal_pose[:,:3]
            curr_obj_mat[:,:3,:3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:,[6,3,4,5]])
            target_hand_pose = curr_obj_mat @ self.hand_rel_mat[self.stage[idx]//6]
            new_target_dof = self.set_hand_pick_pos(target_hand_pose,idx)
            # _reach = self.check_hand_pos_reach(new_target_dof,threshold,_idx)
            _hand_pose_reach = self.check_hand_pose_reach(new_target_dof,threshold,_idx)

            self.stage[_hand_pose_reach]+=1

        ###############################################################################################

        _idx = torch.logical_and(set_hand_placed,torch.logical_not(self.done))

        idx = torch.where(_idx)
        
        if idx[0].shape[0]>0:
            # print(self.stage[idx])
            goal_pose = self.block_goal_list[idx[0],self.stage[idx],:]
            curr_obj_mat = torch.eye(4).unsqueeze(0).repeat(idx[0].shape[0],1,1).to(self.device)
            curr_obj_mat[:,:3,3] = goal_pose[:,:3]
            curr_obj_mat[:,:3,:3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:,[6,3,4,5]])
            target_hand_pose = curr_obj_mat @ self.hand_rel_mat[self.stage[idx]//6]
            new_target_dof = self.set_hand_placed_pos(target_hand_pose,idx)
            # _reach = self.check_hand_pos_reach(new_target_dof,threshold,_idx)
            _reach = self.check_hand_pos_reach(new_target_dof,threshold,_idx)

            self.stage[_reach]+=1

        idx = torch.where(torch.logical_and(set_object,torch.logical_not(self.done)))


        if idx[0].shape[0]>0:
            target_block_pose = self.set_hand_object_pos(idx)
            goal_pose = self.root_state_tensor[self.block_indices[idx[0],self.stage[idx]//6],:7]
            cur_obj_mat = torch.eye(4).unsqueeze(0).repeat(idx[0].shape[0], 1, 1).to(self.device)
            cur_obj_mat[:,:3, 3] = goal_pose[:,:3]
            cur_obj_mat[:,:3, :3] = pytorch3d.transforms.quaternion_to_matrix(goal_pose[:, [6,3,4,5]])

            target_hand_pose = cur_obj_mat @ self.hand_rel_mat[self.stage[idx]//6]

            new_target_dof = self.set_hand_pick_pos(target_hand_pose,idx)

            _reach = self.check_block_pos_reach(target_block_pose,block_threshold,set_object)
            # print(_reach)
            # print(_reach)

            self.stage[_reach]+=1

        self.done = torch.where(self.stage==self.block_goal_list.shape[1]-1,True,False)

        
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
        # print(self.stage)
        

    def simulate(self):
        torch.set_printoptions(sci_mode=False)
        self.frame_count=0
        self.done = torch.zeros((self.num_envs),dtype=torch.long).to(self.device)
        self.set_init_hand_pos()
        self.done_count = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)

        # self.reset_idx()
        for _ in trange(2000):
            if ((self.viewer is None) or (not self.gym.query_viewer_has_closed(self.viewer))):          

                # step the physics
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)

                # print(self.root_state_tensor[self.block_indices[:,0], :7].shape)
                # print(self.block_indices)
                # print(self.step_size_list.shape)

                self.update()

                self.done_count[self.done]+=1

                if self.save:
                    if self.frame_count % 10 == 0 and self.frame_count != 0:
                        self._write_images()

                self.frame_count+=1


                
                # update the viewer
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                self.gym.sync_frame_time(self.sim)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)



if __name__ == "__main__":
    issac = ManoBlockAssembly()
    issac.simulate()
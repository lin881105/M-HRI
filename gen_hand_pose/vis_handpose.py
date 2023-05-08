"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

import os
import time
import yaml
import torch
import pickle
import numpy as np
import open3d as o3d
import pytorch3d.transforms
from tqdm import tqdm

class IsaacSim():
    def __init__(self):
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
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
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_franka_dofs, 2)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

    def create_sim(self):
        # parse arguments
        args = gymutil.parse_arguments(description="Joint control Methods Example")

        args.use_gpu = True
        args.use_gpu_pipeline = True
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_envs = 1

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
        self.gym.prepare_sim(self.sim)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(spacing, 0.75 * spacing, spacing)

        # create franka asset
        asset_root_franka = "../../../assets"
        asset_file_franka = "urdf/franka_description/robots/franka_panda.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        franka_asset = self.gym.load_asset(self.sim, asset_root_franka, asset_file_franka, asset_options)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        
        # create kit asset
        asset_root_kit = "../../../assets"
        asset_file_kit = "urdf/usb/factory_rectangular_hole_8mm.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        kit_asset = self.gym.load_asset(self.sim, asset_root_kit, asset_file_kit, asset_options)
        
        # create usb asset
        asset_root_usb = "../../assets"
        asset_file_usb = "urdf/usb/usb.urdf"

        # set default pose
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0, -0.7, 0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        usb_start_pose = gymapi.Transform()
        usb_start_pose.p = gymapi.Vec3(0.2, -0.125, 0.51)
        usb_start_pose.r = gymapi.Quat(0.4999998, 0.4999998, 0.4996018, 0.5003982)

        # cache some common handles for later use
        self.camera_handles = []
        self.franka_indices, self.kit_indices, self.table_indices = [], [], []
        self.envs = []

        # create and populate the environments
        for i in range(num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            # create table and set properties
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_sim_index = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_sim_index)

            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(60, 33, 0) / 255)
         
            # create kit and set properties
            kit_handle = self.gym.create_actor(env_ptr, kit_asset, kit_start_pose, "kit", i, 2, 1)
            kit_sim_index = self.gym.get_actor_index(env_ptr, kit_handle, gymapi.DOMAIN_SIM)
            self.kit_indices.append(kit_sim_index)

            # create usb and set properties
            usb_handle = self.gym.create_actor(env_ptr, usb_asset, usb_start_pose, 'usb', i, 4, 2)

            self.gym.set_rigid_body_color(env_ptr, usb_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))
            
            # create franka and set properties
            franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 8, 3)
            franka_sim_index = self.gym.get_actor_index(env_ptr, franka_handle, gymapi.DOMAIN_SIM)
            self.franka_indices.append(franka_sim_index)

            self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)

            # create camera and set properties
            camera_handle_side = self.gym.create_camera_sensor(env_ptr, camera_properties)
            camera_position = gymapi.Vec3(0.7, 0, 0.9)
            camera_target = gymapi.Vec3(0, 0, 0.5)
            self.gym.set_camera_location(camera_handle_side, env_ptr, camera_position, camera_target)
            self.camera_handles.append(camera_handle_side)

            camera_handle_robot = self.gym.create_camera_sensor(env_ptr, camera_properties)
            link7_rb = self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, 'panda_hand')
            self.gym.attach_camera_to_body(camera_handle_robot, env_ptr, link7_rb, camera_rel_pose, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles.append(camera_handle_robot)
                
        self.franka_indices = to_torch(self.franka_indices, dtype=torch.long, device=self.device)
        self.kit_indices = to_torch(self.kit_indices, dtype=torch.long, device=self.device)

    def reset(self):
        franka_init_pose = torch.tensor([[1.4561, 0.0723, 0.0594, -0.9704, 0.0594, 1.3842, 1.0995, 0, 0]])
        franka_init_pose = to_torch(franka_init_pose, dtype=torch.float32, device=self.device)
        self.dof_state[:, :, 0] = franka_init_pose
        
        dof_indices = self.franka_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(franka_init_pose),
                                                        gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        
        self.frame = 0

    def simulate(self):
        self.reset()

        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.refresh_dof_state_tensor(self.sim)
            # self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            # self.gym.refresh_net_contact_force_tensor(self.sim)

            # write image
            if self.frame % 500 == 0:
                for i in range(len(self.camera_handles)):
                    self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handles[i], gymapi.IMAGE_COLOR,
                        os.path.join('franka_img', str(i), 'frame_{}.png'.format(self.frame)))

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

            self.frame += 1

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        
if __name__ == "__main__":
    issac = IsaacSim()
    issac.simulate()
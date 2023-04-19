from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time
import scipy.io as sio
from utils import utils
from scipy.spatial.transform import Rotation as R

custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
    {"name": "--headless", "action": "store_true", "help": "Run headless"},
]
args = gymutil.parse_arguments(
    description="Franka block assembly demonstration",
    custom_parameters=custom_parameters,
)



class FrankaBlockAssembly():
    def __init__(self):
        
        # initialize gym
        self.gym = gymapi.acquire_gym()
        self.num_envs = args.num_envs
        
        # IK params
        self.damping = 0.05
        
        self.asset_root = "assets"
        
        # create simulator
    
    def create_sim(self):
        # set torch device
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
            raise Exception("This example can only be used with PhysX")
        # create sim
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        use_viewer = not args.headless
        if use_viewer:
            viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if viewer is None:
                raise Exception("Failed to create viewer")
        else:
            viewer = None
    
    def _create_envs(self):
        
        ################
        # creat assets #
        ################
        
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

        # load block asset
        block_asset_list = []
        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        block_type = ['A.urdf', 'B.urdf', 'C.urdf', 'D.urdf', 'E.urdf']
        for t in block_type:
            block_asset_list.append(self.gym.load_asset(self.sim, self.asset_root, 'urdf/block_assembly/block_' + t, asset_options))

        # load franka asset
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        franka_asset = self.gym.load_asset(self.sim, self.asset_root, franka_asset_file, asset_options)

        # configure franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        franka_lower_limits = franka_dof_props["lower"]
        franka_upper_limits = franka_dof_props["upper"]
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

        # use position drive for all dofs
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:7].fill(100.0)
        franka_dof_props["damping"][:7].fill(40.0)
        
        # grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(800.0)
        franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        franka_num_dofs = self.gym.get_asset_dof_count(franka_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]
        # grippers open
        default_dof_pos[7:] = franka_upper_limits[7:]

        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # send to torch
        default_dof_pos_tensor = to_torch(default_dof_pos, device=self.device)

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        franka_hand_index = franka_link_dict["panda_hand"]

        # configure env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        
        print("Creating %d environments" % self.num_envs)
        
        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.4, 0.0, 0.5 * table_dims.z)

        region_pose = gymapi.Transform()
        
        self.envs = []
        self.block_idxs_list = []
        self.hand_idxs = []
        self.init_pos_list = []
        self.init_rot_list = []
        
        for i in range(self.num_envs):

            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add table
            table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i,0)

            # add region
            # region_x = np.random.uniform(-0.1,0.1)
            # region_y = np.random.uniform(-0.1,0.1)

            region_pose.p.x = table_pose.p.x + region_xy[0]
            region_pose.p.y = table_pose.p.y + region_xy[1]
            region_pose.p.z = table_dims.z #+ 0.001
            region_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            
            region_pose_mat = utils.gymapi_transform2mat(region_pose)

            black = gymapi.Vec3(0.,0.,0.)
            region_handle = gym.create_actor(env, region_asset, region_pose, "target", i, 1)
            gym.set_rigid_body_color(env, region_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, black)

            # add box
            block_handles = []
            block_idxs = []


            for j, idx in enumerate(self.goal_list):

                block_pose = gymapi.Transform()
                block_pose.p.x = table_pose.p.x + rand_xy[j,0]
                block_pose.p.y = table_pose.p.y + rand_xy[j,1]
                block_pose.p.z = table_dims.z + block_height[j]

                r1 = R.from_euler('z', np.random.uniform(-math.pi, math.pi))
                r2 = R.from_matrix(goal_pose[j][:3,:3])
                rot = r1 * r2
                euler = rot.as_euler("xyz", degrees=False)

                block_pose.r = gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2])
                # block_pose=utils.mat2gymapi_transform(block_pos_world[j])
                block_handle = gym.create_actor(env, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i)
                block_handles.append(block_handle)


                # block_pose = utils.mat2gymapi_transform(utils.gymapi_transform2mat(region_pose)@goal_pose[j])
                # block_handle = gym.create_actor(env, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i+1)
                # block_handles.append(block_handle)

                color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                gym.set_rigid_body_color(env, block_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                block_idx = gym.get_actor_rigid_body_index(env, block_handle, 0, gymapi.DOMAIN_SIM)
                block_idxs.append(block_idx)

                block_idxs_list.append(block_idxs)

                # # get global index of box in rigid body state tensor
                # box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
                # block_idxs.append(box_idx)

                # add franka
                franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)

                # set dof properties
                gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

                # set initial dof states
                gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

                # set initial position targets
                gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

                # get inital hand pose
                hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
                hand_pose = gym.get_rigid_transform(env, hand_handle)
                init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
                init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

                # get global index of hand in rigid body state tensor
                hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
                hand_idxs.append(hand_idx)
                    
        
    
    
    def load_goal_data(self):
        mat_file = "goal/block_assembly/goal_A_data.mat"
        mat_dict = sio.loadmat(mat_file)

        self.goal_list = mat_dict["block_list"][0]
        self.goal_pose = mat_dict["block_pose"]
        self.rel_pick_pos = mat_dict["pick_pose"]
        self.rel_place_pos = mat_dict["place_pose"]
        self.block_pos_world = mat_dict["block_world"]
        self.block_height = mat_dict["block_height"]
        
    def check_in_region(region_xy, rand_xy):
        for i in range(rand_xy.shape[0]):
            if np.linalg.norm(region_xy - rand_xy[i]) < 0.08:
                return True
        
        return False

    def check_contact_block(rand_xy):
        for i in range(rand_xy.shape[0]):
            for j in range(i+1, rand_xy.shape[0]):
                if np.linalg.norm(rand_xy[i] - rand_xy[j]) < 0.08:
                    return True
                
        return False
    
    def generate_pose(self):

        region_xy = np.random.uniform([-0.085, -0.085], [0.085, 0.085], 2)

        while True:
            rand_xy = np.random.uniform([-0.13, -0.23], [0.13, 0.23], (3, 2))
            
            if self.check_in_region(region_xy, rand_xy) or self.check_contact_block(rand_xy):
                continue
            else:
                break
            
        print("success generate initial pos!!!")
                
        
        
                
            

            
    

        
        
        

        
        
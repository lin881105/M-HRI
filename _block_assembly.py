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
        self.create_sim()

        self.load_goal_data()

        self.get_pp_pose_tensor()

        self._create_envs()

        self.create_camera()

        self.stage = 0


    
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
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
        else:
            self.viewer = None
    
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
        self.franka_lower_limits = franka_dof_props["lower"]
        self.franka_upper_limits = franka_dof_props["upper"]
        franka_ranges = self.franka_upper_limits - self.franka_lower_limits
        franka_mids = 0.3 * (self.franka_upper_limits + self.franka_lower_limits)

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
        self.num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        self.env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        
        print("Creating %d environments" % self.num_envs)
        
        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.4, 0.0, 0.5 * table_dims.z)

        region_pose = gymapi.Transform()
        
        self.envs = []
        self.block_idxs_list = [[] for _ in range(self.num_envs)]
        self.hand_idxs = []
        self.init_pos_list = []
        self.init_rot_list = []
        self.block_handles_list = [ [] for _ in range(self.num_envs)]
        self.region_pose_list = []
        
        for i in range(self.num_envs):

            # create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)

            # add table
            table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i,0)

            # add region
            region_pose.p.x = table_pose.p.x + self.region_xy[0]
            region_pose.p.y = table_pose.p.y + self.region_xy[1]
            region_pose.p.z = table_dims.z #+ 0.001
            region_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            
            # self.region_pose_mat = utils.gymapi_transform2mat(region_pose)
            self.region_pose_list.append(region_pose)

            black = gymapi.Vec3(0.,0.,0.)
            region_handle = self.gym.create_actor(env, region_asset, region_pose, "target", i, 1)
            self.gym.set_rigid_body_color(env, region_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, black)

            # add box

            for j, idx in enumerate(self.goal_list):

                block_pose = gymapi.Transform()
                block_pose.p.x = table_pose.p.x + self.rand_xy[j,0]
                block_pose.p.y = table_pose.p.y + self.rand_xy[j,1]
                block_pose.p.z = table_dims.z + self.block_height[j]

                r1 = R.from_euler('z', np.random.uniform(-math.pi, math.pi))
                r2 = R.from_matrix(self.goal_pose[j][:3,:3])
                rot = r1 * r2
                euler = rot.as_euler("xyz", degrees=False)

                block_pose.r = gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2])
                # block_pose=utils.mat2gymapi_transform(block_pos_world[j])
                block_handle = self.gym.create_actor(env, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i)
                self.block_handles_list[i].append(block_handle)

                color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                self.gym.set_rigid_body_color(env, block_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                block_idx = self.gym.get_actor_rigid_body_index(env, block_handle, 0, gymapi.DOMAIN_SIM)
                self.block_idxs_list[i].append(block_idx)

            # # get global index of box in rigid body state tensor
            # box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
            # block_idxs.append(box_idx)

            # add franka
            franka_handle = self.gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)

            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)


        # create observation buffer

        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, franka_hand_index - 1, :, :7]
                    
        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = dof_states[:, 0].view(self.num_envs, 9, 1)
        self.dof_vel = dof_states[:, 1].view(self.num_envs, 9, 1)

        # Create a tensor noting whether the hand should return to the initial position
        hand_restart = torch.full([self.num_envs], False, dtype=torch.bool).to(self.device)

        # Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)

    
    
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

        self.region_xy = np.random.uniform([-0.085, -0.085], [0.085, 0.085], 2)

        while True:
            self.rand_xy = np.random.uniform([-0.13, -0.23], [0.13, 0.23], (3, 2))
            
            if self.check_in_region(self.region_xy, self.rand_xy) or self.check_contact_block(self.rand_xy):
                continue
            else:
                break
            
        print("success generate initial pos!!!")
                
    def get_pp_pose_tensor(self):
        
        
        self.goal_pick_pos_list = [[] for _ in range(self.num_envs)]
        self.goal_pick_rot_list = [[] for _ in range(self.num_envs)]
        self.goal_prepick_pos_list = [[] for _ in range(self.num_envs)]
        self.goal_place_pos_list = [[] for _ in range(self.num_envs)]
        self.goal_place_rot_list = [[] for _ in range(self.num_envs)]
        self.goal_preplace_pos_list = [[] for _ in range(self.num_envs)]
        
        for i in range(self.num_envs):
            for j in range(self.goal_list.shape[0]):
                body_states = self.gym.get_actor_rigid_body_states(self.envs[i], self.block_handles_list[i][j], gymapi.STATE_ALL)

                tmp_mat = np.eye(4)
                tmp_mat[:3, :3] = R.from_quat(np.array([body_states["pose"]["r"]["x"],
                                                        body_states["pose"]["r"]["y"],
                                                        body_states["pose"]["r"]["z"],
                                                        body_states["pose"]["r"]["w"]]).reshape(-1)).as_matrix()
                
                tmp_mat[:3, 3] = np.array([body_states["pose"]["p"]["x"], body_states["pose"]["p"]["y"], body_states["pose"]["p"]["z"]]).reshape(-1)

                place_mat = utils.gymapi_transform2mat(self.region_pose_list[i]) @ self.rel_place_pos[i]
                pick_mat = tmp_mat @ self.rel_pick_pos[i]
                
                goal_pick_pose = utils.mat2gymapi_transform(pick_mat)
                goal_place_pose = utils.mat2gymapi_transform(place_mat)


                goal_prepick_pos = torch.Tensor((goal_pick_pose.p.x,goal_pick_pose.p.y,0.7)).reshape(1, 3).to(self.device)
                goal_pick_pos = torch.Tensor((goal_pick_pose.p.x,goal_pick_pose.p.y,goal_pick_pose.p.z)).reshape(1, 3).to(self.device)
                goal_pick_rot = torch.Tensor((goal_pick_pose.r.x,goal_pick_pose.r.y,goal_pick_pose.r.z,goal_pick_pose.r.w)).reshape(1, 4).to(self.device)

                goal_preplace_pos = torch.Tensor((goal_place_pose.p.x,goal_place_pose.p.y,0.7)).reshape(1, 3).to(self.device)
                goal_place_pos = torch.Tensor((goal_place_pose.p.x,goal_place_pose.p.y,goal_place_pose.p.z)).reshape(1, 3).to(self.device)
                goal_place_rot = torch.Tensor((goal_place_pose.r.x,goal_place_pose.r.y,goal_place_pose.r.z,goal_place_pose.r.w)).reshape(1, 4).to(self.device)

                self.goal_pick_pos_list[i].append(goal_pick_pos)
                self.goal_pick_rot_list[i].append(goal_pick_rot)
                self.goal_prepick_pos_list[i].append(goal_prepick_pos)
                self.goal_place_pos_list[i].append(goal_place_pos)
                self.goal_place_rot_list[i].append(goal_place_rot)
                self.goal_preplace_pos_list[i].append(goal_preplace_pos)

    def create_camera(self):
        # point camera at middle env
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)

        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 640
        camera_properties.height = 480

        for i in range(self.envs):

            # Set a fixed position and look-target for the first camera
            # position and target location are in the coordinate frame of the environment
            camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_properties)
            camera_position = gymapi.Vec3(1, 1, 1.0)
            camera_target = gymapi.Vec3(0, 0, 0)
            self.gym.set_camera_location(camera_handle, self.envs[i], camera_position, camera_target)
                
    def quat_axis(q, axis=0):
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)


    def orientation_error(desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


    def control_ik(self, dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u
            
    def simulate(self):
        img = []
        frame_count=0
        step = 0
        to_prepick = False
        to_pick = False
        to_preplace = False
        to_place = False
        picked = False
        placed = False
        max_step = len(self.goal_list)
        op = True

        pick_counter = 0
        place_counter = 0
        to_place_counter = 0
        picked_counter = 0
        placed_counter = 0
        # simulation loop
        while self.viewer is None or not self.gym.query_viewer_has_closed(self.viewer):

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            self.gym.render_all_camera_sensors(self.sim)

            # refresh tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)

            # block_pos = self.rb_states[self.block_idxs_list, :3]
            # block_rot = self.rb_states[self.block_idxs_list, 3:7]
            # print(block_pos[:,:,0])

            hand_pos = self.rb_states[self.hand_idxs, :3]
            hand_rot = self.rb_states[self.hand_idxs, 3:7]
            # hand_vel = self.rb_states[self.hand_idxs, 7:]

            gripper_open = torch.Tensor(self.franka_upper_limits[7:]).to(self.device)
            gripper_close = torch.Tensor(self.franka_lower_limits[7:]).to(self.device)



            if step < max_step:

                # print(step)
                
                if torch.norm(self.goal_prepick_pos_list[:,step] - hand_pos) < 0.001 and torch.norm(self.orientation_error(self.goal_pick_rot_list[:,step],hand_rot))< 0.05 and not picked:
                    to_prepick = True
                else:
                    goal_pos = self.goal_prepick_pos_list[:,step]
                    goal_rot = self.goal_pick_rot_list[:,step]
                
                    
                if to_prepick:
                    goal_pos = self.goal_pick_pos_list[:,step]
                    goal_rot = self.goal_pick_rot_list[:,step]

                

                if torch.norm(self.goal_pick_pos_list[:,step] - hand_pos) < 0.001 and torch.norm(self.orientation_error(self.goal_pick_rot_list[:,step],hand_rot))< 0.05 and to_prepick:
                    to_pick = True

                
                if to_pick:
                    self.pos_action[:,7:9] = gripper_close
                    pick_counter += 1
                    
                    if pick_counter >= 20:
                        picked = True
                
                if picked:
                    goal_pos = self.goal_prepick_pos_list[:,step]
                    goal_rot = self.goal_pick_rot_list[:,step]
                    picked_counter += 1
                    if picked_counter>=30:
                        goal_pos = self.goal_preplace_pos_list[:,step]
                        goal_rot = self.goal_place_rot_list[:,step]



                # print("pos: ",torch.norm(goal_preplace_pos - hand_pos))
                # print("rot: ", torch.norm(orientation_error(goal_place_rot, hand_rot)))
                    
                if torch.norm(goal_preplace_pos_list[step] - hand_pos) < 0.005 and torch.norm(orientation_error(goal_place_rot_list[step], hand_rot)) < 0.05 and picked and not placed:
                    to_preplace = True
                    
                if to_preplace:
                    goal_pos = goal_place_pos_list[step]
                    goal_rot = goal_place_rot_list[step]
                    print(torch.norm(goal_place_pos_list[step] - hand_pos))
                    print(torch.norm(orientation_error(goal_place_rot_list[step], hand_rot)))
                # print(torch.norm(goal_place_pos - hand_pos), torch.norm(orientation_error(goal_place_rot,hand_rot)))
                if torch.norm(goal_place_pos_list[step] - hand_pos) < 0.02 and torch.norm(orientation_error(goal_place_rot_list[step], hand_rot)) < 0.05 and to_preplace:
                    to_place_counter+=1
                    if to_place_counter>=30:
                        to_place = True

                if to_place:
                    pos_action[:,7:9] = gripper_open
                    place_counter+=1
                    if place_counter >= 30:
                        placed = True
                
                if placed:
                    print("placed")
                    pos_action[:,7:9] = gripper_open
                    
                    goal_pos = goal_preplace_pos_list[step]
                    goal_rot = goal_place_rot_list[step]
                    placed_counter += 1
                    if placed_counter>=30:
                        pick_counter = 0
                        place_counter = 0
                        to_place_counter = 0
                        step+=1
                        to_prepick = False
                        to_pick = False
                        to_preplace = False
                        to_place = False
                        picked = False
                        placed = False
                        placed_counter = 0
                        picked_counter = 0

                # compute position and orientation error
            
            if op:
                pos_action[:,7:9] = gripper_open
                op = False
            pos_err = goal_pos - hand_pos
            orn_err = orientation_error(goal_rot, hand_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

            # print(dpose)

            # Deploy control based on type
            if controller == "ik":
                pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
            else:       # osc
                effort_action[:, :7] = control_osc(dpose)
            
            
            # Deploy actions
            
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
            # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

            rgb_filename = "output/RGB/%3d.png" % (frame_count)
            depth_filename = "output/DEPTH/%3d.png" % (frame_count)

            frame_count += 1
            gym.write_camera_image_to_file(sim, envs[0], camera_handle, gymapi.IMAGE_COLOR, rgb_filename)
            gym.write_camera_image_to_file(sim, envs[0], camera_handle, gymapi.IMAGE_DEPTH, depth_filename)

            # update viewer
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)

        # cleanup
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)



        pass
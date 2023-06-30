from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import os
import datetime

import math
import numpy as np
import torch
import time
import scipy.io as sio
from utils import utils
from scipy.spatial.transform import Rotation as R
import shutil
from tqdm import trange
import scipy.io as sio


# custom_parameters = [
#     {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
#     {"name": "--headless", "action": "store_true", "help": "Run headless"},
#     {"name": "--goal", "type": str, "default":'1',"help": ""},
#     {"name": "--save", "action": "store_true"},
# ]
# self.args = gymutil.parse_arguments(
#     description="Franka peg assembly demonstration",
#     custom_parameters=custom_parameters,
# )



class FrankaPegInsertion():

    def __init__(self,args):
        
        # initialize gym
        self.args = args
        self.gym = gymapi.acquire_gym()
        self.num_envs = self.args.num_envs
        
        # IK params
        self.damping = 0.05
        
        self.asset_root = "assets"
        
        
        # create simulator
        self.create_sim()
        
        self._create_image_directories()        
        
        self.load_goal_data()

        self.generate_pose()

        self._create_envs()

        self.create_camera()

        self.get_pp_pose_tensor()
        
        self.stage = 0
    
    def create_sim(self):
        # set torch device
        self.device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")
        # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        use_viewer = not self.args.headless
        if use_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
        else:
            self.viewer = None
    
    def _create_envs(self):

        # ceate ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
        # create table asset
        table_dims = gymapi.Vec3(0.4, 0.6, 0.4)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # create kit assets
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        kit_asset = self.gym.load_asset(self.sim, self.asset_root, 'urdf/peg_insertion/kit.urdf',asset_options)

        # create peg assets
        peg_asset_list = []
        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        # asset_options.flip_visual_attachments = True
        peg_type = ['rectangle', 'square', 'hexagon', 'triangle']
        for t in peg_type:
            peg_asset_list.append(self.gym.load_asset(self.sim, self.asset_root, 'urdf/peg_insertion/' + t + '.urdf', asset_options))


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
        default_dof_pos[7:] = self.franka_upper_limits[7:]

        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # send to torch
        # default_dof_pos_tensor = to_torch(default_dof_pos, device=self.device)

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

        kit_pose = gymapi.Transform()
        
        self.envs = []
        self.peg_idxs_list = [[] for _ in range(self.num_envs)]
        self.hand_idxs = []
        self.init_pos_list = []
        self.init_rot_list = []
        self.peg_handles_list = [ [] for _ in range(self.num_envs)]
        self.kit_pose_list = []
        self.hand_handle_list = []
        self.kit_init_pose_list = []
        self.peg_init_pose_list = [[] for _ in range(self.num_envs)]
        self.peg_color = [[] for _ in range(self.num_envs)]

        for i in range(self.num_envs):

            # create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)

            # add table
            table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i, 0, 1)

            # add kit
            kit_pose.p.x = table_pose.p.x + self.kit_xy[i][0]
            kit_pose.p.y = table_pose.p.y + self.kit_xy[i][1]
            kit_pose.p.z = table_dims.z #+ 0.001
            kit_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi/2, math.pi/2))
            
            kit_pose_mat = utils.gymapi_transform2mat(kit_pose)
            self.kit_pose_list.append(kit_pose_mat)

            self.kit_init_pose_list.append(np.array((kit_pose.p.x,kit_pose.p.y,kit_pose.p.z,
                                              kit_pose.r.x,kit_pose.r.y,kit_pose.r.z,kit_pose.r.w)))

            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            kit_handle = self.gym.create_actor(env, kit_asset, kit_pose, "target", i, 1, 2)
            self.gym.set_rigid_body_color(env, kit_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # add box

            for j, idx in enumerate(self.goal_list):

                peg_pose = gymapi.Transform()
                peg_pose.p.x = table_pose.p.x + self.rand_xy[i][j][0]
                peg_pose.p.y = table_pose.p.y + self.rand_xy[i][j][1]
                peg_pose.p.z = table_dims.z + self.peg_height[0][j]

                r1 = R.from_euler('z', np.random.uniform(-math.pi/2, math.pi/2))
                r2 = R.from_matrix(self.goal_pose[j][:3,:3])
                rot = r1 * r2
                euler = rot.as_euler("xyz", degrees=False)

                peg_pose.r = gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2])

                peg_handle = self.gym.create_actor(env, peg_asset_list[idx], peg_pose, 'peg_' + peg_type[idx], i, 2 ** (j + 2), j + 3)
                self.peg_handles_list[i].append(peg_handle)

                color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                self.gym.set_rigid_body_color(env, peg_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                peg_idx = self.gym.get_actor_rigid_body_index(env, peg_handle, 0, gymapi.DOMAIN_SIM)
                self.peg_idxs_list[i].append(peg_idx)

                self.peg_init_pose_list[i].append(np.array((peg_pose.p.x,peg_pose.p.y,peg_pose.p.z,
                                                             peg_pose.r.x,peg_pose.r.y,peg_pose.r.z,peg_pose.r.w)))
                self.peg_color[i].append(color)
            
            # init_pose = {
            #     "peg_init_pose_world":peg_init_pose_list,
            #     "kit_init_pose_world":kit_init_pose,
            # }

            # sio.savemat(f"data/goal_{self.args.goal}/{self.time_str}/env_{str(i).zfill(5)}/init_pose.mat",init_pose)


            # # get global index of box in rigid body state tensor
            # box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
            # peg_idxs.append(box_idx)

            # add franka
            franka_handle = self.gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2 ** (2 + len(self.goal_list)), 7)

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
            self.hand_handle_list.append(hand_handle)

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
        # hand_restart = torch.full([self.num_envs], False, dtype=torch.bool).to(self.device)

        # Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
    
    def load_goal_data(self):
        mat_file = f"goal/peg_insertion/goal_data/goal_{self.args.goal}_data.mat"
        mat_dict = sio.loadmat(mat_file)

        self.goal_list = mat_dict["peg_list"][0]
        self.goal_pose = mat_dict["peg_pose"]
        self.rel_pick_pos = mat_dict["pick_pose"]
        self.rel_place_pos = mat_dict["place_pose"]
        self.peg_height = mat_dict["peg_height"]
        
    def check_in_kit(self, kit_xy, rand_xy):
        # for j in range(rand_xy.shape[0]):
        #     for i in range(rand_xy.shape[1]):
        #         if np.linalg.norm(kit_xy[j] - rand_xy[j][i]) < 0.08:
        #             return True
        reset_idx = torch.where(torch.norm(kit_xy.unsqueeze(1).repeat(1,len(self.goal_list),1) - rand_xy,dim=2)<0.08,True,False)
        # reset_idx = torch.logical_not(_diff).to(self.device)
   
        # reset_idx = torch.logical_not(_diff).to(self.device)
        
        return reset_idx

    def check_contact_peg(self,rand_xy):
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

        # self.kit_xy = np.random.uniform([-0.085, -0.085], [0.085, 0.085], (self.num_envs,2))
        self.kit_xy = torch.FloatTensor(self.num_envs,2).uniform_(-0.085,0.085).to(self.device)
        self.rand_xy = torch.stack((torch.FloatTensor(self.num_envs,len(self.goal_list)).uniform_(-0.13,0.13),
                                    torch.FloatTensor(self.num_envs,len(self.goal_list)).uniform_(-0.23,0.23)),dim=2).to(self.device)

        while True:
            # self.rand_xy = np.random.uniform([-0.13, -0.23], [0.13, 0.23], (self.num_envs,len(self.goal_list), 2))
            kit_reset_idx = self.check_in_kit(self.kit_xy, self.rand_xy)
            peg_reset_idx = self.check_contact_peg(self.rand_xy)
            # print(kit_reset_idx)
            # print(peg_reset_idx)

            reset_idx = torch.logical_or(kit_reset_idx,peg_reset_idx).to(self.device)

            if torch.all(torch.logical_not(reset_idx)):
                break
            else:
                # print(reset_idx)

                # print(self.rand_xy[reset_idx, :].shape)
                # print(torch.sum(torch.any(reset_idx, dim=1)))
                self.rand_xy[torch.any(reset_idx,dim=1),:, :] = torch.stack((torch.FloatTensor(torch.sum(torch.any(reset_idx, dim=1)), len(self.goal_list)).uniform_(-0.13,0.13),
                                                         torch.FloatTensor(torch.sum(torch.any(reset_idx, dim=1)), len(self.goal_list)).uniform_(-0.23,0.23)),dim=2).to(self.device)
            
        print("success generate initial pos!!!")
                
    def get_pp_pose_tensor(self):
        
        goal_pick_pos_list = [[] for _ in range(self.num_envs)]
        goal_pick_rot_list = [[] for _ in range(self.num_envs)]
        goal_prepick_pos_list = [[] for _ in range(self.num_envs)]
        goal_place_pos_list = [[] for _ in range(self.num_envs)]
        goal_place_rot_list = [[] for _ in range(self.num_envs)]
        goal_preplace_pos_list = [[] for _ in range(self.num_envs)]
        goal_pose_list = [[] for _ in range(self.num_envs)]
        
        for i in range(self.num_envs):
            for j in range(self.goal_list.shape[0]):
                body_states = self.gym.get_actor_rigid_body_states(self.envs[i], self.peg_handles_list[i][j], gymapi.STATE_ALL)

                tmp_mat = np.eye(4)
                tmp_mat[:3, :3] = R.from_quat(np.array([body_states["pose"]["r"]["x"],
                                                        body_states["pose"]["r"]["y"],
                                                        body_states["pose"]["r"]["z"],
                                                        body_states["pose"]["r"]["w"]]).reshape(-1)).as_matrix()
                
                tmp_mat[:3, 3] = np.array([body_states["pose"]["p"]["x"], body_states["pose"]["p"]["y"], body_states["pose"]["p"]["z"]]).reshape(-1)

                place_mat = self.kit_pose_list[i] @ self.rel_place_pos[j]
                pick_mat = tmp_mat @ self.rel_pick_pos[j]
                
                goal_pick_pose = utils.mat2gymapi_transform(pick_mat)
                goal_place_pose = utils.mat2gymapi_transform(place_mat)
                goal_pose_world = self.kit_pose_list[i] @ self.goal_pose[j]
                goal_pose_world = utils.mat2gymapi_transform(goal_pose_world)

                goal_prepick_pos = torch.Tensor((goal_pick_pose.p.x,goal_pick_pose.p.y,0.71)).reshape(1, 3).to(self.device)
                goal_pick_pos = torch.Tensor((goal_pick_pose.p.x,goal_pick_pose.p.y,goal_pick_pose.p.z)).reshape(1, 3).to(self.device)
                goal_pick_rot = torch.Tensor((goal_pick_pose.r.x,goal_pick_pose.r.y,goal_pick_pose.r.z,goal_pick_pose.r.w)).reshape(1, 4).to(self.device)

                goal_preplace_pos = torch.Tensor((goal_place_pose.p.x,goal_place_pose.p.y,0.71)).reshape(1, 3).to(self.device)
                goal_place_pos = torch.Tensor((goal_place_pose.p.x,goal_place_pose.p.y,goal_place_pose.p.z)).reshape(1, 3).to(self.device)
                goal_place_rot = torch.Tensor((goal_place_pose.r.x,goal_place_pose.r.y,goal_place_pose.r.z,goal_place_pose.r.w)).reshape(1, 4).to(self.device)

                goal_pose_world = torch.Tensor((goal_pose_world.p.x,goal_pose_world.p.y,goal_pose_world.p.z,
                                                goal_pose_world.r.x,goal_pose_world.r.y,goal_pose_world.r.z,goal_pose_world.r.w)).reshape(1,7).to(self.device)
                
                goal_pick_pos_list[i].append(goal_pick_pos)
                goal_pick_rot_list[i].append(goal_pick_rot)
                goal_prepick_pos_list[i].append(goal_prepick_pos)
                goal_place_pos_list[i].append(goal_place_pos)
                goal_place_rot_list[i].append(goal_place_rot)            # if self.frame_count % 10 == 0:
            #     self._write_images()

            # self.frame_count+=1
                goal_preplace_pos_list[i].append(goal_preplace_pos)
                goal_pose_list[i].append(goal_pose_world)
        
        self.goal_pick_pos_list = torch.stack([torch.stack(goal_pick_pos_list[i]).squeeze(1).to(self.device) for i in range(self.num_envs)]).to(self.device)
        self.goal_pick_rot_list = torch.stack([torch.stack(goal_pick_rot_list[i]).squeeze(1).to(self.device) for i in range(self.num_envs)]).to(self.device)
        self.goal_prepick_pos_list = torch.stack([torch.stack(goal_prepick_pos_list[i]).squeeze(1).to(self.device) for i in range(self.num_envs)]).to(self.device)
        self.goal_place_pos_list = torch.stack([torch.stack(goal_place_pos_list[i]).squeeze(1).to(self.device) for i in range(self.num_envs)]).to(self.device)
        self.goal_place_rot_list = torch.stack([torch.stack(goal_place_rot_list[i]).squeeze(1).to(self.device) for i in range(self.num_envs)]).to(self.device)
        self.goal_preplace_pos_list = torch.stack([torch.stack(goal_preplace_pos_list[i]).squeeze(1).to(self.device) for i in range(self.num_envs)]).to(self.device)
        self.goal_pose_list = torch.stack([torch.stack(goal_pose_list[i]).squeeze(1).to(self.device) for i in range(self.num_envs)]).to(self.device)

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

        inHand_camera_properties = gymapi.CameraProperties()
        inHand_camera_properties.width = 640
        inHand_camera_properties.height = 480
        inHand_camera_properties.enable_tensors = True



        inHand_camera_rel_pose = gymapi.Transform()
        inHand_camera_rel_pose.p = gymapi.Vec3(0.05, 0, 0)
        inHand_camera_rel_pose.r = gymapi.Quat(0.707388, 0.0005629, 0.706825, 0.0005633)


        self.side_camera_handle_list = []
        self.inHand_camera_handle_list = []

        for i in range(self.num_envs):

            # Set a fixed position and look-target for the first camera
            # position and target location are in the coordinate frame of the environment
            camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_properties)
            camera_position = gymapi.Vec3(0.7, 0.4, 0.8)
            camera_target = gymapi.Vec3(0, 0, 0)
            self.gym.set_camera_location(camera_handle, self.envs[i], camera_position, camera_target)
            self.side_camera_handle_list.append(camera_handle)

            camera_handle_robot = self.gym.create_camera_sensor(self.envs[i], inHand_camera_properties)
            # link7_rb = self.gym.find_actor_rigid_body_handle(self.envs[i], self.hand_handle_list[i], 'panda_hand')
            self.gym.attach_camera_to_body(camera_handle_robot, self.envs[i], self.hand_handle_list[i], inHand_camera_rel_pose, gymapi.FOLLOW_TRANSFORM)
            self.inHand_camera_handle_list.append(camera_handle_robot)

    def _create_image_directories(self):
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        # create root path
        # os.makedirs(os.path.join('data',f'goal_{self.args.goal}',exist_ok=True))
        os.makedirs(os.path.join('data', 'peg_insertion', f'goal_{self.args.goal}',self.time_str),exist_ok=True)
        self.img_pth_root = os.path.join('data', 'peg_insertion', f'goal_{self.args.goal}',self.time_str)
        env_pth = os.path.join(self.img_pth_root, 'env_{}')
        
        # create path for each envs
        for i in range(self.num_envs):
            envid_str = str(i).zfill(5)
            os.mkdir(env_pth.format(envid_str))

            # self.side_img_pth_rgb = os.path.join(env_pth, 'side_rgb')
            # self.side_img_pth_depth = os.path.join(env_pth, 'side_depth')
            # self.side_img_pth_semantic = os.path.join(env_pth, 'side_semantic')

            img_pth_rgb = os.path.join(env_pth, 'rgb')
            img_pth_depth = os.path.join(env_pth, 'depth')
            img_pth_semantic = os.path.join(env_pth, 'semantic')

            # create rgb, depth, semantic path
            os.mkdir(img_pth_rgb.format(envid_str))
            os.mkdir(img_pth_depth.format(envid_str))
            os.mkdir(img_pth_semantic.format(envid_str))

    
            self.side_img_pth_rgb = os.path.join(img_pth_rgb, 'side')
            self.side_img_pth_depth = os.path.join(img_pth_depth, 'side')
            self.side_img_pth_semantic = os.path.join(img_pth_semantic, 'side')

            self.inHand_img_pth_rgb = os.path.join(img_pth_rgb, 'in_hand')
            self.inHand_img_pth_depth = os.path.join(img_pth_depth, 'in_hand')
            self.inHand_img_pth_semantic = os.path.join(img_pth_semantic, 'in_hand')

            os.mkdir(self.side_img_pth_rgb.format(envid_str))
            os.mkdir(self.side_img_pth_depth.format(envid_str))
            os.mkdir(self.side_img_pth_semantic.format(envid_str))      

            os.mkdir(self.inHand_img_pth_rgb.format(envid_str))
            os.mkdir(self.inHand_img_pth_depth.format(envid_str))
            os.mkdir(self.inHand_img_pth_semantic.format(envid_str))
    
    def _write_images(self):

        for i in range(self.num_envs):
            if self.done_counter[i]<50:
                side_rgb_pth = self.side_img_pth_rgb.format(str(i).zfill(5))
                side_depth_pth = self.side_img_pth_depth.format(str(i).zfill(5))
                side_semantic_pth = self.side_img_pth_semantic.format(str(i).zfill(5))

                inHand_rgb_pth = self.inHand_img_pth_rgb.format(str(i).zfill(5))
                inHand_depth_pth = self.inHand_img_pth_depth.format(str(i).zfill(5))
                inHand_semantic_pth = self.inHand_img_pth_semantic.format(str(i).zfill(5))
                frame_id_str = str(self.frame_count//10).zfill(5)

                self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.side_camera_handle_list[i], gymapi.IMAGE_COLOR,
                    os.path.join(side_rgb_pth, 'frame_{}.png'.format(frame_id_str)))
                
                side_depth = self.gym.get_camera_image(self.sim, self.envs[i], self.side_camera_handle_list[i], gymapi.IMAGE_DEPTH)
                side_depth[side_depth == -np.inf] = 0
                np.save(os.path.join(side_depth_pth, 'frame_{}'.format(frame_id_str)), side_depth)

                side_semantic = self.gym.get_camera_image(self.sim, self.envs[i], self.side_camera_handle_list[i], gymapi.IMAGE_SEGMENTATION)

                np.save(os.path.join(side_semantic_pth, 'frame_{}'.format(frame_id_str)), side_semantic)

                self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.inHand_camera_handle_list[i], gymapi.IMAGE_COLOR,
                    os.path.join(inHand_rgb_pth, 'frame_{}.png'.format(frame_id_str)))
                
                inHand_depth = self.gym.get_camera_image(self.sim, self.envs[i], self.inHand_camera_handle_list[i], gymapi.IMAGE_DEPTH)
                inHand_depth[inHand_depth == -np.inf] = 0
                np.save(os.path.join(inHand_depth_pth, 'frame_{}'.format(frame_id_str)), inHand_depth)
                
                inHand_semantic = self.gym.get_camera_image(self.sim, self.envs[i], self.inHand_camera_handle_list[i], gymapi.IMAGE_SEGMENTATION)
                
                # normalized_depth = -255.0 * (depth / np.min(depth + 1e-4))
                # normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
                # normalized_depth_image.save(os.path.join(depth_pth, 'frame_{}.jpg'.format(str(self.progress_buf[0].item()).zfill(4))))

                
                np.save(os.path.join(inHand_semantic_pth, 'frame_{}'.format(frame_id_str)), inHand_semantic)
                
    def quat_axis(q, axis=0):
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)

    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def _orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:,:, 0:3] * torch.sign(q_r[:,:, 3]).unsqueeze(-1)

    def control_ik(self, dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u
    
    def check_place(self,current_pose_list):

        # print(self.goal_pose_list.shape)
        # print(current_pose_list.shape)
        pos_err = torch.norm(self.goal_pose_list[:,:,:3] - current_pose_list[:,:,:3],dim=2)
        rot_err = torch.norm(self._orientation_error(current_pose_list[:,:,3:7],self.goal_pose_list[:,:,3:7]),dim=2)
        # print(pos_err)
        # print(rot_err)

        check_pos = torch.where(pos_err<0.005,1,0)
        check_rot = torch.where(rot_err<0.1,1,0)

        done = torch.sum(torch.logical_and(check_pos,check_rot),dim=1)

        self.reward = done*(1.0/len(self.goal_list))

        # print(self.reward)
            
    def simulate(self):
        
        step = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)
        to_prepick = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)
        to_pick = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)
        picked_done = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)
        to_preplace = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)
        to_place = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)
        to_place_1 = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)
        picked = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)
        placed = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)
        max_step = len(self.goal_list)
        op = True
        ind = torch.arange(step.shape[0]).long().to(self.device)
        pick_counter = torch.zeros(self.num_envs).to(self.device)
        place_counter = torch.zeros(self.num_envs).to(self.device)
        to_place_counter = torch.zeros(self.num_envs).to(self.device)
        picked_counter = torch.zeros(self.num_envs).to(self.device)
        placed_counter = torch.zeros(self.num_envs).to(self.device)

        self.gym.prepare_sim(self.sim)

        goal_pos = self.goal_prepick_pos_list[ind,step]
        goal_rot = self.goal_pick_rot_list[ind,step]

        self.reward = torch.zeros(self.num_envs,dtype=torch.float32).to(self.device)

        self.frame_count = 0

        self.done_counter = torch.zeros(self.num_envs,dtype=torch.long).to(self.device)

        


        # simulation loop
        # while ((self.viewer is None) or (not self.gym.query_viewer_has_closed(self.viewer))) and (self.frame_count<2000):
        for _ in trange(2000):
            if ((self.viewer is None) or (not self.gym.query_viewer_has_closed(self.viewer))):
                # step the physics
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

                self.gym.render_all_camera_sensors(self.sim)

                # refresh tensors
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_jacobian_tensors(self.sim)
                self.gym.refresh_mass_matrix_tensors(self.sim)


                hand_pos = self.rb_states[self.hand_idxs, :3]
                hand_rot = self.rb_states[self.hand_idxs, 3:7]

                peg_pos_list = self.rb_states[self.peg_idxs_list, :3]
                peg_rot_list = self.rb_states[self.peg_idxs_list, 3:7]

                current_pose_list = torch.cat([peg_pos_list,peg_rot_list],dim=2).to(self.device)

                # hand_vel = self.rb_states[self.hand_idxs, 7:]

                gripper_open = torch.Tensor(self.franka_upper_limits[7:]).to(self.device)
                gripper_close = torch.Tensor(self.franka_lower_limits[7:]).to(self.device)


                _to_prepick = torch.logical_and(torch.where((torch.norm(self.goal_prepick_pos_list[ind,step] - hand_pos,dim=1)) < 0.001,True,False),
                                                torch.where((torch.norm(self.orientation_error(self.goal_pick_rot_list[ind,step],hand_rot),dim=1))< 0.05,True,False))
                to_prepick = torch.logical_or(to_prepick,_to_prepick)
                to_prepick = torch.logical_and(to_prepick, torch.logical_not(picked)) 
                to_prepick_not = torch.logical_not(to_prepick)

                # print(torch.norm(self.orientation_error(self.goal_pick_rot_list[ind,step],hand_rot),dim=1))


                goal_pos[to_prepick_not,:] = self.goal_prepick_pos_list[to_prepick_not,step[to_prepick_not]]
                goal_rot[to_prepick_not,:] = self.goal_pick_rot_list[to_prepick_not,step[to_prepick_not]]
                goal_pos[to_prepick,:] = self.goal_pick_pos_list[to_prepick,step[to_prepick]]
                goal_rot[to_prepick,:] = self.goal_pick_rot_list[to_prepick,step[to_prepick]]

                _to_pick = torch.logical_and(torch.where((torch.norm(self.goal_pick_pos_list[ind,step] - hand_pos,dim=1)) < 0.001,True,False),
                                            torch.where((torch.norm(self.orientation_error(self.goal_pick_rot_list[ind,step],hand_rot),dim=1))< 0.05,True,False)) 
                to_pick = torch.logical_or(to_pick, _to_pick)
                to_pick = torch.logical_and(to_pick,to_prepick)

                # print(torch.norm(self.goal_pick_pos_list[ind,step] - hand_pos,dim=1))


                self.pos_action[to_pick,7:9]=gripper_close
                pick_counter[to_pick]+=1

                _picked = torch.where(pick_counter>=20,True,False)
                picked = torch.logical_or(picked, _picked)

                # print(pick_counter)

                goal_pos[picked,:] = self.goal_prepick_pos_list[picked,step[picked]]
                goal_rot[picked,:] = self.goal_pick_rot_list[picked,step[picked]]
                picked_counter[picked]+=1


                _picked_done = torch.where(picked_counter>=30,True,False)
                picked_done = torch.logical_or(picked_done, _picked_done)


                goal_pos[picked_done,:] = self.goal_preplace_pos_list[picked_done,step[picked_done]]
                goal_rot[picked_done,:] = self.goal_place_rot_list[picked_done,step[picked_done]]


                
                _to_preplace = torch.logical_and(torch.where((torch.norm(self.goal_preplace_pos_list[ind,step] - hand_pos,dim=1)) < 0.005,True,False),
                                            torch.where((torch.norm(self.orientation_error(self.goal_place_rot_list[ind,step], hand_rot),dim=1))< 0.05,True,False)) 
                to_preplace = torch.logical_or(to_preplace,_to_preplace)
                to_preplace = torch.logical_and(to_preplace,picked)
                to_preplace = torch.logical_and(to_preplace,torch.logical_not(placed))
                
                
                goal_pos[to_preplace,:] = self.goal_place_pos_list[to_preplace,step[to_preplace]]
                goal_rot[to_preplace,:] = self.goal_place_rot_list[to_preplace,step[to_preplace]]

                _to_place_1 = torch.logical_and(torch.where((torch.norm(self.goal_place_pos_list[ind,step] - hand_pos,dim=1)) < 0.02,True,False),
                                            torch.where((torch.norm(self.orientation_error(self.goal_place_rot_list[ind,step], hand_rot),dim=1))< 0.05,True,False)) 
                to_place_1 = torch.logical_or(_to_place_1,to_place_1)
                to_place_1 = torch.logical_and(to_place_1,to_preplace)

                to_place_counter[to_place_1]+=1

                _to_place = torch.where(to_place_counter>30,True,False)
                to_place = torch.logical_or(to_place,_to_place)

                self.pos_action[to_place,7:9] = gripper_open
                
                place_counter[to_place]+=1

                _placed = torch.where(place_counter>=30,True,False)
                placed = torch.logical_or(placed, _placed)

            
                self.pos_action[placed,7:9] = gripper_open
                goal_pos[placed] = self.goal_preplace_pos_list[placed,step[placed]]
                goal_rot[placed] = self.goal_place_rot_list[placed,step[placed]]
                placed_counter[placed] += 1

                reset = torch.logical_and(torch.where(place_counter>=30,True,False),torch.where(step<max_step-1,True,False))
                pick_counter[reset]=0
                place_counter[reset] = 0
                to_place_counter[reset] = 0
                step[reset]+=1
                to_prepick[reset] = False
                to_pick[reset] = False
                to_preplace[reset] = False
                picked_done[reset] = False
                to_place[reset] = False
                to_place_1[reset] = False
                picked[reset] = False
                placed[reset] = False
                placed_counter[reset] = 0
                picked_counter[reset] = 0

                self.check_place(current_pose_list)


                    
                # else:
                #     # check peg pose
                #     pass

                if op:
                    self.pos_action[:,7:9] = gripper_open
                    op = False
                pos_err = goal_pos - hand_pos


                orn_err = self.orientation_error(goal_rot, hand_rot)
                dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

                # print(dpose)

                # Deploy control
                self.pos_action[:, :7] = self.dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)
                
                # Deploy actions
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

                # save images

                _done_count = torch.where(self.reward>0.99,True,False)
                self.done_counter[_done_count] += 1

                if self.args.save:
                    if self.frame_count % 10 == 0 and self.frame_count != 0:
                        self._write_images()

                self.frame_count+=1

                # update viewer
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
                self.gym.sync_frame_time(self.sim)


        # cleanup
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

        success_env = []

        for i in range(self.num_envs):
            if self.reward[i]<0.99:
                shutil.rmtree(f'data/peg_insertion/goal_{self.args.goal}/{self.time_str}/env_{str(i).zfill(5)}')
            else:
                success_env.append(i)
        
        info = {
            "success_envs":success_env,
            "kit_init_pose":self.kit_init_pose_list,
            "peg_init_pose":self.peg_init_pose_list,
            "peg_color":self.peg_color,
            "img_pth":self.img_pth_root,
        }
        
        # return success_env, self.kit_init_pose_list, self.peg_init_pose_list,self.peg_color,self.img_pth_root
        return info





    
"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

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
import goal.peg_insertion.goal_data as goal_data


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u



def create_peg_actor(gym,env,goal_list,goal_pose_list,i):
    peg_indices = []
    for j, idx in enumerate(goal_list):
        peg_pose = utils.multiply_gymapi_transform(kit_pose, utils.mat2gymapi_transform(goal_pose_list[j]))
        peg_handle = gym.create_actor(env, peg_asset_list[idx], peg_pose, peg_type[idx], i)
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, peg_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        peg_idx = gym.get_actor_index(env, peg_handle,gymapi.DOMAIN_SIM)
        peg_indices.append(peg_idx)
    return peg_indices

# def create_kit_actor(gym,env,goal_list,goal_pose_list,i):
#     kit_indices = []
#     for j, idx in enumerate(goal_list):
#         kit_pose = utils.multiply_gymapi_transform(table_pose, utils.mat2gymapi_transform(goal_pose_list[j]))
#         kit_handle = gym.create_actor(env, peg_asset_list[idx], kit_pose, kit_type[idx], i)
#         color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
#         gym.set_rigid_body_color(env, kit_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
#         kit_idx = gym.get_actor_index(env, kit_handle,gymapi.DOMAIN_SIM)
#         kit_indices.append(kit_idx)
#     return kit_indices

# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik", "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
    {"name": "--headless", "action": "store_true", "help": ""},
    {"name": "--save", "action": "store_true", "help": ""},
    {"name": "--goal", "type": str, "default":'1',"help": ""},
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'


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

# Set controller parameters
# IK params
damping = 0.05

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)


# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

#keyboard event
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "up")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "down")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "right")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "backward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "forward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "turn_right")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "turn_left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "turn_up")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "turn_down")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "gripper_close")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_X, "save")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_B, "quit")


asset_root = "./assets"

# create table asset
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create peg assets
peg_asset_list = []
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
# asset_options.flip_visual_attachments = True
peg_type = ['rectangle', 'square', 'pentagon', 'triangle']
for t in peg_type:
    peg_asset_list.append(gym.load_asset(sim, asset_root, 'urdf/peg_insertion/' + t + '.urdf', asset_options))

# create kit assets
# kit_asset_list = []
# asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
# asset_options.flip_visual_attachments = True
# kit_type = ['pentagon_kit.urdf', 'rectangle_kit.urdf', 'square_kit.urdf', 'triangle_kit.urdf']
# for t in kit_type:
#     kit_asset_list.append(gym.load_asset(sim, asset_root, 'urdf/peg_insertion/' + t, asset_options))

# create kit assets
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
kit_asset = gym.load_asset(sim, asset_root, 'urdf/peg_insertion/kit.urdf',asset_options)

asset_options.fix_base_link = True
sphere_asset = gym.create_sphere(sim, 0.01, asset_options)
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0,0,0)
initial_pose.r = gymapi.Quat(0, 0, 0, 1)

# load franka asset
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

# use position drive for all dofs
if controller == "ik":
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][:7].fill(400.0)
    franka_dof_props["damping"][:7].fill(40.0)
else:       # osc
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    franka_dof_props["stiffness"][:7].fill(0.0)
    franka_dof_props["damping"][:7].fill(0.0)
# grippers
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]
# grippers open
default_dof_pos[7:] = franka_upper_limits[7:]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["panda_hand"]

# configure env grid
num_envs = 1# args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

kit_pose = gymapi.Transform()

envs = []
box_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

i=0
# create env
env = gym.create_env(sim, env_lower, env_upper, num_per_row)
envs.append(env)

# add table
table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

ball_handle = gym.create_actor(env, sphere_asset, initial_pose, "ball", i+1)

# add kit
kit_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
kit_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
kit_pose.p.z = table_dims.z #+ 0.001
kit_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
kit_handle = gym.create_actor(env, kit_asset, kit_pose, "kit", i)
gym.set_rigid_body_color(env, kit_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)


#############################################################################################################

task = goal_data.names[args.goal]

goal = task.goal
goal_pose = task.goal_pose
peg_height = task.peg_height

##############################################################################################################

peg_indices = to_torch(create_peg_actor(gym,env,goal,goal_pose,i),dtype=torch.long,device = device)


        
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

ball_state = gym.get_actor_rigid_body_states(env, ball_handle, gymapi.STATE_ALL)
ball_state["pose"]["p"] = tuple(np.array([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z]))

gym.set_actor_rigid_body_states(env, ball_handle, ball_state, gymapi.STATE_ALL)

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to franka hand
j_eef = jacobian[:, franka_hand_index - 1, :, :7]

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:9, 0].view(num_envs, 9, 1)
dof_vel = dof_states[:9, 1].view(num_envs, 9, 1)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)

camera_properties = gymapi.CameraProperties()
camera_properties.width = 640
camera_properties.height = 480

camera_handle = gym.create_camera_sensor(envs[i], camera_properties)
camera_position = gymapi.Vec3(1, 0.5, 1.0)
camera_target = gymapi.Vec3(0, 0, 0)
gym.set_camera_location(camera_handle, envs[i], camera_position, camera_target)



action = ''
op = False
idx = len(goal)-1
grasp_pos = []
grasp_rot = []
# simulation loop
while viewer is None or not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]

    gripper_open = torch.Tensor(franka_upper_limits[7:]).to(device)
    gripper_close = torch.Tensor(franka_lower_limits[7:]).to(device)
    delta = 0.01

    for evt in gym.query_viewer_action_events(viewer):

        if evt.value > 0:
            action = evt.action
            if action == "gripper_close":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]).to(device)
                if torch.all(pos_action[:,7:9] == gripper_close):
                    pos_action[:,7:9] = gripper_open
                elif torch.all(pos_action[:,7:9] == gripper_open):
                    pos_action[:,7:9] = gripper_close
                action = ''
            if action == "save":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]).to(device)
                print(hand_pos)
                print(hand_rot)
                grasp_pos.append(hand_pos)
                grasp_rot.append(hand_rot)
                root_state_tensor[peg_indices[idx], 2] = 5
                goal_obj_indices = torch.tensor([peg_indices[idx]], dtype=torch.int32, device=device)
                gym.set_actor_root_state_tensor_indexed(sim,
                                                        gymtorch.unwrap_tensor(root_state_tensor),
                                                        gymtorch.unwrap_tensor(goal_obj_indices), len(goal_obj_indices))
                idx-=1
        else :
            action = ''

    if action == "up":
        dpose = torch.Tensor([[[0.],[0.],[1.],[0.],[0.],[0.]]]).to(device) * delta
    elif action == "down":
        dpose = torch.Tensor([[[0.],[0.],[-1.],[0.],[0.],[0.]]]).to(device) * delta
    elif action == "left":
        dpose = torch.Tensor([[[0.],[-1.],[0.],[0.],[0.],[0.]]]).to(device) * delta
    elif action == "right":
        dpose = torch.Tensor([[[0.],[1.],[0.],[0.],[0.],[0.]]]).to(device) * delta
    elif action == "backward":
        dpose = torch.Tensor([[[-1.],[0.],[0.],[0.],[0.],[0.]]]).to(device) * delta
    elif action == "forward":
        dpose = torch.Tensor([[[1.],[0.],[0.],[0.],[0.],[0.]]]).to(device) * delta
    elif action == "turn_left":
        dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[-10.]]]).to(device) * delta
    elif action == "turn_right":
        dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[10.]]]).to(device) * delta
    elif action == "turn_up":
        dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[10.],[0.]]]).to(device) * delta
    elif action == "turn_down":
        dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[-10.],[0.]]]).to(device) * delta
    elif action == "gripper_close":
        dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]).to(device)
        if torch.all(pos_action[:,7:9] == gripper_close):
            pos_action[:,7:9] = gripper_open
        elif torch.all(pos_action[:,7:9] == gripper_open):
            pos_action[:,7:9] = gripper_close
    elif action == "quit":
        break
    else:
        dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]).to(device)


    pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))



    # print(root_state_tensor[4,3:7])

    # update viewer
    gym.step_graphics(sim)
    
    if viewer is not None:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


rel_pick_pos=[]
rel_place_pos=[]

i = len(goal)-1


for pos, quat in zip(grasp_pos, grasp_rot):
    rel_pos = np.linalg.inv(utils.gymapi_transform2mat(kit_pose)) @ utils.tensor_6d_pose2mat(pos, quat)
    rel_place_pos.insert(0,rel_pos)
    # peg_world_pos.insert(0,utils.gymapi_transform2mat(region_pose) @ goal_pose[i-1])
    # print(np.linalg.inv(utils.gymapi_transform2mat(region_pose) @ goal_pose[i-1]) @ utils.tensor_6d_pose2mat(pos, quat))
    # rel_pick_pos.insert(0, np.linalg.inv(utils.gymapi_transform2mat(region_pose) @ goal_pose[i-1]) @ utils.tensor_6d_pose2mat(pos, quat))
    rel_pick_pos.insert(0,np.linalg.inv(goal_pose[i]) @ rel_pos)
    
    i-=1


_goal_data = {
"peg_list": goal, 
"peg_pose": goal_pose,
"pick_pose": rel_pick_pos,
"place_pose": rel_place_pos,
"peg_height": peg_height,
# "hand_pose" : task.hand_pose_list,
# "hand_rel_pose" : task.hand_rel_mat_list,
}

if args.save:
    sio.savemat(f"goal/peg_insertion/goal_data/goal_{args.goal}_data.mat",_goal_data)
    print("Pose saved !!!")


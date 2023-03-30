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


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)




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

# create target_region
region_dims = gymapi.Vec3(0.1,0.1,0.0001)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
region_asset = gym.create_box(sim, region_dims.x,region_dims.y, region_dims.z, asset_options)

# create block assets
block_asset_list = []
asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
block_type = ['A.urdf', 'B.urdf', 'C.urdf', 'D.urdf', 'E.urdf']
for t in block_type:
    block_asset_list.append(gym.load_asset(sim, asset_root, 'urdf/block_assembly/block_' + t, asset_options))

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

box_pose = gymapi.Transform()
region_pose = gymapi.Transform()


envs = []
box_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    ball_handle = gym.create_actor(env, sphere_asset, initial_pose, "ball", i+1)
    
    # add region
    region_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    region_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    region_pose.p.z = table_dims.z #+ 0.001
    region_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))

    # add target region
    black = gymapi.Vec3(0.,0.,0.)
    region_handle = gym.create_actor(env, region_asset, region_pose, "target_reginon", i)
    gym.set_rigid_body_color(env, region_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, black)

    goal_A = [0,0,1]
    goal_A_pose = []

    goal_A_pose_1 = gymapi.Transform()
    goal_A_pose_1.p = gymapi.Vec3(0, 0.0255, 0.03)
    goal_A_pose_1.r = gymapi.Quat.from_euler_zyx(math.pi*0.5, 0, 0)

    goal_A_pose_2 = gymapi.Transform()
    goal_A_pose_2.p = gymapi.Vec3(0, -0.0255, 0.03)
    goal_A_pose_2.r = gymapi.Quat.from_euler_zyx(math.pi*0.5, 0, 0)

    goal_A_pose_3 = gymapi.Transform()
    goal_A_pose_3.p = gymapi.Vec3(0, 0, 0.0745)
    goal_A_pose_3.r = gymapi.Quat.from_euler_zyx(math.pi*0.5,0,math.pi*0.5)

    

    goal_A_pose.append(utils.gymapi_transform2mat(goal_A_pose_1))
    goal_A_pose.append(utils.gymapi_transform2mat(goal_A_pose_2))
    goal_A_pose.append(utils.gymapi_transform2mat(goal_A_pose_3))

    block_list = []
    block_pose_world = []
    
    for j, idx in enumerate(goal_A):
        block_pose = utils.multiply_gymapi_transform(region_pose, utils.mat2gymapi_transform(goal_A_pose[j]))
        block_handle = gym.create_actor(env, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i)
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, block_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        block_pose_world.append(block_pose)
        block_list.append(block_handle)

    # for j in len(goal_A):

    #     body_states = gym.get_actor_rigid_body_states(env, block_list[j], gymapi.STATE_ALL)

    #     tmp_mat = np.eye(4)
    #     tmp_mat[:3, :3] = R.from_quat(np.array([body_states["pose"]["r"]["x"],
    #                                             body_states["pose"]["r"]["y"],
    #                                             body_states["pose"]["r"]["z"],
    #                                             body_states["pose"]["r"]["w"]]).reshape(-1)).as_matrix()
    #     tmp_mat[:3, 3] = np.array([body_states["pose"]["p"]["x"], body_states["pose"]["p"]["y"], body_states["pose"]["p"]["z"]]).reshape(-1)

            
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


action = ''
op = False

grasp_pose = []
grasp_rot = []

camera_properties = gymapi.CameraProperties()
camera_properties.width = 640
camera_properties.height = 480

camera_handle = gym.create_camera_sensor(envs[i], camera_properties)
camera_position = gymapi.Vec3(1, 0.5, 1.0)
camera_target = gymapi.Vec3(0, 0, 0)
gym.set_camera_location(camera_handle, envs[i], camera_position, camera_target)


img = []
frame_count=0


# simulation loop
while viewer is None or not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]
    hand_vel = rb_states[hand_idxs, 7:]

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
                print(hand_pos,hand_rot) 
                grasp_pose.append(hand_pos)
                grasp_rot.append(hand_rot)

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
        print(pos_action[:,7:9])
        if torch.all(pos_action[:,7:9] == gripper_close):
            pos_action[:,7:9] = gripper_open
        elif torch.all(pos_action[:,7:9] == gripper_open):
            pos_action[:,7:9] = gripper_close
    elif action == "quit":
        break
    else:
        dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]).to(device)


    if controller == "ik":
        pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
    else:       # osc
        effort_action[:, :7] = control_osc(dpose)

    # tmp_action = torch.cat((pos_action,rope_pos.squeeze(-1)),1)
    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))
    rgb_filename = "output_grasp/frame%d.png" % (frame_count)
    frame_count += 1
    gym.write_camera_image_to_file(sim, envs[i], camera_handle, gymapi.IMAGE_COLOR, rgb_filename)


    # update viewer
    gym.step_graphics(sim)
    
    if viewer is not None:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

rel_place_pos = []
rel_pick_pos = []

"""
tensor([[0.4119, 0.2724, 0.5762]], device='cuda:0') tensor([[ 0.9994, -0.0153, -0.0325,  0.0020]], device='cuda:0')
tensor([[0.3918, 0.2770, 0.5477]], device='cuda:0') tensor([[ 0.9988, -0.0151, -0.0469,  0.0066]], device='cuda:0')
tensor([[0.4483, 0.2689, 0.5538]], device='cuda:0') tensor([[ 0.9983, -0.0144, -0.0563,  0.0093]], device='cuda:0')
"""


# grasp_pose.append(torch.tensor([[0.4119, 0.2724, 0.5762]], device='cuda:0')) 
# grasp_rot.append(torch.tensor([[ 0.9994, -0.0153, -0.0325,  0.0020]], device='cuda:0'))
# grasp_pose.append(torch.tensor([[0.3918, 0.2770, 0.5477]], device='cuda:0')) 
# grasp_rot.append(torch.tensor([[ 0.9988, -0.0151, -0.0469,  0.0066]], device='cuda:0'))
# grasp_pose.append(torch.tensor([[0.4483, 0.2689, 0.5538]], device='cuda:0')) 
# grasp_rot.append(torch.tensor([[ 0.9983, -0.0144, -0.0563,  0.0093]], device='cuda:0'))



i = len(goal_A_pose)

block_world_pos = []

for pos, quat in zip(grasp_pose, grasp_rot):
    rel_pos = np.linalg.inv(utils.gymapi_transform2mat(region_pose)) @ utils.tensor_6d_pose2mat(pos, quat)
    rel_place_pos.insert(0, rel_pos)
    # block_world_pos.insert(0,utils.gymapi_transform2mat(region_pose) @ goal_A_pose[i-1])
    # print(np.linalg.inv(utils.gymapi_transform2mat(region_pose) @ goal_A_pose[i-1]) @ utils.tensor_6d_pose2mat(pos, quat))
    # rel_pick_pos.insert(0, np.linalg.inv(utils.gymapi_transform2mat(region_pose) @ goal_A_pose[i-1]) @ utils.tensor_6d_pose2mat(pos, quat))
    rel_pick_pos.insert(0, np.linalg.inv(goal_A_pose[i-1]) @ rel_pos)
    
    i-=1




goal_A_data = {
    "block_list": goal_A, 
    "block_pose": goal_A_pose,
    "pick_pose": rel_pick_pos,
    "place_pose": rel_place_pos,
    "block_world": block_world_pos,
}

sio.savemat("goal/block_assembly/goal_A_data.mat",goal_A_data)


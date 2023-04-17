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

import math
import time
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import pytorch3d.transforms

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'


# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.002
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.2
sim_params.physx.max_depenetration_velocity = 1000.0

sim_params.use_gpu_pipeline = False
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.gravity = gymapi.Vec3(0, 0, -9.8)
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z

if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, 0.0)

# add urdf asset
asset_root = "assets"
asset_file = "urdf/mano/zeros/mano_addtips.urdf"
# ycb_asset_file = "urdf/ycb/002_master_chef_can/002_master_chef_can.urdf"
ycb_asset_file = "urdf/block_assembly/block_D.urdf"

# load hand asset
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
mano_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# load block asset
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
ycb_asset = gym.load_asset(sim, asset_root, ycb_asset_file, asset_options)

# load shpere asset
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
sphere_asset = gym.create_sphere(sim, 0.01, asset_options)

# set force sensor on hand
link_name_list = ['link' + str(i).zfill(2) for i in range(7, 53, 3)]

# for link_name in link_name_list:
#     body_idx = gym.find_asset_rigid_body_index(mano_asset, link_name)
#     sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
#     sensor_idx = gym.create_asset_force_sensor(mano_asset, body_idx, sensor_pose)

# block_asset_list = []
# block_type = ['A.urdf', 'B.urdf', 'C.urdf', 'D.urdf', 'E.urdf']
# for t in block_type:
#     block_asset_list.append(gym.load_asset(sim, asset_root, 'urdf/block_assembly/block_' + t, asset_options))

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0, 0, 0)
initial_pose.r = gymapi.Quat(0, 0, 0, 1)

# Create environment 0
env0 = gym.create_env(sim, env_lower, env_upper, 2)
mano_hand0 = gym.create_actor(env0, mano_asset, initial_pose, 'mano', 0, 2)
ycb_obj = gym.create_actor(env0, ycb_asset, initial_pose, 'ycb_obj', 0)

mano_id = gym.get_actor_index(env0, mano_hand0, gymapi.DOMAIN_SIM)
ycb_id = gym.get_actor_index(env0, ycb_obj, gymapi.DOMAIN_SIM)

prop = gym.get_actor_rigid_shape_properties(env0, ycb_obj)[0]
prop.rolling_friction = 1
gym.set_actor_rigid_shape_properties(env0, ycb_obj, [prop])

link_sim_id = []

for link_name in link_name_list:
    hand_idx = gym.find_actor_rigid_body_index(env0, mano_hand0, link_name, gymapi.DOMAIN_SIM)
    link_sim_id.append(hand_idx)

# block_list = []
# for asset, t in zip(block_asset_list, block_type):
#     block_handle = gym.create_actor(env0, asset, initial_pose, 'block_' + t)
#     block_list.append(block_handle)

#     color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
#     gym.set_rigid_body_color(env0, block_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

num_sensors = gym.get_actor_force_sensor_count(env0, mano_hand0)
print('get_actor_force_sensor_count = {}'.format(num_sensors))

# flag = gym.enable_actor_dof_force_sensors(env0, mano_hand0)
# print('enable_actor_dof_force_sensors = {}'.format(flag))

# Configure DOF properties
props = gym.get_actor_dof_properties(env0, mano_hand0)
props["driveMode"].fill(gymapi.DOF_MODE_POS)
props["stiffness"][:3].fill(500)
props["stiffness"][3:].fill(50)
props["damping"][:3].fill(200)
props["damping"][3:].fill(200)
props["friction"].fill(1)
gym.set_actor_dof_properties(env0, mano_hand0, props)

# bad id: 0, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19
data_id = 0

# with open('/home/hcis-s12/Lego_Assembly/Simulator/pybullet/gym_lego/dexycb_data_all.pickle', 'rb') as f:
#     train_data = pickle.load(f)[data_id]

# init_pose = train_data["subgoal_1"]["hand_traj_reach"][0, 0]
# goal_pose = train_data["subgoal_1"]["hand_ref_pose"][0, 0]

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
hand_goal_mat = np.eye(4)
hand_goal_mat[:3, :3] = Rot.from_euler("XYZ", hand_goal_pose[3:6]).as_matrix()
hand_goal_mat[:3, 3] = hand_goal_pose[0:3]

# reset hand and object pose
dof_states = gym.get_actor_dof_states(env0, mano_hand0, gymapi.STATE_ALL)
dof_states["pos"] = hand_goal_pose
gym.set_actor_dof_states(env0, mano_hand0, dof_states, gymapi.STATE_ALL)
gym.set_actor_dof_position_targets(env0, mano_hand0, hand_goal_pose.astype('f'))

# obj_init = train_data["subgoal_1"]["obj_init"]
# obj_final = train_data["subgoal_1"]["obj_final"]

obj_init = np.array([-0.19434147, -0.11968146, 0.01548913, -0.50471319, 0.49524648, -0.4952225 , 0.50472785])
obj_init_mat = np.eye(4)
obj_init_mat[:3, :3] = Rot.from_quat(obj_init[3:7]).as_matrix()
obj_init_mat[:3, 3] = obj_init[0:3]

obj_goal = np.random.uniform(0, 1, 7)
obj_goal[0:3] += 1

initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(*obj_goal[0:3])
initial_pose.r = gymapi.Quat(*obj_goal[3:7])
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
ycb_asset = gym.load_asset(sim, asset_root, ycb_asset_file, asset_options)
ycb_obj_2 = gym.create_actor(env0, ycb_asset, initial_pose, 'ycb_obj', 0, 3)

hand_rel_mat = torch.tensor(np.linalg.inv(obj_init_mat) @ hand_goal_mat, dtype=torch.float32)
step_size = torch.tensor((obj_goal - obj_init) / 1000)

body_states = gym.get_actor_rigid_body_states(env0, ycb_obj, gymapi.STATE_ALL)
body_states["pose"]["p"] = tuple(obj_init[:3])
body_states["pose"]["r"] = tuple(obj_init[3:])
gym.set_actor_rigid_body_states(env0, ycb_obj, body_states, gymapi.STATE_ALL)

# get joint position
jointid_list = [6, 
                9, 12, 15, 16,
                19, 22, 25, 26,
                29, 32, 35, 36,
                39, 42, 45, 46,
                49, 52, 55, 56]

joint_transforms = gym.get_actor_joint_transforms(env0, mano_hand0)[jointid_list]["p"]
joint_pos = np.vstack([joint_transforms["x"], joint_transforms["y"], joint_transforms["z"]]).T
# print(joint_pos)
# print()

# sphere_list = []
# for i in range(joint_pos.shape[0]):
#     initial_pose = gymapi.Transform()
#     initial_pose.p = gymapi.Vec3(joint_pos[i, 0], joint_pos[i, 1], joint_pos[i, 2])
#     initial_pose.r = gymapi.Quat(0, 0, 0, 1)

#     sphere_handle = gym.create_actor(env0, sphere_asset, initial_pose, 'sphere_' + str(i), 1, -1)
#     color = gymapi.Vec3(1, 0, 0)
#     gym.set_rigid_body_color(env0, sphere_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
#     sphere_list.append(sphere_handle)

# get rigidbody position
rigidbodyid_list = [7, 
                    10, 13, 16, 17,
                    20, 23, 26, 27,
                    30, 33, 36, 37,
                    40, 43, 46, 47,
                    50, 53, 56, 57]

rigidbody_transforms = gym.get_actor_rigid_body_states(env0, mano_hand0, gymapi.STATE_ALL)[rigidbodyid_list]["pose"]["p"]
rigidbody_pos = np.vstack([rigidbody_transforms["x"], rigidbody_transforms["y"], rigidbody_transforms["z"]]).T
# print(rigidbody_pos)
# print()

# sphere_list = []
# for i in range(rigidbody_pos.shape[0]):
#     initial_pose = gymapi.Transform()
#     initial_pose.p = gymapi.Vec3(rigidbody_pos[i, 0], rigidbody_pos[i, 1], rigidbody_pos[i, 2])
#     initial_pose.r = gymapi.Quat(0, 0, 0, 1)

#     sphere_handle = gym.create_actor(env0, sphere_asset, initial_pose, 'sphere_' + str(i), 3, 1)
#     color = gymapi.Vec3(1, 0, 0)
#     gym.set_rigid_body_color(env0, sphere_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
#     sphere_list.append(sphere_handle)

# for cnt, block in enumerate(block_list):
#     body_states = gym.get_actor_rigid_body_states(env0, block, gymapi.STATE_ALL)
#     body_states["pose"]["p"] = (0.7, cnt * 0.1, 0)
#     body_states["pose"]["r"] = (0, 0, 0, 1)

#     gym.set_actor_rigid_body_states(env0, block, body_states, gymapi.STATE_ALL)

# Look at the first env
cam_pos = gymapi.Vec3(1, 0, 1.5)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# set wrist goal in object frame
# goal_wrist = np.linalg.inv(Rot.from_quat(obj_final[3:]).as_matrix()) @ (goal_pose[:3] - obj_final[:3])

_dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state_tensor = gymtorch.wrap_tensor(_dof_state_tensor)

_actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
root_state_tensor = gymtorch.wrap_tensor(_actor_root_state_tensor).view(-1, 13)

_net_contact_force = gym.acquire_net_contact_force_tensor(sim)
net_contact_force = gymtorch.wrap_tensor(_net_contact_force)

cnt = 0
# Simulate
while not gym.query_viewer_has_closed(viewer):
    cnt += 1
    print('cnt = ', cnt)

    if cnt == 1000:
        time.sleep(10)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)

    # set object pose
    root_state_tensor[ycb_id, 0:7] += step_size
    goal_obj_indices = torch.tensor([ycb_id]).to(torch.int32)
    gym.set_actor_root_state_tensor_indexed(sim,
                                            gymtorch.unwrap_tensor(root_state_tensor),
                                            gymtorch.unwrap_tensor(goal_obj_indices), len(goal_obj_indices))

    # set hand pose
    cur_obj_mat = torch.eye(4)
    cur_obj_mat[:3, 3] = root_state_tensor[ycb_id, 0:3]
    cur_obj_mat[:3, :3] = pytorch3d.transforms.quaternion_to_matrix(root_state_tensor[ycb_id, 3:7][[3, 0, 1, 2]])

    new_wrist_mat = cur_obj_mat @ hand_rel_mat
    dof_state_tensor[0:3, 0] = new_wrist_mat[:3, 3]
    dof_state_tensor[3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:3, :3], "XYZ")
    # print(dof_state_tensor[3:6,0])
    # exit()
    dof_indices = torch.tensor([mano_id]).to(dtype=torch.int32)
    gym.set_dof_state_tensor_indexed(sim,
                                     gymtorch.unwrap_tensor(dof_state_tensor),
                                     gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
    target = dof_state_tensor[:, 0].clone()
    gym.set_dof_position_target_tensor_indexed(sim,
                                               gymtorch.unwrap_tensor(target),
                                               gymtorch.unwrap_tensor(dof_indices), len(dof_indices))

    # body_states = gym.get_actor_rigid_body_states(env0, ycb_obj, gymapi.STATE_ALL)
    # obj_quat = np.asarray(body_states["pose"]["r"][0].tolist())
    # obj_pos = np.asarray(body_states["pose"]["p"][0].tolist())
    # Fpos_world = Rot.from_quat(obj_quat).as_matrix() @ goal_wrist + obj_pos

    # w = 0
    # goal_pose[:3] = w * goal_pose[:3] + (1 - w) * Fpos_world
    # goal_pose[:3] = Fpos_world

    # gym.set_actor_dof_position_targets(env0, mano_hand0, goal_pose.astype('f'))

    # # visualize force
    # cur_force = net_contact_force[link_sim_id, :]
    # nzero_force = cur_force[cur_force.any(axis=1), :]
    # print('-' * 10)
    # np.set_printoptions(suppress=True)
    # print(nzero_force)
    # print()

    # rigidbody_transforms = gym.get_actor_rigid_body_states(env0, mano_hand0, gymapi.STATE_ALL)[rigidbodyid_list]["pose"]["p"]
    # rigidbody_pos = np.vstack([rigidbody_transforms["x"], rigidbody_transforms["y"], rigidbody_transforms["z"]]).T

    # for i in range(rigidbody_pos.shape[0]):
    #     body_states = gym.get_actor_rigid_body_states(env0, sphere_list[i], gymapi.STATE_ALL)
    #     body_states["pose"]["p"] = tuple(rigidbody_pos[i])
    #     gym.set_actor_rigid_body_states(env0, sphere_list[i], body_states, gymapi.STATE_ALL)

    # gym.clear_lines(viewer)

    # if nzero_force.shape[0] != 0:
    #     for i in np.where(cur_force.any(axis=1))[0]:
    #         vertices = torch.from_numpy(np.vstack([rigidbody_pos[i], rigidbody_pos[i] + cur_force[i].numpy()]))
    #         gym.add_lines(viewer, env0, 1, vertices, torch.tensor([0, 0, 1]))

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
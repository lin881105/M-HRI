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
import pytorch3d
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

gym = gymapi.acquire_gym()
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik", "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
    {"name": "--headless", "action": "store_true", "help": "Run headless"},
]

args = gymutil.parse_arguments(
    description="Joint control Methods Example",
    custom_parameters=custom_parameters,
    )

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

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

use_viewer = not args.headless
if use_viewer:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")
else:
    viewer = None

num_envs = 10
env_spacing = 1.5
max_episode_length = 195





# sim_params.physx.solver_type = 1
# sim_params.physx.num_position_iterations = 4
# sim_params.physx.num_velocity_iterations = 1
# sim_params.physx.contact_offset = 0.005
# sim_params.physx.rest_offset = 0.0
# sim_params.physx.bounce_threshold_velocity = 0.2
# sim_params.physx.max_depenetration_velocity = 1
# sim_params.physx.num_threads = args.num_threads
# sim_params.physx.use_gpu = args.use_gpu




plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)


lower = gymapi.Vec3(-env_spacing, 0.75 * -env_spacing, 0.0)
upper = gymapi.Vec3(env_spacing, 0.75 * env_spacing, env_spacing)

asset_root = "assets"
asset_file_mano = "urdf/mano/zeros/mano_addtips.urdf"

# create mano asset
asset_path_mano = os.path.join(asset_root, asset_file_mano)
asset_root_mano = os.path.dirname(asset_path_mano)
asset_file_mano = os.path.basename(asset_path_mano)

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
mano_asset = gym.load_asset(sim, asset_root_mano, asset_file_mano, asset_options)
num_mano_dofs = gym.get_asset_dof_count(mano_asset)

# create table asset
table_dims = gymapi.Vec3(0.3, 0.5, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create target region
region_dims = gymapi.Vec3(0.1,0.1,0.0001)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
region_asset = gym.create_box(sim, region_dims.x,region_dims.y, region_dims.z, asset_options)

# create block asset
block_asset_list = []
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
# asset_options.fix_base_link = True
block_type = ['A.urdf', 'B.urdf', 'C.urdf', 'D.urdf', 'E.urdf']
for t in block_type:
    block_asset_list.append(gym.load_asset(sim, asset_root, 'urdf/block_assembly/block_' + t, asset_options))


# set mano dof properties
mano_dof_props = gym.get_asset_dof_properties(mano_asset)
mano_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
mano_dof_props["stiffness"][:3].fill(500)
mano_dof_props["stiffness"][3:].fill(50)
mano_dof_props["damping"][:3].fill(200)
mano_dof_props["damping"][3:].fill(200)
mano_dof_props["friction"].fill(1)

mano_dof_lower_limits = mano_dof_props['lower']
mano_dof_upper_limits = mano_dof_props['upper']
mano_dof_lower_limits = to_torch(mano_dof_lower_limits, device=device)
mano_dof_upper_limits = to_torch(mano_dof_upper_limits, device=device)

# set YCB properties
# ycb_rb_props = gym.get_asset_rigid_shape_properties(ycb_asset)
# ycb_rb_props[0].rolling_friction = 1

# set default pose
handobj_start_pose = gymapi.Transform()
handobj_start_pose.p = gymapi.Vec3(0, 0, 0)
handobj_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.45, 0.0, 0.5 * table_dims.z)

region_pose = gymapi.Transform()

# read block goal pos
mat_file = "goal/block_assembly/goal_A_data.mat"
mat_dict = sio.loadmat(mat_file)

block_list = mat_dict["block_list"][0]
goal_pose = mat_dict["block_pose"]
# rel_pick_pos = mat_dict["pick_pose"]
# rel_place_pos = mat_dict["place_pose"]
block_pos_world = mat_dict["block_world"]

# cache some common handles for later use
mano_indices, table_indices = [], []
block_indices = [[] for _ in range(num_envs)]
block_masses = [[] for _ in range(num_envs)]
envs = []

goal_list = []

# create and populate the environments
for i in range(num_envs):
    # create env
    env_ptr = gym.create_env(sim, lower, upper, int(np.sqrt(num_envs)))
    envs.append(env_ptr)

    # create table and set properties
    table_handle = gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 1, 0) # 001
    table_sim_index = gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
    table_indices.append(table_sim_index)

    gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(60, 33, 0) / 255)

    # add region
    region_pose.p.x = table_pose.p.x + region_xy[0]
    region_pose.p.y = table_pose.p.y + region_xy[1]
    region_pose.p.z = table_dims.z #+ 0.001
    region_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))

    region_handle = gym.create_actor(env_ptr, region_asset, region_pose, "target", i, 1, 1) # 001
    gym.set_rigid_body_color(env_ptr, region_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0., 0., 0.))

    for cnt, idx in enumerate(block_list):
        block_pose = gymapi.Transform()
        block_pose.p.x = table_pose.p.x + rand_xy[cnt, 0]
        block_pose.p.y = table_pose.p.y + rand_xy[cnt, 1]
        block_pose.p.z = table_dims.z + 0.03

        r1 = R.from_euler('z', np.random.uniform(-math.pi, math.pi))
        r2 = R.from_matrix(goal_pose[cnt][:3,:3])
        rot = r1 * r2
        euler = rot.as_euler("xyz", degrees=False)

        block_pose.r = gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2])
        # block_pose=utils.mat2gymapi_transform(block_pos_world[cnt])
        block_handle = gym.create_actor(env_ptr, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i, 2 ** (cnt + 1), cnt + 2) # 010
        
        # block_pose = utils.mat2gymapi_transform(utils.gymapi_transform2mat(region_pose)@goal_pose[cnt])
        # block_handle = gym.create_actor(env, block_asset_list[idx], block_pose, 'block_' + block_type[idx], i+1)
        # block_handles.append(block_handle)

        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env_ptr, block_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        block_idx = gym.get_actor_index(env_ptr, block_handle, gymapi.DOMAIN_SIM)
        block_indices[i].append(block_idx)

    # save block goal pose
    goal = []
    for j in range(len(block_list)):
        tmp_pose = utils.mat2gymapi_transform(utils.gymapi_transform2mat(region_pose) @ goal_pose[j])
        goal_place_pose = torch.Tensor((tmp_pose.p.x,tmp_pose.p.y,tmp_pose.p.z,tmp_pose.r.x,tmp_pose.r.y,tmp_pose.r.z,tmp_pose.r.w)).to(device)
        goal_preplace_pose = torch.Tensor((tmp_pose.p.x,tmp_pose.p.y,0.7,tmp_pose.r.x,tmp_pose.r.y,tmp_pose.r.z,tmp_pose.r.w)).to(device)
        goal.append(goal_preplace_pose)
        goal.append(goal_place_pose)
        goal.append(goal_preplace_pose)
    
    goal_list.append(goal)
    # create mano and set properties
    mano_handle = gym.create_actor(env_ptr, mano_asset, handobj_start_pose, "mano", i, 2 ** (len(block_list)), len(block_list) + 2) # 100
    mano_sim_index = gym.get_actor_index(env_ptr, mano_handle, gymapi.DOMAIN_SIM)
    mano_indices.append(mano_sim_index)

    gym.set_actor_dof_properties(env_ptr, mano_handle, mano_dof_props)

mano_indices = to_torch(mano_indices, dtype=torch.long, device=device)
block_indices = to_torch(block_indices, dtype=torch.long, device=device) 


def get_hand_rel_mat():
    obj_init = np.array([-0.19434147, -0.11968146, 0.01548913, -0.50471319, 0.49524648, -0.4952225 , 0.50472785])
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
    hand_goal_mat = np.eye(4)
    hand_goal_mat[:3, :3] = R.from_euler("XYZ", hand_goal_pose[3:6]).as_matrix()
    hand_goal_mat[:3, 3] = hand_goal_pose[0:3]
    return torch.tensor(np.linalg.inv(obj_init_mat) @ hand_goal_mat, dtype=torch.float32)


# def update(:
    

#     goal_pose = goal_list[stage_tensor]
#     init_pose = root_state_tensor[block_indices[:,step],:7]

#     step_size = (goal_pose-init_pose)*0.001

#     root_state_tensor[block_indices[:,step], :7] += step_size
#     goal_obj_indices = torch.tensor([block_indices[:,step]]).to(torch.int32)
#     gym.set_actor_root_state_tensor_indexed(
#         sim,
#         gymtorch.unwrap_tensor(root_state_tensor),
#         gymtorch.unwrap_tensor(goal_obj_indices),
#         len(goal_obj_indices)
#     )

#     # set hand pose
#     cur_obj_mat = torch.eye(4)
#     cur_obj_mat[:3, 3] = root_state_tensor[block_indices[:,step], 0:3]
#     cur_obj_mat[:3, :3] = pytorch3d.transforms.quaternion_to_matrix(root_state_tensor[block_indices[:,step], 3:7][[3, 0, 1, 2]])

#     new_wrist_mat = cur_obj_mat @ get_hand_rel_mat()
#     dof_state[0:3, 0] = new_wrist_mat[:3, 3]
#     dof_state[3:6, 0] = pytorch3d.transforms.matrix_to_euler_angles(new_wrist_mat[:3, :3], "XYZ")
#     dof_indices = torch.tensor([mano_indices]).to(dtype=torch.int32)
#     gym.set_dof_state_tensor_indexed(sim,
#                                     gymtorch.unwrap_tensor(dof_state),
#                                     gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
#     target = dof_state[:, 0].clone()
#     gym.set_dof_position_target_tensor_indexed(sim,
#                                             gymtorch.unwrap_tensor(target),
#                                             gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
    



# Look at the first env
cam_pos = gymapi.Vec3(1, 0, 1.5)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

gym.prepare_sim(sim)

# create observation buffer
_dof_states_tensor = gym.acquire_dof_state_tensor(sim)
dof_states_tensor = gymtorch.wrap_tensor(_dof_states_tensor)

actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

stage_tensor = torch.zeros((num_envs)).to(device)
    
# reset_idx()

step = 0

cnt = 0
while not gym.query_viewer_has_closed(viewer):            
    cnt += 1

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)


    # print(root_state_tensor[block_indices[:,0], :7].shape)
    # print(block_indices)
    print(len(goal_list))

    # print('-' * 10)
    # print(root_state_tensor[ycb_indices, :7])
    # print()

    # process predicted actions
    # actions_tensor = torch.tile(data_hand_ref, (num_envs // data_num, 1))

    # tf_mat = pose7d_to_matrix(root_state_tensor[mano_indices, :7])
    # cur_wrist_mat = pose6d_to_matrix(actions_tensor[:, :6], "XYZ")
    # new_wrist_mat = torch.bmm(torch.linalg.inv(tf_mat), cur_wrist_mat)
    # new_wrist_tensor = matrix_to_pose_6d(new_wrist_mat, "XYZ")

    # actions_tensor[:, :6] = new_wrist_tensor

    # set position target
    # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(actions_tensor))

    # rigidbody_transforms = gym.get_actor_rigid_body_states(env0, mano_hand0, gymapi.STATE_ALL)[rigidbodyid_list]["pose"]["p"]
    # rigidbody_pos = np.vstack([rigidbody_transforms["x"], rigidbody_transforms["y"], rigidbody_transforms["z"]]).T

    # for i in range(rigidbody_pos.shape[0]):
    #     body_states = gym.get_actor_rigid_body_states(env0, sphere_list[i], gymapi.STATE_ALL)
    #     body_states["pose"]["p"] = tuple(rigidbody_pos[i])
    #     gym.set_actor_rigid_body_states(env0, sphere_list[i], body_states, gymapi.STATE_ALL)

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


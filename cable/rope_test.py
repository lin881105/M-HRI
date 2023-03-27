"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Soft Body
---------
Simple import of a URDF with a soft body link and rigid body press mechanism
"""

import math
import random
from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np



# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="FEM Soft Body Example")
# if args.physics_engine != gymapi.SIM_FLEX:
#     print("*** Soft body example only supports FleX")
#     print("*** Run example with --flex flag")
#     quit()

random.seed(7)

# simulation parameters
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 3
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.75             
sim_params.flex.shape_collision_margin = 0.1

# enable Von-Mises stress visualization
sim_params.stress_visualization = True
sim_params.stress_visualization_min = 0.0
sim_params.stress_visualization_max = 1.e+5

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.compute_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# create table asset
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# load urdf for sphere asset used to create softbody
asset_root = "../../assets"
soft_asset_file = "urdf/rope/rope.urdf"

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

# get link index of panda hand, which we will use as end effector
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["panda_hand"]

soft_thickness = 0   # important to add some thickness to the soft body to avoid interpenetrations

asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
asset_options.thickness = soft_thickness
# asset_options.collapse_fixed_joints = True
asset_options.armature = 0.1
# asset_options.disable_gravity = True
# asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

asset_soft_body_count = gym.get_asset_soft_body_count(soft_asset)
asset_soft_materials = gym.get_asset_soft_materials(soft_asset)

# Print asset soft material properties
print('Soft Material Properties:')
for i in range(asset_soft_body_count):
    mat = asset_soft_materials[i]
    print(f'(Body {i}) youngs: {mat.youngs} poissons: {mat.poissons} damping: {mat.damping}')

# set up the env grid
num_envs = 1
spacing = 3.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5,0.0, 0.5 * table_dims.z )

soft_pose = gymapi.Transform()
soft_pose.p = gymapi.Vec3(0.3, 0.5, 0.6)
# cache some common handles for later use
envs = []
soft_actors = []

init_pos_list = []
init_rot_list = []


print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
for i in range(num_envs):

    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)



    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

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


    

    # add soft body + rail actor
    soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 1)
    soft_actors.append(soft_actor)

    # set soft material within a range of default
    actor_default_soft_materials = gym.get_actor_soft_materials(env, soft_actor)
    actor_soft_materials = gym.get_actor_soft_materials(env, soft_actor)
    for j in range(asset_soft_body_count):
        youngs = actor_soft_materials[j].youngs
        actor_soft_materials[j].youngs = random.uniform(youngs * 0.2, youngs * 2.4)

        poissons = actor_soft_materials[j].poissons
        actor_soft_materials[j].poissons = random.uniform(poissons * 0.8, poissons * 1.2)

        damping = actor_soft_materials[j].damping
        # damping is 0, instead we just randomize from scratch
        actor_soft_materials[j].damping = random.uniform(0.0, 0.08)**2

        gym.set_actor_soft_materials(env, soft_actor, actor_soft_materials)

    # enable pd-control on rail joint to allow
    # control of the press using the GUI
    gym.set_joint_target_position(env, gym.get_joint_handle(env, "soft", "rail"), 0.0)

# Point camera at environments
cam_pos = gymapi.Vec3(4.0, 2.8, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

camera_properties = gymapi.CameraProperties()
camera_properties.width = 640
camera_properties.height = 480

# Set a fixed position and look-target for the first camera
# position and target location are in the coordinate frame of the environment
camera_handle = gym.create_camera_sensor(envs[i], camera_properties)
camera_position = gymapi.Vec3(1, 0.5, 1.0)
camera_target = gymapi.Vec3(0, 0, 0)
gym.set_camera_location(camera_handle, envs[i], camera_position, camera_target)

# options
flag_draw_contacts = False
flag_compute_pressure = False

img = []
frame_count=0

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.render_all_camera_sensors(sim)

    if (flag_draw_contacts):
        gym.clear_lines(viewer)
        for env in envs:
            gym.draw_env_soft_contacts(viewer, env, gymapi.Vec3(0.7, 0.2, 0.15), 0.0005, True, True)

    if (flag_compute_pressure):
        # read tetrahedral and triangle data from simulation
        (tet_indices, tet_stress) = gym.get_sim_tetrahedra(sim)
        (tri_indices, tri_parents, tri_normals) = gym.get_sim_triangles(sim)

        # for env in envs:

        #     # get range (start, count) for each soft actor (assumes one soft body)
        #     tet_range = gym.get_actor_tetrahedra_range(env, 0, 0)
        #     tri_range = gym.get_actor_triangle_range(env, 0, 0)

        #     for i in range(tri_range.start, tri_range.start + tri_range.count):

        #         # 'parent' tetrahedron of this triangle
        #         parent = tri_parents[i]
        #         parent_stress = tet_stress[parent]

        #         # convert to numpy
        #         stress = np.matrix([(parent_stress.x.x, parent_stress.y.x, parent_stress.z.x),
        #                             (parent_stress.x.y, parent_stress.y.y, parent_stress.z.y),
        #                             (parent_stress.x.z, parent_stress.y.z, parent_stress.z.z)])

        #         normal = np.array((tri_normals[i].x, tri_normals[i].y, tri_normals[i].z))

        #         # compute force density (since normal is normalized)
        #         force = np.dot(stress, normal)
    rgb_filename = "soft/frame%d.png" % (frame_count)
    frame_count += 1
    gym.write_camera_image_to_file(sim, envs[i], camera_handle, gymapi.IMAGE_COLOR, rgb_filename)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
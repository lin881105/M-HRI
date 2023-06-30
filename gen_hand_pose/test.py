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
from scipy.spatial.transform import Rotation as Rot

from manopth.manolayer import ManoLayer
from mano_pybullet.hand_model import HandModel45

MANO_TO_CONTACT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 6,
    9: 7,
    10:8,
    11:9,
    12:9,
    13:10,
    14:11,
    15:12,
    16:12,
    17:13,
    18:14,
    19:15,
    20:15,
}

def load_dexycb_data(root_dir, foldername):
    pth = os.path.join(root_dir, foldername)
    pose = np.load(pth + "/solve_mano.npz")
    
    with open(pth + "/meta.yml", 'r') as f:
        meta = yaml.safe_load(f)

    extr_file = root_dir + "/calibration/extrinsics_" + meta["extrinsics"] + "/extrinsics.yml"

    with open(extr_file, "r") as f:
        extr = yaml.load(f, Loader=yaml.FullLoader)

    return pose, meta, extr

# process mano
def tag_transformation(q, t, tag_R_inv, tag_t_inv):
    """Transforms 6D pose to tag coordinates."""
    q_trans = np.zeros((*q.shape[:2], 4), dtype=q.dtype)
    t_trans = np.zeros(t.shape, dtype=t.dtype)

    i = np.any(q != 0, axis=2) | np.any(t != 0, axis=2)
    q = q[i]
    t = t[i]

    if q.shape[1] == 4:
        R = Rot.from_quat(q).as_matrix().astype(np.float32)
    if q.shape[1] == 3:
        R = Rot.from_rotvec(q).as_matrix().astype(np.float32)
    R = np.matmul(tag_R_inv, R)
    t = np.matmul(tag_R_inv, t.T).T + tag_t_inv
    q = Rot.from_matrix(R).as_quat().astype(np.float32)

    q_trans[i] = q
    t_trans[i] = t

    return q_trans, t_trans

def mano_to_handoversim(pose, meta, extr):
    tag_T = np.array(extr["extrinsics"]["apriltag"], dtype=np.float32).reshape(3, 4)
    tag_R = tag_T[:, :3]
    tag_t = tag_T[:, 3]
    tag_R_inv = tag_R.T
    tag_t_inv = np.matmul(tag_R_inv, -tag_t)

    # Load MANO model.
    mano = {}
    for k, name in zip(("right", "left"), ("RIGHT", "LEFT")):
        mano_file = os.path.join(
            os.path.dirname(__file__), "manopth", "mano", "models", "MANO_{}.pkl".format(name)
        )
        with open(mano_file, "rb") as f:
            mano[k] = pickle.load(f, encoding="latin1")

    # Process MANO pose.
    mano_betas = []
    root_trans = []
    comp = []
    mean = []
    
    betas = np.zeros(10)
    mano_betas.append(betas)
    v = mano["right"]["shapedirs"].dot(betas) + mano["right"]["v_template"]
    r = mano["right"]["J_regressor"][0].dot(v)[0]
    root_trans.append(r)
    comp.append(mano["right"]["hands_components"])
    mean.append(mano["right"]["hands_mean"])

    root_trans = np.array(root_trans, dtype=np.float32)
    comp = np.array(comp, dtype=np.float32)
    mean = np.array(mean, dtype=np.float32)

    i = np.any(pose["pose_m"] != 0.0, axis=2)

    q = pose["pose_m"][:, :, 0:3]
    t = pose["pose_m"][:, :, 48:51]

    t[i] += root_trans[np.nonzero(i)[1]]
    q, t = tag_transformation(q, t, tag_R_inv, tag_t_inv)
    t[i] -= root_trans[np.nonzero(i)[1]]

    p = pose["pose_m"][:, :, 3:48]
    p = np.einsum("abj,bjk->abk", p, comp) + mean
    p[~i] = 0.0

    q_i = q[i]
    q_i = Rot.from_quat(q_i).as_rotvec().astype(np.float32)
    q = np.zeros((*q.shape[:2], 3), dtype=q.dtype)
    q[i] = q_i
    q = np.dstack((q, p))
    models = {}
    origins = {}
    for o, (s, b) in enumerate(zip(meta["mano_sides"], mano_betas)):
        k = "NaN"
        models[k] = HandModel45(
            left_hand=s == "left", models_dir=os.path.join(os.path.dirname(__file__), "manopth", "mano", "models"), betas=b
        )
        origins[k] = models[k].origins(b)[0]
        sid = np.nonzero(np.any(q[:, o] != 0, axis=1))[0][0]
        eid = np.nonzero(np.any(q[:, o] != 0, axis=1))[0][-1]
        for f in range(sid, eid + 1):
            mano_pose = q[f, o]
            trans = t[f, o]
            angles, basis = models[k].mano_to_angles(mano_pose)
            trans = trans + origins[k] - basis @ origins[k]
            q[f, o, 3:48] = angles
            t[f, o] = trans
    q_i = q[i]
    q_i_base = q_i[:, 0:3]
    q_i_pose = q_i[:, 3:48].reshape(-1, 3)
    q_i_base = Rot.from_rotvec(q_i_base).as_quat().astype(np.float32)
    q_i_pose = Rot.from_euler("XYZ", q_i_pose).as_quat().astype(np.float32)
    q_i_pose = q_i_pose.reshape(-1, 60)
    q_i = np.hstack((q_i_base, q_i_pose))
    q = np.zeros((*q.shape[:2], 64), dtype=q.dtype)
    q[i] = q_i

    # q, t = _resample(q, t, time_step_resample)

    i = np.any(q != 0.0, axis=2)
    q_i = q[i]
    q = np.zeros((*q.shape[:2], 48), dtype=q.dtype)
    for o in range(q.shape[1]):
        q_i_o = q_i[np.nonzero(i)[1] == o]
        q_i_o = q_i_o.reshape(-1, 4)
        q_i_o = Rot.from_quat(q_i_o).as_euler("XYZ").astype(np.float32)
        q_i_o = q_i_o.reshape(-1, 48)
        # https://math.stackexchange.com/questions/463748/getting-cumulative-euler-angle-from-a-single-quaternion
        q_i_o[:, 0:3] = np.unwrap(q_i_o[:, 0:3], axis=0)
        q[i[:, o], o] = q_i_o
    pose_m = np.dstack((t, q))

    return pose_m

def visualize_scene(mano_layer, verts, tag_T, obj_mesh, final_ee):
    # load hand mesh
    mesh = o3d.geometry.TriangleMesh()
    np_vertices = verts.cpu().detach().numpy().reshape(778, 3)
    np_triangles = mano_layer.th_faces.cpu().detach().numpy()
    mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([255/255, 217/255, 179/255])
    mesh.scale(1/1000, center=[0, 0, 0])
    mesh.transform(np.linalg.inv(tag_T))

    # obj mesh already transform to tag coordinate

    # load final ee sphere
    sphere_list = []
    for i in range(21):
        tmp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=20)
        tmp_sphere.translate(final_ee.reshape(21, 3)[i, :])
        tmp_sphere.paint_uniform_color([1, 0, 0])
        sphere_list.append(tmp_sphere)
    
    o3d.visualization.draw_geometries([mesh, obj_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame()] + sphere_list)

def compute_contact(hand_param, tag_T, obj_mesh_o3d):
    hand_param = torch.from_numpy(hand_param)
    mano_layer = ManoLayer(mano_root='manopth/mano/models',
                           flat_hand_mean=False,
                           ncomps=45,
                           use_pca=True)
    
    # transform final joint position to tag coordinate and save
    verts, joints = mano_layer(th_pose_coeffs=hand_param[:, :48], th_trans=hand_param[:, 48:])
    full_joints = torch.zeros([21, 3])

    # convert joint order
    joints = joints[0]
    full_joints[0] = joints[0] # wrist
    full_joints[1:5] = joints[5:9] # forefinger
    full_joints[5:9] = joints[9:13] # middlefinger
    full_joints[9:13] = joints[17:21] # pinky
    full_joints[13:17] = joints[13:17] # ringfinger
    full_joints[17:21] = joints[1:5] # thumb

    final_ee = full_joints.reshape(-1).detach().numpy() / 1000
    final_ee = (np.linalg.inv(tag_T) @ np.vstack([final_ee.reshape(21, 3).T, np.ones(21)]))[:3, :].T

    # visualize scene
    # visualize_scene(mano_layer, verts, tag_T, obj_mesh_o3d, final_ee)

    # compute hand object contact and save
    contact_threshold = 0.015
    obj_pcd = np.asarray(obj_mesh_o3d.vertices)
    ftip_pos = final_ee.reshape(1, -1, 3)

    final_relftip_pos = np.tile(ftip_pos, (obj_pcd.shape[0], 1, 1))

    verts = obj_pcd[:, np.newaxis]
    diff_vert_fpos = np.linalg.norm(final_relftip_pos - verts, axis=-1)

    min_vert_dist = np.min(diff_vert_fpos,axis=0)

    idx_below_thresh = np.where(min_vert_dist < contact_threshold)[0]
    target_idxs = [MANO_TO_CONTACT[idx] for idx in idx_below_thresh]

    target_contacts = np.zeros(16)
    target_contacts[target_idxs] = 1
    target_contacts[-1] = 1

    return final_ee, target_contacts
    
# process ycb
def process_ycb(pose, meta, extr, ycb_name):
    tag_T = np.array(extr["extrinsics"]["apriltag"], dtype=np.float32).reshape(3, 4)
    tag_T = np.vstack([tag_T, np.array([0, 0, 0, 1])])

    ycb_pose = np.load('/home/hcis-s12/Desktop/M-HRI/gen_hand_pose/obj_pose.npy')

    # load open3d object mesh and transform to tag coordinate
    obj_file = f"/home/hcis-s12/Lego_Assembly/Dataset/DexYCB/dex-ycb/data_dexycb/models/{ycb_name}/textured.obj"
    obj_mesh = o3d.io.read_triangle_mesh(obj_file)
    obj_mesh.remove_duplicated_vertices()

    obj_pos_mat = np.eye(4)
    obj_pos_mat[:3, :3] = Rot.from_quat(ycb_pose[1][3:]).as_matrix()
    obj_pos_mat[:3, 3] = ycb_pose[1][:3].T
    obj_mesh.transform(obj_pos_mat)

    return ycb_pose, meta["num_frames"]-1, obj_mesh, tag_T

# run
def run(root_dir, foldername):
    out_dict = {}

    block_type = ["A", "B", "D", "E"]

    for cnt, btype in enumerate(tqdm(block_type)):
        out_dict[cnt] = {}

        # load dexycb_data
        pose, meta, extr = load_dexycb_data(root_dir, foldername)

        # process ycb
        ycb_pose_list, final_frame, obj_mesh_o3d, tag_T = process_ycb(pose, meta, extr, btype)

        # process mano
        # final_ee, target_contacts = compute_contact(pose["pose_m"][grasp_ref_frame], tag_T, obj_mesh_o3d)
        hand_urdf_pose = mano_to_handoversim(pose, meta, extr)

        # output to pickle dict
        out_dict[cnt]["obj_init"] = ycb_pose_list[cnt]
        # out_dict[cnt]["subgoal_1"]["obj_grasp"] = ycb_pose_list[1]
        # out_dict[cnt]["subgoal_1"]["obj_final"] = ycb_pose_list[2]

        # out_dict[cnt]["subgoal_1"]["hand_traj_reach"] = hand_urdf_pose[start_frame:grasp_ref_frame, :, :]
        # out_dict[cnt]["subgoal_1"]["hand_traj_grasp"] = hand_urdf_pose[grasp_ref_frame+1:, :, :]
        out_dict[cnt]["hand_ref_pose"] = hand_urdf_pose[cnt, :, :].reshape(1, 1, 51)
        # out_dict[cnt]["subgoal_1"]["hand_ref_position"] = final_ee
        # out_dict[cnt]["subgoal_1"]["hand_contact"] = target_contacts
    
    # write file
    # with open(f"./test.pickle", "wb") as openfile:
    #     pickle.dump(out_dict, openfile)

    return out_dict

# isaac
def euler_to_quat(euler_angles: torch.Tensor, convention: str):
    rot_mat = pytorch3d.transforms.euler_angles_to_matrix(euler_angles, convention)
    quaternion = pytorch3d.transforms.matrix_to_quaternion(rot_mat)[:, [1, 2, 3, 0]]

    return quaternion

def pose7d_to_matrix(pose7d: torch.Tensor):
    matrix = torch.eye(4).reshape(1, 4, 4).repeat(pose7d.shape[0], 1, 1)
    matrix[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(pose7d[:, [6, 3, 4, 5]])
    matrix[:, :3, 3] = pose7d[:, :3]

    return matrix

def pose6d_to_matrix(pose6d: torch.Tensor, convention: str):
    matrix = torch.eye(4).reshape(1, 4, 4).repeat(pose6d.shape[0], 1, 1)
    matrix[:, :3, :3] = pytorch3d.transforms.euler_angles_to_matrix(pose6d[:, 3:], convention)
    matrix[:, :3, 3] = pose6d[:, :3]

    return matrix

def matrix_to_pose_6d(matrix: torch.Tensor, convention: str):
    pose_6d = torch.zeros(matrix.shape[0], 6)
    pose_6d[:, 3:] = pytorch3d.transforms.matrix_to_euler_angles(matrix[:, :3, :3], convention)
    pose_6d[:, :3] = matrix[:, :3, 3]

    return pose_6d

class IsaacSim():
    def __init__(self, data):
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        # self.num_envs = 20
        self.data = data
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
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

    def create_sim(self):
        # parse arguments
        args = gymutil.parse_arguments(description="Joint control Methods Example")

        args.use_gpu = False
        args.use_gpu_pipeline = False
        self.device = 'cpu' if torch.cuda.is_available() else 'cpu'

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

        self._read_train_data(self.data)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(spacing, 0.75 * spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file_mano = "urdf/mano/zeros/mano_addtips.urdf"
        asset_file_block_list = ["urdf/block_assembly/block_{}.urdf".format(t) for t in ["A", "B", "D", "E"]]

        # create mano asset
        asset_path_mano = os.path.join(asset_root, asset_file_mano)
        asset_root_mano = os.path.dirname(asset_path_mano)
        asset_file_mano = os.path.basename(asset_path_mano)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        mano_asset = self.gym.load_asset(self.sim, asset_root_mano, asset_file_mano, asset_options)
        self.num_mano_dofs = self.gym.get_asset_dof_count(mano_asset)
        
        # create ycb asset
        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        # asset_path_ycb = os.path.join(asset_root, asset_file_ycb)
        # asset_root_ycb = os.path.dirname(asset_path_ycb)
        # asset_file_ycb = os.path.basename(asset_path_ycb)
        # ycb_asset = self.gym.create_box(self.sim, *[0.05, 0.075, 0.05], asset_options)
        block_asset_list = [self.gym.load_asset(self.sim, asset_root, file, asset_options) for file in asset_file_block_list]

        # create table asset
        table_thickness = 0.05
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[2, 1.5, table_thickness], asset_options)

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

        # set default pose
        handobj_start_pose = gymapi.Transform()
        handobj_start_pose.p = gymapi.Vec3(0, 0, 0)
        handobj_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.5, 0.2, 0.5 - table_thickness / 2)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # cache some common handles for later use
        self.mano_indices, self.ycb_indices, self.table_indices = [], [], []
        self.ycb_masses = []
        self.envs = []

        # create and populate the environments
        for i in range(num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            # create table and set properties
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0) # 001
            table_sim_index = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_sim_index)

            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(60, 33, 0) / 255)

            # create ycb and set properties
            ycb_handle = self.gym.create_actor(env_ptr, block_asset_list[i], handobj_start_pose, "YCB", i, 2, 1) # 010
            ycb_sim_index = self.gym.get_actor_index(env_ptr, ycb_handle, gymapi.DOMAIN_SIM)
            self.ycb_indices.append(ycb_sim_index)

            ycb_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, ycb_handle)
            ycb_rb_props[0].mass = 0.3
            self.gym.set_actor_rigid_body_properties(env_ptr, ycb_handle, ycb_rb_props)
            self.ycb_masses.append(ycb_rb_props[0].mass)
            
            # create mano and set properties
            mano_handle = self.gym.create_actor(env_ptr, mano_asset, handobj_start_pose, "mano", i, 4, 2) # 100
            mano_sim_index = self.gym.get_actor_index(env_ptr, mano_handle, gymapi.DOMAIN_SIM)
            self.mano_indices.append(mano_sim_index)

            self.gym.set_actor_dof_properties(env_ptr, mano_handle, mano_dof_props)

        self.mano_indices = to_torch(self.mano_indices, dtype=torch.long, device=self.device)
        self.ycb_indices = to_torch(self.ycb_indices, dtype=torch.long, device=self.device)

        self.ycb_masses = to_torch(self.ycb_masses, dtype=torch.float32, device=self.device)

    def _read_train_data(self, data):
        self.good_data = list(range(len(self.data)))
        self.data_num = len(self.good_data)
        self.num_envs = len(self.good_data)
        self.data_hand_ref = torch.from_numpy(np.array([data[i]["hand_ref_pose"][0, 0] for i in self.good_data]))
        self.data_hand_ref = self.data_hand_ref[[2, 1, 0, 3]]
        self.data_obj_init = torch.from_numpy(np.array([data[i]["obj_init"] for i in self.good_data]))

        # add table shift
        self.data_hand_ref[:, 2] += 0.5
        self.data_obj_init[:, 2] += 0.5
        
        # to torch
        self.data_hand_ref = to_torch(self.data_hand_ref, dtype=torch.float32, device=self.device)
        self.data_obj_init = to_torch(self.data_obj_init, dtype=torch.float32, device=self.device)

    def reset_idx(self):
        # reset hand root pose
        hand_ref_pose = torch.tile(self.data_hand_ref, (self.num_envs // self.data_num, 1)).clone()

        # reset hand pose
        self.dof_state[:, 0] = hand_ref_pose.reshape(-1)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(hand_ref_pose.reshape(-1)))

        # reset object pose
        obj_init_pose = torch.tile(self.data_obj_init, (self.num_envs // self.data_num, 1)).clone()
        
        self.root_state_tensor[self.ycb_indices, 0:3] = obj_init_pose[:, 0:3]
        self.root_state_tensor[self.ycb_indices, 3:7] = obj_init_pose[:, 3:7]
        self.root_state_tensor[self.ycb_indices, 7:13] = torch.zeros_like(self.root_state_tensor[self.ycb_indices, 7:13])
        init_object_indices = self.ycb_indices.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(init_object_indices), len(init_object_indices))

    def simulate(self):
        torch.set_printoptions(sci_mode=False)

        self.reset_idx()

        cnt = 0
        while not self.gym.query_viewer_has_closed(self.viewer):
            if cnt >= self.max_episode_length:
                print('-' * 10)
                print('reset')
                print()

                # new_obj_pose = self.root_state_tensor[self.ycb_indices, :7]
                # break

                self.reset_idx()
                cnt = 0
            
            cnt += 1

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            self.gym.refresh_actor_root_state_tensor(self.sim)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

        for cnt, i in enumerate(self.good_data):
            tmp_pose = new_obj_pose[cnt].numpy().copy()
            tmp_pose[2] -= 0.5

            self.data[i]["obj_init"][:] = tmp_pose

        # # write file
        # with open(f"./test_ft.pickle", "wb") as openfile:
        #     pickle.dump(self.data, openfile)
        

if __name__ == '__main__':
    out_dict = {}
    root_dir = "/home/hcis-s12/Lego_Assembly/Dataset/DexYCB/dex-ycb/data"
    foldername = "20230629_150209_subsample"

    # run
    out_dict = run(root_dir, foldername)
    
    # read file
    with open(f"./test_ft.pickle", "rb") as openfile:
        out_dict = pickle.load(openfile)

    print('-' * 20)
    print('Success dump the data pickle')
    print()

    # visualize in isaac
    issac = IsaacSim(out_dict)
    issac.simulate()

    print('-' * 20)
    print('Success dump the fine tune data pickle')
    print()
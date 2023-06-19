from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

import os
import glob
import time
import yaml
import trimesh
import torch
import pickle
import numpy as np
import open3d as o3d
import pytorch3d.transforms
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot

from manopth.manolayer import ManoLayer
from mano_pybullet.hand_model import HandModel45
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type",type=int,default=0)
args = parser.parse_args()
# def fast_load_obj(file_obj, **kwargs):
#     """
#     Code slightly adapted from trimesh (https://github.com/mikedh/trimesh)
#     Thanks to Michael Dawson-Haggerty for this great library !
#     loads an ascii wavefront obj file_obj into kwargs
#     for the trimesh constructor.
#     vertices with the same position but different normals or uvs
#     are split into multiple vertices.
#     colors are discarded.
#     parameters
#     ----------
#     file_obj : file object
#                    containing a wavefront file
#     returns
#     ----------
#     loaded : dict
#                 kwargs for trimesh constructor
#     """
#     # make sure text is utf-8 with only \n newlines
#     text = file_obj.read()
#     if hasattr(text, 'decode'):
#         text = text.decode('utf-8')
#     text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'
#     meshes = []

#     def append_mesh():
#         # append kwargs for a trimesh constructor
#         # to our list of meshes
#         if len(current['f']) > 0:
#             # get vertices as clean numpy array
#             vertices = np.array(
#                 current['v'], dtype=np.float64).reshape((-1, 3))
#             # do the same for faces
#             faces = np.array(current['f'], dtype=np.int64).reshape((-1, 3))
#             # get keys and values of remap as numpy arrays
#             # we are going to try to preserve the order as
#             # much as possible by sorting by remap key
#             keys, values = (np.array(list(remap.keys())),
#                             np.array(list(remap.values())))
#             # new order of vertices
#             vert_order = values[keys.argsort()]
#             # we need to mask to preserve index relationship
#             # between faces and vertices
#             face_order = np.zeros(len(vertices), dtype=np.int64)
#             face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)
#             # apply the ordering and put into kwarg dict
#             loaded = {
#                 'vertices': vertices[vert_order],
#                 'faces': face_order[faces],
#                 'metadata': {}
#             }
#             # build face groups information
#             # faces didn't move around so we don't have to reindex
#             if len(current['g']) > 0:
#                 face_groups = np.zeros(len(current['f']) // 3, dtype=np.int64)
#                 for idx, start_f in current['g']:
#                     face_groups[start_f:] = idx
#                 loaded['metadata']['face_groups'] = face_groups
#             # we're done, append the loaded mesh kwarg dict
#             meshes.append(loaded)
#     attribs = {k: [] for k in ['v']}
#     current = {k: [] for k in ['v', 'f', 'g']}
#     # remap vertex indexes {str key: int index}
#     remap = {}
#     next_idx = 0
#     group_idx = 0
#     for line in text.split("\n"):
#         line_split = line.strip().split()
#         if len(line_split) < 2:
#             continue
#         if line_split[0] in attribs:
#             # v, vt, or vn
#             # vertex, vertex texture, or vertex normal
#             # only parse 3 values, ignore colors
#             attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
#         elif line_split[0] == 'f':
#             # a face
#             ft = line_split[1:]
#             if len(ft) == 4:
#                 # hasty triangulation of quad
#                 ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
#             for f in ft:
#                 # loop through each vertex reference of a face
#                 # we are reshaping later into (n,3)
#                 if f not in remap:
#                     remap[f] = next_idx
#                     next_idx += 1
#                     # faces are "vertex index"/"vertex texture"/"vertex normal"
#                     # you are allowed to leave a value blank, which .split
#                     # will handle by nicely maintaining the index
#                     f_split = f.split('/')
#                     current['v'].append(attribs['v'][int(f_split[0]) - 1])
#                 current['f'].append(remap[f])
#         elif line_split[0] == 'o':
#             # defining a new object
#             append_mesh()
#             # reset current to empty lists
#             current = {k: [] for k in current.keys()}
#             remap = {}
#             next_idx = 0
#             group_idx = 0
#         elif line_split[0] == 'g':
#             # defining a new group
#             group_idx += 1
#             current['g'].append((group_idx, len(current['f']) // 3))
#     if next_idx > 0:
#         append_mesh()
#     return meshes

# def get_diameter(vp):
#     x = vp[:, 0].reshape((1, -1))
#     y = vp[:, 1].reshape((1, -1))
#     z = vp[:, 2].reshape((1, -1))
#     x_max, x_min, y_max, y_min, z_max, z_min = np.max(
#         x), np.min(x), np.max(y), np.min(y), np.max(z), np.min(z)
#     diameter_x = abs(x_max - x_min)
#     diameter_y = abs(y_max - y_min)
#     diameter_z = abs(z_max - z_min)
#     diameter = np.sqrt(diameter_x**2 + diameter_y**2 + diameter_z**2)
#     return diameter

# def load_objects_brick(obj_root):
#     object_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
#                     'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
#     obj_pc, obj_face, obj_scale, obj_pc_resampled, obj_resampled_faceid = {}, {}, {}, {}, {}
#     for obj_name in object_names:
#         texture_path = os.path.join(obj_root, obj_name, 'textured_simple.obj')
#         texture = fast_load_obj(open(texture_path))[0]
#         obj_pc[obj_name] = texture['vertices']
#         obj_face[obj_name] = texture['faces']
#         obj_scale[obj_name] = get_diameter(texture['vertices'])
#         obj_pc_resampled[obj_name] = np.load(
#             texture_path.replace('textured_simple.obj', 'resampled.npy'))
#         obj_resampled_faceid[obj_name] = np.load(
#             texture_path.replace('textured_simple.obj', 'resample_face_id.npy'))
#         #resample_obj_xyz(texture['vertices'], texture['faces'], texture_path)
#     return obj_pc, obj_face, obj_scale, obj_pc_resampled, obj_resampled_faceid

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

# process mano
def mano_to_handoversim(pose):
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

    i = np.any(pose != 0.0, axis=2)

    q = pose[:, :, 0:3]
    t = pose[:, :, 48:51]

    p = pose[:, :, 3:48]
    p = np.einsum("abj,bjk->abk", p, comp) + mean
    p[~i] = 0.0

    q_i = q[i]
    q = np.zeros((*q.shape[:2], 3), dtype=q.dtype)
    q[i] = q_i
    q = np.dstack((q, p))
    models = {}
    origins = {}
    for o, (s, b) in enumerate(zip(["right"], mano_betas)):
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

def visualize_scene(mano_layer, verts, obj_mesh, final_ee):
    # load hand mesh
    mesh = o3d.geometry.TriangleMesh()
    np_vertices = verts.cpu().detach().numpy().reshape(778, 3)
    np_triangles = mano_layer.th_faces.cpu().detach().numpy()
    mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([255/255, 217/255, 179/255])

    # load final ee sphere
    sphere_list = []
    for i in range(21):
        tmp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001, resolution=20)
        tmp_sphere.translate(final_ee.reshape(21, 3)[i, :])
        tmp_sphere.paint_uniform_color([1, 0, 0])
        sphere_list.append(tmp_sphere)

    # loadd o3d coordinate
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # visualize
    o3d.visualization.draw_geometries([mesh, obj_mesh, coord] + sphere_list)

def compute_contact(hand_param, obj_mesh_o3d):
    hand_param = torch.from_numpy(hand_param)
    mano_layer = ManoLayer(mano_root='manopth/mano/models', flat_hand_mean=True, use_pca=False)
    
    # compute final joint position
    verts, joints = mano_layer.forward(th_trans=hand_param[0, :, 48:51], th_pose_coeffs=hand_param[0, :, 0:48])
    full_joints = torch.zeros([21, 3])
    
    verts /= 1000.0
    joints /= 1000.0

    # convert joint order
    joints = joints[0]
    full_joints[0] = joints[0] # wrist
    full_joints[1:5] = joints[5:9] # forefinger
    full_joints[5:9] = joints[9:13] # middlefinger
    full_joints[9:13] = joints[17:21] # pinky
    full_joints[13:17] = joints[13:17] # ringfinger
    full_joints[17:21] = joints[1:5] # thumb

    final_ee = full_joints.reshape(-1).detach().numpy()

    # visualize scene
    # visualize_scene(mano_layer, verts, obj_mesh_o3d, final_ee)

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
    
# process blocks
def process_blocks(block_name):
    # load open3d object mesh
    obj_file = f'./DexGraspNet/meshdata/{block_name}/coacd/decomposed.obj'
    obj_mesh_o3d = o3d.io.read_triangle_mesh(obj_file)
    obj_mesh_o3d.remove_duplicated_vertices()
    
    block_pose = np.array([0, 0, 0, 0, 0, 0, 1])

    return block_pose, obj_mesh_o3d

# run
def run(root_dir):
    out_dict = {}
    
    data_pth_list = sorted(glob.glob(root_dir))[3:4] # modify to change block type

    for type_id, pth in enumerate(data_pth_list):
        out_dict[type_id] = {}
        mesh_name = os.path.basename(pth).replace('.npy', '')
        data_all = np.load(pth, allow_pickle=True)

        print()
        print('-' * 20)
        print(f'Processing {mesh_name}:')
        print()

        for data_id, data in enumerate(tqdm(data_all)):
            out_dict[type_id][data_id] = {}
            # print('-' * 20)
            # print(data_id)
            # print()

            init_theta = np.concatenate([data['qpos_st']['rot'],
                                         data['qpos_st']['thetas'],
                                         data['qpos_st']['trans']]).reshape(1, 1, 51).astype(np.float32)
            grasp_theta = np.concatenate([data['qpos']['rot'],
                                          data['qpos']['thetas'],
                                          data['qpos']['trans'],]).reshape(1, 1, 51).astype(np.float32)
            
            # process blocks
            block_pose, obj_mesh_o3d = process_blocks(mesh_name)
            
            # process mano
            final_ee, target_contacts = compute_contact(grasp_theta.copy(), obj_mesh_o3d)
            hand_init_pose = mano_to_handoversim(init_theta.copy())
            hand_grasp_pose = mano_to_handoversim(grasp_theta.copy())
            
            # output to pickle dict
            out_dict[type_id][data_id]["obj_init"] = block_pose.copy()
            out_dict[type_id][data_id]["obj_grasp"] = block_pose.copy()
            
            out_dict[type_id][data_id]["hand_init_pose"] = hand_init_pose.copy()
            out_dict[type_id][data_id]["hand_ref_pose"] = hand_grasp_pose.copy()
            out_dict[type_id][data_id]["hand_ref_position"] = final_ee.copy()
            out_dict[type_id][data_id]["hand_contact"] = target_contacts.copy()
    
    # write file
    with open(f"./dexgraspnet_all.pickle", "wb") as openfile:
        pickle.dump(out_dict, openfile)

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
    def __init__(self, data, ycb_name):
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        # self.num_envs = 20
        self.data = data
        self.ycb_name = ycb_name
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

        self._read_train_data()
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(spacing, 0.75 * spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        asset_file_mano = "urdf/mano/zeros/mano_addtips.urdf"
        asset_file_ycb = f"urdf/ycb/{self.ycb_name}/{self.ycb_name}.urdf"

        # create mano asset
        asset_path_mano = os.path.join(asset_root, asset_file_mano)
        asset_root_mano = os.path.dirname(asset_path_mano)
        asset_file_mano = os.path.basename(asset_path_mano)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        mano_asset = self.gym.load_asset(self.sim, asset_root_mano, asset_file_mano, asset_options)
        self.num_mano_dofs = self.gym.get_asset_dof_count(mano_asset)
        
        # create ycb asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        # asset_path_ycb = os.path.join(asset_root, asset_file_ycb)
        # asset_root_ycb = os.path.dirname(asset_path_ycb)
        # asset_file_ycb = os.path.basename(asset_path_ycb)
        ycb_asset = self.gym.create_box(self.sim, *[0.05, 0.075, 0.05], asset_options)

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

        # set YCB properties
        ycb_rb_props = self.gym.get_asset_rigid_shape_properties(ycb_asset)
        ycb_rb_props[0].rolling_friction = 1

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

            # # create table and set properties
            # table_handle = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0) # 001
            # table_sim_index = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            # self.table_indices.append(table_sim_index)

            # self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(60, 33, 0) / 255)
            '''
            # error ???
            # self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_rb_shape_props)
            '''

            # create ycb and set properties
            ycb_handle = self.gym.create_actor(env_ptr, ycb_asset, handobj_start_pose, "YCB", i, 2, 1) # 010
            ycb_sim_index = self.gym.get_actor_index(env_ptr, ycb_handle, gymapi.DOMAIN_SIM)
            self.ycb_indices.append(ycb_sim_index)

            ycb_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, ycb_handle)
            ycb_rb_props[0].mass = 0.3
            self.gym.set_actor_rigid_body_properties(env_ptr, ycb_handle, ycb_rb_props)
            '''
            # error ???
            # self.gym.set_actor_rigid_shape_properties(env_ptr, ycb_handle, ycb_rb_shape_props)
            '''
            self.ycb_masses.append(ycb_rb_props[0].mass)
            
            # create mano and set properties
            mano_handle = self.gym.create_actor(env_ptr, mano_asset, handobj_start_pose, "mano", i, 2, 2) # 100
            mano_sim_index = self.gym.get_actor_index(env_ptr, mano_handle, gymapi.DOMAIN_SIM)
            self.mano_indices.append(mano_sim_index)

            self.gym.set_actor_dof_properties(env_ptr, mano_handle, mano_dof_props)

        self.mano_indices = to_torch(self.mano_indices, dtype=torch.long, device=self.device)
        self.ycb_indices = to_torch(self.ycb_indices, dtype=torch.long, device=self.device)

        self.ycb_masses = to_torch(self.ycb_masses, dtype=torch.float32, device=self.device)

    def _read_train_data(self):
        self.data = self.data[0]
        # self.good_data = list(range(len(self.data)))
        self.good_data = list(range(100))
        self.data_num = len(self.good_data)
        self.num_envs = len(self.good_data)
        self.data_hand_init = torch.from_numpy(np.array([self.data[i]["hand_init_pose"][0, 0] for i in self.good_data]))
        self.data_hand_ref = torch.from_numpy(np.array([self.data[i]["hand_ref_pose"][0, 0] for i in self.good_data]))
        self.data_obj_init = torch.from_numpy(np.array([self.data[i]["obj_init"] for i in self.good_data]))
        
        # to torch
        self.data_hand_init = to_torch(self.data_hand_init, dtype=torch.float32, device=self.device)
        self.data_hand_ref = to_torch(self.data_hand_ref, dtype=torch.float32, device=self.device)
        self.data_obj_init = to_torch(self.data_obj_init, dtype=torch.float32, device=self.device)

        # add table shift
        self.data_hand_init[:, 2] += 0.5
        self.data_hand_ref[:, 2] += 0.5
        self.data_obj_init[:, 2] += 0.5

    def reset_idx(self):
        # reset hand root pose
        hand_ref_pose = torch.tile(self.data_hand_ref, (self.num_envs // self.data_num, 1)).clone()

        # self.root_state_tensor[self.mano_indices, 0:3] = hand_init_pose[:, 0:3]
        # self.root_state_tensor[self.mano_indices, 3:7] = euler_to_quat(hand_init_pose[:, 3:6], "XYZ")
        # self.root_state_tensor[self.mano_indices, 7:13] = torch.zeros_like(self.root_state_tensor[self.mano_indices, 7:13])
        # init_hand_indices = self.mano_indices.to(torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(init_hand_indices), len(init_hand_indices))

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

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

        # for cnt, i in enumerate(self.good_data):
        #     tmp_pose = new_obj_pose[cnt].numpy().copy()
        #     tmp_pose[2] -= 0.5

        #     self.data[i]["subgoal_1"]["obj_init"][:] = tmp_pose

        # # write file
        # with open(f"./{ycb_name}_ft.pickle", "wb") as openfile:
        #     pickle.dump(self.data, openfile)

if __name__ == '__main__':
    root_dir = './DexGraspNet/demo_0416/results/*'

    out_dict = run(root_dir)
    
    # read file
    with open(f"./dexgraspnet_all.pickle", "rb") as openfile:
        out_dict = pickle.load(openfile)

    print('-' * 20)
    print('Success dump the data pickle')
    print()

    # visualize in isaac
    issac = IsaacSim(out_dict, None)
    issac.simulate()

    print('-' * 20)
    print('Success dump the fine tune data pickle')
    print()
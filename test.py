
import os
import math
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def visualize():
    objs = ['square', 'triangle', 'rectangle', 'hexagon']
    ycb_pose = np.load('/home/hcis-s12/Desktop/M-HRI/gen_hand_pose/obj_pose.npy')
    vis = o3d.visualization.Visualizer()    
    vis.create_window()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    for idx, obj in enumerate(objs):

        obj_file = f"/home/hcis-s12/Desktop/M-HRI/assets/urdf/peg_insertion/{obj}.obj"
        obj_mesh = o3d.io.read_triangle_mesh(obj_file)
        obj_mesh.remove_duplicated_vertices()

        obj_pos_mat = np.eye(4)
        obj_pos_mat[:3, :3] = Rot.from_quat(ycb_pose[idx][3:]).as_matrix()
        obj_pos_mat[:3, 3] = ycb_pose[idx][:3].T
        obj_mesh.scale(10, center=obj_mesh.get_center())
        obj_mesh.transform(obj_pos_mat)

        vis.add_geometry(obj_mesh)


    vis.add_geometry(coord)
    vis.run()
    vis.destroy_window()

def main():
    mesh = o3d.io.read_triangle_mesh("assets/urdf/peg_insertion/rectangle.obj")
    
    # mano_mesh = o3d.io.read_triangle_mesh("gen_hand_pose/DexGraspNet/brick/A/textured_simple.obj")
    
    # aabb = mesh.get_axis_aligned_bounding_box()
    # mano_aabb = mano_mesh.get_axis_aligned_bounding_box()
    # print(aabb)
    # print(mano_aabb)

    
    visualize()

main()
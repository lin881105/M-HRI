
import os
import math
import open3d as o3d

def visualize(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

def main():
    mesh = o3d.io.read_triangle_mesh("assets/urdf/block_assembly/type_A.obj")
    mano_mesh = o3d.io.read_triangle_mesh("gen_hand_pose/DexGraspNet/brick/A/textured_simple.obj")
    
    aabb = mesh.get_axis_aligned_bounding_box()
    mano_aabb = mano_mesh.get_axis_aligned_bounding_box()
    print(aabb)
    print(mano_aabb)

    
    # visualize(mesh)

main()
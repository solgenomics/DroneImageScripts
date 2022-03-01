# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/Open3DConvertPLY.py --input_ply_path $file --output_obj_path $outfile.obj --output_pcd_path $outfile.pcd

# import the necessary packages
import argparse
import numpy as np
import csv
import open3d as o3d

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_ply_path", required=True, help="point cloud (.ply) input file")
ap.add_argument("-o", "--output_obj_path", required=True, help="output file (.obj) for point cloud")
ap.add_argument("-p", "--output_pcd_path", required=True, help="output file (.pcd) for point cloud")
ap.add_argument("-q", "--output_gltf_path", required=True, help="output file (.gltf) for point cloud")
ap.add_argument("-r", "--output_vertices_path", required=True, help="output file (.csv) for point cloud vertices")
args = vars(ap.parse_args())

input_ply_path = args["input_ply_path"]
output_obj_path = args["output_obj_path"]
output_pcd_path = args["output_pcd_path"]
output_gltf_path = args["output_gltf_path"]
output_vertices_path = args["output_vertices_path"]

pcd = o3d.io.read_point_cloud(input_ply_path)
o3d.io.write_point_cloud(output_pcd_path, pcd)

mesh = o3d.io.read_triangle_mesh(input_ply_path)
o3d.io.write_triangle_mesh(output_obj_path, mesh)
o3d.io.write_triangle_mesh(output_gltf_path, mesh)

verts = mesh.vertices
# print(verts)

with open(output_vertices_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(verts)

writeFile.close()

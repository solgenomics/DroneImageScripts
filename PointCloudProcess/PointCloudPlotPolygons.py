# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/PointCloudProcess/PointCloudPlotPolygons.py --pointcloud_xyz_file /point_cloud.xyz --plot_polygons_ratio_file /plot_ratio_file.tsv

# import the necessary packages
import argparse
import os
import math
import numpy as np
from statistics import mean
import copy
import time
import open3d as o3d
import pandas as pd
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--pointcloud_xyz_file", required=True, help="image path DSM")
ap.add_argument("-j", "--plot_polygons_ratio_file", required=False, default=0.001, help="voxel size for downsampling")
args = vars(ap.parse_args())

pointcloud_xyz_file = args["pointcloud_xyz_file"]
plot_polygons_ratio_file = args["plot_polygons_ratio_file"]

xyz = np.genfromtxt(pointcloud_xyz_file, delimiter=" ")
print(xyz.shape)
# xyz[:, 0] #span
# xyz[:, 1] #forward
# xyz[:, 2] #height

# values from 0 to (max-min)
xyz[:,0] -= np.min(xyz[:,0])
xyz[:,1] -= np.min(xyz[:,1])
xyz[:,2] -= np.min(xyz[:,2])

z_max = np.max(xyz[:,0])
x_max = np.max(xyz[:,1])
y_max = np.max(xyz[:,2])
print([x_max, y_max, z_max])

input_plot_polygons_ratio_file = pd.read_csv(plot_polygons_ratio_file, sep="\t", header=None)

for index, row in input_plot_polygons_ratio_file.iterrows():
    stock_id = row[0]
    temp_file = row[1]
    x1_ratio = row[2]
    z1_ratio = row[3]
    x2_ratio = row[4]
    z2_ratio = row[5]

    x1 = x_max*x1_ratio
    x2 = x_max*x2_ratio
    z1 = z_max*z1_ratio
    z2 = z_max*z2_ratio

    x_indexes = np.where((xyz[:, 1] > x1) & (xyz[:, 1] < x2))
    y_indexes = np.where((xyz[:, 0] > z1) & (xyz[:, 0] < z2))
    plot_indices = np.intersect1d(x_indexes, y_indexes)
    plot_values = xyz[plot_indices]

    np.savetxt(temp_file, plot_values, delimiter=' ', fmt="%10.5f")

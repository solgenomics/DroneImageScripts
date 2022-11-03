# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/PointCloudProcess/PointCloudPlotPolygons.py --pointcloud_xyz_file /point_cloud.xyz --plot_polygons_ratio_file /plot_ratio_file.tsv --phenotype_ouput_file /pheno_out.csv

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
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--pointcloud_xyz_file", required=True, help="xyz file point cloud")
ap.add_argument("-j", "--plot_polygons_ratio_file", required=True, help="file with plot boundary ratios")
ap.add_argument("-k", "--phenotype_ouput_file", required=True, help="file for output of plot phenotypes")
args = vars(ap.parse_args())

pointcloud_xyz_file = args["pointcloud_xyz_file"]
plot_polygons_ratio_file = args["plot_polygons_ratio_file"]
phenotype_ouput_file = args["phenotype_ouput_file"]

xyz = np.genfromtxt(pointcloud_xyz_file, delimiter=" ")
print(xyz.shape)
# xyz[:, 1] #forward
# xyz[:, 2] #height
# xyz[:, 0] #span

# values from 0 to (max-min)
xyz[:,0] -= np.min(xyz[:,0])
xyz[:,1] -= np.min(xyz[:,1])
xyz[:,2] -= np.min(xyz[:,2])

x_max = np.max(xyz[:,1])
y_max = np.max(xyz[:,2])
z_max = np.max(xyz[:,0])
print([x_max, y_max, z_max])

result_file_lines = [['stock_id','num_points','length_max','height_max','span_max','length_min','height_min','span_min','length_average','height_average','span_average','average_volume','length_density','height_density','span_density','average_density']]

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
    z_indexes = np.where((xyz[:, 0] > z1) & (xyz[:, 0] < z2))
    plot_indices = np.intersect1d(x_indexes, z_indexes)
    print(plot_indices.shape)
    plot_values = xyz[plot_indices]

    np.savetxt(temp_file, plot_values, delimiter=' ', fmt="%10.5f")

    plot_x_vals = plot_values[:,1]
    plot_y_vals = plot_values[:,2]
    plot_z_vals = plot_values[:,0]

    plot_points = len(plot_values)

    x_max_plot = None
    y_max_plot = None
    z_max_plot = None
    x_min_plot = None
    y_min_plot = None
    z_min_plot = None
    if plot_x_vals.shape[0] > 0:
        x_max_plot = np.max(plot_x_vals)
        x_min_plot = np.min(plot_x_vals)
    if plot_y_vals.shape[0] > 0:
        y_max_plot = np.max(plot_y_vals)
        y_min_plot = np.min(plot_y_vals)
    if plot_z_vals.shape[0] > 0:
        z_max_plot = np.max(plot_z_vals)
        z_min_plot = np.min(plot_z_vals)

    x_average_plot = np.average(plot_x_vals)
    y_average_plot = np.average(plot_y_vals)
    z_average_plot = np.average(plot_z_vals)

    average_volume_plot = x_average_plot*y_average_plot*z_average_plot

    x_density_area = 0
    if x_max_plot is not None and x_min_plot is not None and z_max_plot is not None and z_min_plot is not None:
        x_density_area = (x_max_plot-x_min_plot)*(z_max_plot-z_min_plot)
    x_density = None
    if x_density_area > 0:
        x_density = plot_points/x_density_area

    y_density_area = 0
    if x_max_plot is not None and x_min_plot is not None and y_max_plot is not None and y_min_plot is not None:
        y_density_area = (x_max_plot-x_min_plot)*(y_max_plot-y_min_plot)
    y_density = None
    if y_density_area > 0:
        y_density = plot_points/y_density_area

    z_density_area = 0
    if z_max_plot is not None and z_min_plot is not None and y_max_plot is not None and y_min_plot is not None:
        z_density_area = (y_max_plot-y_min_plot)*(z_max_plot-z_min_plot)
    z_density = None
    if z_density_area > 0:
        z_density = plot_points/z_density_area

    average_density = None
    if average_volume_plot > 0:
        average_density = plot_points/average_volume_plot

    result_file_lines.append([
        stock_id,
        plot_points,
        x_max_plot,
        y_max_plot,
        z_max_plot,
        x_min_plot,
        y_min_plot,
        z_min_plot,
        x_average_plot,
        y_average_plot,
        z_average_plot,
        average_volume_plot,
        x_density,
        y_density,
        z_density,
        average_density
    ])

with open(phenotype_ouput_file, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(result_file_lines)
writeFile.close()

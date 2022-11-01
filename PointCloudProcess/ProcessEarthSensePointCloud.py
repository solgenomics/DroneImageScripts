# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/PointCloudProcess/ProcessEarthSensePointCloud.py --earthesense_capture_image_path /0239391-asdh-djak-jgj9/ --voxel_size 0.001 --outlier_nb_neighbors 15 outlier_std_ratio 0.05 --mask_infinite True --side_mask_distance 2 --height_mask_distance 0.00001 --height_mask_max_distance 30

# import the necessary packages
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import interpolate
from statistics import mean
import copy
import time
import open3d as o3d
import pandas as pd
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--earthesense_capture_image_path", required=True, help="image path DSM")
ap.add_argument("-j", "--voxel_size", required=False, default=0.001, help="voxel size for downsampling")
ap.add_argument("-k", "--outlier_nb_neighbors", required=False, default=15, help="statistical outlier removal number of neighbors")
ap.add_argument("-l", "--outlier_std_ratio", required=False, default=0.05, help="statistical outlier removal std")
ap.add_argument("-m", "--mask_infinite", required=False, default=True, help="mask points greater than 65m")
ap.add_argument("-n", "--side_mask_distance", required=False, default=2, help="side distance to mask points in meters")
ap.add_argument("-o", "--height_mask_distance", required=False, default=0.00001, help="height distance below which to mask points in meters")
ap.add_argument("-p", "--height_mask_max_distance", required=False, default=20, help="height distance above which to mask points in meters")
args = vars(ap.parse_args())

directory = args["earthesense_capture_image_path"]
voxel_size = float(args["voxel_size"])
outlier_nb_neighbors = int(args["outlier_nb_neighbors"])
outlier_std_ratio = float(args["outlier_std_ratio"])
mask_infinite = args["mask_infinite"]
side_mask_distance = float(args["side_mask_distance"])
height_mask_distance = float(args["height_mask_distance"])
height_mask_max_distance = float(args["height_mask_max_distance"])
figure_size = (16,4)
print(directory)

def d_cos(angles):
    """takes cosine of angle in degrees"""
    return np.cos(np.radians(angles))
def d_sin(angles):
    """sine of angle in degrees"""
    return np.sin(np.radians(angles))
def d_hav(angles):
    """haversine of theta in degrees: sin(theta/2)**2"""
    return (1-d_cos(angles))/2

R = 6373.0*1000 # Earth Radius in meters

temp = np.genfromtxt(os.path.join(directory, "secondary_lidar_log.csv"), delimiter=",")

uptimes = temp[:,0]
data = temp[:,1:]/1000

if mask_infinite == True:
    data_mask = data < 65
    data = data*data_mask

sample_size = data.shape[1]
angles = (225-np.arange(sample_size)*270/sample_size)*np.pi/180
cos = -np.cos(angles)
sin = np.sin(angles)
sides = cos*data
heights = sin*data

sl_temp = np.genfromtxt(os.path.join(directory, "system_log.csv"), delimiter=",", skip_header=2)
sl_uptimes = sl_temp[:,1]
lats = sl_temp[:,22]
lons = sl_temp[:,23]
origin_lat = lats[0]
origin_lon = lons[0]

relative_lats = lats-origin_lat
relative_lons = lons-origin_lon
a = d_hav(relative_lats) + d_cos(origin_lat)*d_cos(lats)*d_hav(relative_lons)
sl_distances = 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

uptime_distance_interp = interpolate.interp1d(sl_uptimes, sl_distances, bounds_error=False, fill_value=(sl_distances[0], sl_distances[-1]))
distances = uptime_distance_interp(uptimes)

min_uptime = min(uptimes)
max_uptime = max(uptimes)
plot_sides = sides #sides[(uptimes >= min_uptime) & (uptimes <= max_uptime),:]
plot_distances = distances #distances[(uptimes >= min_uptime) & (uptimes <= max_uptime)]
plot_heights = heights #heights[(uptimes >= min_uptime) & (uptimes <= max_uptime),:]

points_original_image_file = os.path.join(directory, "points_original.png")
point_cloud_output = os.path.join(directory,"point_cloud.txt")
point_cloud_side_filtered_output = os.path.join(directory,"point_cloud_side_filtered.xyz")
points_filtered_height_image_file = os.path.join(directory, "points_filtered_height.png")
points_filtered_side_span_image_file = os.path.join(directory, "points_filtered_side_span.png")
points_filtered_side_height_image_file = os.path.join(directory, "points_filtered_side_height.png")
output_json_file = os.path.join(directory, "output_log.json")

fig = plt.figure(figsize=figure_size)
plt.pcolormesh(plot_distances, np.arange(plot_sides.shape[1]), np.abs(plot_sides.T), vmin=0, vmax=1, shading="auto")
plt.xticks(np.arange(min(plot_distances), max(plot_distances)+1, 1.0))
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.savefig(points_original_image_file)
print("::Saved points_original.png")

ext_plot_distances = np.tile(plot_distances, (sample_size,1)).T
length = plot_distances.shape[0]
xyz = np.zeros((length, sample_size, 3))
xyz[:, :, 0] = plot_sides # horizontal
xyz[:, :, 1] = ext_plot_distances # forward
xyz[:, :, 2] = plot_heights # vertical

q = np.reshape(xyz, (length*sample_size,3))
np.savetxt(point_cloud_output, q, delimiter=' ', fmt="%10.5f")
print("::Saved point_cloud.txt")

pcd = o3d.io.read_point_cloud(point_cloud_output, format='xyz')
print(pcd)
pcd_original_num_points = len(pcd.points)

pcd_down = pcd.voxel_down_sample(voxel_size)
print(pcd_down)
pcd_down_num_points = len(pcd_down.points)

#pcd_down, pcd_down_ind = pcd_down.remove_radius_outlier(nb_points=16, radius=radius_feature)
pcd_down_filtered, pcd_down_filtered_ind = pcd_down.remove_statistical_outlier(
    nb_neighbors = outlier_nb_neighbors, #was 20
    std_ratio = outlier_std_ratio #lower is more aggressive, was 2.0
)
print(pcd_down_filtered)
pcd_down_filtered_num_points = len(pcd_down_filtered.points)

xyz_filtered = np.asarray(pcd_down_filtered.points)

xyz_filtered_height_mask = xyz_filtered[:,2] > height_mask_distance
xyz_filtered_height = xyz_filtered[xyz_filtered_height_mask]

xyz_filtered_height_mask = xyz_filtered_height[:,2] < height_mask_max_distance
xyz_filtered_height = xyz_filtered_height[xyz_filtered_height_mask]

pcd_down_filtered.points = o3d.utility.Vector3dVector(xyz_filtered_height) # normals and colors are unchanged
print(pcd_down_filtered)
pcd_down_filtered_height_points = len(pcd_down_filtered.points)

xyz_filtered_side_mask = np.abs(xyz_filtered_height[:,0]) < side_mask_distance
xyz_filtered_height = xyz_filtered_height[xyz_filtered_side_mask]

pcd_down_filtered.points = o3d.utility.Vector3dVector(xyz_filtered_height) # normals and colors are unchanged
print(pcd_down_filtered)
pcd_down_filtered_height_side_points = len(pcd_down_filtered.points)

o3d.io.write_point_cloud(point_cloud_side_filtered_output, pcd_down_filtered)
print("::Saved point_cloud_side_filtered.xyz")

fig = plt.figure(figsize=figure_size)
plt.scatter(xyz_filtered_height[:,1], xyz_filtered_height[:,2], s=1)
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.savefig(points_filtered_height_image_file)
print("::Saved points_filtered_height.png")

fig = plt.figure(figsize=figure_size)
plt.scatter(xyz_filtered_height[:,1], xyz_filtered_height[:,0], s=1)
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.savefig(points_filtered_side_span_image_file)
print("::Saved points_filtered_side_span.png")

fig = plt.figure(figsize=figure_size)
plt.scatter(xyz_filtered_height[:,0], xyz_filtered_height[:,2], s=1)
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.savefig(points_filtered_side_height_image_file)
print("::Saved points_filtered_side_height.png")

output_dictionary = {
    "voxel_size" : voxel_size,
    "outlier_nb_neighbors" : outlier_nb_neighbors,
    "outlier_std_ratio" : outlier_std_ratio,
    "mask_infinite" : mask_infinite,
    "side_mask_distance" : side_mask_distance,
    "height_mask_distance" : height_mask_distance,
    "height_mask_max_distance" : height_mask_max_distance,
    "figure_size" : figure_size,
    "points_original_image_file" : points_original_image_file,
    "point_cloud_output" : point_cloud_output,
    "point_cloud_side_filtered_output" : point_cloud_side_filtered_output,
    "points_filtered_height_image_file" : points_filtered_height_image_file,
    "points_filtered_side_span_image_file" : points_filtered_side_span_image_file,
    "points_filtered_side_height_image_file" : points_filtered_side_height_image_file,
    "pcd_original_num_points" : pcd_original_num_points,
    "pcd_down_num_points" : pcd_down_num_points,
    "pcd_down_filtered_num_points" : pcd_down_filtered_num_points,
    "pcd_down_filtered_height_points" : pcd_down_filtered_height_points,
    "pcd_down_filtered_height_side_points" : pcd_down_filtered_height_side_points
}

with open(output_json_file, "w") as outfile:
    json.dump(output_dictionary, outfile)

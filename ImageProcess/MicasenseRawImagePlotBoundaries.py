# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/MicasenseRawImagePlotBoundaries.py --file_with_image_paths /folder/myimages.csv --file_with_panel_image_paths  /folder/mypanels.csv --output_path /export/myresults.csv --field_layout_path layout.csv --field_layout_params params.csv

# import the necessary packages
import sys
import argparse
import csv
import imutils
import cv2
import numpy as np
import math
import json
from PIL import Image
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils
from micasense.image import Image
from micasense.panel import Panel
import micasense.utils as msutils
from micasense.capture import Capture

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-i", "--file_with_image_paths", required=True, help="file with file paths to the Micasense images in order")
ap.add_argument("-p", "--file_with_panel_image_paths", required=True, help="file with file paths to the Micasense panel images in order")
ap.add_argument("-a", "--field_layout_path", required=True, help="file with field layout")
ap.add_argument("-r", "--field_layout_params", required=True, help="file with layout params")
ap.add_argument("-o", "--output_path", required=True, help="file path where the output will be saved")
args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
file_with_image_paths = args["file_with_image_paths"]
file_with_panel_image_paths = args["file_with_panel_image_paths"]
field_layout_path = args["field_layout_path"]
field_layout_params = args["field_layout_params"]
output_path = args["output_path"]

if sys.version_info[0] < 3:
    raise Exception("Must use Python3. Use python3 in your command line.")

if log_file_path is not None:
    sys.stderr = open(log_file_path, 'a')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

imageNamesAll = []
imageTempNames = []
with open(file_with_image_paths) as fp:
    for line in fp:
        imageName, tempImageName = line.strip().split(",")
        imageNamesAll.append(imageName)
        imageTempNames.append(tempImageName)

panelNames = []
with open(file_with_panel_image_paths) as fp:
    for line in fp:
        imageName = line.strip()
        panelNames.append(imageName)

field_layout = []
with open(file_with_image_paths) as fp:
    for line in fp:
        plot_id, plot_name, plot_number = line.strip().split(",")
        field_layout.append([plot_id, plot_name, plot_number])

field_params = []
with open(field_layout_params) as fp:
    for line in fp:
        param = line.strip()
        field_params.append(param)

first_plot_corner = field_params[0] #north_west, north_east, south_west, south_east
second_plot_direction = field_params[1] #north_to_south, south_to_north, east_to_west, west_to_east
first_plot_orientation = field_params[2] #serpentine, zigzag
corners_obj = json.loads(field_params[3])
corner_gps_obj = json.loads(field_params[4])
rotate_angle = float(field_params[5])
num_rows = int(field_params[6])
num_columns = int(field_params[7])
flight_direction = field_params[8] #rows, columns, #DEPRECATED
plot_width_m = float(field_params[9])
plot_length_m = float(field_params[10])
plot_corners_pixels = field_params[11]
gps_precision_to_mm = float(field_params[12])
start_direction = field_params[13] #north_to_south, south_to_north, east_to_west, west_to_east
turn_direction = field_params[14] #north_to_south, south_to_north, east_to_west, west_to_east
geographic_position = field_params[15] #Q1, Q2, Q3, Q4
image_top_direction = field_params[16] #north, south, east, west

panelCap = Capture.from_filelist(panelNames)
if panelCap.panel_albedo() is not None:
    panel_reflectance_by_band = panelCap.panel_albedo()
else:
    panel_reflectance_by_band = [0.58, 0.59, 0.59, 0.54, 0.58] #RedEdge band_index order
panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)

imageNamesDict = {}
for i in imageNamesAll:
    s = i.split("_")
    k = s[-1].split(".")
    if s[-2] not in imageNamesDict:
        imageNamesDict[s[-2]] = {}
    imageNamesDict[s[-2]][k[0]] = i

imageNameCaptures = []
for i in sorted (imageNamesDict.keys()):
    im = []
    for j in sorted (imageNamesDict[i].keys()):
        imageName = imageNamesDict[i][j]
        img = Image(imageName)
        im.append(img)
    if len(im) > 0:
        imageNameCaptures.append(im)

captures = []
for i in imageNameCaptures:
    im = Capture(i)
    captures.append(im)

match_index = 3 # Index of the band. NIR band
max_alignment_iterations = 1000
warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
pyramid_levels = None # for images with RigRelatives, setting this to 0 or 1 may improve alignment

if log_file_path is not None:
    eprint("Aligning images. Depending on settings this can take from a few seconds to many minutes")
else:
    print("Aligning images. Depending on settings this can take from a few seconds to many minutes")

warp_matrices, alignment_pairs = imageutils.align_capture(
    captures[0],
    ref_index = match_index,
    max_iterations = max_alignment_iterations,
    warp_mode = warp_mode,
    pyramid_levels = pyramid_levels,
    multithreaded = True
)

if log_file_path is not None:
    eprint("Finished Aligning, warp matrices={}".format(warp_matrices))
else:
    print("Finished Aligning, warp matrices={}".format(warp_matrices))

rotated_imgs = []
img_gps_locations = []
for x in captures:
    im_aligned = x.create_aligned_capture(
        irradiance_list = panel_irradiance,
        warp_matrices = warp_matrices,
        match_index = match_index,
        warp_mode = warp_mode
    )
    if log_file_path is not None:
        eprint(im_aligned.shape)
    else:
        print(im_aligned.shape)

    img = Image(im_aligned[:,:,match_index])
    latitude = img.latitude
    longitude = img.longitude
    altitude = img.altitude
    img_gps_locations.append([latitude, longitude, altitude])

    rows,cols,d = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_angle,1)
    rotated_img = cv2.warpAffine(img,M,(cols,rows,d))

    rotated_imgs.append(rotated_img)

img_rows_pixels, img_columns_pixels, d = rotated_imgs[0].shape
img_rows_pixels_half = img_rows_pixels/2
img_columns_pixels_half = img_columns_pixels/2

plot_width_pixel_tl = int(plot_corners_pixels['north_west'][1]['x']) - int(plot_corners_pixels['north_west'][0]['x'])
plot_width_pixel_tr = int(plot_corners_pixels['north_east'][1]['x']) - int(plot_corners_pixels['north_east'][0]['x'])
plot_width_pixel_bl = int(plot_corners_pixels['south_west'][1]['x']) - int(plot_corners_pixels['south_west'][0]['x'])
plot_width_pixel_br = int(plot_corners_pixels['south_east'][1]['x']) - int(plot_corners_pixels['south_east'][0]['x'])
plot_length_pixel_tl = int(plot_corners_pixels['north_west'][1]['y']) - int(plot_corners_pixels['north_west'][0]['y'])
plot_length_pixel_tr = int(plot_corners_pixels['north_east'][1]['y']) - int(plot_corners_pixels['north_east'][0]['y'])
plot_length_pixel_bl = int(plot_corners_pixels['south_west'][1]['y']) - int(plot_corners_pixels['south_west'][0]['y'])
plot_length_pixel_br = int(plot_corners_pixels['south_east'][1]['y']) - int(plot_corners_pixels['south_east'][0]['y'])

plot_width_pixel_avg = int((plot_width_pixel_tl + plot_width_pixel_tr + plot_width_pixel_bl + plot_width_pixel_br)/4)
plot_length_pixel_avg = int((plot_length_pixel_tl + plot_length_pixel_tr + plot_length_pixel_bl + plot_length_pixel_br)/4)

plot_width_pixels_per_m = plot_width_pixel_avg/plot_width_m
plot_length_pixels_per_m = plot_length_pixel_avg/plot_length_m
plot_pixels_per_m_avg = int((plot_width_pixels_per_m + plot_length_pixels_per_m)/2)

plot_pixels_per_gps = int(plot_pixels_per_m_avg * gps_precision_to_mm * 1000)

column_width_gps = 0
column_height_gps = 0
row_width_gps = 0
row_height_gps = 0
column_width_pixels = 0
column_height_pixels = 0
row_width_pixels = 0
row_height_pixels = 0
tl_latitude_to_pixel_sign = 1
tl_longitude_to_pixel_sign = 1
tr_latitude_to_pixel_sign = 1
tr_longitude_to_pixel_sign = 1
bl_latitude_to_pixel_sign = 1
bl_longitude_to_pixel_sign = 1
br_latitude_to_pixel_sign = 1
br_longitude_to_pixel_sign = 1

tl_pixel_x_diff = int(corners_obj['north_west']['x']) - img_rows_pixels_half
tl_pixel_y_diff = int(corners_obj['north_west']['y']) - img_columns_pixels_half
tr_pixel_x_diff = int(corners_obj['north_east']['x']) - img_rows_pixels_half
tr_pixel_y_diff = int(corners_obj['north_east']['y']) - img_columns_pixels_half
bl_pixel_x_diff = int(corners_obj['south_west']['x']) - img_rows_pixels_half
bl_pixel_y_diff = int(corners_obj['south_west']['y']) - img_columns_pixels_half
br_pixel_x_diff = int(corners_obj['south_east']['x']) - img_rows_pixels_half
br_pixel_y_diff = int(corners_obj['south_east']['y']) - img_columns_pixels_half

# Q1 is north of 0 and west of 0 e.g. North America
if geographic_position == 'Q1':

    if start_direction == 'west_to_east' and turn_direction == 'north_to_south':
        if image_top_direction == 'north':

            if tl_pixel_x_diff < 0:
                tl_pixel_x_diff = img_rows_pixels_half - int(corners_obj['north_west']['x'])
                tl_longitude_to_pixel_sign = -1
            if tl_pixel_y_diff < 0:
                tl_pixel_y_diff = img_rows_pixels_half - int(corners_obj['north_west']['y'])
            else:
                tl_latitude_to_pixel_sign = -1

            if tr_pixel_x_diff < 0:
                tr_pixel_x_diff = img_rows_pixels_half - int(corners_obj['north_east']['x'])
                tr_longitude_to_pixel_sign = -1
            if tr_pixel_y_diff < 0:
                tr_pixel_y_diff = img_rows_pixels_half - int(corners_obj['north_east']['y'])
            else:
                tr_latitude_to_pixel_sign = -1

            if bl_pixel_x_diff < 0:
                bl_pixel_x_diff = img_rows_pixels_half - int(corners_obj['south_west']['x'])
                bl_longitude_to_pixel_sign = -1
            if bl_pixel_y_diff < 0:
                bl_pixel_y_diff = img_rows_pixels_half - int(corners_obj['south_west']['y'])
            else:
                bl_latitude_to_pixel_sign = -1

            if br_pixel_x_diff < 0:
                br_pixel_x_diff = img_rows_pixels_half - int(corners_obj['south_east']['x'])
                br_longitude_to_pixel_sign = -1
            if br_pixel_y_diff < 0:
                br_pixel_y_diff = img_rows_pixels_half - int(corners_obj['south_east']['y'])
            else:
                br_latitude_to_pixel_sign = -1

            if first_plot_corner == 'north_west' and second_plot_direction == 'west_to_east':
                field_tl_longitude_gps = float(corner_gps_obj['north_west'][1]) + (tl_pixel_x_diff*tl_longitude_to_pixel_sign/plot_pixels_per_gps)
                field_tl_latitude_gps = float(corner_gps_obj['north_west'][0]) + (tl_pixel_y_diff*tl_latitude_to_pixel_sign/plot_pixels_per_gps)
                field_tr_longitude_gps = float(corner_gps_obj['north_east'][1]) + (tr_pixel_x_diff*tr_longitude_to_pixel_sign/plot_pixels_per_gps)
                field_tr_latitude_gps = float(corner_gps_obj['north_east'][0]) + (tr_pixel_y_diff*tr_latitude_to_pixel_sign/plot_pixels_per_gps)
                field_bl_longitude_gps = float(corner_gps_obj['south_west'][1]) + (bl_pixel_x_diff*bl_longitude_to_pixel_sign/plot_pixels_per_gps)
                field_bl_latitude_gps = float(corner_gps_obj['south_west'][0]) + (bl_pixel_y_diff*bl_latitude_to_pixel_sign/plot_pixels_per_gps)
                field_br_longitude_gps = float(corner_gps_obj['south_east'][1]) + (br_pixel_x_diff*br_longitude_to_pixel_sign/plot_pixels_per_gps)
                field_br_latitude_gps = float(corner_gps_obj['south_east'][0]) + (br_pixel_y_diff*br_latitude_to_pixel_sign/plot_pixels_per_gps)
    
                plot_width_top_gps = (field_tr_longitude_gps - field_tl_longitude_gps)/num_columns
                plot_width_bottom_gps = (field_br_longitude_gps - field_bl_longitude_gps)/num_columns
                plot_width_gps_avg = (plot_width_top_gps + plot_width_bottom_gps)/2

                plot_length_left_gps = (field_tl_latitude_gps - field_bl_latitude_gps)/num_rows
                plot_length_right_gps = (field_tr_latitude_gps - field_br_latitude_gps)/num_rows
                plot_length_gps_avg = (plot_length_left_gps + plot_length_right_gps)/2
                
        if image_top_direction == 'west':

            if tl_pixel_x_diff < 0:
                tl_pixel_x_diff = img_rows_pixels_half - int(corners_obj['north_west']['x'])
                tl_latitude_to_pixel_sign = -1
            if tl_pixel_y_diff < 0:
                tl_pixel_y_diff = img_rows_pixels_half - int(corners_obj['north_west']['y'])
                tl_longitude_to_pixel_sign = -1

            if tr_pixel_x_diff < 0:
                tr_pixel_x_diff = img_rows_pixels_half - int(corners_obj['north_east']['x'])
                tr_latitude_to_pixel_sign = -1
            if tr_pixel_y_diff < 0:
                tr_pixel_y_diff = img_rows_pixels_half - int(corners_obj['north_east']['y'])
                tr_longitude_to_pixel_sign = -1

            if bl_pixel_x_diff < 0:
                bl_pixel_x_diff = img_rows_pixels_half - int(corners_obj['south_west']['x'])
                bl_latitude_to_pixel_sign = -1
            if bl_pixel_y_diff < 0:
                bl_pixel_y_diff = img_rows_pixels_half - int(corners_obj['south_west']['y'])
                bl_longitude_to_pixel_sign = -1

            if br_pixel_x_diff < 0:
                br_pixel_x_diff = img_rows_pixels_half - int(corners_obj['south_east']['x'])
                br_latitude_to_pixel_sign = -1
            if br_pixel_y_diff < 0:
                br_pixel_y_diff = img_rows_pixels_half - int(corners_obj['south_east']['y'])
                br_longitude_to_pixel_sign = -1

            if first_plot_corner == 'north_west' and second_plot_direction == 'west_to_east':
                field_tl_longitude_gps = float(corner_gps_obj['north_west'][1]) + (tl_pixel_y_diff*tl_longitude_to_pixel_sign/plot_pixels_per_gps)
                field_tl_latitude_gps = float(corner_gps_obj['north_west'][0]) + (tl_pixel_x_diff*tl_latitude_to_pixel_sign/plot_pixels_per_gps)
                field_tr_longitude_gps = float(corner_gps_obj['north_east'][1]) + (tr_pixel_y_diff*tr_longitude_to_pixel_sign/plot_pixels_per_gps)
                field_tr_latitude_gps = float(corner_gps_obj['north_east'][0]) + (tr_pixel_x_diff*tr_latitude_to_pixel_sign/plot_pixels_per_gps)
                field_bl_longitude_gps = float(corner_gps_obj['south_west'][1]) + (bl_pixel_y_diff*bl_longitude_to_pixel_sign/plot_pixels_per_gps)
                field_bl_latitude_gps = float(corner_gps_obj['south_west'][0]) + (bl_pixel_x_diff*bl_latitude_to_pixel_sign/plot_pixels_per_gps)
                field_br_longitude_gps = float(corner_gps_obj['south_east'][1]) + (br_pixel_y_diff*br_longitude_to_pixel_sign/plot_pixels_per_gps)
                field_br_latitude_gps = float(corner_gps_obj['south_east'][0]) + (br_pixel_x_diff*br_latitude_to_pixel_sign/plot_pixels_per_gps)
    
                plot_width_top_gps = (field_tl_latitude_gps - field_bl_latitude_gps)/num_columns
                plot_width_bottom_gps = (field_br_latitude_gps - field_br_latitude_gps)/num_columns
                plot_width_gps_avg = (plot_width_top_gps + plot_width_bottom_gps)/2

                plot_length_left_gps = (field_tr_longitude_gps - field_tl_longitude_gps)/num_rows
                plot_length_right_gps = (field_br_longitude_gps - field_bl_longitude_gps)/num_rows
                plot_length_gps_avg = (plot_length_left_gps + plot_length_right_gps)/2
                




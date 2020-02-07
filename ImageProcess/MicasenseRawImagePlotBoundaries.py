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


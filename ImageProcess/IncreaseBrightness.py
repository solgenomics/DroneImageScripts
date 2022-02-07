# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/IncreaseBrightness.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_path", required=True, help="where output image will be saved")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]

#print(input_image)
img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
img_shape = img.shape

#img = img
#img = img.astype(np.uint8)

is_single_channel = 1
if len(img_shape) == 3:
    if img_shape[2] == 3:
        is_single_channel = 0

max_pixel_limit = 20
mean_pixel_limit = 85
fixed_pixel_average = 150

if is_single_channel == 1:
    smallest = np.amin(img)
    biggest = np.amax(img)
    avg = np.mean(img)
    pix_range = biggest - smallest
    print([smallest, biggest, avg, pix_range])

    if avg < mean_pixel_limit:
        diff = fixed_pixel_average - avg
        img = img+diff
        img[img<0] = 0
        img[img>255] = 255

cv2.imwrite(outfile_path, img)

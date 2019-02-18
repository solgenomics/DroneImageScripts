# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/VegetativeIndex/NDVI.py --nir_image_path /folder/myNIRpic.png --red_image_path /folder/myRGBpic.png --input_image_type RGB --outfile_path /export/mychoppedimages/tgi.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--nir_image_path", required=True, help="image path for NIR image")
ap.add_argument("-i", "--red_image_path", required=True, help="image path for Red image. Can be a single red image band or an RGB image, but make sure the input_image_type matches the correct one.")
ap.add_argument("-t", "--input_image_type", required=True, help="image type for input of red image. can be either RGB or Red depending on what kind of image goes into red_image_path.")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
args = vars(ap.parse_args())

nir_input_image = args["nir_image_path"]
red_input_image = args["red_image_path"]
input_image_type = args["input_image_type"]
outfile_path = args["outfile_path"]

nir = cv2.imread(nir_input_image, 0)
r
if input_image_type == 'RGB':
    red_img = cv2.imread(red_input_image)
    b,g,r = cv2.split(red_img)
else:
    r = cv2.imread(red_input_image, 0)

numerator = nir - r
denominator = nir + r
ndvi = np.divide(numerator, denominator)
ndvi[np.isnan(ndvi)] = 0

print(ndvi.shape)
print(ndvi.dtype)

ndvi = (ndvi + 1)*255/2
ndvi = ndvi.astype(np.uint8)

print(ndvi.shape)
print(ndvi.dtype)

cv2.imwrite(outfile_path, ndvi)

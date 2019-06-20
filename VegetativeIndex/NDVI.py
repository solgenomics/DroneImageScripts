# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/VegetativeIndex/NDVI.py --image_path /folder/mypic.png --outfile_path /export/myimages/ndvi.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path for image with NIR as first channel and Red as second channel.")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
args = vars(ap.parse_args())

image_path = args["image_path"]
outfile_path = args["outfile_path"]

input_image = cv2.imread(image_path)
#cv2.imshow("img", input_image)
nir,r,x = cv2.split(input_image)

numerator = nir - r
denominator = nir + r
ndvi = np.divide(numerator, denominator)
ndvi[np.isnan(ndvi)] = 0

#print(ndvi.shape)
#print(ndvi.dtype)

ndvi = ndvi * 255
ndvi = ndvi.astype(np.uint8)

#print(ndvi.shape)
#print(ndvi.dtype)

cv2.imwrite(outfile_path, ndvi)
#cv2.waitKey(0)

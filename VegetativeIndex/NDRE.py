# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/VegetativeIndex/NDRE.py --image_path /folder/mypic.png --outfile_path /export/myimages/ndre.png

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
nir,re,x = cv2.split(input_image)

numerator = nir - re
denominator = nir + re
ndre = np.divide(numerator, denominator)
ndre[np.isnan(ndre)] = 0

#print(ndre.shape)
#print(ndre.dtype)

ndre = ndre * 255
ndre = ndre.astype(np.uint8)

#print(ndre.shape)
#print(ndre.dtype)

cv2.imwrite(outfile_path, ndre)
#cv2.waitKey(0)
